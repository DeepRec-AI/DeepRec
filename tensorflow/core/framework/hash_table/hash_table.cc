/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/hash_table/hash_table.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

#include "tensorflow/core/framework/hash_table/status_collector.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/util/env_var.h"

namespace {
static constexpr float kMaxLoadFactor = 0.5;
static constexpr tensorflow::int64 kPartitionBlockSize = 65536;
static constexpr int kPreAllocIds = 256;
static const int64_t kPreseverdEmptyKey =
    tensorflow::random::New64Configuable();
}

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace tensorflow {

bool HashTable::IdsContainer::GetNext(int64* id) {
  if (id_list_.empty()) {
    while (!region_list_.empty() &&
        (region_list_.front().start == region_list_.front().end)) {
      region_list_.pop_front();
    }
    if (region_list_.empty()) {
      return false;
    }
    *id = region_list_.front().start++;
    return true;
  }
  *id = id_list_.front();
  id_list_.pop_front();
  return true;
}

int64 HashTable::IdsContainer::Size() {
  int64 ret = 0;
  ret += id_list_.size();
  for (auto iter = region_list_.begin(); iter != region_list_.end(); ++iter) {
    ret += (iter->end - iter->start);
  }
  return ret;
}

void HashTable::IdsContainer::Clear() {
  id_list_.clear();
  region_list_.clear();
}

void HashTable::IdsAllocator::GetIds(int64 count, IdsContainer* ids) {
  count -= ids->Size();
  if (count <= 0) return;
  if (free_list_.empty()) {
    ids->region_list_.emplace_back();
    ids->region_list_.back().start = counter_;
    counter_ += count;
    ids->region_list_.back().end = counter_;
    return;
  }
  int64 from_list = count <= free_list_.size() ? count : free_list_.size();
  ids->id_list_.insert(ids->id_list_.end(),
      free_list_.begin(), free_list_.begin() + from_list);
  free_list_.erase(free_list_.begin(), free_list_.begin() + from_list);

  int64 left = count - from_list;
  if (left <= 0) return;
  ids->region_list_.emplace_back();
  ids->region_list_.back().start = counter_;
  counter_ += left;
  ids->region_list_.back().end = counter_;
}

void HashTable::IdsAllocator::GetId(int64* id) {
  if (free_list_.empty()) {
    *id = counter_++;
    return;
  }
  *id = free_list_.front();
  free_list_.pop_front();
}

void HashTable::IdsAllocator::Clear() {
  free_list_.clear();
  counter_ = 0;
}

HashTable::HashTable(int num_worker_threads, bool concurrent_read,
    int slice_size, int id_block_size)
  : slice_size_(slice_size), id_block_size_(id_block_size),
    size_(slice_size), concurrent_read_(concurrent_read) {
  num_tables_ = num_worker_threads;
  LOG(INFO) << "HashTable table splits: " << num_tables_;
  table_locks_.resize(num_tables_);
  ids_container_.resize(num_tables_);
  static char print_once = [] {
    LOG(INFO) << "HashTable preserved dense hash map key: " <<
        kPreseverdEmptyKey << " and " << kPreseverdEmptyKey + 1;
    return '\0';
  }();
  for (int64 i = 0; i < num_tables_; ++i) {
    tables_.emplace_back();
    tables_.back().max_load_factor(kMaxLoadFactor);
    tables_.back().set_empty_key(kPreseverdEmptyKey);
    tables_.back().set_deleted_key(kPreseverdEmptyKey + 1);
  }
}

void HashTable::AddTensible(
    TensibleVariable* tensor, std::function<void(Status)> done) {
  tensor->Ref();
  AddTask([this, done, tensor] {
    tensor->Resize(size_, [this, tensor, done] (Status st) {
      if (st.ok()) {
        tensors_.push_back(tensor);
      }
      done(st);
      RunNext();
    });
  });
}

HashTable::~HashTable() {
  for (auto tensor : tensors_) {
    tensor->Unref();
  }
}

void HashTable::PartitionKeys(
    int64* keys, int64* ids, int64 size,
    int64 offset, int64* partitions,
    std::function<void(Status)> done) {
  for (int64 i = 0; i < num_tables_; ++i) {
    *(partitions + i) = -1;
  }
  auto st = Status::OK();
  for (int64 i = 0; i < size; ++i) {
    auto& key = keys[i];
    if (unlikely(key == kPreseverdEmptyKey || key == kPreseverdEmptyKey + 1)) {
      st = errors::InvalidArgument(
          "Input key is preserved key of dense_hash_map, not supported: ", key);
      break;
    }
    int64 table_idx = KeyToTableIdx(&key);
    *(ids + i) = *(partitions + table_idx);
    *(partitions + table_idx) = i + offset;
  }
  done(st);
}

void HashTable::GetIds(
    int64* keys, int32* freqs, int64* ids, int64 size,
    HashTableAdmitStrategy* admit_strategy,
    std::function<void(std::function<void()>)>* runner,
    std::function<void(Status)> done, bool random) {
  if (size == 0) {
    done(Status::OK());
    return;
  }

  int partition_threads = size / kPartitionBlockSize;
  partition_threads = std::min(partition_threads, num_tables_);
  if (partition_threads == 0) partition_threads = 1;
  int64* partitions = new int64[partition_threads * num_tables_];

  // do in one thread when size is small
  if (!runner || size < id_block_size_ * 2) {
    // partition keys
    Status partition_st;
    auto partition_done = [&partition_st](Status st) {
      partition_st = st;
    };
    PartitionKeys(keys, ids, size, 0, partitions, partition_done);
    if (!partition_st.ok()) {
      done(partition_st);
      delete[] partitions;
      return;
    }
    // get ids by keys
    StatusCollector* done_stc = new StatusCollector(num_tables_, done);
    // pick a random start point to decrease multi-thread lock overhead
    int64 start_idx;
    if (likely(random)) {
      start_idx = random::New64() % num_tables_;
    } else {  // for utest
      start_idx = 0;
    }
    int64 counter = num_tables_;
    while (counter--) {
      int64 table_idx = start_idx % num_tables_;
      ++start_idx;
      GetIdsSimple(keys, freqs, ids, size, partitions, 1,
                   table_idx, admit_strategy, done_stc->AddStatusFunc());
    }
    done_stc->Start();
    delete[] partitions;
    return;
  }

  // do get ids
  StatusCollector* done_stc = new StatusCollector(num_tables_,
      [done, partitions] (Status st) {
        delete[] partitions;
        done(st);
      });

  auto do_get_ids = [this, runner, keys, freqs, ids, size, partitions,
      partition_threads, done_stc, admit_strategy](Status st) {
    if (!st.ok()) {
      for (int64 i = 0; i < num_tables_; ++i) {
        done_stc->AddStatusFunc()(st);
      }
    } else {
      // pick a random start point to decrease multi-thread lock overhead
      int64 start_idx = random::New64() % num_tables_;
      int64 counter = num_tables_;
      while (counter--) {
        int64 table_idx = start_idx % num_tables_;
        ++start_idx;
        (*runner)([this, keys, freqs, ids, size, partitions,
            partition_threads, table_idx, admit_strategy, done_stc]{
          GetIdsSimple(keys, freqs, ids, size, partitions,
              partition_threads, table_idx, admit_strategy,
              done_stc->AddStatusFunc());
        });
      }
    }
    done_stc->Start();
  };

  // partition keys to table splits
  StatusCollector* partition_stc = new StatusCollector(partition_threads, do_get_ids);
  if (partition_threads == 1) {
    PartitionKeys(keys, ids, size, 0, partitions,
        partition_stc->AddStatusFunc());
  } else {
    int64* keys_cur = keys;
    int64* ids_cur = ids;
    int64* partitions_cur = partitions;
    int64 size_cur = size;
    while (size_cur > 0) {
      if (size_cur < kPartitionBlockSize * 2) {
        (*runner)([this, keys, keys_cur, ids_cur, size_cur, partitions_cur, partition_stc]{
          PartitionKeys(keys_cur, ids_cur, size_cur, keys_cur - keys,
              partitions_cur, partition_stc->AddStatusFunc());
        });
        size_cur = 0;
      } else {
        (*runner)([this, keys, keys_cur, ids_cur, partitions_cur, partition_stc]{
          PartitionKeys(keys_cur, ids_cur, kPartitionBlockSize, keys_cur - keys,
              partitions_cur, partition_stc->AddStatusFunc());
        });
        keys_cur += kPartitionBlockSize;
        ids_cur += kPartitionBlockSize;
        size_cur -= kPartitionBlockSize;
        partitions_cur += num_tables_;
      }
    }
  }
  partition_stc->Start();
}

void HashTable::GetIdsSimple(
    int64* keys, int32* freqs, int64* ids, int64 size,
    int64* partitions, int64 partition_threads, int64 table_idx,
    HashTableAdmitStrategy* admit_strategy,
    std::function<void(Status)> done) {
  if (concurrent_read_) {
    GetIdsSimpleForConcurrentRead(keys, freqs, ids, size, partitions, partition_threads,
        table_idx, admit_strategy, done);
  } else {
    GetIdsSimpleForExclusiveAccess(keys, freqs, ids, size, partitions, partition_threads,
        table_idx, admit_strategy, done);
  }
}

void HashTable::GetIdsSimpleForConcurrentRead(
    int64* keys, int32* freqs, int64* ids, int64 size,
    int64* partitions, int64 partition_threads, int64 table_idx,
    HashTableAdmitStrategy* admit_strategy,
    std::function<void(Status)> done) {
  // do find
  int64 new_id_list = -1;
  int64 new_id_size = 0;
  int64 sizex = 0;
  int64 cur_idx;
  {
    tf_shared_lock rlock(table_locks_[table_idx]);
    for (int64 i = 0; i < partition_threads; ++i) {
      int64 next_idx = *(partitions + table_idx);
      partitions += num_tables_;
      while(next_idx != -1) {
        cur_idx = next_idx;
        next_idx = ids[next_idx];
        auto iter = tables_[table_idx].find(keys[cur_idx]);
        if (iter != tables_[table_idx].end() && iter->second != kNotAdmitted) {
          ids[cur_idx] = iter->second;
          sizex = std::max(sizex, ids[cur_idx]);
        } else {
          // do admit
          int32_t freq = (freqs == nullptr) ? 1 : freqs[cur_idx];
          if (admit_strategy == nullptr ||
              admit_strategy->Admit(keys[cur_idx], freq)) {
            ids[cur_idx] = new_id_list;
            new_id_list = cur_idx;
            ++new_id_size;
          } else {
            ids[cur_idx] = kNotAdmitted;
          }
        }
      }
    }
  }
  // do alloc ids
  if (new_id_list != -1) {
    mutex_lock wlock(table_locks_[table_idx]);
    {
      mutex_lock lock(update_mu_);
      ids_allocator_.GetIds(new_id_size, &ids_container_[table_idx]);
    }
    while (new_id_list != -1) {
      cur_idx = new_id_list;
      new_id_list = ids[new_id_list];
      auto iter = tables_[table_idx].find(keys[cur_idx]);
      if (iter != tables_[table_idx].end() && iter->second != kNotAdmitted) {
        ids[cur_idx] = iter->second;
      } else {
        CHECK(ids_container_[table_idx].GetNext(&ids[cur_idx])) <<
            "new_id_size: " << new_id_size;
        tables_[table_idx][keys[cur_idx]] = ids[cur_idx];
      }
      sizex = std::max(sizex, ids[cur_idx]);
    }
  }
  sizex = (sizex / slice_size_ + 1) * slice_size_;
  Resize(sizex, done);
}

void HashTable::GetIdsSimpleForExclusiveAccess(
    int64* keys, int32* freqs, int64* ids, int64 size,
    int64* partitions, int64 partition_threads, int64 table_idx,
    HashTableAdmitStrategy* admit_strategy,
    std::function<void(Status)> done) {
  // do find
  int64 sizex = 0;
  int64 cur_idx;
  {
    mutex_lock lock(table_locks_[table_idx]);
    for (int64 i = 0; i < partition_threads; ++i) {
      int64 next_idx = *(partitions + table_idx);
      partitions += num_tables_;
      while(next_idx != -1) {
        cur_idx = next_idx;
        next_idx = ids[next_idx];
        auto iter = tables_[table_idx].find(keys[cur_idx]);
        // item found
        if (iter != tables_[table_idx].end() && iter->second != kNotAdmitted) {
          ids[cur_idx] = iter->second;
          sizex = std::max(sizex, ids[cur_idx]);
          continue;
        }
        // do admit
        int32_t freq = (freqs == nullptr) ? 1 : freqs[cur_idx];
        if (admit_strategy != nullptr &&
            !admit_strategy->Admit(keys[cur_idx], freq)) {
          ids[cur_idx] = kNotAdmitted;
          continue;
        }
        // do alloc ids
        if (!ids_container_[table_idx].GetNext(&ids[cur_idx])) {
          mutex_lock lock(update_mu_);
          ids_allocator_.GetIds(kPreAllocIds, &ids_container_[table_idx]);
          CHECK(ids_container_[table_idx].GetNext(&ids[cur_idx]));
        }
        tables_[table_idx][keys[cur_idx]] = ids[cur_idx];
        sizex = std::max(sizex, ids[cur_idx]);
      }
    }
  }

  sizex = (sizex / slice_size_ + 1) * slice_size_;
  Resize(sizex, done);
}

void HashTable::Resize(int64 size, std::function<void(Status)> done) {
  if (size_ >= size) {
    done(Status::OK());
    return;
  }
  AddTask([size, done, this] {
    if (size_ >= size) {
      done(Status::OK());
      RunNext();
      return;
    } else {
      StatusCollector* stc = new StatusCollector(tensors_.size(),
      [this, size, done] (Status st) {
        if (st.ok()) {
          size_ = std::max(size_.load(), size);
        }
        done(st);
        RunNext();
      });
      for (auto tensor : tensors_) {
        tensor->Resize(size, [stc] (Status st) {
          stc->AddStatus(st);
        });
      }
      stc->Start();
    }
  });
}

void HashTable::AddTask(std::function<void()> task) {
  bool run;
  {
    mutex_lock lock(task_mu_);
    run = tasks_.empty();
    tasks_.push(task);
  }
  if (run) {
    task();
  }
}

void HashTable::RunNext() {
  std::function<void()> task;
  {
    mutex_lock lock(task_mu_);
    tasks_.pop();
    if (!tasks_.empty()) {
      task = tasks_.front();
    }
  }
  if (task) {
    task();
  }
}

void HashTable::ClearAllTask() {
  mutex_lock lock(task_mu_);
  while (!tasks_.empty()) {
    tasks_.pop();
  }
}

void HashTable::DeleteKeys(
    int64* keys, int64* ids, int64 size,
    const std::function<void(Status)>& done) {
}

void HashTable::DeleteKeysSimple(
    int64* keys, int64* ids, int64 size,
    const std::function<void(Status)>& done) {
  {
    mutex_lock lock(update_mu_);
    for (int64 i = 0; i < size; ++i) {
      int64 table_idx =  KeyToTableIdx(keys + i);
      if (tables_[table_idx].erase(keys[i]) && ids[i] != kNotAdmitted) {
        ids_allocator_.FreeId(ids[i]);
      }
    }
  }

  AddTask([done, this, ids, size] {
    StatusCollector* stc = new StatusCollector(
        tensors_.size(), [this, done] (Status st) {
      done(st);
      RunNext();
    });
    for (auto&& tensor: tensors_) {
      tensor->ClearIds(ids, size, [stc](Status st) {
        stc->AddStatus(st);
      });
    }

    stc->Start();
  });
}

void HashTable::Clear(const std::function<void(Status)>& done) {
  AddTask([this, done] {
    {
      mutex_lock lock(update_mu_);
      for (auto&& tensor : tensors_) {
        tensor->Clear();
      }
      for (int64 i = 0; i < num_tables_; ++i) {
        tables_[i].clear();
      }
      ids_allocator_.Clear();
      for (int64 i = 0; i < num_tables_; ++i) {
        ids_container_[i].Clear();
      }
      size_ = 0;
    }
    done(Status::OK());
    ClearAllTask();
  });
}

std::vector<std::pair<int64, int64>> HashTable::Snapshot() {
  mutex_lock lock(update_mu_);
  int64 size = size_;
  std::vector<std::pair<int64, int64>> ret;
  for (int64 i = 0; i < num_tables_; ++i) {
    for (auto iter = tables_[i].begin(); iter != tables_[i].end(); ++iter) {
      if (iter->second < size) {
        ret.emplace_back(iter->first, iter->second);
      }
    }
  }
  return ret;
}

void HashTable::Snapshot(std::vector<int64>* keys, 
                         std::vector<int64>* ids) {
  mutex_lock lock(update_mu_);
  int64 size = size_;
  keys->reserve(size);
  ids->reserve(size);
  for (int64 i = 0; i < num_tables_; ++i) {
    for (auto iter = tables_[i].begin(); iter != tables_[i].end(); ++iter) {
      if (iter->second < size) {
        keys->push_back(iter->first);
        ids->push_back(iter->second);
      }
    }
  }
}

int64 HashTable::GetIdsWithoutResize(int64* keys, int64* ids, int64 size) {
  mutex_lock lock(update_mu_);
  for (int64 i = 0; i < size; i++) {
    int64 table_idx = KeyToTableIdx(keys + i);
    auto iter = tables_[table_idx].find(keys[i]);
    if (iter != tables_[table_idx].end() && iter->second != kNotAdmitted) {
      ids[i] = iter->second;
    } else {
      int64 new_id;
      ids_allocator_.GetId(&new_id);
      size_++;
      tables_[table_idx][keys[i]] = new_id;
      ids[i] = new_id;
    }
  }
  return size_;
}

namespace {
constexpr int64 kIndexLen = 52;
constexpr int64 kIndexBase = 0xFFFFFFFFFFFFF;

bool Match(int64 key, int64 index) {
  return (key >> kIndexLen) == index;
}

int64 Encode(int64 key, int64 index) {
  return (key & kIndexBase) | (index << kIndexLen);
}

int64 Decode(int64 key) {
  return (key & kIndexBase);
}

}  // namespace

string CoalescedHashTable::ChildName(const string& name) {
  if (str_util::EndsWith(name, "/ids")) {
    return name.substr(0, name.size() - 4);
  }
  return name;
}

Status CoalescedHashTable::ValidChild(const string& name) {
  string child_name = ChildName(name);
  if (index_map_.find(child_name) == index_map_.end()) {
    return errors::NotFound("Child HashTable ", child_name, " not found");
  }
  return Status::OK();
}

// TODO(chihan.hs): extract all children in one traverse
Status CoalescedHashTable::ChildSnapshot(
    const string& name, std::vector<std::pair<int64, int64>>* output) {
  mutex_lock lock(update_mu_);
  int64 size = size_;
  string child_name = ChildName(name);
  int64 index = index_map_[child_name];
  for (int64 i = 0; i < num_tables_; ++i) {
    for (auto iter = tables_[i].begin(); iter != tables_[i].end(); ++iter) {
      if (iter->second < size && Match(iter->first, index)) {
        output->emplace_back(Decode(iter->first), iter->second);
      }
    }
  }
  return Status::OK();
}

std::function<void(int64*,size_t)> CoalescedHashTable::MakeReviserFn(
    const string& name) {
  string child_name = ChildName(name);
  int64 index = index_map_[child_name];
  auto fn = [index] (int64* keys, size_t sz) {
    for (size_t i = 0; i < sz; ++i) {
      keys[i] = Encode(keys[i], index);
    }
  };
  return fn;
}

void CoalescedHashTable::ClearChildren(
    const std::vector<string>& table_names,
    const std::function<void(Status)>& done) {
  std::unordered_set<int64> index_set;
  for (auto&& table_name : table_names) {
    string child_name = ChildName(table_name);
    if (index_map_.find(child_name) == index_map_.end()) {
      continue;
    }
    index_set.insert(index_map_[child_name]);
  }

  auto match_fn = [index_set] (int64 id) {
    return index_set.find((id >> kIndexLen)) != index_set.end();
  };

  AddTask([this, done, match_fn] {
    ClearMatchedKeys(match_fn, [this, done](Status st) {
      done(st);
      ClearAllTask();
    });
  });
}

void CoalescedHashTable::ClearMatchedKeys(
    const std::function<bool(int64)>& match,
    const std::function<void(Status)>& done) {
  mutex_lock lock(update_mu_);
  std::vector<int64> ids;
  for (int64 i = 0; i < num_tables_; ++i) {
    for (auto iter = tables_[i].begin(); iter != tables_[i].end(); ) {
      if (match(iter->first)) {
        ids.push_back(iter->second);
        iter = tables_[i].erase(iter);
        if (ids.back() != kNotAdmitted) {
          ids_allocator_.FreeId(ids.back());
        }
      } else {
        ++iter;
      }
    }
  }
  StatusCollector* stc = new StatusCollector(tensors_.size(), done);
  for (auto&& tensor : tensors_) {
    tensor->ClearIds(ids.data(), ids.size(), stc->AddStatusFunc());
  }
  stc->Start();
}

}  // namespace tensorflow
