/* Copyright 2022 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_HASH_KV_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_HASH_KV_H_

#include <map>
#include <vector>
#include <cstdlib>

#include "sparsehash/dense_hash_map_lockless"
#include "sparsehash/dense_hash_set_lockless"
#include "tensorflow/core/framework/embedding/emb_file_creator.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

template <class V>
class ValuePtr;

template <class K>
struct SsdRecordDescriptor {
  //prefix of embedding file
  tstring file_prefix;
  //keys in ssd storage
  std::vector<K> key_list;
  //file ids of features
  std::vector<int64> key_file_id_list;
  //offsets in the file of features
  std::vector<int64> key_offset_list;
   //files in ssd storage
  std::vector<int64> file_list;
  //number of invalid records in the file
  std::vector<int64> invalid_record_count_list;
  //number of records in the file
  std::vector<int64> record_count_list;
};

namespace embedding {
class EmbPosition {
 public:
  EmbPosition(int o, size_t v, int bo, bool f)
      : offset_(o), version_(v), buffer_offset_(bo), flushed_(f),
        invalid_(false) {}

  EmbPosition()
      : offset_(-1),
        version_(-1),
        buffer_offset_(-1),
        flushed_(false),
        invalid_(false) {}

  void Print() {
    LOG(INFO) << "EmbPosition: "
              << "offset = " << offset_
              << ", version = " << version_
              << ", buffer_offset = " << buffer_offset_
              << ", flushed = " << flushed_;
  }
 public:
  int offset_;
  int buffer_offset_;
  size_t version_;
  bool flushed_;
  bool invalid_;
};

template <class K>
class SSDIterator : public Iterator {
 public:
  SSDIterator(google::dense_hash_map_lockless<K, EmbPosition*>* hash_map,
              const std::vector<EmbFile*>& emb_files, int64 value_len,
              char* write_buffer)
      : emb_files_(emb_files),
        curr_file_(0),
        curr_vec_(0),
        value_len_(value_len),
        write_buffer_(write_buffer) {
    for (auto it : *hash_map) {
      EmbPosition* posi = it.second;
      auto iter = file_map_.find(posi->version_);
      if (iter == file_map_.end()) {
        std::vector<std::pair<K, EmbPosition*>> tmp;
        file_map_[posi->version_] = tmp;
        file_id_vec_.emplace_back(posi->version_);
      }
      file_map_[posi->version_].emplace_back(it);
    }
  }

  virtual ~SSDIterator() {}

  virtual bool Valid() {
    return !(curr_file_ == file_id_vec_.size());
  }

  virtual void SeekToFirst() {
    curr_file_ = 0;
    curr_vec_ = 0;
    if (file_id_vec_.size() > 0) {
      int64 f_id = file_id_vec_[curr_file_];
      emb_files_[f_id]->MapForRead();
    }
  }

  virtual void Next() {
    curr_vec_++;
    int64 f_id = file_id_vec_[curr_file_];
    if (curr_vec_ == file_map_[f_id].size()) {
      emb_files_[f_id]->UnmapForRead();
      curr_vec_ = 0;
      curr_file_++;
      if (curr_file_ < file_id_vec_.size())
        emb_files_[file_id_vec_[curr_file_]]->MapForRead();
    }
  }

  virtual void Key(char* val, int64 dim) {
    int64 f_id = file_id_vec_[curr_file_];
    memcpy((char*)val, &((file_map_[f_id])[curr_vec_].first), dim);
  }

  virtual void Value(char* val, int64 dim, int64 value_offset) {
    int64 f_id = file_id_vec_[curr_file_];
    EmbPosition* posi = (file_map_[f_id])[curr_vec_].second;
    if (posi->flushed_) {
      emb_files_[posi->version_]->
          ReadWithMemcpy(val, dim,
              posi->offset_ + value_offset + sizeof(FixedLengthHeader));
    } else {
      memcpy(val, write_buffer_ + posi->buffer_offset_ +
          value_offset + sizeof(FixedLengthHeader), dim);
    }
  }

  virtual void Freq(char* val, int64 dim) {
    int64 f_id = file_id_vec_[curr_file_];
    EmbPosition* posi = (file_map_[f_id])[curr_vec_].second;
    if (posi->flushed_) {
      emb_files_[posi->version_]->
          ReadWithMemcpy(val, sizeof(FixedLengthHeader),
              posi->offset_);
    } else {
      memcpy(val, write_buffer_ + posi->buffer_offset_,
             sizeof(FixedLengthHeader));
    }
    *((int64*)val) =
        reinterpret_cast<FixedLengthHeader*>(val)->GetFreqCounter();
  }

  virtual void Version(char* val, int64 dim) {
    int64 f_id = file_id_vec_[curr_file_];
    EmbPosition* posi = (file_map_[f_id])[curr_vec_].second;

    if (posi->flushed_) {
      emb_files_[posi->version_]->
          ReadWithMemcpy(val, sizeof(FixedLengthHeader),
              posi->offset_);
    } else {
      memcpy(val, write_buffer_ + posi->buffer_offset_,
             sizeof(FixedLengthHeader));
    }
    *((int64*)val) = 
        reinterpret_cast<FixedLengthHeader*>(val)->GetGlobalStep();
  }

  virtual K Key() {
    int64 f_id = file_id_vec_[curr_file_];
    return (file_map_[f_id])[curr_vec_].first;
  }

  virtual int64 FileId() {
    return file_id_vec_[curr_file_];
  }

  virtual int64 Offset() {
    int64 f_id = file_id_vec_[curr_file_];
    EmbPosition* posi = (file_map_[f_id])[curr_vec_].second;
    return posi->offset_;
  }

 private:
  int64 value_len_;
  int64 curr_file_;
  int64 curr_vec_;
  char* write_buffer_;
  std::map<int64, std::vector<std::pair<K, EmbPosition*>>> file_map_;
  std::vector<int64> file_id_vec_;
  std::vector<EmbFile*> emb_files_;
};

template <class K, class V>
class SSDHashKV : public KVInterface<K, V> {
 public:
  explicit SSDHashKV(const std::string& path, Allocator* alloc)
  : current_version_(0),
    evict_version_(0),
    compaction_version_(0),
    current_offset_(0),
    buffer_cur_(0),
    alloc_(alloc),
    total_app_count_(0),
    val_len_(-1),
    compaction_thread_(nullptr) {
    path_ = io::JoinPath(
        path, "ssd_kv_" + std::to_string(Env::Default()->NowMicros()) + "_");
    hash_map_.max_load_factor(0.8);
    hash_map_.set_empty_key_and_value(-1, nullptr);
    hash_map_.set_counternum(16);
    hash_map_.set_deleted_key(-2);
    evict_file_set_.max_load_factor(0.8);
    evict_file_set_.set_empty_key_and_value(-1, -1);
    evict_file_set_.set_counternum(16);
    evict_file_set_.set_deleted_key(-2);

    new_value_ptr_fn_ = [this](size_t size) {
      return new NormalContiguousValuePtr<V>(alloc_, size);
    };
    is_async_compaction_ = true;
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_SSDHASH_ASYNC_COMPACTION", true,
          &is_async_compaction_));

    std::string io_scheme = "mmap_and_madvise";
    TF_CHECK_OK(ReadStringFromEnvVar(
        "TF_SSDHASH_IO_SCHEME", "mmap_and_madvise", &io_scheme));
    emb_file_creator_ =  EmbFileCreatorFactory::Create(io_scheme);
    EmbFile* ef = emb_file_creator_->Create(path_, current_version_, BUFFER_SIZE);
    emb_files_.emplace_back(ef);

    if (!is_async_compaction_) {
      LOG(INFO) <<
        "Use Sync Compactor in SSDHashKV of Multi-tier Embedding Storage!";
      compaction_fn_ = [this](){Compaction();}; 
      check_buffer_fn_ = [this](){CheckBuffer();};
      save_kv_fn_ = [this](K key, const ValuePtr<V>* value_ptr,
          bool is_compaction=false) {
        SaveKV(key, value_ptr, is_compaction);
      };
    } else {
      LOG(INFO) <<
        "Use Async Compactor in SSDHashKV of Multi-tier Embedding Storage!";
      compaction_fn_ = [](){};
      check_buffer_fn_ = [this](){CheckBufferAsync();};
      save_kv_fn_ = [this](K key, const ValuePtr<V>* value_ptr,
          bool is_compaction=false) {
        SaveKVAsync(key, value_ptr, is_compaction);
      };
      compaction_thread_ = Env::Default()->StartThread(
          ThreadOptions(), "COMPACTION", [this]() {
            CompactionThread();
          });
    }
  }

  void SetTotalDims(int total_dims) override {
    total_dims_ = total_dims;
    val_len_ = sizeof(FixedLengthHeader) + total_dims_ * sizeof(V);
    max_app_count_ = BUFFER_SIZE / val_len_;
    write_buffer_ = new char[BUFFER_SIZE];
    unsigned int max_key_count = 1 + int(BUFFER_SIZE / val_len_);
    key_buffer_ = new K[max_key_count];
    done_ = true;
  }

  Iterator* GetIterator() override {
    return new SSDIterator<K>(&hash_map_, emb_files_, val_len_,
        write_buffer_);
  }

  void SetSsdRecordDescriptor(SsdRecordDescriptor<K>* ssd_rec_desc) {
    mutex_lock l(compact_save_mu_);
    auto ssd_iter =
        reinterpret_cast<SSDIterator<K>*>(GetIterator());
    for (ssd_iter->SeekToFirst(); ssd_iter->Valid(); ssd_iter->Next()) {
      ssd_rec_desc->key_list.emplace_back(ssd_iter->Key());
      ssd_rec_desc->key_file_id_list.emplace_back(ssd_iter->FileId());
      ssd_rec_desc->key_offset_list.emplace_back(ssd_iter->Offset());
    }
    ssd_rec_desc->file_prefix = path_;

    for (auto file: emb_files_) {
      if (file->IsDeleted())
        continue;
      ssd_rec_desc->file_list.emplace_back(file->Version());
      ssd_rec_desc->invalid_record_count_list.emplace_back(
          file->InvalidCount());
      ssd_rec_desc->record_count_list.emplace_back(
          file->Count());
    }

    if (buffer_cur_ > 0) {
      if (!is_async_compaction_) {
        emb_files_[current_version_]->Write(write_buffer_,
            buffer_cur_ * val_len_);
        emb_files_[current_version_]->Flush();
        ++current_version_;
        CreateFile(current_version_);
      } else {
        emb_files_[evict_version_]->Write(write_buffer_,
            buffer_cur_ * val_len_);
        emb_files_[evict_version_]->Flush();
        evict_version_ = ++current_version_;
        CreateFile(evict_version_);
      }
      TF_CHECK_OK(UpdateFlushStatus());
      current_offset_ = 0;
      buffer_cur_ = 0;
    }
  }

  ~SSDHashKV() override {
    if (buffer_cur_ > 0) {
      if (!is_async_compaction_) {
        emb_files_[current_version_]->Write(write_buffer_,
            buffer_cur_ * val_len_);
      } else {
        emb_files_[evict_version_]->Write(write_buffer_,
            buffer_cur_ * val_len_);
        mutex_lock l(shutdown_mu_);
        shutdown_ = true;
        // Need last compaction or not???
        // CompactionAsync();
        delete compaction_thread_;
      }
      buffer_cur_ = 0;
    }
    for (auto it : emb_files_) {
      if (!it->IsDeleted()) {
        it->DeleteFile();
      }
      delete it;
    }
    delete[] write_buffer_;
    delete[] key_buffer_;
  }

  Status UpdateFlushStatus() {
    for (int i = 0; i < buffer_cur_; ++i) {
      auto iter = hash_map_.find_wait_free(key_buffer_[i]);
      if (iter.first == EMPTY_KEY) {
        return errors::NotFound("Unable to find Key: ",
            key_buffer_[i], " in SSDHashKV.");
      } else {
        iter.second->flushed_ = true;
      }
    }
    return Status::OK();
  }

  Status Lookup(K key, ValuePtr<V>** value_ptr) override {
    auto iter = hash_map_.find_wait_free(key);
    if (iter.first == EMPTY_KEY) {
      return errors::NotFound("Unable to find Key: ", key, " in SSDHashKV.");
    } else {
      ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
      EmbPosition* posi = iter.second;
      if (posi->flushed_) {
        emb_files_[posi->version_]->Read((char*)(val->GetPtr()),
            val_len_, posi->offset_);
      } else {
        memcpy((char*)val->GetPtr(),
            write_buffer_ + posi->buffer_offset_, val_len_);
      }
      *value_ptr = val;
      posi->invalid_ = true;
      return Status::OK();
    }
  }

  Status Contains(K key) override {
    auto iter = hash_map_.find_wait_free(key);
    if (iter.first == EMPTY_KEY) {
      return errors::NotFound("Unable to find Key: ", key, " in SSDHashKV.");
    } else {
      return Status::OK();
    }
  }

  Status Insert(K key, const ValuePtr<V>* value_ptr) override {
    return Status::OK();
  }

  Status BatchInsert(const std::vector<K>& keys,
                     const std::vector<ValuePtr<V>*>& value_ptrs) override {
    return BatchCommit(keys, value_ptrs);
  }

  Status BatchCommit(const std::vector<K>& keys,
                     const std::vector<ValuePtr<V>*>& value_ptrs) override {
    compaction_fn_();
    __sync_fetch_and_add(&total_app_count_, keys.size());
    for (int i = 0; i < keys.size(); i++) {
      check_buffer_fn_();
      save_kv_fn_(keys[i], value_ptrs[i], false);
      delete value_ptrs[i];
    }
    return Status::OK();
  }

  Status Commit(K key, const ValuePtr<V>* value_ptr) override {
    compaction_fn_();
    __sync_fetch_and_add(&total_app_count_, 1);
    check_buffer_fn_();
    save_kv_fn_(key, value_ptr, false);
    return Status::OK();
  }

  Status Remove(K key) override {
    if (hash_map_.erase_lockless(key)) {
      return Status::OK();
    } else {
      return errors::NotFound("Unable to find Key: ",
          key, " in SSDHashKV.");
    }
  }

  Status GetSnapshot(std::vector<K>* key_list,
                     std::vector<ValuePtr<V>*>* value_ptr_list) override {
    return Status::OK();
  }

  Status GetSnapshot(
      std::vector<K>* key_list,
      std::vector<EmbFile*>* file_list) {
    int64 bucket_count;
    auto it = hash_map_.GetSnapshot();
    auto hash_map_dump = it.first;
    bucket_count = it.second;
    for (int64 j = 0; j < bucket_count; j++) {
      if (hash_map_dump[j].first != LocklessHashMap<K, V>::EMPTY_KEY_
          && hash_map_dump[j].first != LocklessHashMap<K, V>::DELETED_KEY_) {
        key_list->emplace_back(hash_map_dump[j].first);
        file_list->emplace_back(hash_map_dump[j].second);
      }
    }
    //Free the memory of snapshot allocated by hash map.
    free(hash_map_dump);
    return Status::OK();
  }

  void Import(K* key_list, int64* key_file_id_list,
              int64* key_offset_list, int64 num_of_keys,
              std::map<int64, int64>& file_id_map) {
    for (int i = 0; i < num_of_keys; i++) {
      int64 old_file_id = key_file_id_list[i];
      int64 new_file_id = file_id_map[old_file_id];
      EmbPosition* ep =
          new EmbPosition(key_offset_list[i],
                          new_file_id,
                          0, true);
      hash_map_.insert_lockless(std::move(
          std::pair<K, EmbPosition*>(
              key_list[i], const_cast<EmbPosition*>(ep))));
    }
  }

  void CopyEmbFilesFromCkpt(
      int64* file_list, int64* invalid_record_count_list,
      int64* record_count_list, int64 num_of_files,
      const std::string& old_file_prefix) {
    //delete the file created by constructor
    emb_files_[0]->DeleteFile();
    delete emb_files_[0];
    emb_files_.erase(emb_files_.begin());
    for (int64 i = 0; i < num_of_files; i++) {
      std::stringstream ss;
      ss << old_file_prefix << "/" << file_list[i] << ".emb";
      std::string old_file_path = ss.str();
      EmbFile* f =
          emb_file_creator_->Create(path_, current_version_, BUFFER_SIZE);
      ++current_version_;
      f->LoadExistFile(old_file_path,
                       record_count_list[i],
                       invalid_record_count_list[i]);
      emb_files_.emplace_back(f);
      total_app_count_ += record_count_list[i];
    }
    CreateFile(current_version_);
  }

  int64 Size() const override { return hash_map_.size_lockless(); }

  void FreeValuePtr(ValuePtr<V>* value_ptr) override {
    delete value_ptr;
  }

 private:
  void WriteFile(size_t version, size_t curr_buffer_offset) {
    emb_files_[version]->Write(write_buffer_, curr_buffer_offset);
    emb_files_[version]->Flush();
  }

  void CreateFile(size_t version) {
    emb_files_.emplace_back(
        emb_file_creator_->Create(path_, version, BUFFER_SIZE));
  }

  Status FlushAndUpdate(char* value_buffer, K* id_buffer,
                        EmbPosition** pos_buffer, int64& n_ids,
                        std::vector<int64>& invalid_files) {
    {
      mutex_lock l(mu_);
      compaction_version_ = ++current_version_;
      CreateFile(compaction_version_);
    }

    emb_files_[compaction_version_]->Write(value_buffer, n_ids * val_len_);
    emb_files_[compaction_version_]->AddCount(n_ids);
    emb_files_[compaction_version_]->Flush();

    for (int64 i = 0; i < n_ids; i++) {
      auto iter = hash_map_.insert_lockless(std::move(
        std::pair<K, EmbPosition*>(id_buffer[i], nullptr)));
      if ((*(iter.first)).first == EMPTY_KEY) {
        return errors::NotFound("Unable to find Key: ",
            id_buffer[i], " in SSDHashKV.");
      } else {
        size_t offset = i * val_len_;
        EmbPosition* ep = new EmbPosition(offset, compaction_version_,
            offset, true);
        bool flag = __sync_bool_compare_and_swap(
            &((*(iter.first)).second), pos_buffer[i], ep);
        if (!flag) {
          emb_files_[compaction_version_]->AddInvalidCountAtomic(1);
          if (emb_files_[compaction_version_]->IsNeedToBeCompacted()) {
            evict_file_set_.insert_lockless(compaction_version_);
          }
          delete ep;
        } else {
          pos_out_of_date_compact_.emplace_back(pos_buffer[i]);
        }
      }
    }

    for (int i = 0; i < invalid_files.size(); i++) {
      evict_file_set_.erase_lockless(invalid_files[i]);
    }
    invalid_files.clear();
    n_ids = 0;
    return Status::OK();
  }

  void CheckBuffer() {
    size_t curr_buffer_offset = buffer_cur_ * val_len_;
    if (curr_buffer_offset + val_len_ > BUFFER_SIZE) {
      WriteFile(current_version_, curr_buffer_offset);
      if (emb_files_[current_version_]->Count() >= max_app_count_) {
        ++current_version_;
        current_offset_ = 0;
        CreateFile(current_version_);
      }
      TF_CHECK_OK(UpdateFlushStatus());
      buffer_cur_ = 0;
    }
  }

  void CheckBufferAsync() {
    size_t curr_buffer_offset = buffer_cur_ * val_len_;
    if (curr_buffer_offset + val_len_ > BUFFER_SIZE) {
      WriteFile(evict_version_, curr_buffer_offset);
      TF_CHECK_OK(UpdateFlushStatus());
      mutex_lock l(mu_);
      evict_version_ = ++current_version_;
      current_offset_ = 0;
      CreateFile(evict_version_);
      buffer_cur_ = 0;
    }
  }

  void AppendToWriteBuffer(size_t curr_buffer_offset, K key,
                            const ValuePtr<V>* value_ptr) {
    current_offset_ += val_len_;
    memcpy(write_buffer_ + curr_buffer_offset,
        (char*)value_ptr->GetPtr(), val_len_);
    key_buffer_[buffer_cur_] = key;
    ++buffer_cur_;
  }

  void AppendToPositionRecordQueue(EmbPosition* old_posi) {
    //A parameter that can be adjusted in the future
    if (pos_out_of_date_.size() > CAP_INVALID_POS) {
      EmbPosition* posi = pos_out_of_date_.front();
      delete posi;
      pos_out_of_date_.pop_front();
    }
    pos_out_of_date_.emplace_back(old_posi);
  }

  bool UpdatePosition(EmbPosition** pos, EmbPosition* old_posi,
      EmbPosition* new_posi) {
    bool flag = __sync_bool_compare_and_swap(pos, old_posi, new_posi);
    if (flag) {
      AppendToPositionRecordQueue(old_posi);
    }
    return flag;
  }

  void SaveKV(K key, const ValuePtr<V>* value_ptr,
      bool is_compaction = false) {
    size_t curr_buffer_offset = buffer_cur_ * val_len_;
    EmbPosition* ep = new EmbPosition(current_offset_, current_version_,
                                      curr_buffer_offset, false);
    AppendToWriteBuffer(curr_buffer_offset, key, value_ptr);

    auto iter = hash_map_.insert_lockless(std::move(
        std::pair<K, EmbPosition*>(key, const_cast<EmbPosition*>(ep))));
    emb_files_[ep->version_]->AddCount(1);

    if ((*(iter.first)).second != ep) {
      EmbPosition* old_posi = (*(iter.first)).second;
      int64 version = old_posi->version_;
      if (!is_compaction) {
        emb_files_[version]->AddInvalidCount(1);
        //A parameter that can be adjusted in the future
        if (version != current_version_ &&
            emb_files_[version]->IsNeedToBeCompacted()) {
          evict_file_set_.insert_lockless(version);
        }
      }
      UpdatePosition(&((*(iter.first)).second), old_posi, ep);
    }
  }

  void SaveKVAsync(K key, const ValuePtr<V>* value_ptr,
      bool is_compaction = false) {
    size_t curr_buffer_offset = buffer_cur_ * val_len_;
    EmbPosition* ep = new EmbPosition(current_offset_, evict_version_,
                                      curr_buffer_offset, false);

    AppendToWriteBuffer(curr_buffer_offset, key, value_ptr);
    auto iter = hash_map_.insert_lockless(std::move(
        std::pair<K, EmbPosition*>(key, const_cast<EmbPosition*>(ep))));
    emb_files_[ep->version_]->AddCount(1);

    if ((*(iter.first)).second != ep) {
      bool flag = false;
      EmbPosition* old_posi = nullptr;
      do {
        old_posi = (*(iter.first)).second;
        flag = UpdatePosition(&((*(iter.first)).second), old_posi, ep);
      } while (!flag);

      if (!is_compaction) {
        int version = old_posi->version_;
        emb_files_[version]->AddInvalidCountAtomic(1);
        //A parameter that can be adjusted in the future
        if (version != evict_version_ &&
            emb_files_[version]->IsNeedToBeCompacted()) {
          evict_file_set_.insert_lockless(version);
        }
      }
    }
  }

  void DeleteInvalidFiles() {
    for (auto it : evict_file_map_) {
      emb_files_[it.first]->DeleteFile();
    }
    evict_file_map_.clear();
  }

  void DeleteInvalidRecord() {
    for (auto it: pos_out_of_date_compact_) {
      delete it;
    }
    pos_out_of_date_compact_.clear();
  }

  void LookupValidItems() {
    for (auto it : hash_map_) {
      EmbPosition* posi = it.second;
      auto iter = evict_file_map_.find(posi->version_);
      if (iter != evict_file_map_.end()) {
        (*iter).second.emplace_back(it);
      }
    }
  }

  void InitializeEvictMap() {
    for (auto it : evict_file_set_) {
      std::vector<std::pair<K, EmbPosition*>> tmp;
      evict_file_map_[it] = tmp;
      evict_file_set_.erase_lockless(it);
    }
    LookupValidItems();
  }

  void InitializeEvictMapWithoutErase() {
    for (auto it : evict_file_set_) {
      std::vector<std::pair<K, EmbPosition*>> tmp;
      evict_file_map_[it] = tmp;
    }
    LookupValidItems();
  }

  void MoveToNewFile() {
    ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
    for (auto it : evict_file_map_) {
      EmbFile* file = emb_files_[it.first];
      total_app_count_ -= file->InvalidCount();
      file->MapForRead();
      for (auto it_vec : it.second) {
        EmbPosition* posi = it_vec.second;
        file->ReadWithMemcpy((char*)(val->GetPtr()), val_len_,
            posi->offset_);
        CheckBuffer();
        SaveKV(it_vec.first, val, true);
      }
      file->UnmapForRead();
    }
    delete val;
  }

  void MoveToNewFileAsync() {
    char* compact_buffer = new char[BUFFER_SIZE];
    int64 n_ids = 0;
    std::vector<int64> invalid_files;
    unsigned int max_key_count = 1 + int(BUFFER_SIZE / val_len_);
    K* id_buffer = new K[max_key_count];
    EmbPosition** pos_buffer = new EmbPosition*[max_key_count];
    for (auto it : evict_file_map_) {
      EmbFile* file = emb_files_[it.first];
      __sync_fetch_and_sub(&total_app_count_, file->InvalidCount());
      file->MapForRead();
      for (auto it_vec : it.second) {
        EmbPosition* posi = it_vec.second;
        id_buffer[n_ids] = it_vec.first;
        pos_buffer[n_ids] = posi;
        file->ReadWithMemcpy(compact_buffer + val_len_ * n_ids, val_len_,
            posi->offset_);
        n_ids++;
        if (n_ids == max_app_count_) {
          Status st = FlushAndUpdate(compact_buffer, id_buffer,
              pos_buffer, n_ids, invalid_files);
          if(!st.ok()) {
            LOG(WARNING)<<"FLUSH ERROR: "<<st.ToString();
          }
        }
      }
      file->UnmapForRead();
      invalid_files.emplace_back(it.first);
    }
    Status st = FlushAndUpdate(compact_buffer, id_buffer,
        pos_buffer, n_ids, invalid_files);
    if(!st.ok()) {
      LOG(WARNING)<<"FLUSH ERROR: "<<st.ToString();
    }
    delete[] id_buffer;
    delete[] compact_buffer;
    delete[] pos_buffer;
  }

  void Compaction() {
    int64 hash_size = hash_map_.size_lockless();
    //These parameter that can be adjusted in the future
    if (hash_size * 3 / 2 < total_app_count_ ||
        total_app_count_ - hash_size > CAP_INVALID_ID) {
      // delete the evict_files
      DeleteInvalidFiles();
      // Initialize evict_file_map
      InitializeEvictMap();
      // read embeddings and write to new file
      MoveToNewFile();
    }
  }

  void CompactionAsync() {
    int64 hash_size = hash_map_.size_lockless();
    //These parameter that can be adjusted in the future
    if (hash_size * 3 / 2 < total_app_count_ ||
        total_app_count_ - hash_size > CAP_INVALID_ID) {
      DeleteInvalidRecord();
      // delete the evict_files
      DeleteInvalidFiles();
      // Initialize evict_file_map
      InitializeEvictMapWithoutErase();
      // read embeddings and write to new file
      MoveToNewFileAsync();
    }
  }

  void CompactionThread() {
    if (val_len_ == -1) {
      while (!done_) {
      }
    }
    while (!shutdown_) {
      if (shutdown_mu_.try_lock()) {
        if (!shutdown_) {
          mutex_lock l(compact_save_mu_);
          CompactionAsync();
        }
        shutdown_mu_.unlock();
      }
      Env::Default()->SleepForMicroseconds(1000);
    }
  }

  std::string DebugString() const {
    return strings::StrCat("map info size:", Size(),
                           ", map info bucket_count:",
                           hash_map_.load_factor(),
                           ",map info load_factor:",
                           hash_map_.load_factor(),
                           ", map info max_load_factor:",
                           hash_map_.max_load_factor(),
                           ", map info min_load_factor: ",
                           hash_map_.min_load_factor(),
                           ", evict_version: ", evict_version_,
                           ", compaction_version: ", compaction_version_);
  }

 private:
  size_t val_len_;
  volatile size_t current_version_;
  volatile size_t evict_version_;
  volatile size_t compaction_version_;
  volatile size_t current_offset_;
  volatile size_t buffer_cur_;
  size_t total_app_count_;
  size_t max_app_count_;

  char* write_buffer_;
  K* key_buffer_;
  bool is_async_compaction_;
  Allocator* alloc_;

  int total_dims_;
  std::string path_;
  std::function<ValuePtr<V>*(size_t)> new_value_ptr_fn_;

  typedef google::dense_hash_map_lockless<K, EmbPosition*> LockLessHashMap;
  LockLessHashMap hash_map_;
  mutex mu_;
  mutex shutdown_mu_;
  mutex compact_save_mu_;


  static constexpr int EMPTY_KEY = -1;
  static constexpr int CAP_INVALID_POS = 200000;
  static constexpr int CAP_INVALID_ID = 10000000;
  static constexpr size_t BUFFER_SIZE = 1 << 27;

  std::vector<EmbFile*> emb_files_;
  std::deque<EmbPosition*> pos_out_of_date_;
  std::deque<EmbPosition*> pos_out_of_date_compact_;
  typedef google::dense_hash_set_lockless<K> LocklessHashSet;
  LocklessHashSet evict_file_set_;
  std::map<int64, std::vector<std::pair<K, EmbPosition*>>> evict_file_map_;

  Thread* compaction_thread_;
  volatile bool shutdown_ = false;
  volatile bool done_ = false;
  // std::atomic_flag flag_ = ATOMIC_FLAG_INIT; unused

  std::function<void()> compaction_fn_;
  std::function<void()> check_buffer_fn_;
  std::function<void(K, const ValuePtr<V>*, bool)> save_kv_fn_;
  EmbFileCreator* emb_file_creator_;
};

}  // namespace embedding
}  // namespace tensorflow

#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_HASH_KV_H_
