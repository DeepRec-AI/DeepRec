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

#ifndef TENSORFLOW_CORE_FRAMEWORK_HASH_TABLE_HASH_TABLE_H_
#define TENSORFLOW_CORE_FRAMEWORK_HASH_TABLE_HASH_TABLE_H_

#include <memory>
#include <queue>
#include <string>
#include <stdint.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparsehash/dense_hash_map"
#include "tensorflow/core/framework/hash_table/tensible_variable.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

class HashTableAdmitStrategy {
 public:
  virtual ~HashTableAdmitStrategy() {}
  virtual bool Admit(int64 key) = 0;
  virtual bool Admit(int64 key, int64 freq) { return Admit(key); }
};

class HashTable {
 public:
  explicit HashTable(int num_worker_threads, bool concurrent_read,
      int slice_size = kSliceSize, int id_block_size = kIdBlockSize);
  virtual ~HashTable();
  void AddTensible(TensibleVariable* tensor, std::function<void(Status)> done);
  void GetIds(
      int64* keys, int32* freqs, int64* ids, int64 size,
      HashTableAdmitStrategy* admit_strategy,
      std::function<void(std::function<void()>)>* runner,
      std::function<void(Status)> done, bool random = true);
  void DeleteKeys(
      int64* keys, int64* ids, int64 size,
      const std::function<void(Status)>& done);
  void DeleteKeysSimple(
      int64* keys, int64* ids, int64 size,
      const std::function<void(Status)>& done);
  std::vector<std::pair<int64, int64>> Snapshot();
  void Snapshot(std::vector<int64>* keys, std::vector<int64>* ids);
  int64 GetIdsWithoutResize(int64* keys, int64* ids, int64 size);

  int64 Size() { return size_; }

  void Clear(const std::function<void(Status)>& done);

  const std::vector<TensibleVariable*>& Tensibles() { return tensors_; }

  static constexpr int kNotAdmitted = -1;

 protected:
  void PartitionKeys(
      int64* keys, int64* ids, int64 size,
      int64 offset,
      int64* partitions,
      std::function<void(Status)> done);
  void GetIdsSimple(
      int64* keys, int32* freqs, int64* ids, int64 size,
      int64* partitions, int64 partition_threads, int64 table_idx,
      HashTableAdmitStrategy* admit_strategy,
      std::function<void(Status)> done);
  void GetIdsSimpleForConcurrentRead(
      int64* keys, int32* freqs, int64* ids, int64 size,
      int64* partitions, int64 partition_threads, int64 table_idx,
      HashTableAdmitStrategy* admit_strategy,
      std::function<void(Status)> done);
  void GetIdsSimpleForExclusiveAccess(
      int64* keys, int32* freqs, int64* ids, int64 size,
      int64* partitions, int64 partition_threads, int64 table_idx,
      HashTableAdmitStrategy* admit_strategy,
      std::function<void(Status)> done);
  void Resize(int64 size, std::function<void(Status)> done);

  void AddTask(std::function<void()> task);
  void RunNext();
  void ClearAllTask();

  inline int64 KeyToTableIdx(int64* key) {
    static IdHash hasher;
    return (hasher(*key) >> 54) % num_tables_;
  }

  static constexpr int kSliceSize = 64 << 10;
  static constexpr int kIdBlockSize = 65536;

  int slice_size_;
  int id_block_size_;

  int num_tables_{0};
  mutable std::vector<mutex> table_locks_;
  struct IdHash : public std::hash<int64>
  {
      inline std::size_t operator()(int64 const& i) const noexcept {
        size_t x = (i ^ (i >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        return x;
      }
  };
  std::vector<google::dense_hash_map<int64, int64, IdHash>> tables_;

  class IdsContainer {
   public:
    bool GetNext(int64* id);
    int64 Size();
    void Clear();

    std::deque<int64> id_list_;
    struct Region {
      int64 start;
      int64 end;
    };
    std::deque<Region> region_list_;
  };
  std::vector<IdsContainer> ids_container_;

  class IdsAllocator {
   public:
    void GetIds(int64 count, IdsContainer* ids);
    inline void GetId(int64* id);
    inline void FreeId(int64 id) {
      free_list_.push_back(id);
    }
    void Clear();

   private:
    std::deque<int64> free_list_;
    std::atomic<int64> counter_ = {0};
  };
  mutex update_mu_;
  IdsAllocator ids_allocator_;

  mutex task_mu_;
  std::queue<std::function<void()>> tasks_;
  std::atomic<int64> size_;

  std::vector<TensibleVariable*> tensors_;

  const bool concurrent_read_;
};

class CoalescedHashTable : public HashTable {
 public:
  explicit CoalescedHashTable(int num_worker_threads, bool concurrent_read,
                              const std::vector<string>& children)
    : HashTable(num_worker_threads, concurrent_read), children_(children) {
    for (size_t i = 0; i < children.size(); ++i) {
      index_map_[children[i]] = static_cast<int64>(i);
    }
  }
  const std::vector<string>& children() const { return children_; }
  const std::unordered_map<string, int64>& index_map() const {
    return index_map_;
  }
  string ChildName(const string& name);
  Status ValidChild(const string& name);
  Status ChildSnapshot(const string& name,
                       std::vector<std::pair<int64, int64>>* output);
  std::function<void(int64*,size_t)> MakeReviserFn(const string& name);
  void ClearChildren(const std::vector<string>& table_names,
                     const std::function<void(Status)>& done);
 private:
  void ClearMatchedKeys(const std::function<bool(int64)>& match,
                        const std::function<void(Status)>& done);
  std::vector<string> children_;
  std::unordered_map<string, int64> index_map_;
};

class HashTableResource : public ResourceBase {
 public:
  HashTableResource(int num_worker_threads, bool concurrent_read,
                    const std::vector<string>& children) :
      internal_(nullptr), initialized_(false),
      num_worker_threads_(num_worker_threads),
      concurrent_read_(concurrent_read),
      children_(children) { }

  ~HashTableResource() {
    delete internal_;
  }

  string DebugString() const override {
    return "HashTableResource";
  }

  Status CreateInternal() {
    mutex_lock lock(init_mu_);
    initialized_ = false;
    if (internal_ != nullptr) {
      return errors::FailedPrecondition("HashTable has been initialized");
    }
    if (children_.empty()) {
      internal_ = new HashTable(num_worker_threads_, concurrent_read_);
    } else {
      internal_ = new CoalescedHashTable(num_worker_threads_, concurrent_read_,
                                         children_);
    }
    return Status::OK();
  }

  HashTable* Internal() {
    return internal_;
  }

  bool Initialized() {
    mutex_lock lock(init_mu_);
    return initialized_;
  }

  void SetInitialized(bool initialized) {
    mutex_lock lock(init_mu_);
    initialized_ = initialized;
  }

 private:
  mutex init_mu_;
  HashTable* internal_;
  bool initialized_;
  int num_worker_threads_;
  bool concurrent_read_;
  std::vector<string> children_;
};

class HashTableAdmitStrategyResource : public ResourceBase {
 public:
  explicit HashTableAdmitStrategyResource()
    : internal_(nullptr), initialized_(false) {}

  explicit HashTableAdmitStrategyResource(HashTableAdmitStrategy* internal)
    : internal_(internal) {}

  virtual string DebugString() const {
    return "HashTableAdmitStrategyResource";
  }

  HashTableAdmitStrategy* Internal() const { return internal_.get(); }

  Status CreateInternal(HashTableAdmitStrategy* strategy) {
    mutex_lock lock(mu_);
    initialized_ = false;
    if (internal_.get() != nullptr) {
      return errors::FailedPrecondition("BloomFilter has been initialized");
    }
    internal_.reset(strategy);
    return Status::OK();
  }

  bool Initialized() {
    mutex_lock lock(mu_);
    return initialized_;
  }

  void SetInitialized(bool initialized) {
    mutex_lock lock(mu_);
    initialized_ = initialized;
  }

 private:
  mutex mu_;
  std::unique_ptr<HashTableAdmitStrategy> internal_;
  bool initialized_ = false;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_HASH_TABLE_HASH_TABLE_H_
