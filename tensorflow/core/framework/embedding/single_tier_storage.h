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
======================================================================*/
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SINGLE_TIER_STORAGE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SINGLE_TIER_STORAGE_H_

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/cpu_hash_map_kv.h"
#include "tensorflow/core/framework/embedding/globalstep_shrink_policy.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/l2weight_shrink_policy.h"
#include "tensorflow/core/framework/embedding/layout_creator.h"
#include "tensorflow/core/framework/embedding/leveldb_kv.h"
#include "tensorflow/core/framework/embedding/ssd_hash_kv.h"
#include "tensorflow/core/framework/embedding/storage_config.h"
#include "tensorflow/core/framework/embedding/storage.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
template <class V>
class ValuePtr;

template <class K, class V>
class EmbeddingVar;

template <class K>
struct SsdRecordDescriptor;

namespace embedding {
template<typename K, typename V>
class SingleTierStorage : public Storage<K, V> {
 public:
  SingleTierStorage(const StorageConfig& sc, Allocator* alloc,
      KVInterface<K, V>* kv, LayoutCreator<V>* lc)
      : kv_(kv), alloc_(alloc), layout_creator_(lc),
        Storage<K, V>(sc) {
  }
  
  ~SingleTierStorage() override {
    mutex_lock l(Storage<K, V>::mu_);
    std::vector<K> key_list;
    std::vector<ValuePtr<V>*> value_ptr_list;
    kv_->GetSnapshot(&key_list, &value_ptr_list);
    for (auto value_ptr : value_ptr_list) {
      value_ptr->Destroy(alloc_);
      delete value_ptr;
    }  
    delete kv_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(SingleTierStorage);

  Status Get(K key, ValuePtr<V>** value_ptr) override {
    return kv_->Lookup(key, value_ptr);
  }

  void Insert(const std::vector<K>& keys,
              ValuePtr<V>** value_ptrs) override{
    for (size_t i = 0; i < keys.size(); i++) {
      do {
        Status s = kv_->Insert(keys[i], value_ptrs[i]);
        if (s.ok()) {
          break;
        } else {
          (value_ptrs[i])->Destroy(alloc_);
          delete value_ptrs[i];
        }
      } while (!(kv_->Lookup(keys[i], &value_ptrs[i])).ok());
    }
  }

  void Insert(K key, ValuePtr<V>** value_ptr,
              int64 alloc_len) override {
    do {
      *value_ptr = layout_creator_->Create(alloc_, alloc_len);
      Status s = kv_->Insert(key, *value_ptr);
      if (s.ok()) {
        break;
      } else {
        (*value_ptr)->Destroy(alloc_);
        delete *value_ptr;
      }
    } while (!(kv_->Lookup(key, value_ptr)).ok());
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size) override {
    Status s = kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      return s;
    }

    *value_ptr = layout_creator_->Create(alloc_, size);
    s = kv_->Insert(key, *value_ptr);
    if (s.ok()) {
      return s;
    }
    // Insert Failed, key already exist
    (*value_ptr)->Destroy(alloc_);
    delete *value_ptr;
    return kv_->Lookup(key, value_ptr);
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size, CopyBackFlag &need_copyback) override {
    need_copyback = NOT_COPYBACK;
    return GetOrCreate(key, value_ptr, size);
  }
 
  Status Remove(K key) override {
    return kv_->Remove(key);
  }

  int64 Size() const override {
    return kv_->Size();
  }
  
  int64 Size(int level) const override {
    if (level > 0) {
      LOG(FATAL) << "Unsupport level>0 in SingleTierStorage.";
    }
    return kv_->Size();
  }

  int64 CacheSize() const override {
    LOG(FATAL) << "Unsupport cachesize in SingleTierStorage.";
    return 0;
  }

  int LookupTier(K key) const override {
    Status s = kv_->Contains(key);
    return (s.ok()) ? 0 : -1;
  }

  void CopyEmbeddingsFromCPUToGPU(
      int total, const K* keys,
      const std::list<int64>& copyback_cursor,
      V** memcpy_address, size_t value_len,
      ValuePtr<V> **gpu_value_ptrs,
      V* memcpy_buffer_gpu) override {
    LOG(FATAL) << "Unsupport CopyEmbeddingsFromCPUToGPU in SingleTierStorage.";
  };

  BatchCache<K>* Cache() override {
    LOG(FATAL) << "Unsupport Cache in SingleTierStorage.";
    return nullptr;
  }

  void InitCache(embedding::CacheStrategy cache_strategy) override {
    LOG(FATAL) << "Unsupport InitCache in SingleTierStorage.";
  }

  Status BatchCommit(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) override {
    LOG(FATAL) << "Unsupport BatchCommit in Storage:"
               << typeid(this).name();
    return Status::OK();
  }

  Status Eviction(K* evict_ids, int64 evict_size) override {
    LOG(FATAL) << "Unsupport Eviction in SingleTierStorage.";
    return Status::OK();
  }

  void CreateEmbeddingMemoryPool(
      Allocator* alloc,
      int64 value_len,
      int64 block_size) override {
    return;
  }

  void AllocateMemoryForNewFeatures(
      const std::vector<ValuePtr<V>*>& value_ptr_list) override {
    return;
  }

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<ValuePtr<V>*>* value_ptr_list) override {
    return kv_->GetSnapshot(key_list, value_ptr_list);
  }

  int64 GetSnapshot(std::vector<K>* key_list,
      std::vector<V* >* value_list,
      std::vector<int64>* version_list,
      std::vector<int64>* freq_list,
      const EmbeddingConfig& emb_config,
      FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
      embedding::Iterator** it) override {
    std::vector<ValuePtr<V>*> value_ptr_list;
    std::vector<K> key_list_tmp;
    TF_CHECK_OK(kv_->GetSnapshot(&key_list_tmp, &value_ptr_list));
    if (key_list_tmp.empty()) {
      *it = kv_->GetIterator();
      return 0;
    }
    for (int64 i = 0; i < key_list_tmp.size(); ++i) {
      V* val = value_ptr_list[i]->GetValue(emb_config.emb_index,
        Storage<K, V>::GetOffset(emb_config.emb_index));
      V* primary_val = value_ptr_list[i]->GetValue(
          emb_config.primary_emb_index,
          Storage<K, V>::GetOffset(emb_config.primary_emb_index));
      key_list->emplace_back(key_list_tmp[i]);
      if (emb_config.filter_freq != 0 || emb_config.record_freq) {
        int64 dump_freq = filter->GetFreq(
            key_list_tmp[i], value_ptr_list[i]);
        freq_list->emplace_back(dump_freq);
      }
      if (emb_config.steps_to_live != 0 || emb_config.record_version) {
        int64 dump_version = value_ptr_list[i]->GetStep();
        version_list->emplace_back(dump_version);
      }
      if (val != nullptr && primary_val != nullptr) {
        value_list->emplace_back(val);
      } else if (val == nullptr && primary_val != nullptr) {
        // only forward, no backward
        value_list->emplace_back(reinterpret_cast<V*>(-1));
      } else {
        // feature filtered
        value_list->emplace_back(nullptr);
      }
    } 
    return key_list->size();
  }

  int64 GetSnapshotWithoutFetchPersistentEmb(
      std::vector<K>* key_list,
      std::vector<V* >* value_list,
      std::vector<int64>* version_list,
      std::vector<int64>* freq_list,
      const EmbeddingConfig& emb_config,
      SsdRecordDescriptor<K>* ssd_rec_desc) override {
    LOG(FATAL)<<"The Storage dosen't use presisten memory"
              <<" or this storage hasn't suppported "
              <<" GetSnapshotWithoutFetchPersistentEmb yet";
    return -1;
  }

  void RestoreSsdHashmap(
      K* key_list, int64* key_file_id_list,
      int64* key_offset_list, int64 num_of_keys,
      int64* file_list, int64* invalid_record_count_list,
      int64* record_count_list, int64 num_of_files,
      const std::string& ssd_emb_file_name) override {
    LOG(FATAL)<<"The Storage dosen't have ssd storage.";
  }

  Status Shrink(const EmbeddingConfig& emb_config,
      int64 value_len) override {
    mutex_lock l(Storage<K, V>::mu_);
    L2WeightShrinkPolicy<K, V> policy(emb_config.primary_emb_index,
        Storage<K, V>::GetOffset(emb_config.primary_emb_index),
        kv_, alloc_);
    policy.Shrink(value_len, (V)emb_config.l2_weight_threshold);
    return Status::OK();
  }

  Status Shrink(int64 global_step, int64 steps_to_live) override {
    mutex_lock l(Storage<K, V>::mu_);
    GlobalStepShrinkPolicy<K, V> policy(kv_, alloc_);
    policy.Shrink(global_step, steps_to_live);
    return Status::OK();
  }

  void SetAllocLen(int64 value_len, int slot_num) override {
    while (Storage<K, V>::flag_.test_and_set(std::memory_order_acquire));
    // The start address of every slot should be aligned to 16 bytes,
    // otherwise a coredump will happen in the ApplyOp.
    Storage<K, V>::alloc_len_ = Storage<K, V>::ComputeAllocLen(value_len);

    int64 temp = Storage<K, V>::alloc_len_ * slot_num;
    if (temp > Storage<K, V>::total_dims_) {
      Storage<K, V>::total_dims_ = temp;
      SetTotalDims(Storage<K, V>::total_dims_);
    }
    Storage<K, V>::flag_.clear(std::memory_order_release);
  }

  bool IsMultiLevel() override {
    return false;
  }

  bool IsUseHbm() override {
    return false;
  }

  bool IsUsePersistentStorage() override {
    return false;
  }

  void iterator_mutex_lock() override {
    return;
  }

  void iterator_mutex_unlock() override {
    return;
  }

  void Schedule(std::function<void()> fn) override {
    LOG(FATAL) << "Unsupport Schedule in SingleTierStorage.";
  }

 protected:
  virtual void SetTotalDims(int64 total_dims) = 0;

 protected:
  KVInterface<K, V>* kv_;
  Allocator* alloc_;
  LayoutCreator<V>* layout_creator_;
};

template<typename K, typename V>
class DramStorage : public SingleTierStorage<K, V> {
 public:
  DramStorage(const StorageConfig& sc, Allocator* alloc,
      LayoutCreator<V>* lc) : SingleTierStorage<K, V>(
          sc, alloc, new LocklessHashMap<K, V>(), lc) {
  }
  ~DramStorage() override {}
 
  TF_DISALLOW_COPY_AND_ASSIGN(DramStorage);

 protected:
  void SetTotalDims(int64 total_dims) override {}
};

template<typename K, typename V>
class PmemMemkindStorage : public SingleTierStorage<K, V> {
 public:
  PmemMemkindStorage(const StorageConfig& sc, Allocator* alloc,
      LayoutCreator<V>* lc) : SingleTierStorage<K, V>(
          sc, alloc, new LocklessHashMap<K, V>(), lc) {
  }
  ~PmemMemkindStorage() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(PmemMemkindStorage);
 
 protected:
  void SetTotalDims(int64 total_dims) override {}
};

template<typename K, typename V>
class PmemLibpmemStorage : public SingleTierStorage<K, V> {
 public:
  PmemLibpmemStorage(const StorageConfig& sc, Allocator* alloc,
      LayoutCreator<V>* lc) : SingleTierStorage<K, V>(
          sc, alloc, new LocklessHashMap<K, V>(), lc) {
  }
  ~PmemLibpmemStorage() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(PmemLibpmemStorage);
 
 protected:
  void SetTotalDims(int64 total_dims) override {}
};

template<typename K, typename V>
class LevelDBStore : public SingleTierStorage<K, V> {
 public:
  LevelDBStore(const StorageConfig& sc, Allocator* alloc,
      LayoutCreator<V>* lc) : SingleTierStorage<K, V>(
          sc, alloc, new LevelDBKV<K, V>(sc.path), lc) {
  }
  ~LevelDBStore() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(LevelDBStore);

  Status BatchCommit(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) override {
    return SingleTierStorage<K, V>::kv_->BatchCommit(keys, value_ptrs);
  }

 protected:
  void SetTotalDims(int64 total_dims) override {
    SingleTierStorage<K, V>::kv_->SetTotalDims(total_dims);
  }
};

template<typename K, typename V>
class SsdHashStorage : public SingleTierStorage<K, V> {
 public:
  SsdHashStorage(const StorageConfig& sc, Allocator* alloc,
      LayoutCreator<V>* lc) : SingleTierStorage<K, V>(
          sc, alloc, new SSDHashKV<K, V>(sc.path, alloc), lc) {
  }
  ~SsdHashStorage() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(SsdHashStorage);

  Status BatchCommit(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) override {
    return SingleTierStorage<K, V>::kv_->BatchCommit(keys, value_ptrs);
  }

 protected:
  void SetTotalDims(int64 total_dims) override {
    SingleTierStorage<K, V>::kv_->SetTotalDims(total_dims);
  }
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_H_
