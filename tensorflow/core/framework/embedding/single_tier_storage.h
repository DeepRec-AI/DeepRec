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

#include "tensorflow/core/framework/embedding/dense_hash_map.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/layout_creator.h"
#include "tensorflow/core/framework/embedding/leveldb_kv.h"
#include "tensorflow/core/framework/embedding/lockless_hash_map.h"
#include "tensorflow/core/framework/embedding/ssd_hashkv.h"
#include "tensorflow/core/framework/embedding/storage_config.h"
#include "tensorflow/core/framework/embedding/storage.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
template <class V>
class ValuePtr;

template <class K, class V>
class EmbeddingVar;

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

  Status Get(K key, ValuePtr<V>** value_ptr) override {
    return kv_->Lookup(key, value_ptr);
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
      size_t size, bool &need_copyback) override {
    need_copyback = false;
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

  void CopyBackToGPU(int total, K* keys, int64 size,
      bool* copyback_flags, V** memcpy_address, size_t value_len,
      int *copyback_cursor, ValuePtr<V> **gpu_value_ptrs,
      V* memcpy_buffer_gpu) override {
    LOG(FATAL) << "Unsupport CopyBackToGPU in SingleTierStorage.";
  };

  BatchCache<K>* Cache() override {
    LOG(FATAL) << "Unsupport Cache in SingleTierStorage.";
    return nullptr;
  }

  void InitCacheStrategy(
      embedding::CacheStrategy cache_strategy) override {
    LOG(FATAL) << "Unsupport InitCacheStrategy in SingleTierStorage.";
  }

  Status Commit(K key, const ValuePtr<V>* value_ptr) override {
    LOG(FATAL) << "Unsupport Commit in SingleTierStorage.";
    return Status::OK();
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

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<ValuePtr<V>*>* value_ptr_list) override {
    return kv_->GetSnapshot(key_list, value_ptr_list);
  }

  int64 GetSnapshot(std::vector<K>* key_list,
      std::vector<V* >* value_list,
      std::vector<int64>* version_list,
      std::vector<int64>* freq_list,
      const EmbeddingConfig& emb_config,
      EmbeddingFilter<K, V, EmbeddingVar<K, V>>* filter,
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

  Status Shrink(const EmbeddingConfig& emb_config,
      int64 value_len) override {
    mutex_lock l(Storage<K, V>::mu_);
    std::vector<K> key_list;
    std::vector<ValuePtr<V>*> value_ptr_list;
    TF_CHECK_OK(kv_->GetSnapshot(&key_list, &value_ptr_list));
    std::vector<std::pair<K, ValuePtr<V>*>> to_deleted;
    for (int64 i = 0; i < key_list.size(); ++i) {
      V* val = value_ptr_list[i]->GetValue(emb_config.primary_emb_index,
          Storage<K, V>::GetOffset(emb_config.primary_emb_index));
      if (val != nullptr) {
        V l2_weight = (V)0.0;
        for (int64 j = 0; j < value_len; j++) {
          l2_weight += val[j] * val[j];
        }
        l2_weight *= (V)0.5;
        if (l2_weight < (V)emb_config.l2_weight_threshold) {
          to_deleted.emplace_back(
              std::pair<K, ValuePtr<V>*>(key_list[i], value_ptr_list[i]));
        }
      }
    }
    for (const auto it : to_deleted) {
      // TODO memory recycle
      (it.second)->Destroy(alloc_);
      delete it.second;
      kv_->Remove(it.first);
    }
    return Status::OK();
  }

  Status Shrink(int64 gs, int64 steps_to_live) override {
    mutex_lock l(Storage<K, V>::mu_);
    std::vector<K> key_list;
    std::vector<ValuePtr<V>* > value_ptr_list;
    TF_CHECK_OK(kv_->GetSnapshot(&key_list, &value_ptr_list));
    std::vector<std::pair<K, ValuePtr<V>*>> to_deleted;
    for (int64 i = 0; i < key_list.size(); ++i) {
      int64 version = value_ptr_list[i]->GetStep();
      if (version == -1) {
        value_ptr_list[i]->SetStep(gs);
      } else {
        if (gs - version > steps_to_live) {
          to_deleted.emplace_back(
              std::pair<K, ValuePtr<V>*>(key_list[i], value_ptr_list[i]));
        }
      }
    }
    for (const auto it : to_deleted) {
      // TODO memory recycle
      (it.second)->Destroy(alloc_);
      delete it.second;
      kv_->Remove(it.first);
    }
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
