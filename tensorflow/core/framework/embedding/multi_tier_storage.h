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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_MULTI_TIER_STORAGE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_MULTI_TIER_STORAGE_H_

#include "tensorflow/core/framework/embedding/cache_factory.h"
#include "tensorflow/core/framework/embedding/cache_thread_pool_creator.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/cpu_hash_map_kv.h"
#include "tensorflow/core/framework/embedding/eviction_manager.h"
#include "tensorflow/core/framework/embedding/globalstep_shrink_policy.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/l2weight_shrink_policy.h"
#include "tensorflow/core/framework/embedding/storage_config.h"
#include "tensorflow/core/framework/embedding/storage.h"

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
template<typename V>
class ValuePtr;

template<typename K, typename V>
class EmbeddingVar;

namespace embedding {
template<typename K, typename V>
class MultiTierStorage : public Storage<K, V> {
 public:
  MultiTierStorage(const StorageConfig& sc, const std::string& name)
      : Storage<K, V>(sc), name_(name) {}

  ~MultiTierStorage() override {
    eviction_manager_->DeleteStorage(this);
    for (auto kv : kvs_) {
      delete kv.kv_;
    }
    delete cache_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(MultiTierStorage);

  void SetAllocLen(int64 value_len, int slot_num) override {
    while (Storage<K, V>::flag_.test_and_set(std::memory_order_acquire));
    // The start address of every slot should be aligned to 16 bytes,
    // otherwise a coredump will happen in the ApplyOp.
    Storage<K, V>::alloc_len_ = Storage<K, V>::ComputeAllocLen(value_len);

    int64 temp = Storage<K, V>::alloc_len_ * slot_num;
    if (temp > Storage<K, V>::total_dims_) {
      Storage<K, V>::total_dims_ = temp;
      SetTotalDims(Storage<K, V>::total_dims_);

      cache_capacity_ = Storage<K, V>::storage_config_.size[0]
                        / (Storage<K, V>::total_dims_ * sizeof(V));
      ready_eviction_ = true;
    }
    Storage<K, V>::flag_.clear(std::memory_order_release);
  }

  int64 CacheSize() const override {
    return cache_capacity_;
  }

  BatchCache<K>* Cache() override {
    return cache_;
  }

  void InitCache(embedding::CacheStrategy cache_strategy) override {
    cache_ = CacheFactory::Create<K>(cache_strategy, name_);
    eviction_manager_ = EvictionManagerCreator::Create<K, V>();
    eviction_manager_->AddStorage(this);
    cache_thread_pool_ = CacheThreadPoolCreator::Create();
  }

  void CopyEmbeddingsFromCPUToGPU(
      int total, const K* keys,
      const std::list<int64>& copyback_cursor,
      V** memcpy_address, size_t value_len,
      ValuePtr<V> **gpu_value_ptrs,
      V* memcpy_buffer_gpu) override {
    LOG(FATAL) << "Unsupport CopyEmbeddingsFromCPUToGPU in MultiTierStorage.";
  };

  int64 GetSnapshot(std::vector<K>* key_list,
      std::vector<V* >* value_list,
      std::vector<int64>* version_list,
      std::vector<int64>* freq_list,
      const EmbeddingConfig& emb_config,
      FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
      embedding::Iterator** it) override {
    for (auto kv : kvs_) {
      mutex_lock l(kv.mu_);
      std::vector<ValuePtr<V>* > value_ptr_list;
      std::vector<K> key_list_tmp;
      TF_CHECK_OK(kv.kv_->GetSnapshot(&key_list_tmp, &value_ptr_list));
      if (key_list_tmp.empty()) {
        *it = kv.kv_->GetIterator();
        continue;
      }
      for (int64 i = 0; i < key_list_tmp.size(); ++i) {
        V* val = value_ptr_list[i]->GetValue(emb_config.emb_index,
          Storage<K, V>::GetOffset(emb_config.emb_index));
        V* primary_val = value_ptr_list[i]->GetValue(
            emb_config.primary_emb_index,
            Storage<K, V>::GetOffset(emb_config.primary_emb_index));
        key_list->emplace_back(key_list_tmp[i]);

        int64 dump_freq = filter->GetFreq(
            key_list_tmp[i], value_ptr_list[i]);
        freq_list->emplace_back(dump_freq);

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
    }
    return key_list->size();
  }

  Status Shrink(const EmbeddingConfig& emb_config,
      int64 value_len) override {
    for (auto kv : kvs_) {
      mutex_lock l(kv.mu_);
      L2WeightShrinkPolicy<K, V> policy(emb_config.primary_emb_index,
          Storage<K, V>::GetOffset(emb_config.primary_emb_index),
          kv.kv_, kv.allocator_);
      policy.Shrink(value_len, (V)emb_config.l2_weight_threshold);
    }
    return Status::OK();
  }

  Status Shrink(int64 global_step, int64 steps_to_live) override {
    for (auto kv : kvs_) {
      mutex_lock l(kv.mu_);
      GlobalStepShrinkPolicy<K, V> policy(kv.kv_, kv.allocator_);
      policy.Shrink(global_step, steps_to_live);
    }
    return Status::OK();
  }

  Status BatchCommit(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) override {
    for (auto kv : kvs_) {
      TF_CHECK_OK(kv.kv_->BatchCommit(keys, value_ptrs));
    }
    return Status::OK();
  }

  Status Eviction(K* evict_ids, int64 evict_size) override {
    ValuePtr<V>* value_ptr;
    for (int64 i = 0; i < evict_size; ++i) {
      if (kvs_[0].kv_->Lookup(evict_ids[i], &value_ptr).ok()) {
        TF_CHECK_OK(kvs_[1].kv_->Commit(evict_ids[i], value_ptr));
        TF_CHECK_OK(kvs_[0].kv_->Remove(evict_ids[i]));
        value_ptr->Destroy(kvs_[0].allocator_);
        delete value_ptr;
      }
    }
    return Status::OK();
  }

  int64 Size(int level) const override {
    return kvs_[level].kv_->Size();
  }

  int LookupTier(K key) const override {
    for (int i = 0; i < kvs_.size(); ++i) {
      Status s = kvs_[i].kv_->Contains(key);
      if (s.ok()) {
        return i;
      }
    }
    return -1;
  }

  bool IsMultiLevel() override {
    return true;
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

  void Schedule(std::function<void()> fn) override {
    cache_thread_pool_->Schedule(std::move(fn));
  }

  virtual void BatchEviction() {
    constexpr int EvictionSize = 10000;
    K evic_ids[EvictionSize];
    if (!ready_eviction_)
      return;
    mutex_lock l(kvs_[0].mu_);
    mutex_lock l1(kvs_[1].mu_);
    //Release the memory of invlid valuetprs
    ReleaseInvalidValuePtr();

    int cache_count = cache_->size();
    if (cache_count > cache_capacity_) {
      // eviction
      int k_size = cache_count - cache_capacity_;
      k_size = std::min(k_size, EvictionSize);
      size_t true_size = cache_->get_evic_ids(evic_ids, k_size);
      ValuePtr<V>* value_ptr;
      if (Storage<K, V>::storage_config_.type == StorageType::HBM_DRAM) {
        std::vector<K> keys;
        std::vector<ValuePtr<V>*> value_ptrs;

        for (int64 i = 0; i < true_size; ++i) {
          if (kvs_[0].kv_->Lookup(evic_ids[i], &value_ptr).ok()) {
            TF_CHECK_OK(kvs_[0].kv_->Remove(evic_ids[i]));
            keys.emplace_back(evic_ids[i]);
            value_ptrs.emplace_back(value_ptr);
          }
        }
        BatchCommit(keys, value_ptrs);
      } else {
        for (int64 i = 0; i < true_size; ++i) {
          if (kvs_[0].kv_->Lookup(evic_ids[i], &value_ptr).ok()) {
            TF_CHECK_OK(kvs_[1].kv_->Commit(evic_ids[i], value_ptr));
            TF_CHECK_OK(kvs_[0].kv_->Remove(evic_ids[i]));
            value_ptr_out_of_date_.emplace_back(value_ptr);
          }
        }
      }
    }
  }

 protected:
  virtual void SetTotalDims(int64 total_dims) = 0;

  void ReleaseValues(
      const std::vector<std::pair<KVInterface<K, V>*, Allocator*>>& kvs) {
    for (auto kv : kvs_) {
      std::vector<K> key_list;
      std::vector<ValuePtr<V>*> value_ptr_list;
      kv.kv_->GetSnapshot(&key_list, &value_ptr_list);
      for (auto value_ptr : value_ptr_list) {
        value_ptr->Destroy(kv.allocator_);
        delete value_ptr;
      } 
    }
  }

  void ReleaseValuePtrs(std::vector<ValuePtr<V>*>& value_ptrs,
                        Allocator* allocator) {
    for (int i = 0; i < value_ptrs.size(); i++) {
      value_ptrs[i]->Destroy(allocator);
      delete value_ptrs[i];
    }
    value_ptrs.clear();
  }

  void ReleaseInvalidValuePtr() {
    ReleaseValuePtrs(value_ptr_out_of_date_, kvs_[0].allocator_);
  }

 protected:
  std::vector<KVInterfaceDescriptor<K, V>> kvs_;
  std::vector<ValuePtr<V>*> value_ptr_out_of_date_;
  BatchCache<K>* cache_ = nullptr;

  EvictionManager<K, V>* eviction_manager_;
  thread::ThreadPool* cache_thread_pool_;

  condition_variable shutdown_cv_;
  volatile bool shutdown_ = false;

  int64 cache_capacity_ = -1;
  volatile bool ready_eviction_ = false;

  std::string name_;
  std::vector<mutex> mu_list_;
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_MULTI_TIER_STORAGE_H_
