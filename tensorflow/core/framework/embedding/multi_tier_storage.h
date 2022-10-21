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

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"

#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/lockless_hash_map.h"
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
    delete eviction_thread_;
    for (auto kv : kvs_) {
      delete kv.first;
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
    if (cache_strategy == CacheStrategy::LRU) {
      LOG(INFO) << " Use StorageManager::LRU in multi-tier EmbeddingVariable "
                << name_;
      cache_ = new LRUCache<K>();
    } else {
      LOG(INFO) << "Use StorageManager::LFU in multi-tier EmbeddingVariable "
                << name_;
      cache_ = new LFUCache<K>();
    }
    eviction_thread_ = Env::Default()->StartThread(
        ThreadOptions(), "EmbeddingVariable_Eviction",
        [this]() { BatchEviction(); });
    thread_pool_.reset(
        new thread::ThreadPool(Env::Default(), ThreadOptions(),
          "MultiTier_Embedding_Cache", 2, /*low_latency_hint=*/false));
  }

  void CopyBackToGPU(int total, K* keys, int64 size,
      bool* copyback_flags, V** memcpy_address, size_t value_len,
      int *copyback_cursor, ValuePtr<V> **gpu_value_ptrs,
      V* memcpy_buffer_gpu) override {
    LOG(FATAL) << "Unsupport CopyBackToGPU in MultiTierStorage.";
  };

  int64 GetSnapshot(std::vector<K>* key_list,
      std::vector<V* >* value_list,
      std::vector<int64>* version_list,
      std::vector<int64>* freq_list,
      const EmbeddingConfig& emb_config,
      EmbeddingFilter<K, V, EmbeddingVar<K, V>>* filter,
      embedding::Iterator** it) override {
    for (auto kv : kvs_) {
      std::vector<ValuePtr<V>* > value_ptr_list;
      std::vector<K> key_list_tmp;
      TF_CHECK_OK(kv.first->GetSnapshot(&key_list_tmp, &value_ptr_list));
      if (key_list_tmp.empty()) {
        *it = kv.first->GetIterator();
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
    mutex_lock l(Storage<K, V>::mu_);
    for (auto kv : kvs_) {
      std::vector<K> key_list;
      std::vector<ValuePtr<V>*> value_ptr_list;
      TF_CHECK_OK(kv.first->GetSnapshot(&key_list, &value_ptr_list));
      std::vector<std::pair<K, ValuePtr<V>* > > to_deleted;
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
        (it.second)->Destroy(kv.second);
        delete it.second;
        kv.first->Remove(it.first);
      }
    }
    return Status::OK();
  }

  Status Shrink(int64 gs, int64 steps_to_live) override {
    mutex_lock l(Storage<K, V>::mu_);
    for (auto kv : kvs_) {
      std::vector<K> key_list;
      std::vector<ValuePtr<V>*> value_ptr_list;
      TF_CHECK_OK(kv.first->GetSnapshot(&key_list, &value_ptr_list));
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
        (it.second)->Destroy(kv.second);
        delete it.second;
        kv.first->Remove(it.first);
      }
    }
    return Status::OK();
  }

  Status BatchCommit(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) override {
    for (auto kv : kvs_) {
      TF_CHECK_OK(kv.first->BatchCommit(keys, value_ptrs));
    }
    return Status::OK();
  }

  Status Eviction(K* evict_ids, int64 evict_size) override {
    ValuePtr<V>* value_ptr;
    for (int64 i = 0; i < evict_size; ++i) {
      if (kvs_[0].first->Lookup(evict_ids[i], &value_ptr).ok()) {
        TF_CHECK_OK(kvs_[1].first->Commit(evict_ids[i], value_ptr));
        TF_CHECK_OK(kvs_[0].first->Remove(evict_ids[i]));
        value_ptr->Destroy(kvs_[0].second);
        delete value_ptr;
      }
    }
    return Status::OK();
  }

  int64 Size(int level) const override {
    return kvs_[level].first->Size();
  }

  int LookupTier(K key) const override {
    for (int i = 0; i < kvs_.size(); ++i) {
      Status s = kvs_[i].first->Contains(key);
      if (s.ok()) {
        return i;
      }
    }
    return -1;
  }

  bool IsMultiLevel() override {
    return true;
  }

  void Schedule(std::function<void()> fn) override {
    thread_pool_->Schedule(std::move(fn)); 
  }

 protected:
  virtual void SetTotalDims(int64 total_dims) = 0;

  void ReleaseValues(
      const std::vector<std::pair<KVInterface<K, V>*, Allocator*>>& kvs) {
    for (auto kv : kvs_) {
      std::vector<K> key_list;
      std::vector<ValuePtr<V>*> value_ptr_list;
      kv.first->GetSnapshot(&key_list, &value_ptr_list);
      for (auto value_ptr : value_ptr_list) {
        value_ptr->Destroy(kv.second);
        delete value_ptr;
      } 
    }
  }

  void ShutdownEvictionThread() {
    mutex_lock l(Storage<K, V>::mu_);
    shutdown_cv_.notify_all();
    shutdown_ = true;
  }

 private:
  void BatchEviction() {
    constexpr int EvictionSize = 10000;
    if (cache_capacity_ == -1) {
      while (!ready_eviction_) {
        // why lock here, volitile is enough..TODO Review
        // mutex_lock l(mu_);
        // Sleep 1ms
        Env::Default()->SleepForMicroseconds(1000);
      }
    }
    K evic_ids[EvictionSize];
    while (!shutdown_) {
      mutex_lock l(Storage<K, V>::mu_);
      if (shutdown_) {
        return;
      }
      // add WaitForMilliseconds() for sleep if necessary
      const int kTimeoutMilliseconds = 1;
      WaitForMilliseconds(&l, &shutdown_cv_, kTimeoutMilliseconds);
      for (int i = 0; i < value_ptr_out_of_date_.size(); i++) {
        value_ptr_out_of_date_[i]->Destroy(kvs_[0].second);
        delete value_ptr_out_of_date_[i];
      }
      value_ptr_out_of_date_.clear();
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
          timespec start, end;

          clock_gettime(CLOCK_MONOTONIC, &start);
          for (int64 i = 0; i < true_size; ++i) {
            if (kvs_[0].first->Lookup(evic_ids[i], &value_ptr).ok()) {
              TF_CHECK_OK(kvs_[0].first->Remove(evic_ids[i]));
              keys.emplace_back(evic_ids[i]);
              value_ptrs.emplace_back(value_ptr);
            }
          }

          BatchCommit(keys, value_ptrs);
          clock_gettime(CLOCK_MONOTONIC, &end);
          LOG(INFO) << "Total Evict Time: "
                    << (double)(end.tv_sec - start.tv_sec) *
                       EnvTime::kSecondsToMillis +
                       (end.tv_nsec - start.tv_nsec) /
                       EnvTime::kMillisToNanos<< "ms";
        } else {
          for (int64 i = 0; i < true_size; ++i) {
            if (kvs_[0].first->Lookup(evic_ids[i], &value_ptr).ok()) {
              LOG(INFO) << "evic_ids[i]: " << evic_ids[i];
              TF_CHECK_OK(kvs_[1].first->Commit(evic_ids[i], value_ptr));
              TF_CHECK_OK(kvs_[0].first->Remove(evic_ids[i]));
              value_ptr_out_of_date_.emplace_back(value_ptr);
            }
          }
        }
      }
    }
  }

 protected:
  std::vector<std::pair<KVInterface<K, V>*, Allocator*>> kvs_;
  std::vector<ValuePtr<V>*> value_ptr_out_of_date_;
  BatchCache<K>* cache_ = nullptr;

  Thread* eviction_thread_ = nullptr;;
  std::unique_ptr<thread::ThreadPool> thread_pool_;

  condition_variable shutdown_cv_;
  volatile bool shutdown_ = false;

  int64 cache_capacity_ = -1;
  volatile bool ready_eviction_ = false;

  std::string name_;
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_MULTI_TIER_STORAGE_H_
