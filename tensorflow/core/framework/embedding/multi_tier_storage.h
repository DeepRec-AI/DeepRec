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
#include "tensorflow/core/framework/embedding/embedding_var_context.h"
#include "tensorflow/core/framework/embedding/embedding_var_restore.h"
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

template <class K>
struct SsdRecordDescriptor;

namespace embedding {
template<typename K, typename V>
class MultiTierStorage : public Storage<K, V> {
 public:
  MultiTierStorage(const StorageConfig& sc, const std::string& name)
      : Storage<K, V>(sc), name_(name) {}

  virtual ~MultiTierStorage() {
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

  Status BatchCommit(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) override {
    LOG(FATAL)<<"BatchCommit isn't supported by MultiTierStorage.";
    return Status::OK();
  }

  Status GetSnapshot(std::vector<K>* key_list,
                     std::vector<ValuePtr<V>*>* value_ptr_list) override {
    LOG(FATAL)<<"Can't get snapshot of MultiTierStorage.";
  }

  void CopyEmbeddingsFromCPUToGPU(
      int total, const K* keys,
      const std::list<int64>& copyback_cursor,
      V** memcpy_address, size_t value_len,
      ValuePtr<V> **gpu_value_ptrs,
      V* memcpy_buffer_gpu,
      se::Stream* compute_stream,
      EventMgr* event_mgr,
      const DeviceBase::CpuWorkerThreads* worker_threads) override {
    LOG(FATAL) << "Unsupport CopyEmbeddingsFromCPUToGPU in MultiTierStorage.";
  };

  Status Contains(K key) override {
    LOG(FATAL)<<"Contains is not support in MultiTierStorage.";
    return Status::OK();
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

  void AllocateMemoryForNewFeatures(
      ValuePtr<V>** value_ptr_list,
      int64 num_of_value_ptrs) override {
    return;
  }

  void Schedule(std::function<void()> fn) override {
    cache_thread_pool_->Schedule(std::move(fn));
  }

  virtual Status Eviction(K* evict_ids, int64 evict_size) override {
    LOG(FATAL)<<"Eviction isn't support by "<<typeid(this).name();
    return Status::OK();
  }

  virtual void BatchEviction() {
    constexpr int EvictionSize = 10000;
    K evic_ids[EvictionSize];
    if (!ready_eviction_)
      return;
    int cache_count = cache_->size();
    if (cache_count > cache_capacity_) {
      // eviction
      int k_size = cache_count - cache_capacity_;
      k_size = std::min(k_size, EvictionSize);
      size_t true_size = cache_->get_evic_ids(evic_ids, k_size);
      EvictionWithDelayedDestroy(evic_ids, true_size);
    }
  }

  void UpdateCache(const Tensor& indices,
                   const Tensor& indices_counts) override {
    Schedule([this, indices, indices_counts]() {
      cache_->update(indices, indices_counts);
    });
  }

  void UpdateCache(const Tensor& indices) override {
    Schedule([this, indices]() {
      cache_->update(indices);
    });
  }

  virtual bool IsUseHbm() override {
    return false;
  }

  void AddToCachePrefetchList(const Tensor& indices) override {
    Schedule([this, indices]() {
      cache_->add_to_prefetch_list(indices);
    });
  }

  void AddToCache(const Tensor& indices) override {
    Schedule([this, indices]() {
      cache_->add_to_cache(indices);
    });
  }

 protected:
  Status RestoreFeatures(int64 key_num, int bucket_num, int64 partition_id,
                         int64 partition_num, int64 value_len, bool is_filter,
                         bool is_incr, const EmbeddingConfig& emb_config,
                         const Eigen::GpuDevice* device,
                         FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
                         RestoreBuffer& restore_buff) override {
    Status s = filter->Restore(key_num, bucket_num, partition_id,
                               partition_num, value_len, is_filter,
                               false/*to_dram*/, is_incr, restore_buff);
 
    if (emb_config.is_primary()) {
      K* key_buff = (K*)restore_buff.key_buffer;
      V* value_buff = (V*)restore_buff.value_buffer;
      int64* version_buff = (int64*)restore_buff.version_buffer;
      int64* freq_buff = (int64*)restore_buff.freq_buffer;
      if (cache_) {
        cache_->update(key_buff, key_num, version_buff, freq_buff);
        auto cache_size = CacheSize();
        if (cache_->size() > cache_size) {
          int64 evict_size = cache_->size() - cache_size;
          std::vector<K> evict_ids(evict_size);
          size_t true_size =
              cache_->get_evic_ids(evict_ids.data(), evict_size);
          Eviction(evict_ids.data(), true_size);
        }
      }
      return s;
    }
    return s;
  }
 
  virtual void SetTotalDims(int64 total_dims) = 0;

  void DeleteFromEvictionManager() {
    eviction_manager_->DeleteStorage(this);
  }

  void ReleaseValuePtrs(std::deque<ValuePtr<V>*>& value_ptrs,
                        Allocator* allocator) {
    constexpr int CAP_INVALID_VALUEPTR = 64 * 1024;
    if (value_ptrs.size() > CAP_INVALID_VALUEPTR) {
      int64 num_of_deleted_value_ptrs =
          value_ptrs.size() - CAP_INVALID_VALUEPTR;
      for (int i = 0; i < num_of_deleted_value_ptrs; i++) {
        ValuePtr<V>* value_ptr = value_ptrs.front();
        value_ptr->Destroy(allocator);
        delete value_ptr;
        value_ptrs.pop_front();
      }
    }
  }

  void ReleaseInvalidValuePtr(Allocator* allocator) {
    ReleaseValuePtrs(value_ptr_out_of_date_, allocator);
  }

  void KeepInvalidValuePtr(ValuePtr<V>* value_ptr) {
    value_ptr_out_of_date_.emplace_back(value_ptr);
  }

#if GOOGLE_CUDA
  void CopyEmbeddingsFromDramToHbm(const EmbeddingVarContext<GPUDevice>& context,
                                   const K* keys,
                                   ValuePtr<V>** value_ptr_list,
                                   std::list<int64>& copyback_cursors,
                                   const std::vector<int64>& memory_index,
                                   const std::vector<ValuePtr<V>*>& gpu_value_ptrs,
                                   int value_len);
#endif //GOOGL_CUDA
 private:
  virtual Status EvictionWithDelayedDestroy(K* evict_ids, int64 evict_size) {}

 protected:
  std::deque<ValuePtr<V>*> value_ptr_out_of_date_;
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
