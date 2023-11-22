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

#if GOOGLE_CUDA
#include "tensorflow/core/framework/embedding/batch.h"
#endif

namespace tensorflow {
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

  virtual void Init() override {
    cache_capacity_ = Storage<K, V>::storage_config_.size[0]
                      / (total_dim() * sizeof(V));
    ready_eviction_ = true;
  }

  int64 CacheSize() const override {
    return cache_capacity_;
  }

  BatchCache<K>* Cache() override {
    return cache_;
  }

  void InitCache(embedding::CacheStrategy cache_strategy) override {
    if (cache_ == nullptr) {
      cache_ = CacheFactory::Create<K>(cache_strategy, name_);
      eviction_manager_ = EvictionManagerCreator::Create<K, V>();
      eviction_manager_->AddStorage(this);
      cache_thread_pool_ = CacheThreadPoolCreator::Create();
    }
  }

  Status BatchCommit(const std::vector<K>& keys,
      const std::vector<void*>& value_ptrs) override {
    LOG(FATAL)<<"BatchCommit isn't supported by MultiTierStorage.";
    return Status::OK();
  }

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<void*>* value_ptr_list) override {
    LOG(FATAL)<<"Can't get snapshot of MultiTierStorage.";
    return Status::OK();
  }

  Status GetShardedSnapshot(
      std::vector<std::vector<K>>& key_list,
      std::vector<std::vector<void*>>& value_ptr_list,
      int partition_id, int partition_nums) override {
    LOG(FATAL)<<"Can't get sharded snapshot of MultiTierStorage.";
    return Status::OK();
  }

  void CopyEmbeddingsFromCPUToGPU(
      int total, const K* keys,
      const std::list<int64>& copyback_cursor,
      V** memcpy_address, size_t value_len,
      void **gpu_value_ptrs,
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
  virtual int total_dim() = 0;

  void DeleteFromEvictionManager() {
    eviction_manager_->DeleteStorage(this);
  }

  void ReleaseValuePtrs(std::deque<void*>& value_ptrs,
                        FeatureDescriptor<V>* feat_desc) {
    constexpr int CAP_INVALID_VALUEPTR = 64 * 1024;
    if (value_ptrs.size() > CAP_INVALID_VALUEPTR) {
      int64 num_of_deleted_value_ptrs =
          value_ptrs.size() - CAP_INVALID_VALUEPTR;
      for (int i = 0; i < num_of_deleted_value_ptrs; i++) {
        void* value_ptr = value_ptrs.front();
        feat_desc->Deallocate(value_ptr);
        value_ptrs.pop_front();
      }
    }
  }

  void ReleaseInvalidValuePtr(FeatureDescriptor<V>* feat_desc) {
    ReleaseValuePtrs(value_ptr_out_of_date_, feat_desc);
  }

  void KeepInvalidValuePtr(void* value_ptr) {
    value_ptr_out_of_date_.emplace_back(value_ptr);
  }

#if GOOGLE_CUDA
  void CopyEmbeddingsFromDramToHbm(const EmbeddingVarContext<GPUDevice>& context,
                                   const K* keys,
                                   void** value_ptr_list,
                                   std::list<int64>& copyback_cursors,
                                   const std::vector<int64>& memory_index,
                                   const std::vector<void*>& gpu_value_ptrs,
                                   int value_len,
                                   FeatureDescriptor<V>* hbm_feat_desc,
                                   FeatureDescriptor<V>* dram_feat_desc);
#endif //GOOGL_CUDA
 private:
  virtual Status EvictionWithDelayedDestroy(K* evict_ids, int64 evict_size) {}

 protected:
  std::deque<void*> value_ptr_out_of_date_;
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

#if GOOGLE_CUDA
template <class V>
void CopyEmbeddingFromHbmToDram(
    const std::vector<void*>& hbm_value_ptrs,
    const std::vector<void*>& dram_value_ptrs,
    Allocator* gpu_alloc,
    FeatureDescriptor<V>* hbm_feat_desc,
    FeatureDescriptor<V>* dram_feat_desc) {
  int batch_size = hbm_value_ptrs.size();
    V** dev_value_address;

  dev_value_address = (V**)gpu_alloc->AllocateRaw(
      Allocator::kAllocatorAlignment, batch_size * sizeof(V*));
  Allocator* cpu_alloc = ev_allocator();
  V** value_address = (V**)cpu_alloc->AllocateRaw(
      Allocator::kAllocatorAlignment, sizeof(V*) * batch_size);

  V* batch_data_place;
  V* dev_batch_data_place;
  int total_dim = dram_feat_desc->total_dim();
  dev_batch_data_place = (V*)gpu_alloc->AllocateRaw(
      Allocator::kAllocatorAlignment, sizeof(V) * batch_size * total_dim);
  batch_data_place = (V *)cpu_alloc->AllocateRaw(
      Allocator::kAllocatorAlignment, sizeof(V) * batch_size * total_dim);
  // Copy GPU addresses V*
  for(int i = 0; i < batch_size; ++i) {
    value_address[i] = hbm_feat_desc->GetEmbedding(hbm_value_ptrs[i], 0);
  }
  cudaMemcpyAsync(dev_value_address, value_address,
                  sizeof(V*) * batch_size,
                  cudaMemcpyHostToDevice);

  // Launch Kernel,Copy data to continuous place
  int block_dim = 128;
  void* args[] = { (void*)&dev_value_address,
      (void*)&dev_batch_data_place, (void*)&total_dim,
      (void*)&batch_size};

  cudaLaunchKernel((void *)BatchCopy<V>,
                    (batch_size * total_dim + block_dim - 1) / block_dim,
                    block_dim, args, 0, NULL);

  cudaMemcpyAsync(batch_data_place, dev_batch_data_place,
                  sizeof(V) * batch_size * total_dim,
                  cudaMemcpyDeviceToHost);

  cudaEvent_t is_finish_;
  cudaEventCreate(&is_finish_);
  cudaEventRecord(is_finish_);
  cudaEventSynchronize(is_finish_);
  cudaEventDestroy(is_finish_);
  
  for(int i = 0; i < batch_size; ++i) {
    memcpy(dram_feat_desc->GetEmbedding(dram_value_ptrs[i], 0),
        &batch_data_place[i * total_dim], total_dim * sizeof(V));
  }

  cpu_alloc->DeallocateRaw(value_address);
  cpu_alloc->DeallocateRaw(batch_data_place);
  gpu_alloc->DeallocateRaw(dev_value_address);
  gpu_alloc->DeallocateRaw(dev_batch_data_place);
}
#endif //GOOGL_CUDA
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_MULTI_TIER_STORAGE_H_
