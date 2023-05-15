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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_STORAGE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_STORAGE_H_

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/framework/embedding/lockless_hash_map_cpu.h"
#include "tensorflow/core/framework/embedding/multi_tier_storage.h"
#include "tensorflow/core/framework/embedding/single_tier_storage.h"
#include "tensorflow/core/framework/embedding/hbm_storage_iterator.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {
using se::DeviceMemoryBase;
using se::Stream;

template <class V>
class ValuePtr;

void SyncWithEventMgr(se::Stream* stream, EventMgr* event_mgr);

namespace embedding {
template<typename K, typename V>
class HbmDramStorage : public MultiTierStorage<K, V> {
 public:
  HbmDramStorage(const StorageConfig& sc, Allocator* gpu_alloc,
      Allocator* cpu_alloc, LayoutCreator<V>* lc, const std::string& name)
      : gpu_alloc_(gpu_alloc),
        MultiTierStorage<K, V>(sc, name) {
    hbm_ = new HbmStorageWithCpuKv<K, V>(sc, gpu_alloc, lc);
    dram_ = new DramStorage<K, V>(sc, cpu_alloc, lc, new LocklessHashMapCPU<K, V>(gpu_alloc));
  }

  ~HbmDramStorage() override {
    MultiTierStorage<K, V>::DeleteFromEvictionManager();
    delete hbm_;
    delete dram_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(HbmDramStorage);

  Status Get(K key, ValuePtr<V>** value_ptr) override {
    Status s = hbm_->Get(key, value_ptr);
    if (!s.ok()) {
      s = dram_->Get(key, value_ptr);
    }
    return s;
  }

  void Insert(K key, ValuePtr<V>* value_ptr) override {
    hbm_->Insert(key, value_ptr);
  }

  void Insert(K key, ValuePtr<V>** value_ptr,
              size_t alloc_len) override {
    hbm_->Insert(key, value_ptr, alloc_len);
  }

  void InsertToDram(K key, ValuePtr<V>** value_ptr,
              int64 alloc_len) override {
    dram_->Insert(key, value_ptr, alloc_len);
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size) override {
    Status s = hbm_->Get(key, value_ptr);
    if (s.ok()) {
      return s;
    }
    ValuePtr<V>* gpu_value_ptr = hbm_->CreateValuePtr(size);
    {
      mutex_lock l(memory_pool_mu_);
      gpu_value_ptr->SetPtr(embedding_mem_pool_->Allocate());
    }
    s = dram_->Get(key, value_ptr);
    if (s.ok()) {
      // copy dram value to hbm
      V* cpu_data_address = (*value_ptr)->GetValue(0, 0);
      V* gpu_data_address = gpu_value_ptr->GetValue(0, 0);
      cudaMemcpy(gpu_data_address, cpu_data_address,
          size * sizeof(V), cudaMemcpyHostToDevice);
      *value_ptr = gpu_value_ptr;
      memcpy(gpu_value_ptr->GetPtr(),
             (*value_ptr)->GetPtr(),
             sizeof(FixedLengthHeader));
    }

    s = hbm_->TryInsert(key, *value_ptr);
    if (s.ok()) {
      return s;
    }
    // Insert Failed, key already exist
    {
      mutex_lock l(memory_pool_mu_);
      embedding_mem_pool_->Deallocate((*value_ptr)->GetValue(0, 0));
    }
    delete *value_ptr;
    return hbm_->Get(key, value_ptr);
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size, CopyBackFlag &need_copyback) override {
    need_copyback = NOT_COPYBACK;
    Status s = hbm_->Get(key, value_ptr);
    if (s.ok()) {
      return s;
    }
    s = dram_->Get(key, value_ptr);
    if (s.ok()) {
      need_copyback = COPYBACK;
      return s;
    }

    hbm_->Insert(key, value_ptr, size);
    return Status::OK();
  }

  void ImportToHbm(
      K* ids, int64 size, int64 value_len, int64 emb_index) override {
    V* memcpy_buffer_cpu = new V[size * value_len];
    V** value_address = new V*[size];
    V* memcpy_buffer_gpu =
        (V*)gpu_alloc_->AllocateRaw(
            Allocator::kAllocatorAlignment,
            size * value_len * sizeof(V));
    V* dev_value_address =
        (V*)gpu_alloc_->AllocateRaw(
            Allocator::kAllocatorAlignment,
            size * sizeof(V*));
    ValuePtr<V>** gpu_value_ptrs = new ValuePtr<V>*[size];
    ValuePtr<V>** cpu_value_ptrs = new ValuePtr<V>*[size];
    {
      //Mutex with other Import Ops
      mutex_lock l(memory_pool_mu_);
      for (int64 i = 0; i < size; i++) {
        dram_->Get(ids[i], &cpu_value_ptrs[i]);
        gpu_value_ptrs[i] = hbm_->CreateValuePtr(value_len);
        V* val_ptr = embedding_mem_pool_->Allocate();
        gpu_value_ptrs[i]->SetPtr(val_ptr);
        memcpy((char *)gpu_value_ptrs[i]->GetPtr(),
               (char *)cpu_value_ptrs[i]->GetPtr(),
               sizeof(FixedLengthHeader));
      }
    }
    //Split from above for loop for minize the cost of mutex lock
    //TODO: Speed up with intra parallelism
    std::vector<ValuePtr<V>*> invalid_value_ptrs;
    for (int64 i = 0; i < size; i++) {
      memcpy(memcpy_buffer_cpu + i * value_len,
          cpu_value_ptrs[i]->GetValue(emb_index,
              Storage<K, V>::GetOffset(emb_index)), value_len * sizeof(V));
      Status s = hbm_->TryInsert(ids[i], gpu_value_ptrs[i]);
      if (!s.ok()) {
        invalid_value_ptrs.emplace_back(gpu_value_ptrs[i]);
        hbm_->Get(ids[i], &gpu_value_ptrs[i]);
      }
      gpu_value_ptrs[i]->SetInitialized(emb_index);
      value_address[i] = gpu_value_ptrs[i]->GetValue(
          emb_index, Storage<K, V>::GetOffset(emb_index));
    }
    cudaMemcpy(memcpy_buffer_gpu, memcpy_buffer_cpu,
        size * value_len * sizeof(V), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_value_address, value_address,
        size * sizeof(V*), cudaMemcpyHostToDevice);
    {
      mutex_lock l(memory_pool_mu_);
      embedding_mem_pool_->Deallocate(invalid_value_ptrs);
    }
    int block_dim = 128;
      void* args[] = {
          (void*)&dev_value_address,
          (void*)&memcpy_buffer_gpu,
          (void*)&value_len,
          (void*)&size};

    cudaLaunchKernel(
          (void *)BatchUnpack<V>,
          (size + block_dim - 1) / block_dim * value_len,
          block_dim,
          args, 0, NULL);
    cudaDeviceSynchronize();

    delete[] memcpy_buffer_cpu;
    delete[] cpu_value_ptrs;
    delete[] gpu_value_ptrs;
    delete[] value_address;
    gpu_alloc_->DeallocateRaw(dev_value_address);
    gpu_alloc_->DeallocateRaw(memcpy_buffer_gpu);
  }

  void CopyEmbeddingsFromCPUToGPU(
      int total, const K* keys,
      const std::list<int64>& copyback_cursor,
      V** memcpy_address, size_t value_len,
      ValuePtr<V> **gpu_value_ptrs, V* memcpy_buffer_gpu,
      se::Stream* compute_stream,
      EventMgr* event_mgr,
      const DeviceBase::CpuWorkerThreads* worker_threads) override {
    auto memcpy_buffer_cpu = TypedAllocator::Allocate<V>(cpu_allocator(),
        total * value_len, AllocationAttributes());
    int64* memory_index = new int64[total];
    int64 i = 0;
    auto it = copyback_cursor.cbegin();
    {
      //Mutex with eviction thread
      mutex_lock l(memory_pool_mu_);
      for ( ; it != copyback_cursor.cend(); ++it, ++i) {
        int64 j = *it;
        memory_index[i] = j;
        ValuePtr<V>* gpu_value_ptr = hbm_->CreateValuePtr(value_len);
        V* val_ptr = embedding_mem_pool_->Allocate();
        bool flag = gpu_value_ptr->SetPtr(val_ptr);
        if (!flag) {
          embedding_mem_pool_->Deallocate(val_ptr);
        }
        memcpy((char *)gpu_value_ptr->GetPtr(),
               (char *)memcpy_address[j] - sizeof(FixedLengthHeader),
               sizeof(FixedLengthHeader));
        gpu_value_ptrs[i] = gpu_value_ptr;
      }
    }
    //Split from above for loop for minize the cost of mutex lock
    auto do_work = [memory_index, memcpy_address,
                    memcpy_buffer_cpu, gpu_value_ptrs,
                    value_len, this] (int64 start, int64 limit) {
      for (int i = start; i < limit; i++) {
        int j = memory_index[i];
        memcpy(memcpy_buffer_cpu + i * value_len,
               memcpy_address[j], value_len * sizeof(V));
      }
    };
    Shard(worker_threads->num_threads, worker_threads->workers, total,
          1000, do_work);
    DeviceMemoryBase gpu_dst_ptr(
        memcpy_buffer_gpu, total * value_len * sizeof(V));
    compute_stream->ThenMemcpy(
        &gpu_dst_ptr, memcpy_buffer_cpu, total * value_len * sizeof(V));
    SyncWithEventMgr(compute_stream, event_mgr);
    TypedAllocator::Deallocate(
        cpu_allocator(), memcpy_buffer_cpu, total * value_len);
    delete[] memory_index;
  }

  Status Remove(K key) override {
    hbm_->Remove(key);
    dram_->Remove(key);
    return Status::OK();
  }

  int64 Size() const override {
    int64 total_size = hbm_->Size();
    total_size += dram_->Size();
    return total_size;
  }

  int64 Size(int level) const override {
    if (level == 0) {
      return hbm_->Size();
    } else if (level == 1) {
      return dram_->Size();
    } else {
      return -1;
    }
  }

  int LookupTier(K key) const override {
    Status s = hbm_->Contains(key);
    if (s.ok())
      return 0;
    s = dram_->Contains(key);
    if (s.ok())
      return 1;
    return -1;
  }

  bool IsUseHbm() override {
    return true;
  }

  bool IsSingleHbm() override {
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

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<ValuePtr<V>* >* value_ptr_list) override {
    {
      mutex_lock l(*(hbm_->get_mutex()));
      TF_CHECK_OK(hbm_->GetSnapshot(key_list, value_ptr_list));
    }
    {
      mutex_lock l(*(dram_->get_mutex()));
      TF_CHECK_OK(dram_->GetSnapshot(key_list, value_ptr_list));
    }
    return Status::OK();
  }

  int64 GetSnapshot(std::vector<K>* key_list,
      std::vector<V* >* value_list,
      std::vector<int64>* version_list,
      std::vector<int64>* freq_list,
      const EmbeddingConfig& emb_config,
      FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
      embedding::Iterator** it) override {
    std::vector<ValuePtr<V>*> hbm_value_ptr_list, dram_value_ptr_list;
    std::vector<K> temp_hbm_key_list, temp_dram_key_list;
    // Get Snapshot of HBM storage
    {
      mutex_lock l(*(hbm_->get_mutex()));
      TF_CHECK_OK(hbm_->GetSnapshot(&temp_hbm_key_list,
                                    &hbm_value_ptr_list));
    }
    // Get Snapshot of DRAM storage.
    {
      mutex_lock l(*(dram_->get_mutex()));
      TF_CHECK_OK(dram_->GetSnapshot(&temp_dram_key_list,
                                     &dram_value_ptr_list));
    }
    *it = new HbmDramIterator<K, V>(temp_hbm_key_list,
                                    temp_dram_key_list,
                                    hbm_value_ptr_list,
                                    dram_value_ptr_list,
                                    Storage<K, V>::alloc_len_,
                                    gpu_alloc_,
                                    emb_config.emb_index);
    // This return value is not the exact number of IDs
    // because the two tables intersect.
    return temp_hbm_key_list.size() + temp_dram_key_list.size();
  }

  Status Shrink(int64 value_len) override {
    hbm_->Shrink(value_len);
    dram_->Shrink(value_len);
    return Status::OK();
  }

  Status Shrink(int64 global_step, int64 steps_to_live) override {
    hbm_->Shrink(global_step, steps_to_live);
    dram_->Shrink(global_step, steps_to_live);
    return Status::OK();
  }

  void CreateEmbeddingMemoryPool(
      Allocator* alloc,
      int64 value_len,
      int64 block_size) override {
    embedding_mem_pool_ =
       new EmbeddingMemoryPool<V>(alloc, value_len, block_size);
  }

  void AllocateMemoryForNewFeatures(
      const std::vector<ValuePtr<V>*>& value_ptr_list) override {
    //Mutex with eviction thread
    mutex_lock l(memory_pool_mu_);
    for (auto it : value_ptr_list) {
      V* val_ptr = embedding_mem_pool_->Allocate();
      bool flag = it->SetPtr(val_ptr);
      if (!flag) {
        embedding_mem_pool_->Deallocate(val_ptr);
      }
    }
  }

  void AllocateMemoryForNewFeatures(
     ValuePtr<V>** value_ptr_list,
     int64 num_of_value_ptrs) override {
    //Mutex with other ImportOps
    mutex_lock l(memory_pool_mu_);
    for (int64 i = 0; i < num_of_value_ptrs; i++) {
      V* val_ptr = embedding_mem_pool_->Allocate();
      bool flag = value_ptr_list[i]->SetPtr(val_ptr);
      if (!flag) {
        embedding_mem_pool_->Deallocate(val_ptr);
      }
    }
  }

  void BatchEviction() override {
    constexpr int EvictionSize = 10000;
    K evic_ids[EvictionSize];
    if (!MultiTierStorage<K, V>::ready_eviction_) {
      return;
    }
    mutex_lock l(*(hbm_->get_mutex()));
    mutex_lock l1(*(dram_->get_mutex()));

    int64 cache_count = MultiTierStorage<K, V>::cache_->size();
    if (cache_count > MultiTierStorage<K, V>::cache_capacity_) {
      // eviction
      int k_size = cache_count - MultiTierStorage<K, V>::cache_capacity_;
      k_size = std::min(k_size, EvictionSize);
      size_t true_size =
          MultiTierStorage<K, V>::cache_->get_evic_ids(evic_ids, k_size);
      ValuePtr<V>* value_ptr;
      std::vector<K> keys;
      std::vector<ValuePtr<V>*> value_ptrs;

      for (int64 i = 0; i < true_size; ++i) {
        if (hbm_->Get(evic_ids[i], &value_ptr).ok()) {
          keys.emplace_back(evic_ids[i]);
          value_ptrs.emplace_back(value_ptr);
        }
      }
      dram_->BatchCommit(keys, value_ptrs);
      {
        //Mutex with main thread
        mutex_lock l_mem(memory_pool_mu_);
        embedding_mem_pool_->Deallocate(value_ptrs);
      }
      for (auto it : keys) {
        TF_CHECK_OK(hbm_->Remove(it));
      }
    }
  }

 protected:
  void SetTotalDims(int64 total_dims) override {
    dram_->SetTotalDims(total_dims);
  }

 private:
  HbmStorageWithCpuKv<K, V>* hbm_ = nullptr;
  DramStorage<K, V>* dram_ = nullptr;
  EmbeddingMemoryPool<V>* embedding_mem_pool_ = nullptr;
  Allocator* gpu_alloc_;
  mutex memory_pool_mu_; //ensure thread safety of embedding_mem_pool_
};
} // embedding
} // tensorflow

#endif  // GOOGLE_CUDA
#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_STORAGE_H_
