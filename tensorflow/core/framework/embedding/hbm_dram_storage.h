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
#include "tensorflow/core/framework/embedding/lockless_hash_map_cpu.h"
#include "tensorflow/core/framework/embedding/multi_tier_storage.h"
#include "tensorflow/core/framework/embedding/hbm_storage_iterator.h"

namespace tensorflow {
template <class V>
class ValuePtr;

namespace embedding {
template<typename K, typename V>
class HbmDramStorage : public MultiTierStorage<K, V> {
 public:
  HbmDramStorage(const StorageConfig& sc, Allocator* gpu_alloc,
      Allocator* cpu_alloc, LayoutCreator<V>* lc, const std::string& name)
      : cpu_alloc_(cpu_alloc), gpu_alloc_(gpu_alloc),
        layout_creator_(lc), MultiTierStorage<K, V>(sc, name) {
    hbm_kv_ = new LocklessHashMap<K, V>();
    dram_kv_ = new LocklessHashMapCPU<K, V>(gpu_alloc);

    MultiTierStorage<K, V>::kvs_.emplace_back(
        KVInterfaceDescriptor<K, V>(hbm_kv_, gpu_alloc_, hbm_mu_));
    MultiTierStorage<K, V>::kvs_.emplace_back(
        KVInterfaceDescriptor<K, V>(dram_kv_, cpu_alloc, dram_mu_));

  }

  ~HbmDramStorage() override {
    ReleaseValues({std::make_pair(hbm_kv_, gpu_alloc_),
                   std::make_pair(dram_kv_, cpu_alloc_)});
  }

  TF_DISALLOW_COPY_AND_ASSIGN(HbmDramStorage);

  Status Get(K key, ValuePtr<V>** value_ptr) override {
    Status s = hbm_kv_->Lookup(key, value_ptr);
    if (!s.ok()) {
      s = dram_kv_->Lookup(key, value_ptr);
    }
    return s;
  }

  void Insert(const std::vector<K>& keys,
              ValuePtr<V>** value_ptrs) override {
    for (size_t i = 0; i < keys.size(); i++) {
      do {
        Status s = hbm_kv_->Insert(keys[i], value_ptrs[i]);
        if (s.ok()) {
          break;
        } else {
          (value_ptrs[i])->Destroy(gpu_alloc_);
          delete value_ptrs[i];
        }
      } while (!(hbm_kv_->Lookup(keys[i], &value_ptrs[i])).ok());
    }
  }

  void Insert(K key, ValuePtr<V>** value_ptr,
              int64 alloc_len) override {
    do {
      *value_ptr = layout_creator_->Create(gpu_alloc_, alloc_len);
      Status s = hbm_kv_->Insert(key, *value_ptr);
      if (s.ok()) {
        break;
      } else {
        (*value_ptr)->Destroy(gpu_alloc_);
        delete *value_ptr;
      }
    } while (!(hbm_kv_->Lookup(key, value_ptr)).ok());
  }

  void InsertToDram(K key, ValuePtr<V>** value_ptr,
              int64 alloc_len) override {
    do {
      *value_ptr = new NormalContiguousValuePtr<V>(ev_allocator(), alloc_len);
      Status s = dram_kv_->Insert(key, *value_ptr);
      if (s.ok()) {
        break;
      } else {
        (*value_ptr)->Destroy(ev_allocator());
        delete *value_ptr;
      }
    } while (!(dram_kv_->Lookup(key, value_ptr)).ok());
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size) override {
    Status s = hbm_kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      return s;
    }
    s = dram_kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      // copy dram value to hbm
      auto gpu_value_ptr = layout_creator_->Create(gpu_alloc_, size);
      gpu_value_ptr->SetPtr(embedding_mem_pool_->Allocate());
      V* cpu_data_address = (*value_ptr)->GetValue(0, 0);
      V* gpu_data_address = gpu_value_ptr->GetValue(0, 0);
      cudaMemcpy(gpu_data_address, cpu_data_address,
          size * sizeof(V), cudaMemcpyHostToDevice);
      *value_ptr = gpu_value_ptr;
      return s;
    }

    *value_ptr = layout_creator_->Create(gpu_alloc_, size);
    s = hbm_kv_->Insert(key, *value_ptr);
    if (s.ok()) {
      (*value_ptr)->SetPtr(embedding_mem_pool_->Allocate());
      return s;
    }
    // Insert Failed, key already exist
    delete *value_ptr;
    return hbm_kv_->Lookup(key, value_ptr);
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size, CopyBackFlag &need_copyback) override {
    need_copyback = NOT_COPYBACK;
    Status s = hbm_kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      return s;
    }
    s = dram_kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      need_copyback = COPYBACK;
      return s;
    }

    *value_ptr = layout_creator_->Create(gpu_alloc_, size);
    s = hbm_kv_->Insert(key, *value_ptr);
    if (s.ok()) {
      return s;
    }
    // Insert Failed, key already exist
    (*value_ptr)->Destroy(gpu_alloc_);
    delete *value_ptr;
    return hbm_kv_->Lookup(key, value_ptr);
  }

  void ImportToHbm(K* ids, int64 size, int64 value_len, int64 emb_index) override {
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
        dram_kv_->Lookup(ids[i], &cpu_value_ptrs[i]);
        gpu_value_ptrs[i] = layout_creator_->Create(gpu_alloc_, value_len);
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
      Status s = hbm_kv_->Insert(ids[i], gpu_value_ptrs[i]);
      if (!s.ok()) {
        invalid_value_ptrs.emplace_back(gpu_value_ptrs[i]);
        hbm_kv_->Lookup(ids[i], &gpu_value_ptrs[i]);
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
      ValuePtr<V> **gpu_value_ptrs, V* memcpy_buffer_gpu) override {
    auto memcpy_buffer_cpu = (V*)malloc(total * value_len * sizeof(V));
    int64* memory_index = new int64[total];
    int64 i = 0;
    auto it = copyback_cursor.cbegin();
    {
      //Mutex with eviction thread
      mutex_lock l(memory_pool_mu_);
      for ( ; it != copyback_cursor.cend(); ++it, ++i) {
        int64 j = *it;
        memory_index[i] = j;
        ValuePtr<V>* gpu_value_ptr =
            layout_creator_->Create(gpu_alloc_, value_len);
        V* val_ptr = embedding_mem_pool_->Allocate();
        bool flag = gpu_value_ptr->SetPtr(val_ptr);
        if (!flag) {
          embedding_mem_pool_->Deallocate(val_ptr);
        }
        memcpy((char *)gpu_value_ptr->GetPtr(),
               (char *)memcpy_address[j] - sizeof(FixedLengthHeader),
               sizeof(FixedLengthHeader));
        V* gpu_data_address = gpu_value_ptr->GetValue(0, 0);
        gpu_value_ptrs[i] = gpu_value_ptr;
      }
    }
    //Split from above for loop for minize the cost of mutex lock
    //TODO: Speed up with intra parallelism
    for (int i = 0; i < total; i++) {
      int j = memory_index[i];
      memcpy(memcpy_buffer_cpu + i * value_len,
             memcpy_address[j], value_len * sizeof(V));
    }

    cudaMemcpy(memcpy_buffer_gpu, memcpy_buffer_cpu,
        total * value_len * sizeof(V), cudaMemcpyHostToDevice);
    free(memcpy_buffer_cpu);
    delete[] memory_index;
  }

  Status Remove(K key) override {
    hbm_kv_->Remove(key);
    dram_kv_->Remove(key);
    return Status::OK();
  }

  int64 Size() const override {
    int64 total_size = hbm_kv_->Size();
    total_size += dram_kv_->Size();
    return total_size;
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
      mutex_lock l(hbm_mu_);
      TF_CHECK_OK(hbm_kv_->GetSnapshot(key_list, value_ptr_list));
    }
    {
      mutex_lock l(dram_mu_);
      TF_CHECK_OK(dram_kv_->GetSnapshot(key_list, value_ptr_list));
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
      mutex_lock l(hbm_mu_);
      TF_CHECK_OK(hbm_kv_->GetSnapshot(&temp_hbm_key_list,
                                       &hbm_value_ptr_list));
    }
    // Get Snapshot of DRAM storage.
    {
      mutex_lock l(dram_mu_);
      TF_CHECK_OK(dram_kv_->GetSnapshot(&temp_dram_key_list,
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

  void ReleaseValues(
      const std::vector<
          std::pair<KVInterface<K, V>*, Allocator*>>& kvs) {
    for (int i = 0; i < kvs.size(); i++) {
      std::vector<K> key_list;
      std::vector<ValuePtr<V>*> value_ptr_list;
      kvs[i].first->GetSnapshot(&key_list, &value_ptr_list);
      if (i == 0) {
        delete embedding_mem_pool_;
      }
      for (auto value_ptr : value_ptr_list) {
        if (i != 0)
          value_ptr->Destroy(kvs[i].second);
        delete value_ptr;
      } 
    }
  }

  void BatchEviction() override {
    constexpr int EvictionSize = 10000;
    K evic_ids[EvictionSize];
    if (!MultiTierStorage<K, V>::ready_eviction_) {
      return;
    }
    mutex_lock l(hbm_mu_);
    mutex_lock l1(dram_mu_);

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
        if (hbm_kv_->Lookup(evic_ids[i], &value_ptr).ok()) {
          keys.emplace_back(evic_ids[i]);
          value_ptrs.emplace_back(value_ptr);
        }
      }
      dram_kv_->BatchCommit(keys, value_ptrs);
      {
        //Mutex with main thread
        mutex_lock l_mem(memory_pool_mu_);
        embedding_mem_pool_->Deallocate(value_ptrs);
      }
      for (auto it : keys) {
        TF_CHECK_OK(hbm_kv_->Remove(it));
      }
    }
  }

 protected:
  void SetTotalDims(int64 total_dims) override {
    dram_kv_->SetTotalDims(total_dims);
  }

 private:
  KVInterface<K, V>* hbm_kv_;
  KVInterface<K, V>* dram_kv_;
  EmbeddingMemoryPool<V>* embedding_mem_pool_;
  Allocator* gpu_alloc_;
  Allocator* cpu_alloc_;
  LayoutCreator<V>* layout_creator_;
  mutex hbm_mu_; //must be locked before dram_mu_ is locked;
  mutex dram_mu_;
  mutex memory_pool_mu_; //ensure thread safety of embedding_mem_pool_
};
} // embedding
} // tensorflow

#endif  // GOOGLE_CUDA
#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_STORAGE_H_
