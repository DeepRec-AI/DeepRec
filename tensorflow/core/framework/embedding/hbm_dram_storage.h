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
    MultiTierStorage<K, V>::ReleaseValues(
        {std::make_pair(hbm_kv_, gpu_alloc_),
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
      return s;
    }
    // Insert Failed, key already exist
    (*value_ptr)->Destroy(gpu_alloc_);
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

  void CopyBackToGPU(int total, const K* keys,
      const std::list<int64>& copyback_cursor,
      V** memcpy_address, size_t value_len,
      ValuePtr<V> **gpu_value_ptrs, V* memcpy_buffer_gpu) override {
    auto memcpy_buffer_cpu = (V*)malloc(total * value_len * sizeof(V));
    int64 i = 0;
    auto it = copyback_cursor.cbegin();
    for ( ; it != copyback_cursor.cend(); ++it, ++i) {
      ValuePtr<V>* gpu_value_ptr = layout_creator_->Create(gpu_alloc_, value_len);
      //Copy Header Info
      int64 j = *it;
      memcpy((char *)gpu_value_ptr->GetPtr(),
             (char *)memcpy_address[j] - sizeof(FixedLengthHeader),
             sizeof(FixedLengthHeader));
      V* cpu_data_address = memcpy_address[j];
      V* gpu_data_address = gpu_value_ptr->GetValue(0, 0);
      memcpy(memcpy_buffer_cpu + i * value_len,
            cpu_data_address, value_len * sizeof(V));
      gpu_value_ptrs[i] = gpu_value_ptr;
    }

    cudaMemcpy(memcpy_buffer_gpu, memcpy_buffer_cpu,
        total * value_len * sizeof(V), cudaMemcpyHostToDevice);
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

 protected:
  void SetTotalDims(int64 total_dims) override {
    dram_kv_->SetTotalDims(total_dims);
  }

 private:
  KVInterface<K, V>* hbm_kv_;
  KVInterface<K, V>* dram_kv_;
  Allocator* gpu_alloc_;
  Allocator* cpu_alloc_;
  LayoutCreator<V>* layout_creator_;
  mutex hbm_mu_; //must be locked before dram_mu_ is locked;
  mutex dram_mu_;
};
} // embedding
} // tensorflow

#endif  // GOOGLE_CUDA
#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_STORAGE_H_
