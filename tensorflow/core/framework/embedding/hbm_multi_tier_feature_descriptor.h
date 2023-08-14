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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_MULTI_TIER_FEATURE_DESCRIPTOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_MULTI_TIER_FEATURE_DESCRIPTOR_H_
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/framework/embedding/feature_descriptor_impl.h"
#include "tensorflow/core/framework/embedding/embedding_memory_pool.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace embedding {
template <class V>
class NormalFeatureDescriptorImpl;

template<class V>
class HbmMultiTierFeatureDescriptorImpl
    : public FeatureDescriptorImpl<V> {
 public:
  HbmMultiTierFeatureDescriptorImpl(
      Allocator* alloc, int64 slot_num,
      bool need_record_freq,
      bool need_record_version)
      : dram_alloc_bytes_(sizeof(V*)),
        hbm_alloc_(alloc),
        dram_alloc_(ev_allocator()),
        FeatureDescriptorImpl<V>(slot_num,
                                 need_record_freq,
                                 need_record_version) {
    FeatureDescriptorImpl<V>::CreateFreqAndVersionDescriptor(&dram_alloc_bytes_);
  }

  ~HbmMultiTierFeatureDescriptorImpl() {}
  
  bool InitSlotInfo(int emb_index, int64 embedding_dim,
                    const std::pair<V*, int64>& default_value) override {
    bool is_compute_alloc_bytes =
        FeatureDescriptorImpl<V>::SetEmbeddingInfo(
            emb_index, embedding_dim, default_value);
    if (is_compute_alloc_bytes) {
      FeatureDescriptorImpl<V>::ComputeAllocBytes(&hbm_alloc_bytes_);
      embedding_mem_pool_.reset(
        new EmbeddingMemoryPool<V>(hbm_alloc_,
                                   hbm_alloc_bytes_ / sizeof(V),
                                   1024 * 1024 * 64));
    }
    return is_compute_alloc_bytes;
  }

  V* GetEmbedding(void *val, int emb_index) override {
    return *((V**)val) +
        FeatureDescriptorImpl<V>::slot_infos_[emb_index].embedding_offset;
  }

  void* Allocate() override {
    void* val = dram_alloc_->AllocateRaw(
        Allocator::kAllocatorAlignment, dram_alloc_bytes_);
    mutex_lock l(memory_pool_mu_);
    *((V**)val) = embedding_mem_pool_->Allocate();
    FeatureDescriptorImpl<V>::InitFreqAndVersion(val);
    return val;
  }

  void Deallocate(void* val) override {
    mutex_lock l(memory_pool_mu_);
    embedding_mem_pool_->Deallocate(*((V**)val));
    dram_alloc_->DeallocateRaw(val);
  }

  void Deallocate(const std::vector<void*>& value_ptrs) override {
    mutex_lock l(memory_pool_mu_);
    for (auto ptr: value_ptrs) {
      embedding_mem_pool_->Deallocate(*((V**)ptr));
      dram_alloc_->DeallocateRaw(ptr);
    }
  }
  void SetDefaultValue(void* val, int64 key) override {
    LOG(FATAL)<<"Can't call SetDefaultValue(void* val, int64 key,"
              <<"int default_value_len) in HbmMultiTierFeatureDescriptor.";
  }

  void SetAllocator(Allocator* alloc) override {
    hbm_alloc_ = alloc;
  }

  template <class K>
  void SetDefaultValues(
      const K* keys,
      const std::list<int64>& init_cursor,
      void** value_ptrs,
      se::Stream* compute_stream,
      EventMgr* event_mgr,
      const Eigen::GpuDevice& gpu_device);

  int data_bytes() override {
    return dram_alloc_bytes_;
  }
 public:
  friend class NormalFeatureDescriptorImpl<V>;
 protected:
  int dram_alloc_bytes_;
  int hbm_alloc_bytes_ = 0;
  mutex memory_pool_mu_; //ensure thread safety of embedding_mem_pool_
  Allocator* hbm_alloc_;
  Allocator* dram_alloc_;
  std::unique_ptr<EmbeddingMemoryPool<V>> embedding_mem_pool_;
};
} //namespace embedding
} //namespace tensorflow

#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_MULTI_TIER_FEATURE_DESCRIPTOR_H_
