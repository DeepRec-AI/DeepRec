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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_NORMAL_FEATURE_DESCRIPTOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_NORMAL_FEATURE_DESCRIPTOR_H_
#include <list>
#include "tensorflow/core/framework/embedding/feature_descriptor_impl.h"

namespace tensorflow {
namespace embedding {
#if GOOGLE_CUDA
template <class V>
class HbmMultiTierFeatureDescriptorImpl;
#endif

template<class V>
class NormalFeatureDescriptorImpl: public FeatureDescriptorImpl<V> {
 public:
  NormalFeatureDescriptorImpl(Allocator* alloc, int64 slot_num,
                          bool need_record_freq,
                          bool need_record_version)
      : alloc_bytes_(0),
        alloc_(alloc),
        FeatureDescriptorImpl<V>(slot_num,
                                 need_record_freq,
                                 need_record_version) {}
  
  NormalFeatureDescriptorImpl(NormalFeatureDescriptorImpl<V>* feat_desc_impl)
      : alloc_(feat_desc_impl->alloc_),
        FeatureDescriptorImpl<V>(feat_desc_impl) {}

  NormalFeatureDescriptorImpl(
      HbmMultiTierFeatureDescriptorImpl<V>* feat_desc_impl)
      : alloc_bytes_(0),
        alloc_(feat_desc_impl->dram_alloc_),
        FeatureDescriptorImpl<V>(feat_desc_impl) {}

  ~NormalFeatureDescriptorImpl() {}
  
  bool InitSlotInfo(int emb_index, int64 embedding_dim,
                    const std::pair<V*, int64>& default_value) override {
    bool is_compute_alloc_bytes = FeatureDescriptorImpl<V>::SetEmbeddingInfo(
        emb_index, embedding_dim, default_value);
    if (is_compute_alloc_bytes) {
      FeatureDescriptorImpl<V>::ComputeAllocBytes(&alloc_bytes_);
      FeatureDescriptorImpl<V>::CreateFreqAndVersionDescriptor(&alloc_bytes_);
    }
    return is_compute_alloc_bytes;
  }

  bool InitSlotInfo(FeatureDescriptorImpl<V>* feat_desc_impl) override {
    FeatureDescriptorImpl<V>::SetSlotInfo(feat_desc_impl);
    FeatureDescriptorImpl<V>::ComputeAllocBytes(&alloc_bytes_);
    FeatureDescriptorImpl<V>::SetFreqAndVersionOffset(&alloc_bytes_);
    return true;
  }

  V* GetEmbedding(void *val, int emb_index) override {
    return reinterpret_cast<V*>(val)
        + FeatureDescriptorImpl<V>::slot_infos_[emb_index].embedding_offset;
  }

  void* Allocate() override {
    void* val = alloc_->AllocateRaw(
        Allocator::kAllocatorAlignment, alloc_bytes_);
    FeatureDescriptorImpl<V>::InitFreqAndVersion(val);
    return val;
  }

  void Deallocate(void* val) override {
    alloc_->DeallocateRaw(val);
  }

  void Deallocate(const std::vector<void*>& value_ptrs) override {
    for (auto val: value_ptrs) {
      Deallocate(val);
    }
  }

  void SetValue(void* val, int64 emb_index, V* value) override {
    V* val_ptr = GetEmbedding(val, emb_index);
    memcpy(val_ptr, value,
        sizeof(V) * FeatureDescriptorImpl<V>::slot_infos_[emb_index].default_value_len);
  }

  void SetDefaultValue(void* val, int64 index) override {
    for (int i = 0; i < FeatureDescriptorImpl<V>::slot_infos_.size(); i++) {
      V* val_ptr = GetEmbedding(val, i);
      FeatureDescriptorImpl<V>::SetDefaultValue((void*)val_ptr, i, index);
    }
  }

#if GOOGLE_CUDA
  template <class K>
  void SetDefaultValues(
      const K* keys,
      const std::list<int64>& init_cursor,
      void** value_ptrs,
      se::Stream* compute_stream,
      EventMgr* event_mgr,
      const Eigen::GpuDevice& gpu_device) {
    LOG(FATAL)<<"Can't call SetDefaultValue(const K*, const std::list<int64>&,"
              <<"void**, se::Stream*, EventMgr*, const Eigen::GpuDevice&)"
              <<" in HbmMultiTierFeatureDescriptor.";
  }
#endif

  void SetAllocator(Allocator* alloc) override {
    alloc_ = alloc;
  }

  int data_bytes() override {
    return alloc_bytes_;
  }

 private:
  int alloc_bytes_;
  Allocator* alloc_;
};
} //namespace embedding
} //namespace tensorflow

#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_NORMAL_FEATURE_DESCRIPTOR_H_
