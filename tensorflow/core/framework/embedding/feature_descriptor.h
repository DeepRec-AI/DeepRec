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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FEATURE_DESCRIPTOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FEATURE_DESCRIPTOR_H_
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/counter_filter_descriptor_impl.h"
#include "tensorflow/core/framework/embedding/dynamic_dim_feature_descriptor_impl.h"
#include "tensorflow/core/framework/embedding/feature_descriptor_impl.h"
#include "tensorflow/core/framework/embedding/hbm_multi_tier_feature_descriptor.h"
#include "tensorflow/core/framework/embedding/normal_feature_descriptor.h"
#include <list>

namespace tensorflow {
namespace embedding {

template <class V>
class HbmMultiTierFeatureDescriptorImpl;

template <class V>
class NormalFeatureDescriptorImpl;

template <class V>
class CounterFilterDescriptorImpl;

template <class V>
class FeatureDescriptor {
 public:
  FeatureDescriptor(
      int64 block_num,
      int64 slot_num,
      Allocator* alloc,
      StorageType storage_type,
      bool need_record_freq,
      bool need_record_version,
      const std::pair<bool, int64>& filter_info) {
    if (block_num > 1) {
      feat_desc_impl_.reset(
          new DynmaicDimDescriptorImpl<V>(
              alloc, block_num * slot_num));
    } else if (filter_info.first) {
      feat_desc_impl_.reset(
          new CounterFilterDescriptorImpl<V>(
              alloc, slot_num,
              need_record_freq,
              need_record_version,
              filter_info.second,
              storage_type));
    } else if (storage_type == StorageType::HBM_DRAM || 
               storage_type == StorageType::HBM_DRAM_SSDHASH) {
      feat_desc_impl_.reset(
          new HbmMultiTierFeatureDescriptorImpl<V>(
              alloc, slot_num,
              need_record_freq,
              need_record_version));
    } else {
      feat_desc_impl_.reset(
          new NormalFeatureDescriptorImpl<V>(
              alloc, slot_num,
              need_record_freq,
              need_record_version));
    }
  }

  FeatureDescriptor(FeatureDescriptor<V>* feat_desc) {
    if (typeid(*(feat_desc->feat_desc_impl_.get())) == 
        typeid(CounterFilterDescriptorImpl<V>*)) {
      feat_desc_impl_.reset(
        new CounterFilterDescriptorImpl<V>(
          dynamic_cast<CounterFilterDescriptorImpl<V>*>(
              feat_desc->feat_desc_impl_.get())));
    }
    else if (typeid(*(feat_desc->feat_desc_impl_.get())) ==
        typeid(HbmMultiTierFeatureDescriptorImpl<V>)) {
      feat_desc_impl_.reset(
          new NormalFeatureDescriptorImpl<V>(
              dynamic_cast<HbmMultiTierFeatureDescriptorImpl<V>*>(
                  feat_desc->feat_desc_impl_.get())));
    }
    else {
      feat_desc_impl_.reset(
          new NormalFeatureDescriptorImpl<V>(
              dynamic_cast<NormalFeatureDescriptorImpl<V>*>(
                  feat_desc->feat_desc_impl_.get())));
    }
  }

  bool InitSlotInfo(int emb_index, int64 embedding_dim,
                    const std::pair<V*, int64>& default_value) {
    return feat_desc_impl_->InitSlotInfo(
        emb_index, embedding_dim, default_value);
  }

  bool InitSlotInfo(FeatureDescriptor<V>* feat_desc) {
    return feat_desc_impl_->InitSlotInfo(feat_desc->feat_desc_impl_.get());
  }

  V* GetEmbedding(void *val, int emb_index) {
    return feat_desc_impl_->GetEmbedding(val, emb_index);
  }

  void* Allocate() {
    return feat_desc_impl_->Allocate();
  }

  void* Allocate(int64 freq) {
    return feat_desc_impl_->Allocate(freq);
  }

  void Deallocate(void* val) {
    feat_desc_impl_->Deallocate(val);
  }

  void Deallocate(const std::vector<void*>& value_ptrs) {
    feat_desc_impl_->Deallocate(value_ptrs);
  }

  void SetDefaultValue(void* val, int64 index) {
    feat_desc_impl_->SetDefaultValue(val, index);
  }

  void SetValue(void* val, int64 emb_index, V* value) {
    feat_desc_impl_->SetValue(val, emb_index, value);
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
    reinterpret_cast<HbmMultiTierFeatureDescriptorImpl<V>*>(feat_desc_impl_.get())->SetDefaultValues(
        keys, init_cursor, value_ptrs,
        compute_stream, event_mgr, gpu_device);
  }
#endif

  void SetAllocator(Allocator* alloc) {
    feat_desc_impl_->SetAllocator(alloc);
  }

  int data_bytes() {
    return feat_desc_impl_->data_bytes();
  }

  int64 GetFreq(void* val) {
    return feat_desc_impl_->GetFreq(val);
  }

  int64 GetVersion(void* val) {
    return feat_desc_impl_->GetVersion(val);
  }

  void SetFreq(void* val, int64 freq) {
    feat_desc_impl_->SetFreq(val, freq);
  }

  void UpdateVersion(void* val, int64 version) {
    feat_desc_impl_->UpdateVersion(val, version);
  }

  void AddFreq(void* val, int64 freq) {
    feat_desc_impl_->AddFreq(val, freq);
  }

  int total_dim() {
    return feat_desc_impl_->total_dim();
  }
  
  bool IsAdmit(void* val) {
    return feat_desc_impl_->IsAdmit(val);
  }

  void* Admit(void* val) {
    return feat_desc_impl_->Admit(val);
  }


 protected:
  std::unique_ptr<FeatureDescriptorImpl<V>> feat_desc_impl_;
};
} //namespace embedding
} //namespace tensorflow

#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FEATURE_DESCRIPTOR_H_
