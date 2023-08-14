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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_COUNTER_FILTER_DESCRIPTOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_COUNTER_FILTER_DESCRIPTOR_H_
#include <list>
#include "tensorflow/core/framework/embedding/feature_descriptor_impl.h"

namespace tensorflow {
namespace embedding {
template <class V>
class HbmMultiTierFeatureDescriptorImpl;

template <class V>
class NormalFeatureDescriptorImpl;

template <class V>
class CounterFilterDescriptorImpl: public FeatureDescriptorImpl<V> {
 public:
  CounterFilterDescriptorImpl(
      Allocator* alloc,
      int64 slot_num,
      bool need_record_freq,
      bool need_record_version,
      int64 filter_freq,
      StorageType storage_type) 
      : filter_freq_(filter_freq),
        is_record_freq_(need_record_freq),
        FeatureDescriptorImpl<V>(slot_num,
                                 need_record_freq,
                                 need_record_version) {
    if (filter_freq >= (1L << version_offset_bits_)) {
      LOG(FATAL)<<"Filter freqeuncy threshold shouldn't bigger than 2^12.";
    }

    if (storage_type == StorageType::HBM_DRAM || 
        storage_type == StorageType::HBM_DRAM_SSDHASH) {
#if GOOGLE_CUDA
      feat_desc_impl_.reset(
          new HbmMultiTierFeatureDescriptorImpl<V>(
              alloc, slot_num,
              need_record_freq,
              need_record_version));
#endif //GOOGLE_CUDA
    } else {
      feat_desc_impl_.reset(
          new NormalFeatureDescriptorImpl<V>(
              alloc, slot_num,
              need_record_freq,
              need_record_version));
    }
  }

  CounterFilterDescriptorImpl(CounterFilterDescriptorImpl<V>* feat_desc_impl)
      : filter_freq_(feat_desc_impl->filter_freq_),
        FeatureDescriptorImpl<V>(feat_desc_impl) {
#if GOOGLE_CUDA
    if (typeid(*(feat_desc_impl->feat_desc_impl_.get())) == 
        typeid(HbmMultiTierFeatureDescriptorImpl<V>*)){
      feat_desc_impl_.reset(
          new NormalFeatureDescriptorImpl<V>(
              dynamic_cast<HbmMultiTierFeatureDescriptorImpl<V>*>(
                  feat_desc_impl->feat_desc_impl_.get())));
    } else {
#endif //GOOGLE_CUDA
      feat_desc_impl_.reset(
          new NormalFeatureDescriptorImpl<V>(
              dynamic_cast<NormalFeatureDescriptorImpl<V>*>(
                  feat_desc_impl->feat_desc_impl_.get())));
#if GOOGLE_CUDA
    }
#endif //GOOGLE_CUDA
  }

  ~CounterFilterDescriptorImpl() {}

  bool InitSlotInfo(int emb_index, int64 embedding_dim,
      const std::pair<V*, int64>& default_value) override {
    return feat_desc_impl_->InitSlotInfo(
        emb_index, embedding_dim, default_value);
  }

  bool InitSlotInfo(FeatureDescriptorImpl<V>* feat_desc_impl) override {
    return feat_desc_impl_->InitSlotInfo(feat_desc_impl);
  }

  V* GetEmbedding(void* val, int emb_index) override {
    return feat_desc_impl_->GetEmbedding(val, emb_index);
  }

  bool IsAdmit(void* val) override {
    return (GetFlag(val) == 0);
  }

  void* Admit(void* val) override {
    if (!IsAdmit(val)) {
      return feat_desc_impl_->Allocate();
    }
  }

  void* Allocate() override {
    uint64* val = (uint64*)alloc_->AllocateRaw(
        Allocator::kAllocatorAlignment, alloc_bytes_);
    uint64 flag = 1L << flag_offset_bits_;
    uint64 version = (0xffffffffffffffff << version_offset_bits_);
    uint64 freq = 0;
    *val = version + freq;
    val = (uint64*)((uint64)val | flag);
    return (void*)val;
  }

  void* Allocate(int64 freq) override {
    if (freq < filter_freq_) {
      return Allocate();
    } else {
      return feat_desc_impl_->Allocate();
    }
  }

  void Deallocate(void* val) override {
    if (IsAdmit(val)) {
      feat_desc_impl_->Deallocate(val);
    } else {
      void* tmp = GetPtr(val);
      alloc_->DeallocateRaw(tmp);
    }
  }

  void Deallocate(const std::vector<void*>& vals) override {
    for (auto val: vals) {
      if (IsAdmit(val)) {
        feat_desc_impl_->Deallocate(val);
      } else {
        void* tmp = GetPtr(val);
        alloc_->DeallocateRaw(tmp);
      }
    }
  }

  void AddFreq(void* val, int64 count) override {
    uint64* tmp = (uint64*)GetPtr(val);
    if (!IsAdmit(val)) {
      __sync_fetch_and_add(tmp, count);
    } else {
      feat_desc_impl_->AddFreq(val, count);
    }
  }

  void SetAllocator(Allocator* alloc) override {
    feat_desc_impl_->SetAllocator(alloc);
  }

  void SetValue(void* val, int64 emb_index, V* value) {
    if (IsAdmit(val)) {
      feat_desc_impl_->SetValue(val, emb_index, value);
    }
  }

  void SetDefaultValue(void* val, int64 key) override {
    feat_desc_impl_->SetDefaultValue(val, key);
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
    feat_desc_impl_->SetDefaultValues(
        keys, init_cursor,
        value_ptrs, compute_stream,
        event_mgr, gpu_device);
  }
#endif

  int64 GetFreq(void* val) override {
    if (!IsAdmit(val)) {
      void* tmp = GetPtr(val);
      return *((uint64*)tmp) &
             ((1L << version_offset_bits_) - 1);
    } else {
      if (is_record_freq_) {
        return feat_desc_impl_->GetFreq(val);
      } else {
        return filter_freq_;
      } 
    }
  }

  int64 GetVersion(void* val) override {
    if (!IsAdmit(val)) {
      void* tmp = GetPtr(val);
      int64 version = *(uint64*)tmp >> version_offset_bits_;
      if (version == 0xffffffffffff) {
        version = -1;
      }
      return version;
    } else {
      return feat_desc_impl_->GetVersion(val);
    }
  }

  void UpdateVersion(void* val, int64 version) override {
    if (!IsAdmit(val)) {
      void* tmp_ptr = GetPtr(val);
      uint64 tmp_val = 0;
      uint64 result  = 0;
      do {
        tmp_val = *(uint64*)tmp_ptr;
        version = version << version_offset_bits_;
        uint64 freq = tmp_val & ((1L << version_offset_bits_) - 1);
        result = version + freq;
      } while(!__sync_bool_compare_and_swap((uint64*)tmp_ptr, tmp_val, result));
    } else {
      feat_desc_impl_->UpdateVersion(val, version);
    }
  }

  void SetFreq(void* val, int64 freq) override {
    uint64* tmp_ptr = (uint64*)GetPtr(val);
    if (!IsAdmit(val)) {
      uint64 tmp = *tmp_ptr;
      tmp = ~((1L << version_offset_bits_) - 1) & tmp;
      tmp += freq;
      __sync_bool_compare_and_swap(tmp_ptr, *tmp_ptr, tmp);
    } else {
      feat_desc_impl_->SetFreq(val, freq);
    }
  }

  int data_bytes() override {
    return alloc_bytes_;
  }
 private:
  uint64 GetFlag(void* val) {
    return (uint64)val >> flag_offset_bits_;
  }

  void* GetPtr(void* val) {
    return (void*)((uint64)val & ((1L << flag_offset_bits_) - 1));
  }

  int64 filter_freq_;
  int alloc_bytes_ = 8;
  Allocator* alloc_ = ev_allocator();
  const int freq_offset_bits_ = 0;
  const int version_offset_bits_ = 16;
  const int flag_offset_bits_ = 48;
  std::unique_ptr<FeatureDescriptorImpl<V>> feat_desc_impl_;
  bool is_record_freq_;
};
} //namespace embedding
} //namespace tensorflow

#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_COUNTER_FILTER_DESCRIPTOR_H_
