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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_DYNAMIC_DIM_DESCRIPTOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_DYNAMIC_DIM_DESCRIPTOR_H_
#include <list>
#include <bitset>
#include <atomic>
#include "tensorflow/core/framework/embedding/feature_descriptor_impl.h"

namespace tensorflow {
namespace embedding {
constexpr int COLUMN_BITSET_BYTES = 5;
constexpr int COLUMN_BITSET_SIZE = COLUMN_BITSET_BYTES * 8;

struct MetaHeader {
  volatile unsigned char embed_num;
  unsigned char value_type;
  unsigned char header_size;
  unsigned char column_bitset[COLUMN_BITSET_BYTES];

  static const int kEmbeddingNumStartIndex = 0;
  static const int kValueTypeStartIndex =
      kEmbeddingNumStartIndex + sizeof(char);
  static const int kHeaderSizeStartIndex =
      kValueTypeStartIndex + sizeof(char);
  static const int kColumnBitsetIndex =
      kHeaderSizeStartIndex + sizeof(char);

  inline unsigned int GetEmbeddingNum() {
    return (unsigned int) embed_num;
  }

  inline void SetEmbeddingNum(size_t s) {
    embed_num = (unsigned char)s;
  }

  inline std::bitset<COLUMN_BITSET_SIZE> GetColumnBitset() {
    unsigned long meta = ((unsigned long*)this)[0];
    std::bitset<COLUMN_BITSET_SIZE> bs(meta >> (8 * kColumnBitsetIndex));
    return bs;
  }

  inline void SetColumnBitset(const std::bitset<COLUMN_BITSET_SIZE>& bs,
      unsigned int embnum) {
    ((unsigned long*)(this))[0] =
      (bs.to_ulong() << (8 * kColumnBitsetIndex)) |
      (header_size << (8 * kHeaderSizeStartIndex)) |
      (value_type << (8 * kValueTypeStartIndex)) |
      (embnum << (8 * kEmbeddingNumStartIndex));
  }

  inline unsigned int GetHeaderSize() {
    return (unsigned int) header_size;
  }

  inline void SetHeaderSize(size_t size) {
    header_size = (unsigned char)size;
  }
};

template <class V>
class DynmaicDimDescriptorImpl: public FeatureDescriptorImpl<V> {
using FeatureDescriptorImpl<V>::slot_infos_;
 public:
  DynmaicDimDescriptorImpl(
      Allocator* alloc,
      int64 slot_num) 
      : alloc_bytes_(sizeof(std::atomic_flag) +
                     sizeof(MetaHeader) +
                     sizeof(V*) * slot_num),
        header_offset_bytes_(sizeof(V*) * slot_num),
        flag_offset_bytes_(sizeof(MetaHeader) +
                           sizeof(V*) * slot_num),
        FeatureDescriptorImpl<V>(slot_num,
                                 false,
                                 false) {
    FeatureDescriptorImpl<V>::CreateFreqAndVersionDescriptor(&alloc_bytes_);
  }
  ~DynmaicDimDescriptorImpl() {}

  bool InitSlotInfo(int emb_index, int64 embedding_dim,
      const std::pair<V*, int64>& default_value) override {
    return FeatureDescriptorImpl<V>::SetEmbeddingInfo(
        emb_index, embedding_dim, default_value);
  } 

  V* GetEmbedding(void* val, int emb_index) override {
		MetaHeader* meta = (MetaHeader*)(val + header_offset_bytes_);
    unsigned int embnum = (unsigned int)meta->embed_num;
    auto metadata = meta->GetColumnBitset();
    
    if (!metadata.test(emb_index)) {
      std::atomic_flag* flag= (std::atomic_flag*)(val + flag_offset_bytes_);
      while(flag->test_and_set(std::memory_order_acquire));
      metadata = meta->GetColumnBitset();
      if (metadata.test(emb_index)) {
        flag->clear(std::memory_order_release);
        return ((V**)val)[emb_index];
      }
      embnum++ ;
      int64 alloc_value_len = slot_infos_[emb_index].embedding_dim;
      V* tensor_val = (V*)alloc_->AllocateRaw(
          Allocator::kAllocatorAlignment, sizeof(V) * alloc_value_len);
      V* default_v = (V*)slot_infos_[emb_index].default_value;
      memcpy(tensor_val, default_v,
             sizeof(V) * slot_infos_[emb_index].default_value_len);
      ((V**)val)[emb_index] = tensor_val;

      metadata.set(emb_index);
      // NOTE:if we use ((unsigned long*)((char*)ptr_ + 1))[0] = metadata.to_ulong();
      // the ptr_ will be occaionally  modified from 0x7f18700912a0 to 0x700912a0
      // must use  ((V**)ptr_ + 1 + 1)[emb_index] = tensor_val;  to avoid
      //LOG(INFO)<<"emb_num: "<<embnum;
      meta->SetColumnBitset(metadata, embnum);
      flag->clear(std::memory_order_release);
      return tensor_val;
    } else {
      return ((V**)val)[emb_index];
    }
  }

  bool IsAdmit(void* val) override {
    return true;
  }

  void* Admit(void* val) override {}

  void* Allocate() override {
    void* val = alloc_->AllocateRaw(
        Allocator::kAllocatorAlignment, alloc_bytes_);
    memset(val, 0, alloc_bytes_);
    new ((char*)val + header_offset_bytes_) MetaHeader();
    return val;
  }

  void Deallocate(void* val) override {
    MetaHeader* meta = (MetaHeader*)(val + header_offset_bytes_);
    unsigned int embnum = (unsigned int)meta->GetEmbeddingNum();
    //LOG(INFO)<<"emb_num in deallocate: "<<embnum;
    auto metadata = meta->GetColumnBitset();
    for (int i = 0; i< embnum; i++) {
      if (metadata.test(i)) {
        V* val_ptr = ((V**)((int64*)val + meta->GetHeaderSize()))[i];
        if (val_ptr != nullptr) {
          alloc_->DeallocateRaw(val_ptr);
        }
      }
    }
  }

  void Deallocate(const std::vector<void*>& vals) override {
    for (auto val: vals) {
      Deallocate(val);
    }
  }

  void AddFreq(void* val, int64 count) override {}

  void SetAllocator(Allocator* alloc) override {
    alloc_ = alloc;
  }

  void SetDefaultValue(void* val, int64 key) override {}

  void SetValue(void* val, int64 emb_index, V* value) override {
    V* val_ptr = GetEmbedding(val, emb_index);
    memcpy(val_ptr, value,
        sizeof(V) * FeatureDescriptorImpl<V>::slot_infos_[emb_index].default_value_len);
  }

#if GOOGLE_CUDA
  template <class K>
  void SetDefaultValues(
      const K* keys,
      const std::list<int64>& init_cursor,
      void** value_ptrs,
      se::Stream* compute_stream,
      EventMgr* event_mgr,
      const Eigen::GpuDevice& gpu_device) {}
#endif

  int64 GetFreq(void* val) override {}

  int64 GetVersion(void* val) override {}

  void UpdateVersion(void* val, int64 version) override {}

  void SetFreq(void* val, int64 freq) override {}

  int data_bytes() override {
    return alloc_bytes_;
  }
 private:
  int alloc_bytes_ = 0;
  int header_offset_bytes_ = 0;
  int flag_offset_bytes_ = 0;
  Allocator* alloc_ = ev_allocator();
};
} //namespace embedding
} //namespace tensorflow

#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_COUNTER_FILTER_DESCRIPTOR_H_
