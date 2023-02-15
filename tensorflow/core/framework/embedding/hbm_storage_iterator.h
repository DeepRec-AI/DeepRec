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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_STORAGE_ITERATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_STORAGE_ITERATOR_H_

#if GOOGLE_CUDA
#include "tensorflow/core/framework/embedding/storage.h"
namespace tensorflow {

template <class V>
class ValuePtr;

namespace embedding {
class Iterator;

template<class K, class V>
class PartitionedCheckpointData {
 public:
  PartitionedCheckpointData() {
    key_list_parts.resize(kSavedPartitionNum);
    value_list_parts.resize(kSavedPartitionNum);
    version_list_parts.resize(kSavedPartitionNum);
    freq_list_parts.resize(kSavedPartitionNum);
    key_filter_list_parts.resize(kSavedPartitionNum);
    version_filter_list_parts.resize(kSavedPartitionNum);
    freq_filter_list_parts.resize(kSavedPartitionNum);
  }

  ~PartitionedCheckpointData() {
  }

  void EmplaceToPartList(K key, ValuePtr<V>* value_ptr, bool is_on_hbm,
                         int64 emb_index, int64 emb_offset) {
    int64 part_id = key % kSavedPartitionNum;
    V* val = value_ptr->GetValue(emb_index, emb_offset);
    V* primary_val = value_ptr->GetValue(0, 0);

    int64 freq = value_ptr->GetFreq();
    int64 version = value_ptr->GetStep();
    if (primary_val == nullptr) {
      // id is filtered by feature filter.
      key_filter_list_parts[part_id].emplace_back(key);
      freq_filter_list_parts[part_id].emplace_back(freq);
      version_filter_list_parts[part_id].emplace_back(version);
    } else {
      if (val != nullptr) {
        key_list_parts[part_id].emplace_back(key);
        freq_list_parts[part_id].emplace_back(freq);
        version_list_parts[part_id].emplace_back(version);
        value_list_parts[part_id].emplace_back(
            std::pair<V*, bool>(val, is_on_hbm));
      }
    }
  }

  void GenerateKeyList(std::vector<K>* output_key_list) {
    MergePartList<K>(key_list_parts, output_key_list);
  }

  void GenerateFilteredKeyList(std::vector<K>* output_filter_key_list) {
    MergePartList<K>(key_filter_list_parts, output_filter_key_list);
  }

  void GenerateValueList(
      std::vector<std::pair<V*, bool>>* output_value_list,
      std::vector<V*>* hbm_ptr_list) {
    for (int i = 0; i < kSavedPartitionNum; i++) {
      for (int j = 0; j < value_list_parts[i].size(); j++) {
        output_value_list->emplace_back(value_list_parts[i][j]);
        if (value_list_parts[i][j].second)
          hbm_ptr_list->emplace_back(value_list_parts[i][j].first);
      }
    }
  }

  void GenerateFreqList(std::vector<int64>* output_freq_list) {
    MergePartList<int64>(freq_list_parts, output_freq_list);
  }

  void GenerateFilteredFreqList(
      std::vector<int64>* output_filter_freq_list) {
    MergePartList<int64>(freq_filter_list_parts, output_filter_freq_list);
  }

  void GenerateVersionList(
      std::vector<int64>* output_version_list) {
    MergePartList<int64>(version_list_parts, output_version_list);
  }

  void GenerateFilteredVersionList(
      std::vector<int64>* output_filter_version_list) {
    MergePartList<int64>(version_filter_list_parts,
                         output_filter_version_list);
  }

  void GeneratePartOffset(std::vector<int32>* part_offset) {
    for (int64 i = 0; i < kSavedPartitionNum; i++) {
      (*part_offset)[i + 1] = (*part_offset)[i] + key_list_parts[i].size();
    }
  }

  void GeneratePartFilterOffset(std::vector<int32>* part_filter_offset) {
    for (int64 i = 0; i < kSavedPartitionNum; i++) {
      (*part_filter_offset)[i + 1] = (*part_filter_offset)[i]
                                     + key_filter_list_parts[i].size();
    }
  }

 private:
  template<class T>
  void MergePartList(
      const std::vector<std::vector<T>>& part_list,
      std::vector<T> *output_list) {
    for (int i = 0; i < kSavedPartitionNum; i++) {
      for (int j = 0; j < part_list[i].size(); j++) {
        output_list->emplace_back(part_list[i][j]);
      }
    }
  }

  std::vector<std::vector<K>> key_list_parts;
  std::vector<std::vector<std::pair<V*, bool>>> value_list_parts;
  std::vector<std::vector<int64>> version_list_parts;
  std::vector<std::vector<int64>> freq_list_parts;
  std::vector<std::vector<K>> key_filter_list_parts;
  std::vector<std::vector<int64>> version_filter_list_parts;
  std::vector<std::vector<int64>> freq_filter_list_parts;
};

template<class K, class V>
class HbmDramIterator: public Iterator {
 public:
  HbmDramIterator(
              const std::vector<K>& hbm_key_list,
              const std::vector<K>& dram_key_list,
              const std::vector<ValuePtr<V>*>& hbm_value_ptr_list,
              const std::vector<ValuePtr<V>*>& dram_value_ptr_list,
              int64 value_len,
              Allocator* alloc,
              int64 emb_index):
              value_len_(value_len),
              alloc_(alloc),
              cursor_(0),
              hbm_ptr_cursor_(0),
              fill_buffer_st_(0),
              fill_buffer_ed_(0),
              emb_index_(emb_index) {
    part_offset_.resize(kSavedPartitionNum + 1);
    part_offset_[0] = 0;
    part_filter_offset_.resize(kSavedPartitionNum + 1);
    part_filter_offset_[0] = 0;
    emb_offset_ = value_len_ * emb_index_;
    std::set<K> hbm_keys;

    PartitionedCheckpointData<K, V> ckpt_data;
    for (int64 i = 0; i < hbm_key_list.size(); i++) {
      ckpt_data.EmplaceToPartList(
          hbm_key_list[i], hbm_value_ptr_list[i], true,
          emb_index_, emb_offset_);
      hbm_keys.insert(hbm_key_list[i]);
    }
    for (int64 i = 0; i < dram_key_list.size(); i++) {
      if (hbm_keys.find(dram_key_list[i]) == hbm_keys.end()) {
        ckpt_data.EmplaceToPartList(
            dram_key_list[i], dram_value_ptr_list[i], false,
            emb_index_, emb_offset_);
      }
    }

    ckpt_data.GenerateKeyList(&key_list_);
    ckpt_data.GenerateValueList(&value_list_, &hbm_ptr_list_);
    ckpt_data.GenerateFreqList(&freq_list_);
    ckpt_data.GenerateVersionList(&version_list_);
    ckpt_data.GeneratePartOffset(&part_offset_);

    ckpt_data.GenerateFilteredKeyList(&filtered_key_list_);
    ckpt_data.GenerateFilteredFreqList(&filtered_freq_list_);
    ckpt_data.GenerateFilteredVersionList(&filtered_version_list_);
    ckpt_data.GeneratePartFilterOffset(&part_filter_offset_);

    dev_addr_list_ = (V**)alloc_->AllocateRaw(Allocator::kAllocatorAlignment,
        buffer_capacity_ / value_len_ * sizeof(V*));
    dev_embedding_buffer_ = (V*)alloc_->AllocateRaw(Allocator::kAllocatorAlignment,
        buffer_capacity_ * sizeof(V));
    local_addr_list_ = new V*[buffer_capacity_ / value_len_];
  }

  ~HbmDramIterator() {
    alloc_->DeallocateRaw(dev_addr_list_);
    alloc_->DeallocateRaw(dev_embedding_buffer_);
    delete[] local_addr_list_;
  }

  virtual bool Valid() {
    return !(cursor_ == current_key_list_->size());
  }

  virtual void SeekToFirst() {
    cursor_ = 0;
    hbm_ptr_cursor_ = 0;
    fill_buffer_st_ = 0;
    fill_buffer_ed_ = 0;
  }

  virtual void SwitchToFilteredFeatures() {
    current_key_list_ = &filtered_key_list_;
    current_freq_list_ = &filtered_freq_list_;
    current_version_list_ = &filtered_version_list_;
  }

  virtual void SwitchToAdmitFeatures() {
    current_key_list_ = &key_list_;
    current_freq_list_ = &freq_list_;
    current_version_list_ = &version_list_;
  }

  virtual void Next() {
    cursor_++;
  }

  virtual void Key(char* val, int64 dim) {
    *((int64*)val) = (*current_key_list_)[cursor_];
  }

  virtual void Value(char* val, int64 dim, int64 value_offset) {
    if (value_list_[cursor_].second) {
      if (hbm_ptr_cursor_ == fill_buffer_ed_) {
        FillEmbeddingBuffer();
      }
      memcpy(val,
             embedding_buffer_ +
                (hbm_ptr_cursor_ - fill_buffer_st_) * value_len_,
             dim);
      V* tmp = (V*)val;
      hbm_ptr_cursor_++;
    } else {
      memcpy(val, value_list_[cursor_].first, dim);
      V* tmp = (V*)val;
    }
  }

  virtual void Freq(char* val, int64 dim) {
    *((int64*)val) = (*current_freq_list_)[cursor_];
  }

  virtual void Version(char* val, int64 dim) {
    *((int64*)val) = (*current_version_list_)[cursor_];
  }

  virtual void SetPartOffset(int32* part_offset_ptr) {
    for (int64 i = 0; i < kSavedPartitionNum + 1; i++) {
      part_offset_ptr[i] = part_offset_[i];
    }
  }

  virtual void SetPartFilterOffset(int32* part_offset_ptr) {
    for (int64 i = 0; i < kSavedPartitionNum + 1; i++) {
      part_offset_ptr[i] = part_filter_offset_[i];
    }
  }

 private:
  void FillEmbeddingBuffer() {
    int64 total_num = std::min(
        buffer_capacity_ / value_len_,
        (int64)(hbm_ptr_list_.size() - hbm_ptr_cursor_));
    fill_buffer_st_ = hbm_ptr_cursor_;
    for (int64 i = 0; i < total_num; i++) {
      local_addr_list_[i] = hbm_ptr_list_[fill_buffer_st_ + i];
    }
    cudaMemcpy(dev_addr_list_,
               local_addr_list_,
               sizeof(V*) * total_num,
               cudaMemcpyHostToDevice);
    int block_dim = 128;
    void* args[] = {(void*)&dev_addr_list_,
                     (void*)&dev_embedding_buffer_,
                     (void*)&value_len_,
                     (void*)&total_num,
                     nullptr,
                     nullptr};
    cudaLaunchKernel((void *)BatchCopy<V>,
                     (total_num + block_dim - 1) / block_dim * value_len_,
                     block_dim, args, 0, NULL);
    cudaDeviceSynchronize();
    cudaMemcpy(embedding_buffer_,
               dev_embedding_buffer_,
               sizeof(V) * total_num * value_len_,
               cudaMemcpyDeviceToHost);
    fill_buffer_ed_ = fill_buffer_st_ + total_num;
  }

  std::vector<K> key_list_;
  std::vector<std::pair<V*, bool>> value_list_;
  std::vector<int64> freq_list_;
  std::vector<int64> version_list_;
  std::vector<int32> part_offset_;
  std::vector<K> filtered_key_list_;
  std::vector<int64> filtered_freq_list_;
  std::vector<int64> filtered_version_list_;
  std::vector<int32> part_filter_offset_;
  std::vector<V*> hbm_ptr_list_;

  const static int64 buffer_capacity_ = 1024 * 1024 * 1;
  V embedding_buffer_[buffer_capacity_];
  V** dev_addr_list_;
  V* dev_embedding_buffer_;
  V** local_addr_list_;
  Allocator* alloc_;
  int64 value_len_;
  int64 cursor_;
  int64 hbm_ptr_cursor_;
  int64 fill_buffer_st_;
  int64 fill_buffer_ed_;
  int64 emb_index_;
  int64 emb_offset_;
  std::vector<K>* current_key_list_;
  std::vector<int64>* current_freq_list_;
  std::vector<int64>* current_version_list_;
};

} // embedding
} // tensorflow

#endif // GOOGLE_CUDA
#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_STORAGE_ITERATOR_H_