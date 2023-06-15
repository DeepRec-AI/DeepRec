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
template<class K, class  V>
class HbmValueIterator: public ValueIterator<V> {
 public:
  HbmValueIterator(
      const std::vector<K>& key_list,
      const std::vector<ValuePtr<V>*>& value_ptr_list,
      int64 emb_index,
      int64 value_len,
      Allocator* alloc)
      : value_len_(value_len),
        alloc_(alloc) {
    int64 emb_offset = value_len_ * emb_index;
    std::vector<std::list<V*>> value_parts_vec(kSavedPartitionNum);
    for (int64 i = 0; i < key_list.size(); i++) {
      for (int part_id = 0; part_id < kSavedPartitionNum; part_id++) {
        if (key_list[i] % kSavedPartitionNum == part_id) {
          value_parts_vec[part_id].emplace_back(
              value_ptr_list[i]->GetValue(emb_index, emb_offset));
          break;
        }
      }
    }

    for (int64 i = 0; i < kSavedPartitionNum; i++) {
      values_.splice(values_.end(), value_parts_vec[i]);
    }

    values_iter_ = values_.begin();

    num_of_embs_ = buffer_capacity_ / value_len_;
    dev_addr_list_ = (V**)alloc_->AllocateRaw(
        Allocator::kAllocatorAlignment,
        num_of_embs_ * sizeof(V*));
    dev_embedding_buffer_ = (V*)alloc_->AllocateRaw(
        Allocator::kAllocatorAlignment,
        buffer_capacity_ * sizeof(V));

    FillEmbeddingBuffer();
  }

  ~HbmValueIterator() {
    alloc_->DeallocateRaw(dev_addr_list_);
    alloc_->DeallocateRaw(dev_embedding_buffer_);
  }

  V* Next() {
    if (buffer_cursor_ == num_of_embs_) {
      FillEmbeddingBuffer();
      buffer_cursor_ = 0;
    }

    V* val = embedding_buffer_ + value_len_ * buffer_cursor_;
    counter_++;
    values_iter_++;
    buffer_cursor_++;
    return val;
  }

 private:
  void FillEmbeddingBuffer() {
    int64 total_num = std::min(
        num_of_embs_,
        (int64)(values_.size() - counter_));
    std::vector<V*> local_addr_list(total_num);
    auto iter = values_iter_;
    for (int64 i = 0; i < total_num; i++) {
      local_addr_list[i] = *iter;
      iter++;
    }
    cudaMemcpy(dev_addr_list_,
               local_addr_list.data(),
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
  }
 private:
  std::list<V*> values_;
  typename std::list<V*>::iterator values_iter_;
  const static int64 buffer_capacity_ = 1024 * 1024 * 1;
  V embedding_buffer_[buffer_capacity_];
  int64 counter_ = 0;
  int64 buffer_cursor_ = 0;
  int64 value_len_;
  int64 num_of_embs_ = 0;
  Allocator* alloc_;
  V** dev_addr_list_;
  V* dev_embedding_buffer_;
};

} // embedding
} // tensorflow

#endif // GOOGLE_CUDA
#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_STORAGE_ITERATOR_H_
