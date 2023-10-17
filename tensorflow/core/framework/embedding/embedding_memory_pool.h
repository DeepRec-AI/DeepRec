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
=======================================================================*/

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_MEMORY_POOL_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_MEMORY_POOL_H_
#include <deque>

namespace tensorflow {
namespace embedding {
template<typename V>
class EmbeddingMemoryPool {
 public:
  explicit EmbeddingMemoryPool(
      Allocator* alloc,
      int64 value_len,
      int64 block_size): alloc_(alloc),
                         value_len_(value_len),
                         block_size_(block_size) {
    embs_per_block_ = block_size_ / (sizeof(V) * value_len_);
    CreateBlock();
  }

  ~EmbeddingMemoryPool() {
    for (auto it : block_list_) {
      alloc_->DeallocateRaw(it);
    }
  }

  V* Allocate() {
    if (free_ptr_queue_.size() == 0) {
      CreateBlock();
    }
    V* ptr = free_ptr_queue_.front();
    free_ptr_queue_.pop_front();
    return ptr;
  }

  void Deallocate(std::vector<void*> value_ptrs) {
    int64 prev_size = value_ptrs_queue_.size();
    for (auto it : value_ptrs) {
      value_ptrs_queue_.emplace_back(it);
    }
    if (value_ptrs_queue_.size() > embs_per_block_) {
      int64 n = value_ptrs_queue_.size() - embs_per_block_;
      n = std::min(prev_size, n);
      for (int64 i = 0; i < n; i++) {
        void* val = value_ptrs_queue_.front();
        free_ptr_queue_.emplace_back((V*)val);
        value_ptrs_queue_.pop_front();
      }
    }
  }

  void Deallocate(V* ptr) {
    free_ptr_queue_.emplace_back(ptr);
  }

 private:
  void CreateBlock() {
    V* dev_addr =
        (V*)alloc_->AllocateRaw(
            Allocator::kAllocatorAlignment,
            sizeof(V) * value_len_ * embs_per_block_);
    block_list_.emplace_back(dev_addr);
    for (int64 i = 0; i < embs_per_block_; i++) {
      free_ptr_queue_.emplace_back(dev_addr + i * value_len_);
    }
  }

  int64 block_size_;
  int64 value_len_;
  int64 embs_per_block_;
  Allocator* alloc_;
  std::deque<V*> free_ptr_queue_;
  std::deque<void*> value_ptrs_queue_;
  std::vector<V*> block_list_;
};
} //embedding
} //tensorflow

#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_MEMORY_POOL_H_
