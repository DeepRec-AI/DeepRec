/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/gpu/gpu_adjustable_allocator.h"

namespace tensorflow {

void GPUAdjustableAllocator::GetMemPoolStats(BFCAllocator* bfc_allocator,
                                             int64_t* deviceMemPoolSize,
                                             int64_t* deviceMemStable) {
  size_t region_size = 0;
  size_t in_use_size = 0;
  bool region_in_use = 0;
  size_t total_size = 0;
  for (const auto& region : bfc_allocator->region_manager_.regions()) {
    size_t h = bfc_allocator->region_manager_.get_handle(region.ptr());
    region_in_use = false;
    region_size = region.memory_size();
    total_size += region_size;
    while (h != bfc_allocator->kInvalidChunkHandle) {
      const BFCAllocator::Chunk* c = bfc_allocator->ChunkFromHandle(h);
      if (c->in_use()) {
        region_in_use = true;
        break;
      }
      h = c->next;
    }
    if (region_in_use) {
      in_use_size += region_size;
    }
  }
  *deviceMemPoolSize = total_size;
  *deviceMemStable = in_use_size;
}

size_t GPUAdjustableAllocator::AdjustMemoryLimit(size_t new_memory_limit,
                                                 BFCAllocator* bfc_allocator) {
  mutex_lock l(bfc_allocator->lock_);
  if (new_memory_limit >= bfc_allocator->total_region_allocated_bytes_) {
    // 1) new_memory_limit >= memory_limit_ : grow memory size
    // 2) memory_limit_ > new_memory_limit >= total_region_allocated_bytes_:
    //    shrink, but don't need to free memory
    // In both cases, no action needed by changing the memory limit
    bfc_allocator->memory_limit_ = new_memory_limit;
    bfc_allocator->stats_.bytes_limit = new_memory_limit;
  } else {
    // total_region_allocated_bytes_ > new_memory_limit:
    // shrink, need to free memory
    size_t free_res = FreeEmptyMemory(
        new_memory_limit, bfc_allocator);
    if (free_res <= new_memory_limit) {
      bfc_allocator->memory_limit_ = new_memory_limit;
      bfc_allocator->stats_.bytes_limit = new_memory_limit;
      LOG(INFO) << "successful";
    } else {
      bfc_allocator->memory_limit_ = free_res;
      bfc_allocator->stats_.bytes_limit = free_res;
      LOG(INFO) << "fail in memory free : ";
      LOG(INFO) << "in-used memory : "
                << strings::HumanReadableNumBytes(bfc_allocator->stats_.bytes_in_use)
                << " bytes.";
      LOG(INFO) << "new memory limit : "
                << strings::HumanReadableNumBytes(new_memory_limit)
                << " bytes.";
      LOG(INFO) << "current memory limit : "
                << strings::HumanReadableNumBytes(bfc_allocator->memory_limit_)
                << " bytes.";
    }
  }
  return bfc_allocator->memory_limit_;
}

size_t GPUAdjustableAllocator::FreeEmptyMemory(size_t target_memory_bytes,
                                               BFCAllocator* bfc_allocator) {
  int64_t to_free_bytes =
      bfc_allocator->total_region_allocated_bytes_ - target_memory_bytes;
  DCHECK_GE(to_free_bytes, 0);
  if (to_free_bytes == 0) return true;

  const auto& regions = bfc_allocator->region_manager_.regions();
  std::vector<size_t> tag;
  tag.reserve(regions.size());
  for (size_t i = 0; i < regions.size(); i++) {
    const BFCAllocator::AllocationRegion& ar = regions[i];
    LOG(INFO) << "memory region: [" << ar.ptr() << "] size : "
              << strings::HumanReadableNumBytes(ar.memory_size());
    BFCAllocator::ChunkHandle h = ar.get_handle(ar.ptr());
    BFCAllocator::Chunk* c = bfc_allocator->ChunkFromHandle(h);
    if (ar.is_single_chunk(c->size) && !c->in_use()) {
      void *p = ar.ptr();
      size_t memory_size = ar.memory_size();
      tag.push_back(i);
      bfc_allocator->RemoveFreeChunkFromBin(h);
      bfc_allocator->sub_allocator_->Free(p, memory_size);
      CHECK_GE(bfc_allocator->total_region_allocated_bytes_, memory_size);
      bfc_allocator->total_region_allocated_bytes_ -= memory_size;
      LOG(INFO) << "free memory region: [" << p << "] size : "
                << strings::HumanReadableNumBytes(memory_size)
                << " region ptr: " << ar.ptr()
                << " total_region_allocated_bytes_: "
                << strings::HumanReadableNumBytes(
                    bfc_allocator->total_region_allocated_bytes_);
      to_free_bytes -= static_cast<int64_t>(memory_size);
      if (to_free_bytes <= 0) break;
    }
  }

  // clean state in region manager
  for (int64_t i = static_cast<int64_t>(tag.size()) - 1; i >= 0; i--) {
    const BFCAllocator::AllocationRegion& ar = regions[tag[i]];
    LOG(INFO) << "delete memory region ptr: [" << ar.ptr() << "]";
    auto regions_ptr =
      const_cast<std::vector<BFCAllocator::AllocationRegion>*>(&(bfc_allocator->region_manager_.regions()));
    auto it = regions_ptr->begin();
    bfc_allocator->region_manager_.RemoveAllocationRegion(it + tag[i]);
  }

  size_t remain_bytes = std::max(
      target_memory_bytes - bfc_allocator->total_region_allocated_bytes_,
      size_t{1});
  bfc_allocator->curr_region_allocation_bytes_ = BFCAllocator::RoundedBytes(
      std::min(remain_bytes, size_t{1024 * 1024}));
  if (to_free_bytes > 0) {
    // the target cannot be reached after releasing all chunks
    LOG(INFO) << "fail to free, to free bytes: "
              << strings::HumanReadableNumBytes(to_free_bytes);
  }
  return bfc_allocator->total_region_allocated_bytes_;
}

}  // namespace tensorflow
