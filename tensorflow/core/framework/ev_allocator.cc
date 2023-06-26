/* Copyright 2021 The DeepRec Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/framework/ev_allocator.h"

namespace tensorflow {

// Cache first invocation to port::AvailableRam, as it can be expensive.
static int64_t LargeAllocationWarningBytes() {
  static int64_t value = static_cast<int64>(port::AvailableRam() *
                                            kLargeAllocationWarningThreshold);
  return value;
}

static int64_t TotalAllocationWarningBytes() {
  static int64_t value = static_cast<int64>(port::AvailableRam() *
                                            kTotalAllocationWarningThreshold);
  return value;
}

namespace {

static constexpr size_t kPageShift = 12;
static constexpr size_t kPageSize = (1 << kPageShift);    // 4KB page by default
static constexpr size_t kPageCount = kChunkSize / kPageSize;

class CPUChunk : public Chunk<CPUChunk> {
public:
  CPUChunk(size_t chunk_size, size_t slot_size)
     : Chunk<CPUChunk>(chunk_size, slot_size) {} 

  ~CPUChunk() {
    port::AlignedFree(start_);
  }
 
  void GetMemBlock() override {
    start_ = (char *)port::AlignedMalloc(chunk_size_, kPageSize);
  }
};

template<>
void PageMap<CPUChunk>::Init() {
  page_shift_ = kPageShift;
  npages_ = kPageCount;

  InitInternal();
}

class CPUEVAllocator : public EVAllocator<CPUChunk> {
public:
  CPUEVAllocator() = default;
  ~CPUEVAllocator() override = default;

  string Name() override { return "ev_allocator"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    num_bytes = AlignedSize(num_bytes);

    if (num_bytes > kChunkSize) {
      LOG(FATAL) << "Allocation of " << num_bytes << " exceeds "
                 << kChunkSize << " in EVAllocator.";
    }

    if (num_bytes > LargeAllocationWarningBytes() &&
        single_allocation_warning_count_ < kMaxSingleAllocationWarnings) {
      ++single_allocation_warning_count_;
      LOG(WARNING) << "Allocation of " << num_bytes << " exceeds "
                   << 100 * kLargeAllocationWarningThreshold
                   << "% of system memory.";
    }

    void* p = impl_.Allocate(num_bytes);
    if (ev_allocator_collect_stats) {
      const std::size_t alloc_size = impl_.AllocatedSize(p);
      mutex_lock l(mu_);
      ++stats_.num_allocs;
      stats_.bytes_in_use += alloc_size;
      stats_.peak_bytes_in_use =
          std::max<int64>(stats_.peak_bytes_in_use, stats_.bytes_in_use);
      stats_.largest_alloc_size =
          std::max<int64>(stats_.largest_alloc_size, alloc_size);

      if (stats_.bytes_in_use > TotalAllocationWarningBytes() &&
          total_allocation_warning_count_ < kMaxTotalAllocationWarnings) {
        ++total_allocation_warning_count_;
        LOG(WARNING) << "Total allocated memory " << stats_.bytes_in_use
                     << "exceeds " << 100 * kTotalAllocationWarningThreshold
                     << "% of system memory";
      }
    }
    return p;
  }

  size_t BatchAllocateRaw(size_t num, size_t alignment,
      size_t num_bytes, void** ret) override {
    num_bytes = AlignedSize(num_bytes);

    if (num_bytes > kChunkSize) {
      LOG(FATAL) << "Allocation of " << num_bytes << " exceeds "
                 << kChunkSize << " in EVAllocator.";
    }

    if (num_bytes > LargeAllocationWarningBytes() &&
        single_allocation_warning_count_ < kMaxSingleAllocationWarnings) {
      ++single_allocation_warning_count_;
      LOG(WARNING) << "Allocation of " << num_bytes << " exceeds "
                   << 100 * kLargeAllocationWarningThreshold
                   << "% of system memory.";
    }

    auto allocated_num = impl_.BatchAllocate(num, num_bytes, ret);
    if (allocated_num == 0) {
      LOG(WARNING) << "Can't allocate num:"
                   << num << ", num_bytes:" << num_bytes;
      return 0;
    }

    if (ev_allocator_collect_stats) {
      auto p = ret[0];
      const std::size_t alloc_size = impl_.AllocatedSize(p);
      mutex_lock l(mu_);
      stats_.num_allocs += allocated_num;
      stats_.bytes_in_use += alloc_size * allocated_num;
      stats_.peak_bytes_in_use =
          std::max<int64>(stats_.peak_bytes_in_use, stats_.bytes_in_use);
      stats_.largest_alloc_size =
          std::max<int64>(stats_.largest_alloc_size, alloc_size);

      if (stats_.bytes_in_use > TotalAllocationWarningBytes() &&
          total_allocation_warning_count_ < kMaxTotalAllocationWarnings) {
        ++total_allocation_warning_count_;
        LOG(WARNING) << "Total allocated memory " << stats_.bytes_in_use
                     << "exceeds " << 100 * kTotalAllocationWarningThreshold
                     << "% of system memory";
      }
    }
    return allocated_num;
  }  
};

class EVAllocatorFactory : public AllocatorFactory {
 public:
  Allocator* CreateAllocator() override { return CreateEVAllocator(); }

  Allocator* CreateEVAllocator() override {
    return new CPUEVAllocator;
  }

  SubAllocator* CreateSubAllocator(int numa_node) override {
    return new EVSubAllocator(new CPUEVAllocator);
  }

 private:
  class EVSubAllocator : public SubAllocator {
   public:
    explicit EVSubAllocator(CPUEVAllocator* ev_allocator)
        : SubAllocator({}, {}), ev_allocator_(ev_allocator) {}

    void* Alloc(size_t alignment, size_t num_bytes) override {
      return ev_allocator_->AllocateRaw(alignment, num_bytes);
    }

    void Free(void* ptr, size_t num_bytes) override {
      ev_allocator_->DeallocateRaw(ptr);
    }

   private:
    CPUEVAllocator* ev_allocator_;
  };
};

REGISTER_MEM_ALLOCATOR("EVAllocator", 20, EVAllocatorFactory);
  
} // end of anonymous namespace
  
} // end of namespace tensorflow
