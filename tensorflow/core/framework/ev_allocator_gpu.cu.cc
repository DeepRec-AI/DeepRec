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

#ifdef GOOGLE_CUDA
#include <cuda_runtime.h>
#include "tensorflow/core/framework/ev_allocator.h"

namespace tensorflow {

namespace {

static constexpr size_t kPageShift = 8;
static constexpr size_t kPageSize = (1 << kPageShift);    // 256B page by default
static constexpr size_t kPageCount = kChunkSize / kPageSize;

class GPUChunk : public Chunk<GPUChunk> {
public:
  GPUChunk(size_t chunk_size, size_t slot_size)
    : Chunk<GPUChunk>(chunk_size, slot_size) {}

  ~GPUChunk() {
    cudaFree(start_);
  }

  void GetMemBlock() override {
    cudaMalloc(&start_, chunk_size_);
  }
};

template<>
void PageMap<GPUChunk>::Init() {
  page_shift_ = kPageShift;
  npages_ = kPageCount;

  InitInternal();
}

class GPUEVAllocator : public EVAllocator<GPUChunk> {
public:
  GPUEVAllocator() = default;
  ~GPUEVAllocator() override = default;

  string Name() override { return "gpu_ev_allocator"; }
};

class GPUEVAllocatorFactory : public AllocatorFactory {
public:
  Allocator* CreateAllocator() override { return CreateGPUEVAllocator(); }

  Allocator* CreateGPUEVAllocator() override {
    return new GPUEVAllocator;
  }

  SubAllocator* CreateSubAllocator(int numa_node) override {
    return new GPUEVSubAllocator(new GPUEVAllocator);
  }

private:
  class GPUEVSubAllocator : public SubAllocator {
  public:
    explicit GPUEVSubAllocator(GPUEVAllocator* gpu_ev_allocator)
      : SubAllocator({}, {}), gpu_ev_allocator_(gpu_ev_allocator) {}

    void* Alloc(size_t alignment, size_t num_bytes) override {
      return gpu_ev_allocator_->AllocateRaw(alignment, num_bytes);
    }

    void Free(void *ptr, size_t num_bytes) override {
      gpu_ev_allocator_->DeallocateRaw(ptr);
    }

  private:
    GPUEVAllocator* gpu_ev_allocator_;
  };
};

REGISTER_MEM_ALLOCATOR("GPUEVAllocator", 20, GPUEVAllocatorFactory);

} // end of anonymous namespace
  
} // end of namespace tensorflow

#endif // GOOGLE_CUDA
