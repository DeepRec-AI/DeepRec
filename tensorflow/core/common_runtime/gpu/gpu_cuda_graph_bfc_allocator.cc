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
==============================================================================*/
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_cuda_graph_bfc_allocator.h"

#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

void CudaGraphGPUBFCAllocator::EnableCudaGraphModeMem() {
  enable_cuda_graph_capture_ = true;
}

void CudaGraphGPUBFCAllocator::DisableCudaGraphModeMem() {
  enable_cuda_graph_capture_ = false;
}

CudaGraphGPUBFCAllocator::~CudaGraphGPUBFCAllocator() {
  std::unordered_set<void*> remained_mems =
      cuda_graph_mode_mem_->GetDeallocatedMems();
  for (auto e : remained_mems) {
    GPUBFCAllocator::DeallocateRaw(e);
  }
}

void* CudaGraphGPUBFCAllocator::AllocateRaw(
    size_t alignment, size_t num_bytes,
    const AllocationAttributes& allocation_attr) {
  if (enable_cuda_graph_capture_) {
    VLOG(2) << "ask reused bytes: " << num_bytes;
    void* ret = cuda_graph_mode_mem_->GetReuseMem(num_bytes);
    if (ret) {
      VLOG(2) << "get reused bytes: " << num_bytes;
      return ret;
    }
  }

  void* ret =
      GPUBFCAllocator::AllocateRaw(alignment, num_bytes, allocation_attr);
  if (ret) {
    VLOG(2) << "bfc allocate mem of bytes: ";
  }
  if (enable_cuda_graph_capture_) {
    cuda_graph_mode_mem_->Add(ret, num_bytes, true);
  }
  return ret;
}

void CudaGraphGPUBFCAllocator::DeallocateRaw(void* ptr) {
  if (cuda_graph_mode_mem_.get() != nullptr &&
      cuda_graph_mode_mem_->ContainMem(ptr)) {
    cuda_graph_mode_mem_->MarkDeallocated(ptr);
    return;
  }
  GPUBFCAllocator::DeallocateRaw(ptr);
}

CudaGraphGPUBFCAllocator::CudaGraphGPUBFCAllocator(
    GPUMemAllocator* sub_allocator, size_t total_memory, const string& name)
    : CudaGraphGPUBFCAllocator(sub_allocator, total_memory, GPUOptions(),
                               name) {
}

CudaGraphGPUBFCAllocator::CudaGraphGPUBFCAllocator(
    GPUMemAllocator* sub_allocator, size_t total_memory,
    const GPUOptions& gpu_options, const string& name)
    : GPUBFCAllocator(sub_allocator, total_memory, gpu_options, name) {
  cuda_graph_mode_mem_ =
      std::unique_ptr<CudaGraphModeMem>(new CudaGraphModeMem());
}
}  // namespace tensorflow
#endif  // GOOGLE_CUDA
