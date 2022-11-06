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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_CUDA_GRAPH_GPU_GPU_BFC_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_CUDA_GRAPH_GPU_GPU_BFC_ALLOCATOR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cuda_graph_mode_mem.h"
#include "tensorflow/core/common_runtime/gpu/gpu_mem_allocator.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// A GPU bfc memory allocator that accommodates cuda graph.
class CudaGraphGPUBFCAllocator : public GPUBFCAllocator {
 public:
  CudaGraphGPUBFCAllocator(GPUMemAllocator* sub_allocator, size_t total_memory,
                           const string& name);
  CudaGraphGPUBFCAllocator(GPUMemAllocator* sub_allocator, size_t total_memory,
                           const GPUOptions& gpu_options, const string& name);
  ~CudaGraphGPUBFCAllocator() override;

  TF_DISALLOW_COPY_AND_ASSIGN(CudaGraphGPUBFCAllocator);

  void EnableCudaGraphModeMem();
  void DisableCudaGraphModeMem();
  void CloseCudaGraphModeMem();

  void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr) override;
  void DeallocateRaw(void* ptr) override;

 private:
  std::unique_ptr<CudaGraphModeMem> cuda_graph_mode_mem_;
  bool enable_cuda_graph_capture_ = false;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_CUDA_GRAPH_GPU_GPU_BFC_ALLOCATOR_H_
#endif  // GOOGLE_CUDA
