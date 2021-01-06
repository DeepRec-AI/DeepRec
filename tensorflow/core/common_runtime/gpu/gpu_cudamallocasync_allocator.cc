/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#endif  // GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_cudamallocasync_allocator.h"

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

GPUcudaMallocAsyncAllocator::GPUcudaMallocAsyncAllocator(
    Allocator* allocator, PlatformGpuId platform_gpu_id, size_t pool_size,
    bool reserve_memory)
    : base_allocator_(allocator), cuda_stream_(nullptr) {
  stream_exec_ =
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();

  se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  cudaError_t cerr = cudaStreamCreate(&cuda_stream_);
  if (cerr != cudaSuccess) {
    LOG(ERROR) << "could not allocate CUDA stream for context : "
               << cudaGetErrorString(cerr);
  }

  CUmemoryPool pool = nullptr;
  cerr = cudaDeviceGetDefaultMemPool(&pool, platform_gpu_id.value());
  if (cerr != cudaSuccess) {
    LOG(ERROR) << "could not get the default CUDA pool : "
               << cudaGetErrorString(cerr);
  }

  cerr = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                 (void*)&pool_size);
  if (cerr != cudaSuccess) {
    LOG(ERROR) << "could not set the default CUDA pool memory threshold : "
               << cudaGetErrorString(cerr);
  }
  VLOG(2) << "GPUcudaMallocAsyncAllocator PoolSize " << pool_size;
  if (reserve_memory) {
    void* ptr = AllocateRaw(0, pool_size);
    DeallocateRaw(ptr);
    VLOG(2) << "GPUcudaMallocAsyncAllocator Pre-filled the pool";
  }
  //TODO: check that the GPU and platform support this feature. Otherwise return a good error message.

  //TODO: If in TF_DETERMINISTIC mode, Set the properties
  //      cudaMemPoolAttrReuseAllowOpportunistic and
  //      cudaMemPoolAttrReuseAllowInternalDepedencies to zero to make
  //      the allocator 100% deterministic.
}

GPUcudaMallocAsyncAllocator::~GPUcudaMallocAsyncAllocator() {
  delete base_allocator_;
  cuStreamDestroy(cuda_stream_);
}

void* GPUcudaMallocAsyncAllocator::AllocateRaw(size_t alignment,
                                               size_t num_bytes) {
#ifdef GOOGLE_CUDA
  se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  void* rv = 0;
  cudaError_t res = cudaMallocAsync(&rv, num_bytes, cuda_stream_);
  if (res != cudaSuccess) {
    LOG(ERROR) << "cudaMallocAsync failed to allocate " << num_bytes
               << ". Error: " << cudaGetErrorString(res);
    return nullptr;
  }
  return rv;
#else
  return nullptr;
#endif  // GOOGLE_CUDA
}
void GPUcudaMallocAsyncAllocator::DeallocateRaw(void* ptr) {
#ifdef GOOGLE_CUDA
  cudaError_t res = cudaFreeAsync(ptr, cuda_stream_);
  if (res != cudaSuccess) {
    LOG(ERROR) << "cudaFreeAsync failed to free " << ptr
               << ". Error: " << cudaGetErrorString(res);
  }
#endif  // GOOGLE_CUDA
}

absl::optional<AllocatorStats> GPUcudaMallocAsyncAllocator::GetStats() {
  return base_allocator_->GetStats();
}

bool GPUcudaMallocAsyncAllocator::TracksAllocationSizes() const {
  return false;
}

}  // namespace tensorflow
