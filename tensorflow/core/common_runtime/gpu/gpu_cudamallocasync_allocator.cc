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
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

GPUcudaMallocAsyncAllocator::GPUcudaMallocAsyncAllocator(
    Allocator* allocator, PlatformGpuId platform_gpu_id, size_t pool_size,
    bool reserve_memory)
    : base_allocator_(allocator), cuda_stream_(nullptr),
      name_(absl::StrCat("gpu_async_", platform_gpu_id.value())) {
  stream_exec_ =
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();

#if CUDA_VERSION < 11020
  LOG(ERROR) << "TF_GPU_ALLOCATOR=cuda_malloc_async need CUDA 11.2 or higher to compile.";
#else

  se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  int cuda_malloc_async_supported;
  cudaDeviceGetAttribute(&cuda_malloc_async_supported,
                         cudaDevAttrMemoryPoolsSupported,
                         platform_gpu_id.value());
  if (!cuda_malloc_async_supported) {
    LOG(ERROR) << "TF_GPU_ALLOCATOR=cuda_malloc_async isn't currently supported."
               << " Possible causes: device not supported, driver too old, "
               << " OS not supported, CUDA version too old.";
  }

  cudaError_t cerr = cudaStreamCreate(&cuda_stream_);
  if (cerr != cudaSuccess) {
    LOG(ERROR) << "could not allocate CUDA stream for context : "
               << cudaGetErrorString(cerr);
  }

  cerr = cudaDeviceGetDefaultMemPool(&pool_, platform_gpu_id.value());
  if (cerr != cudaSuccess) {
    LOG(ERROR) << "could not get the default CUDA pool : "
               << cudaGetErrorString(cerr);
  }
  VLOG(1) << Name() << " CudaMallocAsync initialized on platform: "
          << platform_gpu_id.value() << " with pool size of: "
          << pool_size << " this ptr: " << this;
  cerr = cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold,
                                 (void*)&pool_size);
  if (cerr != cudaSuccess) {
    LOG(ERROR) << "could not set the default CUDA pool memory threshold : "
               << cudaGetErrorString(cerr);
  }

  // If in TF_DETERMINISTIC_OPS is set, then make the allocator behave
  // determistically.
  bool deterministic_ops = false;
  TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("TF_DETERMINISTIC_OPS",
                                             /*default_val=*/false,
                                             &deterministic_ops));
  if (deterministic_ops) {
    cudaMemPoolSetAttribute(pool_, cudaMemPoolReuseAllowOpportunistic, 0);
    cudaMemPoolSetAttribute(pool_, cudaMemPoolReuseAllowInternalDependencies, 0);
  }
#endif

  VLOG(2) << Name() << " GPUcudaMallocAsyncAllocator PoolSize " << pool_size;
  if (reserve_memory) {
    void* ptr = AllocateRaw(0, pool_size);
    DeallocateRaw(ptr);
    VLOG(2) << Name() << " GPUcudaMallocAsyncAllocator Pre-filled the pool";
  }
}

GPUcudaMallocAsyncAllocator::~GPUcudaMallocAsyncAllocator() {
  delete base_allocator_;
  cuStreamDestroy(cuda_stream_);
}

void* GPUcudaMallocAsyncAllocator::AllocateRaw(size_t alignment,
                                               size_t num_bytes) {
#if CUDA_VERSION < 11020 || !defined(GOOGLE_CUDA)
  return nullptr;
#else
  se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  void* rv = 0;
  cudaError_t res = cudaMallocFromPoolAsync(&rv, num_bytes, pool_, cuda_stream_);
  if (res != cudaSuccess) {
    LOG(ERROR) << Name() << " cudaMallocAsync failed to allocate " << num_bytes
               << ". Error: " << cudaGetErrorString(res);
    return nullptr;
  }
  VLOG(10) << Name() << " Allocated " << num_bytes << " at " << rv;
  return rv;
#endif
}
void GPUcudaMallocAsyncAllocator::DeallocateRaw(void* ptr) {
#if CUDA_VERSION < 11020 || !defined(GOOGLE_CUDA)
#else
  cudaError_t res = cudaFreeAsync(ptr, cuda_stream_);
  if (res != cudaSuccess) {
    LOG(ERROR) << "cudaFreeAsync failed to free " << ptr
               << ". Error: " << cudaGetErrorString(res);
  }
  VLOG(10) << Name() << " Freed ptr: " << ptr;
#endif  // GOOGLE_CUDA
}

absl::optional<AllocatorStats> GPUcudaMallocAsyncAllocator::GetStats() {
  return base_allocator_->GetStats();
}

bool GPUcudaMallocAsyncAllocator::TracksAllocationSizes() const {
  return false;
}

}  // namespace tensorflow
