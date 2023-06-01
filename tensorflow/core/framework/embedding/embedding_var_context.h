/* Copyright 2019 The DeepRec Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_CONTEXT_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_CONTEXT_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow{
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template<typename Device>
struct EmbeddingVarContext;

template<>
struct EmbeddingVarContext<CPUDevice> {
 public:
  EmbeddingVarContext<CPUDevice>(OpKernelContext* op_ctx)
      : worker_threads(op_ctx->device()->tensorflow_cpu_worker_threads()) {}

  const DeviceBase::CpuWorkerThreads* worker_threads;
};

#if GOOGLE_CUDA
template<>
struct EmbeddingVarContext<GPUDevice> {
 public:
  EmbeddingVarContext<GPUDevice>(OpKernelContext* op_ctx)
      : worker_threads(op_ctx->device()->tensorflow_cpu_worker_threads()),
        compute_stream(op_ctx->op_device_context()->stream()),
        event_mgr(op_ctx->device()->tensorflow_gpu_device_info()->event_mgr),
        gpu_allocator(op_ctx->device()->GetAllocator(AllocatorAttributes())),
        gpu_device(op_ctx->eigen_gpu_device()) {}

  const DeviceBase::CpuWorkerThreads* worker_threads = nullptr;
  se::Stream* compute_stream = nullptr;
  EventMgr* event_mgr = nullptr;
  Allocator* gpu_allocator= nullptr;
  const GPUDevice& gpu_device;
};
#endif  // GOOGLE_CUDA
} //namespace tensorflow

#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_CONTEXT_H_
