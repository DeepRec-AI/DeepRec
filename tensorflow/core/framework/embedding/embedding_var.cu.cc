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
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/framework/embedding/embedding_var.h"

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
using se::DeviceMemoryBase;
using se::Stream;

void SyncWithEventMgr(se::Stream* stream,
      EventMgr* event_mgr) {
  volatile bool is_kernel_finish = false;
  event_mgr->ThenExecute(stream,
      [&is_kernel_finish]() {is_kernel_finish = true;});
  while(!is_kernel_finish) {}
}

template <class K, class V>
void EmbeddingVar<K, V>::CopyEmbeddingsToBuffer(
    V* val_base, int64 size, V** memcpy_address,
    se::Stream* compute_stream, EventMgr* event_mgr,
    const Eigen::GpuDevice& gpu_device) {
  int block_dim = 128;
  V** dev_value_address = (V**)GetBuffer(size);
  DeviceMemoryBase gpu_dst_ptr(dev_value_address, size * sizeof(V*));
  compute_stream->ThenMemcpy(&gpu_dst_ptr, memcpy_address, size * sizeof(V*));

  int limit = size;
  int length = ValueLen();
  TF_CHECK_OK(GpuLaunchKernel(
      embedding::BatchCopy<V>,
      (limit + block_dim - 1) / block_dim * length, block_dim, 0,
      gpu_device.stream(), dev_value_address, val_base, length, limit));
  SyncWithEventMgr(compute_stream, event_mgr);
}
#define REGISTER_KERNELS(ktype, vtype)                              \
  template void EmbeddingVar<ktype, vtype>::CopyEmbeddingsToBuffer( \
      vtype*, int64, vtype**, se::Stream*, EventMgr*,               \
      const Eigen::GpuDevice& gpu_device);
#define REGISTER_KERNELS_ALL(type) \
  REGISTER_KERNELS(int32, type);   \
  REGISTER_KERNELS(int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
