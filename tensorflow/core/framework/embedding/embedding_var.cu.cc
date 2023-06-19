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
void EmbeddingVar<K, V>::SetDefaultValueOfNewFeatures(
    const K* keys, int64 size, const std::list<int64>& init_cursor,
    V** memcpy_address, se::Stream* compute_stream, EventMgr* event_mgr,
    const Eigen::GpuDevice& gpu_device) {
  if (init_cursor.size() > 0) {
    int64 total = init_cursor.size();
    V** value_address = nullptr;
    value_address = TypedAllocator::Allocate<V*>(cpu_allocator(), total * 2,
                                                 AllocationAttributes());
    V** default_value_address = value_address + total;
    V** dev_value_address = nullptr;
    dev_value_address =
        TypedAllocator::Allocate<V*>(alloc_, total * 2, AllocationAttributes());
    V** dev_default_value_address = dev_value_address + total;
    int64 i = 0;
    auto it = init_cursor.cbegin();
    for (; it != init_cursor.cend(); ++it, ++i) {
      ValuePtr<V>* value_ptr =
          reinterpret_cast<ValuePtr<V>*>(memcpy_address[*it]);
      value_address[i] =
          *((V**)((char*)(value_ptr->GetPtr()) + sizeof(FixedLengthHeader))) +
          storage_->GetOffset(emb_config_.emb_index);
      default_value_address[i] =
          default_value_ +
          (keys[i] % emb_config_.default_value_dim) % value_len_;
    }
    DeviceMemoryBase gpu_dst_ptr(dev_value_address, total * 2 * sizeof(V*));
    compute_stream->ThenMemcpy(&gpu_dst_ptr, value_address,
                               total * 2 * sizeof(V*));
    int block_dim = 128;
    TF_CHECK_OK(GpuLaunchKernel(
        embedding::CopyEmbedding<V>,
        (total * value_len_ + block_dim - 1) / block_dim,
        block_dim, 0, gpu_device.stream(), dev_default_value_address,
        dev_value_address, value_len_, total));
    SyncWithEventMgr(compute_stream, event_mgr);
    // Set init meta of ValuePtrs
    for (auto it = init_cursor.cbegin(); it != init_cursor.cend(); ++it) {
      ValuePtr<V>* value_ptr =
          reinterpret_cast<ValuePtr<V>*>(memcpy_address[*it]);
      value_ptr->SetInitialized(emb_config_.emb_index);
      memcpy_address[*it] = value_ptr->GetValue(
          emb_config_.emb_index,
          storage_->GetOffset(emb_config_.emb_index));
    }
    TypedAllocator::Deallocate(alloc_, dev_value_address, total * 2);
    TypedAllocator::Deallocate(cpu_allocator(), value_address, total * 2);
  }
}

#define REGISTER_KERNELS(ktype, vtype)                                        \
  template void EmbeddingVar<ktype, vtype>::SetDefaultValueOfNewFeatures(     \
      const ktype*, int64, const std::list<int64>&, vtype**,                  \
      se::Stream*, EventMgr*, const Eigen::GpuDevice& gpu_device);
#define REGISTER_KERNELS_ALL(type) \
  REGISTER_KERNELS(int32, type);   \
  REGISTER_KERNELS(int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

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

template <class K, class V>
void EmbeddingVar<K, V>::CopyEmbeddingsFromCPUToGPU(
    const K* keys, const std::list<int64>& copyback_cursor, V** memcpy_address,
    se::Stream* compute_stream, EventMgr* event_mgr,
    const Eigen::GpuDevice& gpu_device,
    const DeviceBase::CpuWorkerThreads* worker_threads,
    int64* output_value_ptrs) {
  if (copyback_cursor.size() > 0) {
    int64 total = copyback_cursor.size();
    size_t value_len = emb_config_.total_num(storage_->GetAllocLen());
    V* memcpy_buffer_gpu = nullptr;
    ValuePtr<V>** gpu_value_ptrs = new ValuePtr<V>*[total];
    memcpy_buffer_gpu = (V*)alloc_->AllocateRaw(Allocator::kAllocatorAlignment,
                                                total * value_len * sizeof(V));
    storage_->CopyEmbeddingsFromCPUToGPU(
        total, keys, copyback_cursor, memcpy_address, value_len, gpu_value_ptrs,
        memcpy_buffer_gpu, compute_stream, event_mgr, worker_threads);

    V** value_address = (V**)cpu_allocator()->AllocateRaw(
        Allocator::kAllocatorAlignment, sizeof(V*) * total);
    V** dev_value_address = (V**)alloc_->AllocateRaw(Allocator::kAllocatorAlignment,
                                                 sizeof(V*) * total);
    std::vector<K> copyback_keys(total);
    int64 i = 0;
    auto it = copyback_cursor.cbegin();
    for (; it != copyback_cursor.cend(); ++it, ++i) {
      bool init;
      // Get the curosr
      int64 cursor = *it & 0x0fffffffffffffff;
      gpu_value_ptrs[i]->SetInitialized(emb_config_.emb_index);
      memcpy_address[cursor] = LookupOrCreateEmb(gpu_value_ptrs[i], init);
      value_address[i] = memcpy_address[cursor];
      copyback_keys[i] = keys[cursor];
    }
    DeviceMemoryBase gpu_dst_ptr(dev_value_address, total * sizeof(V*));
    compute_stream->ThenMemcpy(&gpu_dst_ptr, value_address, total * sizeof(V*));

    int block_dim = 128;
    TF_CHECK_OK(GpuLaunchKernel(
        embedding::BatchUnpack<V>, (total + block_dim - 1) / block_dim * value_len,
        block_dim, 0, gpu_device.stream(), dev_value_address, memcpy_buffer_gpu,
        value_len, total));

    auto do_insert = [this, copyback_keys, gpu_value_ptrs, value_len](
                         int64 start, int64 limit) {
      for (int64 i = start; i < limit; i++)
        storage_->Insert(copyback_keys[i], gpu_value_ptrs[i]);
    };
    Shard(worker_threads->num_threads, worker_threads->workers,
          copyback_keys.size(), 100000, do_insert);
    if (output_value_ptrs != nullptr) {
      auto it = copyback_cursor.cbegin();
      for (int64 i = 0; it != copyback_cursor.cend(); ++it, ++i) {
        int64 cursor = *it & 0x0fffffffffffffff;
        output_value_ptrs[cursor] = (int64)gpu_value_ptrs[i];
      }
    }
    SyncWithEventMgr(compute_stream, event_mgr);

    alloc_->DeallocateRaw(dev_value_address);
    alloc_->DeallocateRaw(memcpy_buffer_gpu);
    cpu_allocator()->DeallocateRaw(value_address);
    delete[] gpu_value_ptrs;
  }
}
#define REGISTER_KERNELS(ktype, vtype)                                        \
  template void EmbeddingVar<ktype, vtype>::CopyEmbeddingsFromCPUToGPU(       \
      const ktype*, const std::list<int64>&, vtype**, se::Stream*, EventMgr*, \
      const Eigen::GpuDevice&, const DeviceBase::CpuWorkerThreads*, int64*);
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
