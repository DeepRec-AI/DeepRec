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
#include "tensorflow/core/framework/embedding/multi_tier_storage.h"
#include "tensorflow/core/framework/embedding/hbm_multi_tier_feature_descriptor.h"

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
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
using se::DeviceMemoryBase;
using se::Stream;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
void SyncWithEventMgr(se::Stream* stream,
      EventMgr* event_mgr);

namespace embedding{
template <class K, class V>
void MultiTierStorage<K, V>::CopyEmbeddingsFromDramToHbm(
    const EmbeddingVarContext<GPUDevice>& ctx,
    const K* keys,
    void** value_ptr_list,
    std::list<int64>& copyback_cursor,
    const std::vector<int64>& memory_index,
    const std::vector<void*>& gpu_value_ptrs,
    int value_len,
    FeatureDescriptor<V>* hbm_feat_desc,
    FeatureDescriptor<V>* dram_feat_desc) {
  if (copyback_cursor.size() > 0) {
    int total = copyback_cursor.size();
    //Alocate memcpy buffer on CPU and GPU.
    Allocator* gpu_alloc = ctx.gpu_allocator;
    V* memcpy_buffer_gpu = (V*)gpu_alloc->AllocateRaw(
        Allocator::kAllocatorAlignment,
        total * value_len * sizeof(V));
    V* memcpy_buffer_cpu = (V*)cpu_allocator()->AllocateRaw(
        Allocator::kAllocatorAlignment,
        total * value_len * sizeof(V));

    //Copy embeddings on CPU to bufer on CPU
    auto do_work = [memory_index,
                    memcpy_buffer_cpu, value_ptr_list,
                    gpu_value_ptrs,
                    dram_feat_desc,
                    value_len, this] (int64 start, int64 limit) {
      for (int i = start; i < limit; i++) {
        int j = memory_index[i];
        memcpy(memcpy_buffer_cpu + i * value_len,
               dram_feat_desc->GetEmbedding(value_ptr_list[j], 0),
               value_len * sizeof(V));
        value_ptr_list[j] = gpu_value_ptrs[i];
      }
    };
    auto worker_threads = ctx.worker_threads;
    Shard(worker_threads->num_threads,
          worker_threads->workers, total,
          1000, do_work);

    //Copy embeddings from CPU buffer to GPU buffer
    auto compute_stream = ctx.compute_stream;
    auto event_mgr = ctx.event_mgr;
    DeviceMemoryBase gpu_buffer_dst_ptr(
        memcpy_buffer_gpu, total * value_len * sizeof(V));
    compute_stream->ThenMemcpy(
        &gpu_buffer_dst_ptr, memcpy_buffer_cpu, total * value_len * sizeof(V));
    SyncWithEventMgr(compute_stream, event_mgr);
                                                
    //Copy addr of embeddings on GPU to GPU
    V** value_address = (V**)cpu_allocator()->AllocateRaw(
        Allocator::kAllocatorAlignment, sizeof(V*) * total);
    V** dev_value_address = (V**)gpu_alloc->AllocateRaw(
        Allocator::kAllocatorAlignment, sizeof(V*) * total);
    int64 i = 0;
    auto it = copyback_cursor.cbegin();
    for (; it != copyback_cursor.cend(); ++it, ++i) {
      // Get the cursor
      int64 cursor = *it;
      value_address[i] = hbm_feat_desc->GetEmbedding(gpu_value_ptrs[i], 0);
    }
    DeviceMemoryBase gpu_addr_dst_ptr(dev_value_address, total * sizeof(V*));
    compute_stream->ThenMemcpy(&gpu_addr_dst_ptr, value_address, total * sizeof(V*));

    //Copy each embedding to corresponding address
    int block_dim = 128;
    TF_CHECK_OK(GpuLaunchKernel(
        BatchUnpack<V>, (total + block_dim - 1) / block_dim * value_len,
        block_dim, 0, ctx.gpu_device.stream(),
        dev_value_address, memcpy_buffer_gpu,
        value_len, total));
    SyncWithEventMgr(compute_stream, event_mgr);

    gpu_alloc->DeallocateRaw(dev_value_address);
    gpu_alloc->DeallocateRaw(memcpy_buffer_gpu);
    cpu_allocator()->DeallocateRaw(value_address);
    cpu_allocator()->DeallocateRaw(memcpy_buffer_cpu);
  }
}
#define REGISTER_KERNELS(ktype, vtype)                                        \
  template void MultiTierStorage<ktype, vtype>::CopyEmbeddingsFromDramToHbm(       \
      const EmbeddingVarContext<GPUDevice>&, const ktype*, void**,\
      std::list<int64>&, const std::vector<int64>&,\
      const std::vector<void*>&, int, FeatureDescriptor<vtype>*,\
      FeatureDescriptor<vtype>*);
#define REGISTER_KERNELS_ALL(type) \
  REGISTER_KERNELS(int32, type);   \
  REGISTER_KERNELS(int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

template <class TValue>
template <class K>
void HbmMultiTierFeatureDescriptorImpl<TValue>::SetDefaultValues(
    const K* keys, const std::list<int64>& init_cursor,
    void** value_ptrs, se::Stream* compute_stream, EventMgr* event_mgr,
    const Eigen::GpuDevice& gpu_device) {
  if (init_cursor.size() > 0) {
    int64 total = init_cursor.size();
    TValue** value_address = nullptr;
    value_address = TypedAllocator::Allocate<TValue*>(cpu_allocator(), total * 2,
                                                 AllocationAttributes());
    TValue** default_value_address = value_address + total;
    TValue** dev_value_address = nullptr;
    dev_value_address =
        TypedAllocator::Allocate<TValue*>(hbm_alloc_, total * 2, AllocationAttributes());
    TValue** dev_default_value_address = dev_value_address + total;
    for (int emb_index = 0; emb_index < FeatureDescriptorImpl<TValue>::slot_infos_.size(); emb_index++) {
      int64 i = 0;
      auto it = init_cursor.cbegin();
      for (; it != init_cursor.cend(); ++it, ++i) {
        value_address[i] = GetEmbedding(value_ptrs[*it], emb_index);
        default_value_address[i] =
            FeatureDescriptorImpl<TValue>::GetDefaultValuePtr(emb_index, keys[i]);
      }
      DeviceMemoryBase gpu_dst_ptr(dev_value_address, total * 2 * sizeof(TValue*));
      compute_stream->ThenMemcpy(&gpu_dst_ptr, value_address,
                                total * 2 * sizeof(TValue*));
      int block_dim = 128;
      int value_len = FeatureDescriptorImpl<TValue>::slot_infos_[emb_index].default_value_len;
      TF_CHECK_OK(GpuLaunchKernel(
          embedding::CopyEmbedding<TValue>,
          (total * value_len + block_dim - 1) / block_dim,
          block_dim, 0, gpu_device.stream(), dev_default_value_address,
          dev_value_address, value_len, total));
      SyncWithEventMgr(compute_stream, event_mgr);  
    }
    
    TypedAllocator::Deallocate(hbm_alloc_, dev_value_address, total * 2);
    TypedAllocator::Deallocate(cpu_allocator(), value_address, total * 2);
  }
}

#define REGISTER_KERNELS(ktype, vtype)                                        \
  template void HbmMultiTierFeatureDescriptorImpl<vtype>::SetDefaultValues(     \
      const ktype*, const std::list<int64>&, void**,\
      se::Stream*, EventMgr*, const Eigen::GpuDevice& gpu_device);
#define REGISTER_KERNELS_ALL(type) \
  REGISTER_KERNELS(int32, type);   \
  REGISTER_KERNELS(int64, type)
#define REGISTER_KERNELS_CPU(type) REGISTER_KERNELS_ALL(type)
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_CPU)
#undef REGISTER_KERNELS_CPU
#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS
} // namespace embedding
} // namespace tensorflow

#endif //GOOGLE_CUDA
