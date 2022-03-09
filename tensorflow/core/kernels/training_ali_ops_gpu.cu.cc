/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/training_ali_ops.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {
template <typename Value>
__global__ void kv_sparse_apply_adagrad_kernel(int32* item_idxs,
                                               int64 dim,
                                               Value** d_banks,
                                               bool** d_flags,
                                               int32 var_slot_idx,
                                               int32 acc_slot_idx,
                                               int32 bank_num,
                                               int32 slot_num,
                                               int32 bank_size,
                                               Value lr,
                                               const Value* grad,
                                               Value* var_default_v,
                                               Value* acc_default_v,
                                               int32 var_default_v_num,
                                               int32 acc_default_v_num) {
  auto item_idx = blockIdx.x;
  auto item_pos = item_idxs[item_idx];
  auto bank_idx = item_pos / bank_size;
  auto offset_in_bank = item_pos % bank_size;
  auto var_slot_offset = bank_idx * slot_num + var_slot_idx;
  auto acc_slot_offset = bank_idx * slot_num + acc_slot_idx;

  if (bank_idx < bank_num && offset_in_bank < bank_size) {
    if (var_default_v != nullptr && d_flags[var_slot_offset][offset_in_bank] == false) {
      d_flags[var_slot_offset][offset_in_bank] = true;
      for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
        d_banks[var_slot_offset][offset_in_bank * dim + id] = var_default_v[(item_idx % var_default_v_num) * dim + id];
      }
    }
    if (acc_default_v != nullptr && d_flags[acc_slot_offset][offset_in_bank] == false) {
      d_flags[acc_slot_offset][offset_in_bank] = true;
      for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
        d_banks[acc_slot_offset][offset_in_bank * dim + id] = acc_default_v[(item_idx % acc_default_v_num) * dim + id];
      }
    }
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      auto tmp_offset = offset_in_bank * dim + id;
      Value g = grad[item_idx * dim + id];
      Value* acc = &d_banks[acc_slot_offset][tmp_offset];
      (*acc) += g * g;
      d_banks[var_slot_offset][tmp_offset] -= lr * g * rsqrtf(*acc);
    }
  }
}

template <typename TKey, typename T>
struct KvSparseApplyAdagrad<GPUDevice, TKey, T> {
  void operator()(int32 num_items,
                  Allocator* alloc,
                  EmbeddingVarGPU<TKey, T>* var,
                  EmbeddingVarGPU<TKey, T>* accum,
                  const TKey* key_base,
                  const T* grad,
                  T lr,
                  int64 gs,
                  cudaStream_t stream) {
    int32* item_idxs = TypedAllocator::Allocate<int32>(alloc, num_items, AllocationAttributes());
    var->LookupOrCreateKey(key_base, item_idxs, num_items, stream, gs);
    auto const block_size = 256;
    auto const grid_size = num_items;
    TF_CHECK_OK(GpuLaunchKernel(kv_sparse_apply_adagrad_kernel<T>,
                                grid_size, block_size, 0, stream,
                                item_idxs, var->ValueLen(), var->kv()->d_bank_ptrs, var->kv()->d_existence_flag_ptrs,
                                var->EmbIdx(), accum->EmbIdx(),
                                var->kv()->mem_bank_num, var->SlotNum(), var->kv()->initial_bank_size,
                                lr, grad, var->DefaultValuePtr(), accum->DefaultValuePtr(),
                                var->GetDefaultValueDim(), accum->GetDefaultValueDim()));
    TypedAllocator::Deallocate(alloc, item_idxs, num_items);
  }
};
}  // namespace functor

template struct functor::KvSparseApplyAdagrad<GPUDevice, int32, float>;
template struct functor::KvSparseApplyAdagrad<GPUDevice, int32, double>;
template struct functor::KvSparseApplyAdagrad<GPUDevice, int64, float>;
template struct functor::KvSparseApplyAdagrad<GPUDevice, int64, double>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
