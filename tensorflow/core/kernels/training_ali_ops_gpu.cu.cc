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

#if GOOGLE_CUDA
#if TF_ENABLE_GPU_EV
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/training_ali_ops_gpu.h"
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
  bool var_stored = d_flags[var_slot_offset][offset_in_bank];
  bool acc_stored = d_flags[acc_slot_offset][offset_in_bank];
  __syncthreads();

  if (var_default_v != nullptr && var_stored == false) {
    d_flags[var_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[var_slot_offset][offset_in_bank * dim + id] = var_default_v[(item_idx % var_default_v_num) * dim + id];
    }
  }
  if (acc_default_v != nullptr && acc_stored == false) {
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
                                var->SlotNum(), var->kv()->initial_bank_size,
                                lr, grad, var->DefaultValuePtr(), accum->DefaultValuePtr(),
                                var->GetDefaultValueDim(), accum->GetDefaultValueDim()));
    TypedAllocator::Deallocate(alloc, item_idxs, num_items);
  }
};

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
  for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
__inline__ __device__
T blockReduceSum(T val)
{
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if(lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
  val = warpReduceSum(val);
  return val;
}

template <typename Value>
__global__ void kv_sparse_apply_ftrl_kernel(int32* item_idxs,
                                            int64 dim,
                                            Value** d_banks,
                                            bool** d_flags,
                                            int32 var_slot_idx,
                                            int32 acc_slot_idx,
                                            int32 linear_slot_idx,
                                            int32 slot_num,
                                            int32 bank_size,
                                            Value lr_scalar,
                                            const Value* grad,
                                            Value* var_default_v,
                                            Value* acc_default_v,
                                            Value* linear_default_v,
                                            int32 var_default_v_num,
                                            int32 acc_default_v_num,
                                            int32 linear_default_v_num,
                                            Value l1_scalar,
                                            Value l2_scalar,
                                            Value lr_power_scalar,
                                            bool has_l2_shrinkage,
                                            Value l2_shrinkage_scalar) {
  auto item_idx = blockIdx.x;
  auto item_pos = item_idxs[item_idx];
  auto bank_idx = item_pos / bank_size;
  auto offset_in_bank = item_pos % bank_size;
  auto var_slot_offset = bank_idx * slot_num + var_slot_idx;
  auto acc_slot_offset = bank_idx * slot_num + acc_slot_idx;
  auto linear_slot_offset = bank_idx * slot_num + linear_slot_idx;
  extern __shared__ __align__(sizeof(Value)) unsigned char shared[];
  Value* new_acc = reinterpret_cast<Value*>(shared);
  __shared__ Value linear_sqr_sum;
  bool var_stored = d_flags[var_slot_offset][offset_in_bank];
  bool acc_stored = d_flags[acc_slot_offset][offset_in_bank];
  bool linear_stored = d_flags[linear_slot_offset][offset_in_bank];
  __syncthreads();

  if (var_default_v != nullptr && var_stored == false) {
    d_flags[var_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[var_slot_offset][offset_in_bank * dim + id] = var_default_v[(item_idx % var_default_v_num) * dim + id];
    }
  }
  if (acc_default_v != nullptr && acc_stored == false) {
    d_flags[acc_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[acc_slot_offset][offset_in_bank * dim + id] = acc_default_v[(item_idx % acc_default_v_num) * dim + id];
    }
  }
  if (linear_default_v != nullptr && linear_stored == false) {
    d_flags[linear_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[linear_slot_offset][offset_in_bank * dim + id] = linear_default_v[(item_idx % linear_default_v_num) * dim + id];
    }
  }
  Value linear_tmp = 0;
  for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
    auto tmp_offset = offset_in_bank * dim + id;
    Value* var_p = &d_banks[var_slot_offset][tmp_offset];
    Value g = grad[item_idx * dim + id];
    Value gg;
    if (has_l2_shrinkage) {
      gg = g + 2 * l2_shrinkage_scalar * (*var_p);
    } else {
      gg = g;
    }
    Value* acc_p = &d_banks[acc_slot_offset][tmp_offset];
    new_acc[id] = *acc_p + gg * gg;
    Value* linear_p = &d_banks[linear_slot_offset][tmp_offset];
    if (lr_power_scalar == -0.5) {
      (*linear_p) += gg - (sqrtf(new_acc[id]) - sqrtf(*acc_p)) / lr_scalar * (*var_p);
    } else {
      (*linear_p) += gg - (powf(new_acc[id], -lr_power_scalar) - powf(*acc_p, -lr_power_scalar)) / lr_scalar * (*var_p);
    }
    linear_tmp += (*linear_p) * (*linear_p);
  }
  linear_tmp = blockReduceSum<Value>(linear_tmp);
  if (threadIdx.x == 0) {
    linear_sqr_sum = linear_tmp;
  }
  __syncthreads();
  Value linear_norm = sqrtf(linear_sqr_sum);
  for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
    auto tmp_offset = offset_in_bank * dim + id;
    Value* var_p = &d_banks[var_slot_offset][tmp_offset];
    Value* acc_p = &d_banks[acc_slot_offset][tmp_offset];
    Value* linear_p = &d_banks[linear_slot_offset][tmp_offset];
    Value g = grad[item_idx * dim + id];
    if (linear_norm > l1_scalar) {
      if (lr_power_scalar == -0.5) {
        auto eta_rec = sqrtf(new_acc[id]) / lr_scalar;
        auto coef = (l1_scalar - linear_norm)  /
                      ((eta_rec + 2 * l2_scalar) * linear_norm);
        *var_p = coef * (*linear_p);
      } else {
        auto eta_rec = powf(new_acc[id], -lr_power_scalar) / lr_scalar;
        auto coef = (l1_scalar - linear_norm)  /
                      ((eta_rec + 2 * l2_scalar) * linear_norm);
        *var_p = coef * (*linear_p);
      }
    } else {
      *var_p = 0;
    }
    (*acc_p) += g * g;
  }
}

template <typename TKey, typename T>
struct KvSparseApplyFtrl<GPUDevice, TKey, T> {
  void operator()(int32 num_items,
                  Allocator* alloc,
                  EmbeddingVarGPU<TKey, T>* var,
                  EmbeddingVarGPU<TKey, T>* accum,
                  EmbeddingVarGPU<TKey, T>* linear,
                  const TKey* key_base,
                  const T* grad,
                  T lr,
                  T l1,
                  T l2,
                  T lr_power,
                  bool has_l2_shrinkage,
                  T l2_shrinkage,
                  cudaStream_t stream) {
    int32* item_idxs = TypedAllocator::Allocate<int32>(alloc, num_items, AllocationAttributes());
    var->LookupOrCreateKey(key_base, item_idxs, num_items, stream);
    auto const block_size = 256;
    auto const grid_size = num_items;
    TF_CHECK_OK(GpuLaunchKernel(kv_sparse_apply_ftrl_kernel<T>,
                                grid_size, block_size, (var->ValueLen()) * sizeof(T), stream,
                                item_idxs, var->ValueLen(), var->kv()->d_bank_ptrs, var->kv()->d_existence_flag_ptrs,
                                var->EmbIdx(), accum->EmbIdx(), linear->EmbIdx(),
                                var->SlotNum(), var->kv()->initial_bank_size,
                                lr, grad, var->DefaultValuePtr(), accum->DefaultValuePtr(), linear->DefaultValuePtr(),
                                var->GetDefaultValueDim(), accum->GetDefaultValueDim(), linear->GetDefaultValueDim(),
                                l1, l2, lr_power, has_l2_shrinkage, l2_shrinkage));
    TypedAllocator::Deallocate(alloc, item_idxs, num_items);
  }
};
}  // namespace functor

template struct functor::KvSparseApplyAdagrad<GPUDevice, int32, float>;
template struct functor::KvSparseApplyAdagrad<GPUDevice, int32, double>;
template struct functor::KvSparseApplyAdagrad<GPUDevice, int64, float>;
template struct functor::KvSparseApplyAdagrad<GPUDevice, int64, double>;
template struct functor::KvSparseApplyFtrl<GPUDevice, int32, float>;
template struct functor::KvSparseApplyFtrl<GPUDevice, int32, double>;
template struct functor::KvSparseApplyFtrl<GPUDevice, int64, float>;
template struct functor::KvSparseApplyFtrl<GPUDevice, int64, double>;

}  // end namespace tensorflow

#endif  // TF_ENABLE_GPU_EV
#endif  // GOOGLE_CUDA
