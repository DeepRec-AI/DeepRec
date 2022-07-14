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
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/training_ali_ops.h"
#include "tensorflow/core/kernels/training_ali_ops_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {
template <typename T>
__device__ T impl_sqrt(T x) {
  return sqrt(x);
}
template <typename T>
__device__ T impl_rsqrt(T x) {
  return rsqrt(x);
}
template <>
__device__ Eigen::half impl_sqrt(Eigen::half x) {
  return __float2half(sqrt(__half2float(x)));
}
template <>
__device__ Eigen::half impl_rsqrt(Eigen::half x) {
  return __float2half(rsqrt(__half2float(x)));
}

#if TENSORFLOW_USE_GPU_EV
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

template <typename T>
__global__ void KvSparseApplyAdamAsyncKernel(int32 *item_idxs,
                                              int64 dim,
                                              T **d_banks,
                                              bool **d_flags,
                                              int32 var_slot_idx,
                                              int32 v_slot_idx,
                                              int32 m_slot_idx,
                                              int32 slot_num,
                                              int32 bank_size,
                                              const T *beta1_scalar,
                                              const T *beta2_scalar,
                                              const T *beta1_power_scalar,
                                              const T *beta2_power_scalar,
                                              const T *epsilon_scalar,
                                              const T *lr_scalar,
                                              const T *grad,
                                              T *var_default_v,
                                              T *v_default_v,
                                              T *m_default_v,
                                              int32 var_default_v_num,
                                              int32 v_default_v_num,
                                              int32 m_default_v_num,
                                              bool apply_sparse_rmsprop) {
  const T lr = *lr_scalar;
  const T beta1 = *beta1_scalar;
  const T beta2 = *beta2_scalar;
  const T beta1_power = *beta1_power_scalar;
  const T beta2_power = *beta2_power_scalar;
  const T epsilon = *epsilon_scalar;

  auto item_idx = blockIdx.x;
  auto item_pos = item_idxs[item_idx];
  auto bank_idx = item_pos / bank_size;
  auto offset_in_bank = item_pos % bank_size;
  auto var_slot_offset = bank_idx*slot_num + var_slot_idx;
  auto v_slot_offset = bank_idx*slot_num + v_slot_idx;
  auto m_slot_offset = bank_idx*slot_num + m_slot_idx;
  bool var_stored = d_flags[var_slot_offset][offset_in_bank];
  bool v_stored = d_flags[v_slot_offset][offset_in_bank];
  bool m_stored = d_flags[m_slot_offset][offset_in_bank];
  const T alpha = 
      lr*sqrt(static_cast<T>(1)-beta2_power) / (static_cast<T>(1) - beta1_power);
  __syncthreads();

  if (var_default_v != nullptr && var_stored == false) {
    d_flags[var_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[var_slot_offset][offset_in_bank*dim + id] = var_default_v[(item_idx%var_default_v_num)*dim + id];
    }
  }
  if (v_default_v != nullptr && v_stored == false) {
    d_flags[v_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[v_slot_offset][offset_in_bank*dim + id] = v_default_v[(item_idx%v_default_v_num)*dim + id];
    }
  }
  if (m_default_v != nullptr && m_stored == false) {
    d_flags[m_slot_offset][offset_in_bank] = true;
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      d_banks[m_slot_offset][offset_in_bank*dim + id] = m_default_v[(item_idx%m_default_v_num)*dim + id];
    }
  }

  if (apply_sparse_rmsprop) {
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      auto tmp_offset = offset_in_bank*dim + id;
      T grad_a = grad[item_idx*dim + id];
      T &var_a = d_banks[var_slot_offset][tmp_offset];
      T &v_a = d_banks[v_slot_offset][tmp_offset];
      T &m_a = d_banks[m_slot_offset][tmp_offset];

      v_a = v_a*beta2 + grad_a*grad_a*(static_cast<T>(1)-beta2);
      m_a = m_a*beta1 + rsqrt(v_a+epsilon)*lr*grad_a;
      var_a -= m_a;
    }
  } else {
    for (auto id = threadIdx.x; id < dim; id += blockDim.x) {
      auto tmp_offset = offset_in_bank*dim + id;
      T grad_a = grad[item_idx*dim + id];
      T &var_a = d_banks[var_slot_offset][tmp_offset];
      T &v_a = d_banks[v_slot_offset][tmp_offset];
      T &m_a = d_banks[m_slot_offset][tmp_offset];

      m_a = m_a*beta1 + grad_a*(static_cast<T>(1) - beta1);
      v_a = v_a*beta2 + grad_a*grad_a*(static_cast<T>(1)-beta2);
      var_a -= (m_a*alpha) / (sqrt(v_a)+epsilon);
    }
  }
}

template <typename T, typename Tindex, typename Tstep>
struct KvSparseApplyAdamAsync<GPUDevice, T, Tindex, Tstep> {
  Status operator()(const GPUDevice &d,
                    EmbeddingVarGPU<Tindex, T> *var, 
                    EmbeddingVarGPU<Tindex, T> *m, 
                    EmbeddingVarGPU<Tindex, T> *v, 
                    typename TTypes<T>::Scalar beta1_power_scalar, 
                    typename TTypes<T>::Scalar beta2_power_scalar, 
                    typename TTypes<Tindex>::ConstVec indices_vec, 
                    typename TTypes<T>::ConstMatrix grad, 
                    typename TTypes<T>::ConstScalar lr_scalar, 
                    typename TTypes<T>::ConstScalar beta1_scalar, 
                    typename TTypes<T>::ConstScalar beta2_scalar, 
                    typename TTypes<T>::ConstScalar epsilon_scalar, 
                    typename TTypes<Tstep>::ConstScalar global_step_scalar, 
                    bool apply_sparse_rmsprop, const int64 inner_dim, 
                    Allocator *alloc) {
    const int32 N = indices_vec.dimension(0);
    if (N <= 0) return Status::OK();

    if (inner_dim > 0) {
      const int64 global_step = global_step_scalar();
      int32 *item_idxs = TypedAllocator::Allocate<int32>(alloc, N, AllocationAttributes());
      var->LookupOrCreateKey(indices_vec.data(), item_idxs, N, d.stream(), global_step);
      auto const block_size = 256;
      auto const grid_size = N;
        TF_CHECK_OK(GpuLaunchKernel(KvSparseApplyAdamAsyncKernel<T>, 
                            grid_size, block_size, 0, d.stream(),
                            item_idxs, var->ValueLen(), var->kv()->d_bank_ptrs,
                            var->kv()->d_existence_flag_ptrs, var->EmbIdx(),
                            v->EmbIdx(), m->EmbIdx(), var->SlotNum(), 
                            var->kv()->initial_bank_size, beta1_scalar.data(), 
                            beta2_scalar.data(), beta1_power_scalar.data(), 
                            beta2_power_scalar.data(), epsilon_scalar.data(), 
                            lr_scalar.data(), grad.data(), var->DefaultValuePtr(), 
                            v->DefaultValuePtr(), m->DefaultValuePtr(), 
                            var->GetDefaultValueDim(), v->GetDefaultValueDim(), 
                            m->GetDefaultValueDim(), apply_sparse_rmsprop));
      TypedAllocator::Deallocate(alloc, item_idxs, N);
    }

    if (!apply_sparse_rmsprop) {
      beta1_power_scalar.device(d) = beta1_power_scalar * beta1_scalar;
      beta2_power_scalar.device(d) = beta2_power_scalar * beta2_scalar;
    }

    return Status::OK();
}
};

#endif // TENSORFLOW_USE_GPU_EV

template <typename T>
__global__ __launch_bounds__(1024) void ApplyAdamAsyncKernel(
                            T *var, T *m, T *v, T *beta1_power, T *beta2_power, 
                            const T *lr_scalar, const T *beta1_scalar, 
                            const T *beta2_scalar, const T *epsilon_scalar, 
                            const T *grad, const bool use_nesterov, 
                            const int32 grad_size) {
  T lr = *lr_scalar;
  T beta1 = *beta1_scalar;
  T beta2 = *beta2_scalar;
  T epsilon = *epsilon_scalar;
  T alpha = lr * sqrt(static_cast<T>(1) - *beta2_power) / 
            (static_cast<T>(1) - *beta1_power);

  // beta1 == μ
  // beta2 == ν
  // v     == n
  // var   == θ
  GPU_1D_KERNEL_LOOP(index, grad_size) {
    m[index] = m[index]*beta1 + grad[index]*(static_cast<T>(1) - beta1);
    v[index] = v[index]*beta2 + grad[index]*grad[index]*(static_cast<T>(1)-beta2);
    if (use_nesterov) {
      var[index] -= ((grad[index]*(static_cast<T>(1)-beta1) + beta1*m[index])*alpha) / 
                    (sqrt(v[index])+epsilon);
    } else {
      var[index] -= (m[index]*alpha) / (sqrt(v[index]) + epsilon);
    }
  }
}

template <typename T>
struct ApplyAdamAsync<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::Scalar beta1_power,
                  typename TTypes<T>::Scalar beta2_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad, bool use_nesterov) {
  int32 grad_size = grad.size();

  GpuLaunchConfig config = GetGpuLaunchConfig(grad_size, d);
  GpuLaunchKernel(ApplyAdamAsyncKernel<T>, config.block_count,
                         config.thread_per_block, 0, d.stream(),
                         var.data(), m.data(), v.data(), beta1_power.data(),
                         beta2_power.data(), lr.data(), beta1.data(), 
                         beta2.data(), epsilon.data(), grad.data(), 
                         use_nesterov, grad_size);
  // update beta1_power && beta2_power
  beta1_power.device(d) = beta1_power * beta1;
  beta2_power.device(d) = beta2_power * beta2;
}

};

template <typename T, typename Tindex>
__global__ __launch_bounds__(1024) void SparseApplyAdamAsyncKernel(
                  T *var, T *m, T *v, T *beta1_power, T *beta2_power, 
                  const T *grad, const Tindex *indices_vec, const T *lr_scalar,
                  const T *beta1_scalar, const T *beta2_scalar, 
                  const T *epsilon_scalar, const Tindex first_dim_size,
                  const Tindex grad_size, const Tindex indices_size, 
                  const bool apply_sparse_rmsprop) {
  const T lr = *lr_scalar;
  const T beta1 = *beta1_scalar;
  const T beta2 = *beta2_scalar;
  const T epsilon = *epsilon_scalar;

  const T alpha = lr*sqrt(static_cast<T>(1)- (*beta2_power))/(static_cast<T>(1)- (*beta1_power));
  Tindex col_size = grad_size / indices_size;

  if (apply_sparse_rmsprop) {
    GPU_1D_KERNEL_LOOP(grad_index, grad_size) {
      Tindex grad_row_id = grad_index / col_size;
      Tindex param_row = indices_vec[grad_row_id];
      if (param_row < 0 || param_row >= first_dim_size) {
        // Ignore indices that are out of range
        continue;
      }

      // Index of var, m and v
      Tindex param_index = param_row*col_size + grad_index%col_size;
      T &v_a = v[param_index];
      T &m_a = m[param_index];
      T &var_a = var[param_index];
      const T &grad_a = grad[grad_index];

      v_a = v_a*beta2 + grad_a*grad_a*(static_cast<T>(1)-beta2);
      m_a = m_a*beta1 + (v_a + impl_rsqrt(epsilon)*lr*grad_a);
      var_a -= m_a;
    }
  } else {
    GPU_1D_KERNEL_LOOP(grad_index, grad_size) {
      Tindex grad_row_id = grad_index / col_size;
      Tindex param_row = indices_vec[grad_row_id];
      if (param_row < 0 || param_row >= first_dim_size) {
        // Ignore indices that are out of range
        continue;
      }

      // Index of var, m and v
      Tindex param_index = param_row*col_size + grad_index%col_size;
      T &v_a = v[param_index];
      T &m_a = m[param_index];
      T &var_a = var[param_index];
      const T &grad_a = grad[grad_index];
      v_a = v_a*beta2 + grad_a*grad_a*(static_cast<T>(1)-beta2);
      m_a = m_a*beta1 + grad_a*(static_cast<T>(1)-beta1);
      var_a -= (m_a*alpha) / (sqrt(v_a) + epsilon);
    }
  }
}

template <typename T, typename Tindex>
struct SparseApplyAdamAsync<GPUDevice, T, Tindex> {
  Status operator()(const GPUDevice &d, typename TTypes<T>::Matrix var,
                  typename TTypes<T>::Matrix m, typename TTypes<T>::Matrix v,
                  typename TTypes<T>::Scalar beta1_power_scalar,
                  typename TTypes<T>::Scalar beta2_power_scalar,
                  typename TTypes<T>::ConstScalar lr_scalar,
                  typename TTypes<T>::ConstScalar beta1_scalar,
                  typename TTypes<T>::ConstScalar beta2_scalar,
                  typename TTypes<T>::ConstScalar epsilon_scalar,
                  typename TTypes<T>::ConstMatrix grad,
                  typename TTypes<Tindex>::ConstVec indices_vec,
                  bool apply_sparse_rmsprop, int64 inner_dim) {
    const Tindex N = indices_vec.dimension(0);
    if (N <= 0) return Status::OK();

    const Tindex grad_size = grad.size();
    const Tindex first_dim_size = static_cast<Tindex>(var.dimension(0));
    const Tindex indices_size = indices_vec.size();

    GpuLaunchConfig config = GetGpuLaunchConfig(grad_size, d);

    GpuLaunchKernel(SparseApplyAdamAsyncKernel<T, Tindex>,
                    config.block_count, config.thread_per_block, 0, d.stream(),
                    var.data(), m.data(), v.data(), beta1_power_scalar.data(),
                    beta2_power_scalar.data(), grad.data(), indices_vec.data(),
                    lr_scalar.data(), beta1_scalar.data(), beta2_scalar.data(),
                    epsilon_scalar.data(), first_dim_size, grad_size, 
                    indices_size, apply_sparse_rmsprop);
    if (!apply_sparse_rmsprop) {
      beta1_power_scalar.device(d) = beta1_power_scalar * beta1_scalar;
      beta2_power_scalar.device(d) = beta2_power_scalar * beta2_scalar;
    }

    return Status::OK();
  }
};

}  // namespace functor

#if TENSORFLOW_USE_GPU_EV
template struct functor::KvSparseApplyAdagrad<GPUDevice, int32, float>;
template struct functor::KvSparseApplyAdagrad<GPUDevice, int32, double>;
template struct functor::KvSparseApplyAdagrad<GPUDevice, int64, float>;
template struct functor::KvSparseApplyAdagrad<GPUDevice, int64, double>;
template struct functor::KvSparseApplyFtrl<GPUDevice, int32, float>;
template struct functor::KvSparseApplyFtrl<GPUDevice, int32, double>;
template struct functor::KvSparseApplyFtrl<GPUDevice, int64, float>;
template struct functor::KvSparseApplyFtrl<GPUDevice, int64, double>;
#define EXPLICITLY_INSTANTIATE_FUNCTOR(T) \
  template struct functor::KvSparseApplyAdamAsync<GPUDevice, T, int32, int32>; \
  template struct functor::KvSparseApplyAdamAsync<GPUDevice, T, int32, int64>; \
  template struct functor::KvSparseApplyAdamAsync<GPUDevice, T, int64, int32>; \
  template struct functor::KvSparseApplyAdamAsync<GPUDevice, T, int64, int64>;
EXPLICITLY_INSTANTIATE_FUNCTOR(float);
EXPLICITLY_INSTANTIATE_FUNCTOR(double);
#undef EXPLICITLY_INSTANTIATE_FUNCTOR
#endif // TENSORFLOW_USE_GPU_EV

template struct functor::ApplyAdamAsync<GPUDevice, Eigen::half>;
template struct functor::ApplyAdamAsync<GPUDevice, float>;
template struct functor::ApplyAdamAsync<GPUDevice, double>;

#define EXPLICITLY_INSTANTIATE_FUNCTOR(T) \
  template struct functor::SparseApplyAdamAsync<GPUDevice, T, int32>; \
  template struct functor::SparseApplyAdamAsync<GPUDevice, T, int64>;
EXPLICITLY_INSTANTIATE_FUNCTOR(Eigen::half);
EXPLICITLY_INSTANTIATE_FUNCTOR(float);
EXPLICITLY_INSTANTIATE_FUNCTOR(double);
#undef EXPLICITLY_INSTANTIATE_FUNCTOR

}  // end namespace tensorflow
#endif  // GOOGLE_CUDA
