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
======================================================================*/

#if GOOGLE_CUDA

#include "tensorflow/core/framework/embedding/batch.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
template<class V>
__global__ void BatchCopy(V** batch, V* val_base, int value_len,
    int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / value_len;
  int item_pos = i % value_len;

  if (i < limit * value_len) {
    val_base[i] = *(batch[item_id] + item_pos);
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T) \
   template __global__ void BatchCopy<T>(T**, T*, int, int);
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_int32(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_int64(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX

template<class V>
__global__ void BatchUnpack(V** dev_value_address,
    V* memcpy_buffer_gpu, int value_len, int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / value_len;
  int item_pos = i % value_len;

  if (i < limit * value_len) {
    *(dev_value_address[item_id] + item_pos) = memcpy_buffer_gpu[i];
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T)                       \
  template __global__ void BatchUnpack<T>(T**, T*, int, int);
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_int32(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_int64(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX

template<class V>
__global__ void SparseApplyAdagradGPU(V** a, V** v, V* g, V lr,
    int embedding_dim, long long int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / embedding_dim;
  int item_pos = i % embedding_dim;

  if (i < limit * embedding_dim) {
    *(a[item_id] + item_pos) += g[i] * g[i];
    *(v[item_id] + item_pos) -=
        lr * g[i] * rsqrt(*(a[item_id] + item_pos));
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T)                \
  template __global__ void SparseApplyAdagradGPU<T>( \
    T**, T**, T*, T, int, long long int);
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX

template<class V>
__global__ void SparseApplyAdamGPU(V** var, V** m, V** v,
    V* g, V alpha, V beta1, V beta2, V epsilon,
    int embedding_dim, long long int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / embedding_dim;
  int item_pos = i % embedding_dim;

  if (i < limit * embedding_dim) {
    *(m[item_id] + item_pos) +=
        (g[i] - (*(m[item_id] + item_pos))) * (1.0 - beta1);
    *(v[item_id] + item_pos) +=
        (g[i] * g[i] - (*(v[item_id] + item_pos))) * (1.0 - beta2);
    *(var[item_id] + item_pos) -=
        (*(m[item_id] + item_pos) * alpha) /
        (sqrt(*(v[item_id] + item_pos)) + epsilon);
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T)                \
  template __global__ void SparseApplyAdamGPU<T>( \
    T**, T**, T**, T*, T, \
    T, T, T, int, long long int);
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX

template<class V>
__global__ void SparseApplyAdamAsyncGPU(
    V** var, V** m, V** v,
    V* g, V lr, V beta1, V beta2, V epsilon,
    V* beta1_power_ptr, V* beta2_power_ptr,
    int embedding_dim, long long int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / embedding_dim;
  int item_pos = i % embedding_dim;

  if (i < limit * embedding_dim) {
    V beta1_power = *beta1_power_ptr;
    V beta2_power = *beta2_power_ptr;
    const V alpha = lr *
        sqrt(static_cast<V>(1) - beta2_power) /
        (static_cast<V>(1) - beta1_power);
    *(m[item_id] + item_pos) = *(m[item_id] + item_pos) * beta1 +
        g[i] * (1 - beta1);
    *(v[item_id] + item_pos) = *(v[item_id] + item_pos) * beta2 +
        g[i] * g[i] * (1 - beta2);
    *(var[item_id] + item_pos) -=
        (*(m[item_id] + item_pos) * alpha) /
        (sqrt(*(v[item_id] + item_pos)) + epsilon);
  }
  __syncthreads();

  if (i == 0) {
    *beta1_power_ptr *= beta1;
    *beta2_power_ptr *= beta2;
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T)                \
  template __global__ void SparseApplyAdamAsyncGPU<T>( \
    T**, T**, T**, T*, T, \
    T, T, T, T*, T*, int, long long int);
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX

template<class V>
__global__ void SparseApplyAdamAsyncSparseRmspropGPU(
    V** var, V** m, V** v,
    V* g, V lr, V beta1, V beta2, V epsilon,
    int embedding_dim, long long int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / embedding_dim;
  int item_pos = i % embedding_dim;

  if (i < limit * embedding_dim) {
    *(v[item_id] + item_pos) =  *(v[item_id] + item_pos) * beta2 +
        g[i] * g[i] * (1.0 - beta2);
    *(m[item_id] + item_pos) = *(m[item_id] + item_pos) * beta1 +
        rsqrt(*(v[item_id] + item_pos) + epsilon) *
        lr * g[i];
    *(var[item_id] + item_pos) -= *(m[item_id] + item_pos);
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T)                \
  template __global__ void SparseApplyAdamAsyncSparseRmspropGPU<T>( \
    T**, T**, T**, T*, T, \
    T, T, T, int, long long int);
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX

template<class V>
__global__ void SparseApplyAdamWGPU(V** var, V** m, V** v,
    V* g, V alpha, V beta1, V beta2, V epsilon,
    V weight_decay, int embedding_dim, long long int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / embedding_dim;
  int item_pos = i % embedding_dim;

  if (i < limit * embedding_dim) {
    *(m[item_id] + item_pos) +=
        (g[i] - *(m[item_id] + item_pos)) * (1.0 - beta1);
    *(v[item_id] + item_pos) +=
        (g[i] * g[i] - *(v[item_id] + item_pos)) * (1.0 - beta2);
    *(var[item_id] + item_pos) -=
        (*(m[item_id] + item_pos) * alpha) /
        (sqrt(*(v[item_id] + item_pos)) + epsilon) +
        weight_decay * (*(var[item_id] + item_pos));
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T)                \
  template __global__ void SparseApplyAdamWGPU<T>( \
    T**, T**, T**, T*, T, \
    T, T, T, T, int, long long int);
TF_CALL_float(REGISTER_KERNELS_ALL_INDEX)
TF_CALL_double(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX

template<class V>
__global__ void CopyEmbedding(V** batch, V** batch_data_space,
    int total_dims, int limit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int item_id = i / total_dims;
  int item_pos = i % total_dims;

  if (i < limit  * total_dims) {
    *(batch_data_space[item_id] + item_pos) = *(batch[item_id] + item_pos);
  }
}

#define REGISTER_KERNELS_ALL_INDEX(T) \
   template __global__ void CopyEmbedding<T>(T**, T**, int, int);
TF_CALL_FLOAT_TYPES(REGISTER_KERNELS_ALL_INDEX)
#undef REGISTER_KERNELS_ALL_INDEX
}  // namespace tensorflow
#endif  // GOOGLE_CUDA
