/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_TRAINING_ALI_OPS_GPU_H_
#define TENSORFLOW_CORE_KERNELS_TRAINING_ALI_OPS_GPU_H_

#if GOOGLE_CUDA
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/kernels/kv_variable_ops_gpu.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename TKey, typename T>
struct KvSparseApplyAdagrad {
  void operator()(int32 num_items,
                  Allocator* alloc,
                  EmbeddingVarGPU<TKey, T>* var,
                  EmbeddingVarGPU<TKey, T>* accum,
                  const TKey* key_base,
                  const T* grad,
                  T lr,
                  int64 gs,
                  cudaStream_t stream);
};

template <typename Device, typename TKey, typename T>
struct KvSparseApplyFtrl {
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
                  cudaStream_t stream);
};
}  // end namespace functor
}  // end namespace tensorflow
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_TRAINING_ALI_OPS_GPU_H_
