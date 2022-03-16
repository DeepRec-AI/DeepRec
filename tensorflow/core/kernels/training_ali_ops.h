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

#ifndef TENSORFLOW_CORE_KERNELS_TRAINING_ALI_OPS_H_
#define TENSORFLOW_CORE_KERNELS_TRAINING_ALI_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#if GOOGLE_CUDA
#include "tensorflow/core/kernels/kv_variable_ops_gpu.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {
namespace functor {

#if GOOGLE_CUDA
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
#endif  // GOOGLE_CUDA

template <typename Device, typename T>
struct ApplyAdagradDecay {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad,
                  bool need_decay,
                  typename TTypes<T>::ConstScalar decay_rate,
                  typename TTypes<T>::ConstScalar decay_baseline);
};

template <typename Device, typename T>
struct ApplyAdamAsync {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::Flat beta1_power,
                  typename TTypes<T>::Flat beta2_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad, bool use_nesterov);
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TRAINING_ALI_OPS_H_
