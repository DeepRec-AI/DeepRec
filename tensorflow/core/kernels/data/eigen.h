/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_EIGEN_H_
#define TENSORFLOW_CORE_KERNELS_DATA_EIGEN_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/version.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// NOTE: EIGEN_MAX_ALIGN_BYTES is 64 in TF 1.x. See:
// DeepRec/third_party/eigen.BUILD#L67
#if EIGEN_MAX_ALIGN_BYTES == 0
#define CHECK_EIGEN_ALIGN(...) (true)
#else
#define CHECK_EIGEN_ALIGN(...) \
  (0 == reinterpret_cast<intptr_t>(__VA_ARGS__) % EIGEN_MAX_ALIGN_BYTES)
#endif

#endif  // TENSORFLOW_CORE_KERNELS_DATA_EIGEN_H_
