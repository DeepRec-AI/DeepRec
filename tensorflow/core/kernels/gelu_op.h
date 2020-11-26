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

#ifndef TENSORFLOW_CORE_KERNELS_GELU_OP_H_
#define TENSORFLOW_CORE_KERNELS_GELU_OP_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

namespace internal {
constexpr double kCoeff = 0.044715;
constexpr double kSqrtHalf = 0.7071067811865476;
constexpr double kTwoRsqrtPi = 1.1283791670955126;
constexpr double kAlpha = kSqrtHalf * kTwoRsqrtPi;
}  // namespace internal

namespace functor {

// Functor used by GeluOp to do the computations.
template <typename Device, typename T>
struct Gelu {
  // Computes Gelu activation.
  //
  // features: any shape.
  // approximate: whether to enable approximation.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  bool approximate, typename TTypes<T>::Tensor activations) {
    const T one = static_cast<T>(1);
    const T half = static_cast<T>(0.5);
    if (approximate) {
      // y = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
      activations.device(d) =
          half * features *
          (one +
           (static_cast<T>(internal::kAlpha) *
            (features + static_cast<T>(internal::kCoeff) * features.cube()))
               .tanh());
    } else {
      // y = x * normcdf(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
      activations.device(d) =
          half * features *
          (one + (features * static_cast<T>(internal::kSqrtHalf)).erf());
    }
  }
};

// Functor used by GeluGradOp to do the computations.
template <typename Device, typename T>
struct GeluGrad {
  // Computes GeluGrad backprops.
  //
  // gradients: gradients backpropagated to the Gelu op.
  // features: inputs that were passed to the Gelu op.
  // approximate: whether to enable approximation.
  // backprops: gradients to backpropagate to the Gelu inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor features, bool approximate,
                  typename TTypes<T>::Tensor backprops) {
    const T one = static_cast<T>(1);
    const T half = static_cast<T>(0.5);
    if (approximate) {
      const T kBeta = static_cast<T>(internal::kAlpha) *
                        static_cast<T>(internal::kCoeff) * static_cast<T>(3);
      const auto y =
          (static_cast<T>(internal::kAlpha) *
           ((static_cast<T>(internal::kCoeff) * features.cube()) + features))
              .tanh();
      backprops.device(d) =
          ((-features * y.square() + features) *
               (kBeta * features.square() + static_cast<T>(internal::kAlpha)) +
           one + y) *
          gradients * half;
    } else {
      backprops.device(d) =
          gradients *
          (static_cast<T>(internal::kAlpha * 0.5) * features *
               (-features.square() * half).exp() +
           (half * (one + (features * static_cast<T>(internal::kSqrtHalf)).erf())));
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_GELU_OP_H_
