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

#ifndef TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_H_
#define TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_H_

// This file requires the following include because it uses GpuAtomicMax:
// #include "tensorflow/core/util/gpu_kernel_helper.h"

// Unfortunately we can't add the #include, since it breaks compilation for
// non-GPU targets. This only breaks in clang, because it's more strict for
// template code and GpuAtomicMax is used in template context.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"

#include "tensorflow/core/kernels/segment_reduction_ops_util.h"

namespace tensorflow {

namespace functor {

template <typename Device, typename T, typename Index, typename InitialValueF,
          typename ReductionF>
struct UnsortedSegmentFunctor {
  void operator()(OpKernelContext* ctx, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  typename TTypes<T, 2>::ConstTensor data,
                  typename TTypes<T, 2>::Tensor output);
};

// initial value functors
template <typename T>
struct Zero {
  EIGEN_STRONG_INLINE T operator()() const { return T(0); }
};

template <typename T>
struct One {
  EIGEN_STRONG_INLINE T operator()() const { return T(1); }
};

template <typename T>
struct Lowest {
  EIGEN_STRONG_INLINE T operator()() const {
    return Eigen::NumTraits<T>::lowest();
  }
};

template <typename T>
struct Highest {
  EIGEN_STRONG_INLINE T operator()() const {
    return Eigen::NumTraits<T>::highest();
  }
};

}  // namespace functor

// The UnsortedSegmentReduction OpKernel. The DeviceReductionFunctor
// is the device specific implementation of the reduction. These device
// specific implementations are templated themselves with the corresponding
// initial value functors and reduction functors.
template <typename T, typename Index, typename DeviceReductionFunctor>
class UnsortedSegmentReductionOp : public OpKernel {
 public:
  explicit UnsortedSegmentReductionOp(OpKernelConstruction* context)
      : OpKernel(context), reduction_functor_(DeviceReductionFunctor()) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& data = context->input(0);
    const Tensor& segment_ids = context->input(1);
    const Tensor& num_segments = context->input(2);
    if (!UnsortedSegmentReductionDoValidation(this, context, data, segment_ids,
                                              num_segments)) {
      return;
    }
    const auto segment_flat = segment_ids.flat<Index>();
    const int64 output_rows = internal::SubtleMustCopy(static_cast<int64>(
        num_segments.dtype() == DT_INT32 ? num_segments.scalar<int32>()()
                                         : num_segments.scalar<int64>()()));
    OP_REQUIRES(context, output_rows >= 0,
                errors::InvalidArgument("Input num_segments == ", output_rows,
                                        " must not be negative."));
    TensorShape output_shape;
    output_shape.AddDim(output_rows);
    for (int i = segment_ids.dims(); i < data.dims(); i++) {
      output_shape.AddDim(data.dim_size(i));
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_flat = output->flat_outer_dims<T>();
    auto data_flat = data.flat_inner_outer_dims<T, 2>(segment_ids.dims() - 1);
    reduction_functor_(context, segment_ids.shape(), segment_flat, data_flat,
                       output_flat);
  }

 protected:
  DeviceReductionFunctor reduction_functor_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_H_
