/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/sparse_fill_empty_rows_op_util.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename T>
class SparseFillEmptyRowsOp : public OpKernel {
 public:
  explicit SparseFillEmptyRowsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* indices_t;
    const Tensor* values_t;
    const Tensor* dense_shape_t;
    const Tensor* default_value_t;
    OP_REQUIRES_OK(context, context->input("indices", &indices_t));
    OP_REQUIRES_OK(context, context->input("values", &values_t));
    OP_REQUIRES_OK(context, context->input("dense_shape", &dense_shape_t));
    OP_REQUIRES_OK(context, context->input("default_value", &default_value_t));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(dense_shape_t->shape()),
                errors::InvalidArgument("dense_shape must be a vector, saw: ",
                                        dense_shape_t->shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(indices_t->shape()),
                errors::InvalidArgument("indices must be a matrix, saw: ",
                                        indices_t->shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(values_t->shape()),
                errors::InvalidArgument("values must be a vector, saw: ",
                                        values_t->shape().DebugString()));
    OP_REQUIRES(context, indices_t->dim_size(0) == values_t->dim_size(0),
                errors::InvalidArgument("The length of `values` (",
                    values_t->dim_size(0),
                    ") must match the first dimension of `indices` (",
                    indices_t->dim_size(0), ")."));
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(default_value_t->shape()),
        errors::InvalidArgument("default_value must be a scalar, saw: ",
                                default_value_t->shape().DebugString()));

    Tensor empty_row_indicator_t;
    Tensor reverse_index_map_t;
    Tensor output_indices_t;
    Tensor output_values_t;
    ParallelSparseFillEmptyRows<T>(context, indices_t, values_t, dense_shape_t,
        default_value_t, &empty_row_indicator_t, &reverse_index_map_t,
        &output_indices_t, &output_values_t);
    context->set_output("empty_row_indicator", empty_row_indicator_t);
    context->set_output("reverse_index_map", reverse_index_map_t);
    context->set_output("output_indices", output_indices_t);
    context->set_output("output_values", output_values_t);
  }
};

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("SparseFillEmptyRows")     \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          SparseFillEmptyRowsOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

template <typename T>
class SparseFillEmptyRowsGradOp : public OpKernel {
 public:
  explicit SparseFillEmptyRowsGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* reverse_index_map_t;
    const Tensor* grad_values_t;
    OP_REQUIRES_OK(context,
                   context->input("reverse_index_map", &reverse_index_map_t));
    OP_REQUIRES_OK(context, context->input("grad_values", &grad_values_t));

    const CPUDevice& d = context->eigen_device<CPUDevice>();

    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(reverse_index_map_t->shape()),
        errors::InvalidArgument("reverse_index_map must be a vector, saw: ",
                                reverse_index_map_t->shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(grad_values_t->shape()),
                errors::InvalidArgument("grad_values must be a vector, saw: ",
                                        grad_values_t->shape().DebugString()));

    const auto reverse_index_map = reverse_index_map_t->vec<int64>();
    const auto grad_values = grad_values_t->vec<T>();

    const int64 N = reverse_index_map_t->shape().dim_size(0);
    const int64 N_full = grad_values_t->shape().dim_size(0);

    Tensor* d_values_t;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "d_values", TensorShape({N}), &d_values_t));
    auto d_values = d_values_t->vec<T>();
    Tensor* d_default_value_t;
    OP_REQUIRES_OK(context,
                   context->allocate_output("d_default_value", TensorShape({}),
                                            &d_default_value_t));
    T& d_default_value = d_default_value_t->scalar<T>()();
    d_default_value = T();

    Tensor visited_t;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_BOOL, TensorShape({N_full}), &visited_t));
    auto visited = visited_t.vec<bool>();
    visited.device(d) = visited.constant(false);

    for (int i = 0; i < N; ++i) {
      // Locate the index of the output of the forward prop associated
      // with this location in the input of the forward prop.  Copy
      // the gradient into it.  Mark it as visited.
      int64 reverse_index = reverse_index_map(i);
      OP_REQUIRES(
          context, 0 <= reverse_index && reverse_index < N_full,
          errors::InvalidArgument("Elements in reverse index must be in [0, ",
                                  N_full, ") but got ", reverse_index));
      d_values(i) = grad_values(reverse_index);
      visited(reverse_index) = true;
    }
    for (int j = 0; j < N_full; ++j) {
      // The default value gradient gets the accumulated remainder of
      // the backprop values (since the default value was used to fill
      // in these slots in the forward calculation).
      if (!visited(j)) {
        d_default_value += grad_values(j);
      }
    }
  }
};

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("SparseFillEmptyRowsGrad") \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          SparseFillEmptyRowsGradOp<type>)

TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS
}  // namespace tensorflow
