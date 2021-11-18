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

#define EIGEN_USE_THREADS

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/sorter.h"
#include "tensorflow/core/util/sparse/dim_comparator.h"

namespace tensorflow {

template <typename T>
class SparseValidCutoffOp : public OpKernel {
 public:
  explicit SparseValidCutoffOp(OpKernelConstruction* context) : OpKernel(context) {
    string side;
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
    OP_REQUIRES_OK(context, context->GetAttr("length", &length_));
    OP_REQUIRES_OK(context, context->GetAttr("side", &side));
    InitCutFunc(side, length_);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* indices_t;
    const Tensor* values_t;
    const Tensor* dense_shape_t;
    OP_REQUIRES_OK(context, context->input("indices", &indices_t));
    OP_REQUIRES_OK(context, context->input("values", &values_t));
    OP_REQUIRES_OK(context, context->input("dense_shape", &dense_shape_t));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(dense_shape_t->shape()),
                errors::InvalidArgument("dense_shape must be a vector, saw: ",
                                        dense_shape_t->shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(indices_t->shape()),
                errors::InvalidArgument("indices must be a matrix, saw: ",
                                        indices_t->shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(values_t->shape()),
                errors::InvalidArgument("values must be a vector, saw: ",
                                        values_t->shape().DebugString()));

    auto indices = indices_t->matrix<int64>();
    auto values = values_t->vec<T>();
    auto dense_shape = dense_shape_t->vec<int64>();
    const int input_rank = dense_shape.size();
    const int cutoff_axis = (axis_ < 0) ? input_rank + axis_ : axis_;
    OP_REQUIRES(context, cutoff_axis >= 1 && cutoff_axis < input_rank,
                errors::InvalidArgument("Cutoff dimension must be in range [",
                                        -input_rank, ", ", input_rank,
                                        "), and should not be 0, got ", axis_));

    // count valid size per sample
    int total_columns = 1;
    for (int i = 0; i < cutoff_axis; ++i) {
      total_columns *= dense_shape(i);
    }
    std::vector<int64> sample_max_count(total_columns);
    for (int i = 0; i < values.size(); ++i) {
      int current_column_id = GetColumnId(indices, dense_shape, i, cutoff_axis);
      sample_max_count[current_column_id] = std::max(sample_max_count[current_column_id], indices(i, cutoff_axis) + 1);
    }
    // count output indices and values size
    int count = 0;
    for (int i = 0; i < values.size(); ++i) {
      int current_column_id = GetColumnId(indices, dense_shape, i, cutoff_axis);
      if (Cutoff(sample_max_count[current_column_id], indices(i, cutoff_axis)) >= 0) {
        ++count;
      }
    }
    // set output value and indices
    Tensor* indices_out_t;
    Tensor* values_out_t;
    Tensor* dense_shape_out_t;
    context->allocate_output(0, {count, dense_shape.size()}, &indices_out_t);
    context->allocate_output(1, {count}, &values_out_t);
    context->allocate_output(2, {dense_shape.size()}, &dense_shape_out_t);
    auto indices_out = indices_out_t->matrix<int64>();
    auto values_out = values_out_t->vec<T>();
    auto dense_shape_out = dense_shape_out_t->vec<int64>();
    count = 0;
    for (int i = 0; i < values.size(); ++i) {
      int current_column_id = GetColumnId(indices, dense_shape, i, cutoff_axis);
      int64 cutoff_idx = Cutoff(sample_max_count[current_column_id],
                                indices(i, cutoff_axis));
      if (cutoff_idx >= 0) {
        for (int j = 0; j < dense_shape.size(); ++j) {
          indices_out(count, j) = (j == cutoff_axis) ? cutoff_idx : indices(i, j);
        }
        values_out(count) = values(i);
        ++count;
      }
    }
    for (int i = 0; i < dense_shape.size(); ++i) {
      dense_shape_out(i) = (i == cutoff_axis) ? length_ : dense_shape(i);
    }
  }

 private:
  int axis_;
  int length_;
  std::function<int64(int64, int64)> cutoff_func_;

  void InitCutFunc(string side, int cutoff_length) {
    if (side == "right") {
      cutoff_func_ = [cutoff_length](int64 max_count, int64 cur_indice) {
        return cur_indice >= cutoff_length ? -1 : cur_indice;
      };
    } else {
      cutoff_func_ = [cutoff_length](int64 max_count, int64 cur_indice) {
        return cur_indice - (max_count - cutoff_length);
      };
    }
  }

  int64 Cutoff(const int64 max_count,
               const int64 cur_indice) {
    if (max_count <= length_) {
      return cur_indice;
    }
    return cutoff_func_(max_count, cur_indice);
  }

  int GetColumnId(typename TTypes<int64>::ConstMatrix indices,
                  typename TTypes<int64>::ConstVec dense_shape,
                  int idx,
                  int cutoff_axis) {
    int output_id = indices(idx, 0);
    for (int i = 1; i < cutoff_axis; ++i) {
      output_id = output_id * dense_shape(i - 1) + indices(idx, i);
    }
    return output_id;
  }
};

#define REGISTER_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SparseValidCutoff").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseValidCutoffOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS
}  // namespace tensorflow
