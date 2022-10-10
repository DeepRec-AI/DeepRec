/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
n
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

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/sparse_fill_empty_rows_op_util.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {
template <typename T, typename Tindex>
struct ParallelSparseFillEmptyRows<CPUDevice, T, Tindex> {
  void operator()(const CPUDevice& d,
		  OpKernelContext *context,
		  Allocator *allocator,
		  typename TTypes<Tindex>::ConstMatrix indices,
		  typename TTypes<T>::ConstVec values,
		  typename TTypes<T>::ConstScalar default_val,
		  typename TTypes<Tindex>::Vec reverse_index_map,
		  Tensor *output_indices_t,
		  Tensor *output_values_t,
		  std::vector<Tindex> &scratch,
		  const std::vector<Tindex> &offset,
		  const Tindex dense_rows,
		  const Tindex N,
		  TTypes<bool>::Vec empty_row_indicator,
		  const int rank) {
    for (Tindex row = 0; row < dense_rows; ++row) {
      // Scratch here describes the number of elements in this dense row
      empty_row_indicator(row) = (scratch[row] == 0);
      // In filled version, each row has at least one element.
      scratch[row] = std::max(scratch[row], Tindex{1});
      // Update scratch to represent the number of elements up to and
      // including dense_row + 1:
      //   scratch(0) == #{elements of row 0}
      //   scratch(1) == #{elements of row 1} + #{elements of row 0}
      // ...
      //   scratch(i) == starting index for elements in row i+1.
      if (row > 0)
	scratch[row] += scratch[row-1];
    }

    const Tindex N_full = scratch[dense_rows-1];
    TensorShape output_indices_shape({N_full, rank});
    OP_REQUIRES_OK(context,
		   context->allocate_temp(DataTypeToEnum<Tindex>::v(),
					  output_indices_shape,
					  output_indices_t));
    auto output_indices = output_indices_t->matrix<Tindex>();
    output_indices.device(d) = output_indices.constant(0);

    OP_REQUIRES_OK(context,
		   context->allocate_temp(DataTypeToEnum<T>::v(),
					  TensorShape({N_full}),
					  output_values_t));
    auto output_values = output_values_t->vec<T>();
    output_values.device(d) = output_values.constant(default_val());

    // Fill in values for rows that are not missing
    auto RunTask = [&indices, &scratch, &offset, &output_indices, &values,
		    &output_values, &reverse_index_map, &rank]
      (Tindex start, Tindex end) {
      for (Tindex i = start; i < end; ++i) {
	const Tindex row = indices(i, 0);
	const Tindex output_i = ((row == 0) ? 0 : scratch[row-1]) + offset[i];
	std::copy_n(&indices(i, 0), rank, &output_indices(output_i, 0));
	output_values(output_i) = values(i);
	// We'll need this reverse index map to backprop correctly
	reverse_index_map(i) = output_i;
      }
    };
    
    const Eigen::TensorOpCost cost(0, 0, 10);
    d.parallelFor(N, cost, RunTask);
    // Fill in values for rows that are missing
    for (Tindex row = 0; row < dense_rows; ++row) {
      if (empty_row_indicator(row)) {
	const Tindex starting_index = (row == 0) ? 0 : scratch[row-1];
	// Remaining index values were set to zero already.
	// The value at this index was set to default_value already.
	// Just need to set the row index in the right location.	
	output_indices(starting_index, 0) = row;
      }
    }
  }
};
}  // end of namespace functor

template <typename Device, typename T, typename Tindex>
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
    const Tindex N = indices_t->shape().dim_size(0);
    const int rank = indices_t->shape().dim_size(1);
    const Tindex dense_rows = dense_shape_t->vec<Tindex>()(0);

    OP_REQUIRES_OK(context,
		   context->allocate_temp(DT_BOOL, TensorShape({dense_rows}),
					  &empty_row_indicator_t));
    
    OP_REQUIRES_OK(context,
		   context->allocate_temp(DataTypeToEnum<Tindex>::v(), TensorShape({N}),
					  &reverse_index_map_t));
    if (dense_rows == 0) {
      OP_REQUIRES(context, N == 0,
                  errors::InvalidArgument(
                      "Received SparseTensor with dense_shape[0] = 0" 
                      "but indices.shape[0] = ", N));
      TensorShape output_indices_shape({0, rank});
      OP_REQUIRES_OK(context,
		     context->allocate_temp(DataTypeToEnum<Tindex>::v(), output_indices_shape,
					    &output_indices_t));
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::v(),
                                            TensorShape({0}), &output_values_t));
      // Exit early, noting more to do.
    } else {
        std::vector<Tindex> scratch(dense_rows, 0);
	std::vector<Tindex> offset_vec(N, 0);
	auto indices = indices_t->matrix<Tindex>();
        for (int i = 0; i < N; ++i) {
          const Tindex row = indices(i, 0);
          OP_REQUIRES(context, row >= 0 && row < dense_rows,
        		  errors::InvalidArgument("indices(", i, ", 0) is invalid: ",
        					  row, " >= ", dense_rows));
          offset_vec[i] = scratch[row]++;
        }
        
        const Device& device = context->template eigen_device<Device>();
	Allocator *allocator = context->get_allocator(AllocatorAttributes());
	functor::ParallelSparseFillEmptyRows<Device, T, Tindex>()
	  (device, context, allocator, indices_t->matrix<Tindex>(), values_t->vec<T>(),
	   default_value_t->scalar<T>(), reverse_index_map_t.vec<Tindex>(),
	   &output_indices_t, &output_values_t, scratch, offset_vec, dense_rows,
	   N, empty_row_indicator_t.vec<bool>(), rank);
    }
    
    context->set_output("empty_row_indicator", empty_row_indicator_t);
    context->set_output("reverse_index_map", reverse_index_map_t);
    context->set_output("output_indices", output_indices_t);
    context->set_output("output_values", output_values_t);
  }
};

#define REGISTER_KERNELS(D, T, Tindex)					\
  REGISTER_KERNEL_BUILDER(Name("SparseFillEmptyRows")			\
			  .Device(DEVICE_##D)				\
			  .TypeConstraint<T>("T")			\
			  .HostMemory("indices")			\
			  .HostMemory("dense_shape"),			\
                          SparseFillEmptyRowsOp<D##Device, T, Tindex>)
#define REGISTER_KERNELS_TINDEX_INT64(D, T) REGISTER_KERNELS(D, T, int64)
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS_TINDEX_INT64(CPU, T)

TF_CALL_ALL_TYPES(REGISTER_CPU_KERNELS)

#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex)					\
  template <>								\
  void ParallelSparseFillEmptyRows<GPUDevice, T, Tindex>::operator()(	\
    const GPUDevice& d,							\
    OpKernelContext *context,						\
    Allocator *allocator,						\
    typename TTypes<Tindex>::ConstMatrix indices,			\
    typename TTypes<T>::ConstVec values,				\
    typename TTypes<T>::ConstScalar default_val,			\
    typename TTypes<Tindex>::Vec reverse_index_map,			\
    Tensor *output_indices_t,						\
    Tensor *output_values_t,						\
    std::vector<Tindex> &scratch,					\
    const std::vector<Tindex> &offset,					\
    const Tindex dense_rows,						\
    const Tindex N,							\
    TTypes<bool>::Vec empty_row_indicator,				\
    const int rank);							\
 extern template struct ParallelSparseFillEmptyRows<GPUDevice, T, Tindex>;

#define DECLARE_GPU_SPEC_TINDEX_INT64(T) DECLARE_GPU_SPEC(T, int64)

TF_CALL_POD_TYPES(DECLARE_GPU_SPEC_TINDEX_INT64)
#undef DECLARE_GPU_SPEC_TINDEX_INT64
#undef DECLARE_GPU_SPEC
}  // end of namespace functor

#define REGISTER_GPU_KERNEL(T) REGISTER_KERNELS_TINDEX_INT64(GPU, T)

TF_CALL_POD_TYPES(REGISTER_GPU_KERNEL)
#undef REGISTER_GPU_KERNEL
#endif // GOOGLE_CUDA
#undef REGISTER_KERNELS_TINDEX_INT64
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
