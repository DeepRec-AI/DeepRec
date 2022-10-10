#ifndef TENSORFLOW_KERNELS_SPARSE_FILL_EMPTY_ROWS_UTIL_H_
#define TENSORFLOW_KERNELS_SPARSE_FILL_EMPTY_ROWS_UTIL_H_

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/typed_allocator.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace functor {
template <typename Device, typename T, typename Tindex>
struct ParallelSparseFillEmptyRows {
  void operator()(const Device& d,
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
		  const int rank);
  
};
}  // end namespace functor

// template<typename T>
// void SparseFillEmptyRows(OpKernelContext* context,
//     const Tensor* indices_t, const Tensor* values_t,
//     const Tensor* dense_shape_t, const Tensor* default_value_t,
//     Tensor* empty_row_indicator_t, Tensor* reverse_index_map_t,
//     Tensor* output_indices_t, Tensor* output_values_t) {
//   const CPUDevice& d = context->eigen_device<CPUDevice>();
//   const T& default_value = default_value_t->scalar<T>()();
//   const auto indices = indices_t->matrix<int64>();
//   const auto values = values_t->vec<T>();
//   const auto dense_shape = dense_shape_t->vec<int64>();

//   const int64 N = indices_t->shape().dim_size(0);
//   const int64 dense_rows = dense_shape(0);
//   OP_REQUIRES_OK(context, context->allocate_temp(DT_BOOL, TensorShape({dense_rows}),
//         empty_row_indicator_t));
//   auto empty_row_indicator = empty_row_indicator_t->vec<bool>();
//   OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({N}),
//         reverse_index_map_t));
//   auto reverse_index_map = reverse_index_map_t->vec<int64>();

//   int rank = indices_t->shape().dim_size(1);

//   if (dense_rows == 0) {
//     OP_REQUIRES(
//         context, N == 0,
//         errors::InvalidArgument("Received SparseTensor with dense_shape[0] = "
//           "0 but indices.shape[0] = ",
//           N));
//     TensorShape output_indices_shape({0, rank});
//     OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, output_indices_shape,
//           output_indices_t));
//     OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
//           TensorShape({0}), output_values_t));
//     // Exit early, nothing more to do.
//     return;
//   }

//   Tensor scratch_t;
//   OP_REQUIRES_OK(context,
//       context->allocate_temp(DT_INT64, TensorShape({dense_rows}),
//         &scratch_t));
//   auto scratch = scratch_t.vec<int64>();
//   scratch.device(d) = scratch.constant(0);
//   for (int i = 0; i < N; ++i) {
//     const int64 row = indices(i, 0);
//     OP_REQUIRES(context, row >= 0 && row < dense_rows,
//         errors::InvalidArgument("indices(", i, ", 0) is invalid: ",
//           row, " >= ", dense_rows));
//     ++scratch(indices(i, 0));
//   }
//   for (int row = 0; row < dense_rows; ++row) {
//     // Scratch here describes the number of elements in this dense row
//     empty_row_indicator(row) = (scratch(row) == 0);
//     // In filled version, each row has at least one element.
//     scratch(row) = std::max(scratch(row), int64{1});
//     // Update scratch to represent the number of elements up to and
//     // including dense_row + 1:
//     //  scratch(0) == #{elements of row 0}
//     //  scratch(1) == #{elements of row 1} + #{elements of row 0}
//     //  ..
//     //  scratch(i) == starting index for elements in row i + 1.
//     if (row > 0) {
//       scratch(row) += scratch(row - 1);
//     }
//   }

//   const int64 N_full = scratch(dense_rows - 1);
//   TensorShape output_indices_shape({N_full, rank});

//   OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, output_indices_shape,
//         output_indices_t));
//   auto output_indices = output_indices_t->matrix<int64>();
//   output_indices.device(d) = output_indices.constant(0);

//   OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
//         TensorShape({N_full}), output_values_t));
//   auto output_values = output_values_t->vec<T>();
//   output_values.device(d) = output_values.constant(default_value);

//   Tensor filled_count_t;
//   OP_REQUIRES_OK(context,
//       context->allocate_temp(DT_INT64, TensorShape({dense_rows}),
//         &filled_count_t));
//   auto filled_count = filled_count_t.vec<int64>();
//   filled_count.device(d) = filled_count.constant(0);

//   // Fill in values for rows that are not missing
//   for (int64 i = 0; i < N; ++i) {
//     const int64 row = indices(i, 0);
//     int64& offset = filled_count(row);
//     const int64 output_i = ((row == 0) ? 0 : scratch(row - 1)) + offset;
//     offset++;  // Increment the filled count for this row.
//     std::copy_n(&indices(i, 0), rank, &output_indices(output_i, 0));
//     output_values(output_i) = values(i);
//     // We'll need this reverse index map to backprop correctly.
//     reverse_index_map(i) = output_i;
//   }

//   // Fill in values for rows that are missing
//   for (int64 row = 0; row < dense_rows; ++row) {
//     const int64 row_count = filled_count(row);
//     if (row_count == 0) {  // We haven't filled this row
//       const int64 starting_index = (row == 0) ? 0 : scratch(row - 1);
//       // Remaining index values were set to zero already.
//       // The value at this index was set to default_value already.
//       // Just need to set the row index in the right location.
//       output_indices(starting_index, 0) = row;
//     }
//   } 
// } 

// template<typename T>
// void ParallelSparseFillEmptyRows(OpKernelContext* context,
//     const Tensor* indices_t, const Tensor* values_t,
//     const Tensor* dense_shape_t, const Tensor* default_value_t,
//     Tensor* empty_row_indicator_t, Tensor* reverse_index_map_t,
//     Tensor* output_indices_t, Tensor* output_values_t) {
//   const CPUDevice& d = context->eigen_device<CPUDevice>();
//   const T& default_value = default_value_t->scalar<T>()();
//   const auto indices = indices_t->matrix<int64>();
//   const auto values = values_t->vec<T>();
//   const auto dense_shape = dense_shape_t->vec<int64>();

//   const int64 N = indices_t->shape().dim_size(0);
//   const int64 dense_rows = dense_shape(0);
//   OP_REQUIRES_OK(context, context->allocate_temp(DT_BOOL, TensorShape({dense_rows}),
//         empty_row_indicator_t));
//   auto empty_row_indicator = empty_row_indicator_t->vec<bool>();
//   OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({N}),
//          reverse_index_map_t));
//   auto reverse_index_map = reverse_index_map_t->vec<int64>();

//   int rank = indices_t->shape().dim_size(1);

//   if (dense_rows == 0) {
//     OP_REQUIRES(
//         context, N == 0,
//         errors::InvalidArgument("Received SparseTensor with dense_shape[0] = "
//           "0 but indices.shape[0] = ",
//           N));
//     TensorShape output_indices_shape({0, rank});
//     OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, output_indices_shape,
//           output_indices_t));
//     OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
//           TensorShape({0}), output_values_t));
//     // Exit early, nothing more to do.
//     return;
//   }

//   Tensor scratch_t;
//   OP_REQUIRES_OK(context,
//       context->allocate_temp(DT_INT64, TensorShape({dense_rows, 2}),
//         &scratch_t));
//   auto scratch = scratch_t.matrix<int64>();
//   scratch.device(d) = scratch.constant(0);

//   Tensor offset_t;
//   OP_REQUIRES_OK(context,
//       context->allocate_temp(DT_INT64, TensorShape({N}),
//         &offset_t));
//   auto offset = offset_t.vec<int64>();
//   offset.device(d) = offset.constant(0);

//   for (int i = 0; i < N; ++i) {
//     const int64 row = indices(i, 0);
//     OP_REQUIRES(context, row >= 0 && row < dense_rows,
//         errors::InvalidArgument("indices(", i, ", 0) is invalid: ",
//           row, " >= ", dense_rows));
//     offset(i) = scratch(row, 0)++;
//   }

//   for (int row = 0; row < dense_rows; ++row) {
//     // Scratch here describes the number of elements in this dense row
//     empty_row_indicator(row) = (scratch(row, 0) == 0);
//     // In filled version, each row has at least one element.
//     scratch(row, 1) = std::max(scratch(row, 0), int64{1});
//     // Update scratch to represent the number of elements up to and
//     // including dense_row + 1:
//     //  scratch(0) == #{elements of row 0}
//     //  scratch(1) == #{elements of row 1} + #{elements of row 0}
//     //  ..
//     //  scratch(i) == starting index for elements in row i + 1.
//     if (row > 0) {
//       scratch(row, 1) += scratch(row - 1, 1);
//     }
//   }

//   const int64 N_full = scratch(dense_rows - 1, 1);
//   TensorShape output_indices_shape({N_full, rank});

//   OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, output_indices_shape,
//         output_indices_t));
//   auto output_indices = output_indices_t->matrix<int64>();
//   output_indices.device(d) = output_indices.constant(0);

//   OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
//         TensorShape({N_full}), output_values_t));
//   auto output_values = output_values_t->vec<T>();
//   output_values.device(d) = output_values.constant(default_value);

// // Fill in values for rows that are not missing
//   auto RunTask = [&indices, &scratch, &offset, &output_indices,
//        &values, &output_values, &reverse_index_map, &rank]
//        (int64 start, int64 end) {
//      for (int64 i = start; i < end; ++i) {
//        const int64 row = indices(i, 0);
//        const int64 output_i = ((row == 0) ? 0 : scratch(row - 1, 1)) + offset(i);
//        std::copy_n(&indices(i, 0), rank, &output_indices(output_i, 0));
//        output_values(output_i) = values(i);
//        // We'll need this reverse index map to backprop correctly.
//        reverse_index_map(i) = output_i;
//      }
//   };

//   auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
//   const int64 element_cost = 10;
//   Shard(worker_threads->num_threads - 1,
//       worker_threads->workers, N, element_cost, RunTask);

//   // Fill in values for rows that are missing
//   for (int64 row = 0; row < dense_rows; ++row) {
//     //const int64 row_count = filled_count(row);
//     const int64 row_count = scratch(row, 0);
//     if (row_count == 0) {  // We haven't filled this row
//       const int64 starting_index = (row == 0) ? 0 : scratch(row - 1, 1);
//       // Remaining index values were set to zero already.
//       // The value at this index was set to default_value already.
//       // Just need to set the row index in the right location.
//       output_indices(starting_index, 0) = row;
//     }
//   } 
// }

}  // end of namespace tensorflow

#endif // end of TENSORFLOW_KERNELS_SPARSE_FILL_EMPTY_ROWS_UTIL_H_
