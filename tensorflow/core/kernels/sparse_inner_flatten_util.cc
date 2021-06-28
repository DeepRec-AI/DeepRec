#include "tensorflow/core/kernels/sparse_inner_flatten_util.h"

namespace tensorflow {
void SparseInnerFlatten(OpKernelContext* context,
                        const Tensor& input_indices_t,
                        const Tensor& input_dense_shape_t,
                        const Tensor& new_rank_t,
                        Tensor& output_indices,
                        Tensor& output_shape) {
  auto new_rank = new_rank_t.scalar<int64>()();
  const auto dims = input_dense_shape_t.NumElements();
  Tensor new_dense_shape_t;
  if (new_rank < dims) {
    OP_REQUIRES_OK(context,
        context->allocate_temp(input_dense_shape_t.dtype(),
          TensorShape({new_rank}), &new_dense_shape_t));

    const auto dense_shape = input_dense_shape_t.vec<int64>();
    auto new_dense_shape = new_dense_shape_t.vec<int64>();
    for (auto i = 0; i < new_rank - 1; ++i) {
      new_dense_shape(i) = dense_shape(i);
    }

    int64 last_dim = 1;
    for (auto i = new_rank - 1; i < dims; ++i) {
      auto val = dense_shape(i);
      if (val != -1) {
        last_dim *= val;
      }
    }
    new_dense_shape(new_rank - 1) = last_dim;
  } else if (new_rank == dims) {
    new_dense_shape_t = input_dense_shape_t;
  } else {
    LOG(FATAL) << "SparseInnerFlatten's Inputs has rank("
               << dims << ") less than new_rank("
               << new_rank << ")";
  }

  const int64 nnz = input_indices_t.shape().dim_size(0);
  const int64 output_rank = new_dense_shape_t.NumElements();

  OP_REQUIRES_OK(context, context->allocate_temp(input_indices_t.dtype(),
        TensorShape({nnz, output_rank}),
        &output_indices));

  OP_REQUIRES_OK(context, context->allocate_temp(input_dense_shape_t.dtype(),
        TensorShape({output_rank}),
        &output_shape));

  Reshape(context, input_indices_t, input_dense_shape_t, new_dense_shape_t,
      &output_indices, &output_shape);
}
}  // namespace tensorflow
