#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_INNER_FLATTEN_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_INNER_FLATTEN_UTIL_H_

#include <memory>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/kernels/reshape_util.h"

namespace tensorflow {

void SparseInnerFlatten(OpKernelContext* context,
    const Tensor& indices_t, const Tensor& dense_shape_t,
    const Tensor& new_rank_t, Tensor& output_indices,
    Tensor& output_shape);

}  // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_SPARSE_INNER_FLATTEN_UTIL_H_
