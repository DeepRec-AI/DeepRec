#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/sparse_inner_flatten_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

class SparseInnerFlattenOp : public OpKernel {
 public:
  explicit SparseInnerFlattenOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_indices;
    OP_REQUIRES_OK(context, context->input("indices", &input_indices));

    const Tensor* input_dense_shape;
    OP_REQUIRES_OK(context, context->input("dense_shape", &input_dense_shape));
    OP_REQUIRES(context, IsLegacyVector(input_dense_shape->shape()),
        errors::InvalidArgument("input_dense_shape must be 1-D, not ",
          input_dense_shape->shape().DebugString()));
    
    const Tensor* input_new_rank_t;
    OP_REQUIRES_OK(context, context->input("new_rank", &input_new_rank_t));
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(input_new_rank_t->shape()),
        errors::InvalidArgument("new_rank must be a scalar. Found: ",
          input_new_rank_t->shape().DebugString()));

    Tensor output_indices(input_indices->dtype());
    Tensor output_shape(input_dense_shape->dtype());
    SparseInnerFlatten(context, *input_indices, *input_dense_shape,
        *input_new_rank_t, output_indices, output_shape); 
    context->set_output(0, output_indices);
    context->set_output(1, output_shape);
  }
};

REGISTER_KERNEL_BUILDER(Name("SparseInnerFlatten")
                            .Device(DEVICE_CPU)
                            .HostMemory("indices")
                            .HostMemory("dense_shape")
                            .HostMemory("new_rank"),
                        SparseInnerFlattenOp);
}  // namespace tensorflow
