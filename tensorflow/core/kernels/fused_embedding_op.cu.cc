#include <exception>

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

class SparseFusedEmbeddingGPUOp : public OpKernel {
 public:
  explicit SparseFusedEmbeddingGPUOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor const* values_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("values", &values_tensor));
    Tensor const* indices_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("indices", &indices_tensor));
    Tensor const* dense_shape_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("dense_shape", &dense_shape_tensor));

    Tensor const* emb_variable = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("emb_variable", &emb_variable));

    auto dense_shape_flat = dense_shape_tensor->flat<int64_t>();
    int64_t batch_size = dense_shape_flat[0];
    int64 emb_vec_size_dim = emb_variable.shape().dim_size(1);

    TensorShape emb_vector_tensor_shape;

    emb_vector_tensor_shape = TensorShape(
        std::vector<tensorflow::int64>({batch_size, emb_vec_size_dim}));
    Tensor* emb_vector_tensor = nullptr;
    // allocate output
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, emb_vector_tensor_shape,
                                             &emb_vector_tensor));
  }

 private:
  TensorShape emb_vector_tensor_shape_;
}

REGISTER_KERNEL_BUILDER(
    Name("SparseFusedEmbedding").Device(DEVICE_GPU).HostMemory("dense_shape"),
    SparseFusedEmbeddingGPUOp);

}  // namespace tensorflow