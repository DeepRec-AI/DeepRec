#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <exception>

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

namespace {

__global__ void CalcPerElementInBatchValuesOffset(const int64_t* indices,
                                                  int* value_offset,
                                                  const int64_t indices_num) {
  const int thread_offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_offset < indices_num) {
    const int element_in_batch = int(indices[2 * thread_offset]);
    atomicMax(value_offset, element_in_batch);
  }
}

enum Combiner { Mean, Sum, Sqrt };

template <Combiner combiner>
__forceinline__ __device__ float Combine(const int64_t* values,
                                         const float* emb_variable,
                                         const int value_offset,
                                         const int feature_num,
                                         const int embedding_vec_size);

template <>
__forceinline__ __device__ float Combine<Sqrt>(const int64_t* values,
                                               const float* emb_variable,
                                               const int value_offset,
                                               const int feature_num,
                                               const int embedding_vec_size) {
  float out = 0.0f;
  for (int i = 0; i < feature_num; i++) {
    int64_t value_index = values[value_offset + i];
    out += emb_variable[value_index * embedding_vec_size + threadIdx.x];
  }
  return out / sqrtf(feature_num);
}

template <Combiner combiner>
__global__ void DoEmbedding(const int64_t* values, const float* emb_variable,
                            const int* values_offset, float* embedding_vector,
                            const int embedding_vec_size,
                            const int64_t batch_size) {
  if (blockIdx.x < batch_size && threadIdx.x < embedding_vec_size) {
    int value_offset = values_offset[blockIdx.x];
    int feature_num = values_offset[blockIdx.x + 1] - value_offset;

    // combine
    const float out = Combine<combiner>(values, emb_variable, value_offset,
                                        feature_num, embedding_vec_size);

    // store the embedding vector
    embedding_vector[blockIdx.x * embedding_vec_size + threadIdx.x] = out;
  }
}

}  // namespace

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

    Tensor const* emb_variable_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("emb_variable", &emb_variable_tensor));

    auto dense_shape = dense_shape_tensor->flat<int64>().data();
    const size_t batch_size = dense_shape[0];
    const int indices_num = indices_tensor->shape().dim_size(0);
    int64 emb_vec_size_dim = emb_variable_tensor->shape().dim_size(1);

    TensorShape emb_vector_tensor_shape;

    emb_vector_tensor_shape = TensorShape(std::vector<int64>(
        {static_cast<long long>(batch_size), emb_vec_size_dim}));
    Tensor* emb_vector_tensor = nullptr;
    // allocate output
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, emb_vector_tensor_shape,
                                             &emb_vector_tensor));

    // allocate offset temp tensor
    TensorShape values_offset_tensor_shape;
    values_offset_tensor_shape = TensorShape(std::vector<int64>({batch_size}));

    Tensor* values_offset_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, values_offset_tensor_shape,
                                           values_offset_tensor));

    // blocks and threads
    {
      const int threads = 1024;
      int blocks = indices_num / threads;
      blocks = indices_num % threads == 0 ? blocks : blocks + 1;

      // calculate values offset
      CalcPerElementInBatchValuesOffset<<<
          blocks, threads, 0, ctx->eigen_device<GPUDevice>().stream()>>>(
          reinterpret_cast<const int64_t*>(
              indices_tensor->flat<int64>().data()),
          values_offset_tensor->flat<int32>().data(), indices_num);
    }

    {
      const int blocks = int(batch_size);
      const int threads = int(emb_vec_size_dim);
      DoEmbedding<Sqrt>
          <<<blocks, threads, 0, ctx->eigen_device<GPUDevice>().stream()>>>(
              reinterpret_cast<const int64_t*>(
                  values_tensor->flat<int64>().data()),
              reinterpret_cast<const float*>(
                  emb_variable_tensor->flat<float>().data()),
              reinterpret_cast<const int*>(
                  values_offset_tensor->flat<int32>().data()),
              reinterpret_cast<float*>(emb_vector_tensor->flat<float>().data()),
              int(emb_vec_size_dim), batch_size);
    }
  }

 private:
  TensorShape emb_vector_tensor_shape_;
};

REGISTER_KERNEL_BUILDER(
    Name("SparseFusedEmbedding").Device(DEVICE_GPU).HostMemory("dense_shape"),
    SparseFusedEmbeddingGPUOp);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM