#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <exception>
#include <string>

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

namespace {

enum Combiner { Mean, Sum, Sqrt };

template <Combiner combiner>
__forceinline__ __device__ float Combine(const float* emb_variable,
                                         const int64_t* values,
                                         const int value_offset,
                                         const int feature_num,
                                         const int emb_vec_size);

template <>
__forceinline__ __device__ float Combine<Sqrt>(const float* emb_variable,
                                               const int64_t* values,
                                               const int value_offset,
                                               const int feature_num,
                                               const int emb_vec_size) {
  float out = 0.0f;
  for (int i = 0; i < feature_num; i++) {
    int64_t value_index = values[value_offset + i];
    out += emb_variable[int(value_index) * emb_vec_size + threadIdx.x];
  }
  return out / sqrtf(feature_num);
}

template <Combiner combiner>
__forceinline__ __device__ float CombineGrad(const float* top_grad,
                                             const int64_t element_row,
                                             const int feature_num,
                                             const int emb_vec_size);

template <>
__forceinline__ __device__ float CombineGrad<Sqrt>(const float* top_grad,
                                                   const int64_t element_row,
                                                   const int feature_num,
                                                   const int emb_vec_size) {
  const float grad = top_grad[int(element_row) * emb_vec_size + threadIdx.x];
  return grad / sqrtf(feature_num);
}

__global__ void SetToIntMaxSTG128(int* values_offset, const int batch_size) {
  const int thread_offset = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  const int int_max = 0x7fffffff;
  if (thread_offset + 4 < batch_size) {
    int4 four = make_int4(int_max, int_max, int_max, int_max);
    *((int4*)(values_offset + thread_offset)) = four;
  } else if (thread_offset < batch_size) {
    for (int i = thread_offset; i < batch_size; i++) {
      values_offset[i] = int_max;
    }
  }
}

__global__ void CalcPerElementRowInBatchValuesOffset(const int64_t* indices,
                                                     int* values_offset,
                                                     const int64_t nnz) {
  const int thread_offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_offset < int(nnz)) {
    const int64_t element_row = indices[2 * thread_offset];
    atomicMin(values_offset + int(element_row), thread_offset);
  }
}

template <Combiner combiner>
__global__ void EmbeddingLookUp(const float* emb_variable,
                                const int64_t* values, const int* values_offset,
                                float* embedding_vector, const int emb_vec_size,
                                const int64_t batch_size, const int64_t nnz) {
  if (blockIdx.x < int(batch_size) && threadIdx.x < emb_vec_size) {
    int value_offset = values_offset[blockIdx.x];
    int feature_num;
    if (blockIdx.x == int(batch_size) - 1) {
      feature_num = int(nnz) - value_offset;
    } else {
      feature_num = values_offset[blockIdx.x + 1] - value_offset;
    }
    // combine
    const float out = Combine<combiner>(emb_variable, values, value_offset,
                                        feature_num, emb_vec_size);

    // store the embedding vector
    embedding_vector[blockIdx.x * emb_vec_size + threadIdx.x] = out;
  }
}

template <Combiner combiner>
__global__ void DoEmbeddingGrad(const float* top_grad, const int64_t* values,
                                const int* values_offset, float* grad_values,
                                int64_t* grad_indices, const int emb_vec_size,
                                const int64_t batch_size, const int64_t nnz) {
  if (blockIdx.x < int(batch_size) && threadIdx.x < emb_vec_size) {
    const int value_offset = values_offset[blockIdx.x];
    int feature_num;
    if (blockIdx.x == int(batch_size) - 1) {
      feature_num = int(nnz) - value_offset;
    } else {
      feature_num = values_offset[blockIdx.x + 1] - value_offset;
    }
    float grad = CombineGrad<combiner>(top_grad, int64_t(blockIdx.x),
                                       feature_num, emb_vec_size);
    for (int i = 0; i < feature_num; i++) {
      grad_values[(value_offset + i) * emb_vec_size + threadIdx.x] = grad;
      if (threadIdx.x == 0) {
        const int64_t index = values[value_offset + i];
        grad_indices[value_offset + i] = index;
      }
    }
  }
}

}  // namespace

class FusedEmbeddingSparseLookUpGPUOp : public OpKernel {
 public:
  explicit FusedEmbeddingSparseLookUpGPUOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiner", &combiner_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto stream = ctx->eigen_device<GPUDevice>().stream();

    Tensor const* values_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_values", &values_tensor));
    Tensor const* indices_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_indices", &indices_tensor));
    Tensor const* dense_shape_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_dense_shape", &dense_shape_tensor));
    Tensor const* emb_variable_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("emb_variable", &emb_variable_tensor));

    auto dense_shape = dense_shape_tensor->flat<int64>().data();
    const size_t batch_size = dense_shape[0];
    const int64 nnz = indices_tensor->shape().dim_size(0);
    const int64 emb_vec_size = emb_variable_tensor->shape().dim_size(1);

    TensorShape emb_vector_tensor_shape;

    emb_vector_tensor_shape = TensorShape(
        std::vector<int64>({static_cast<long long>(batch_size), emb_vec_size}));
    Tensor* emb_vector_tensor = nullptr;
    // allocate output
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, emb_vector_tensor_shape,
                                             &emb_vector_tensor));

    // allocate offset tensor
    TensorShape values_offset_tensor_shape =
        TensorShape(std::vector<int64>({static_cast<long long>(batch_size)}));

    Tensor* values_offset_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, values_offset_tensor_shape,
                                             &values_offset_tensor));

    {
      const int threads = 1024;
      int blocks = batch_size / threads;
      blocks = batch_size % threads == 0 ? blocks : blocks + 1;
      SetToIntMaxSTG128<<<blocks, threads, 0, stream>>>(
          values_offset_tensor->flat<int>().data(), int(batch_size));
    }
    {
      const int threads = 1024;
      int blocks = nnz / threads;
      blocks = nnz % threads == 0 ? blocks : blocks + 1;

      // calculate values offset
      CalcPerElementRowInBatchValuesOffset<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const int64_t*>(
              indices_tensor->flat<int64>().data()),
          values_offset_tensor->flat<int>().data(), nnz);
    }
    {
      const int blocks = int(batch_size);
      const int threads = int(emb_vec_size);
      if (combiner_ == "sqrt") {
        EmbeddingLookUp<Sqrt><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float*>(
                emb_variable_tensor->flat<float>().data()),
            reinterpret_cast<const int64_t*>(
                values_tensor->flat<int64>().data()),
            values_offset_tensor->flat<int>().data(),
            reinterpret_cast<float*>(emb_vector_tensor->flat<float>().data()),
            int(emb_vec_size), batch_size, nnz);
      }
    }
  }

 private:
  std::string combiner_;
};

REGISTER_KERNEL_BUILDER(Name("FusedEmbeddingSparseLookUp")
                            .Device(DEVICE_GPU)
                            .HostMemory("sp_dense_shape"),
                        FusedEmbeddingSparseLookUpGPUOp);

class FusedEmbeddingSparseLookUpGradOp : public OpKernel {
 public:
  explicit FusedEmbeddingSparseLookUpGradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiner", &combiner_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto stream = ctx->eigen_device<GPUDevice>().stream();

    Tensor const* top_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("top_grad", &top_grad_tensor));
    Tensor const* values_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_values", &values_tensor));
    Tensor const* values_offset_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_values_offset", &values_offset_tensor));

    const int64 emb_vec_size = top_grad_tensor->shape().dim_size(1);
    const int64 batch_size = top_grad_tensor->shape().dim_size(0);
    const int64 nnz = values_tensor->shape().dim_size(0);

    Tensor* grad_emb_weight_sp_values_tensor;
    TensorShape grad_emb_weight_sp_values_tensor_shape =
        TensorShape(std::vector<int64>({nnz, emb_vec_size}));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, grad_emb_weight_sp_values_tensor_shape,
                                  &grad_emb_weight_sp_values_tensor));

    Tensor* grad_emb_weight_sp_indice_tensor;
    TensorShape grad_emb_weight_sp_indice_tensor_shape =
        TensorShape(std::vector<int64>({nnz}));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, grad_emb_weight_sp_indice_tensor_shape,
                                  &grad_emb_weight_sp_indice_tensor));

    {
      const int blocks = int(batch_size);
      const int threads = int(emb_vec_size);

      if (combiner_ == "sqrt") {
        DoEmbeddingGrad<Sqrt><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float*>(
                top_grad_tensor->flat<float>().data()),
            reinterpret_cast<const int64_t*>(
                values_tensor->flat<int64>().data()),
            values_offset_tensor->flat<int>().data(),
            reinterpret_cast<float*>(
                grad_emb_weight_sp_values_tensor->flat<float>().data()),
            reinterpret_cast<int64_t*>(
                grad_emb_weight_sp_indice_tensor->flat<int64>().data()),
            emb_vec_size, batch_size, nnz);
      }
    }
  }

 private:
  std::string combiner_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedEmbeddingSparseLookUpGrad").Device(DEVICE_GPU),
    FusedEmbeddingSparseLookUpGradOp);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM