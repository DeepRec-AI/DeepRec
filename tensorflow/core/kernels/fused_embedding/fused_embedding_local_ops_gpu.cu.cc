#include <exception>
#include <string>

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/fused_embedding/fused_embedding_common.cu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

namespace {

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
                                float* embedding_vector, const float max_norm,
                                const int emb_vec_size,
                                const int64_t batch_size, const int64_t nnz) {
  __shared__ float l2_sum[1];

  int value_offset = values_offset[blockIdx.x];
  int feature_num;
  if (blockIdx.x == int(batch_size) - 1) {
    feature_num = int(nnz) - value_offset;
  } else {
    feature_num = values_offset[blockIdx.x + 1] - value_offset;
  }
  float out = 0.0f;
  for (int i = 0; i < feature_num; i++) {
    float emb_element =
        emb_variable[int(values[value_offset + i]) * emb_vec_size +
                     threadIdx.x];
    if (max_norm >= 0.0f) {
      // calc l2 norm of this emb row(per block) and compare with max_norm.
      // if greater than max_norm, then clip every element with factor
      // max_norm / l2norm
      if (threadIdx.x == 0) {
        l2_sum[0] = 0.0f;
      }
      __syncthreads();
      atomicAdd(l2_sum, emb_element * emb_element);
      __syncthreads();
      float l2_norm = sqrtf(l2_sum[0]);
      if (l2_norm > max_norm) {
        emb_element *= max_norm / l2_norm;
      }
    }
    out += emb_element;
  }

  // combine
  out = Combine<combiner, int>(out, feature_num);

  // store the embedding vector
  embedding_vector[blockIdx.x * emb_vec_size + threadIdx.x] = out;
}

template <Combiner combiner>
__global__ void DoEmbeddingGrad(const float* top_grad,
                                const float* emb_variable,
                                const int64_t* values, const int* values_offset,
                                float* grad_values, const float max_norm,
                                const int emb_vec_size,
                                const int64_t batch_size, const int64_t nnz) {
  __shared__ float l2_sum[1];
  const int value_offset = values_offset[blockIdx.x];
  int feature_num;
  if (blockIdx.x == int(batch_size) - 1) {
    feature_num = int(nnz) - value_offset;
  } else {
    feature_num = values_offset[blockIdx.x + 1] - value_offset;
  }
  float grad = top_grad[blockIdx.x * emb_vec_size + threadIdx.x];
  grad = CombineGrad<combiner>(grad, feature_num);
  for (int i = 0; i < feature_num; i++) {
    float grad_i = grad;
    if (max_norm > 0.0f) {
      float emb_element =
          emb_variable[int(values[value_offset + i]) * emb_vec_size +
                       threadIdx.x];
      if (threadIdx.x == 0) {
        l2_sum[0] = 0.0f;
      }
      __syncthreads();
      atomicAdd(l2_sum, emb_element * emb_element);
      __syncthreads();
      float l2_norm = sqrtf(l2_sum[0]);
      if (l2_norm > max_norm) {
        grad_i *= max_norm / l2_norm;
      }
    }
    grad_values[(value_offset + i) * emb_vec_size + threadIdx.x] = grad_i;
  }
}

}  // namespace

class FusedEmbeddingLocalSparseLookUpGPU : public OpKernel {
 public:
  explicit FusedEmbeddingLocalSparseLookUpGPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_norm", &max_norm_));
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

    TensorShape emb_vectors_tensor_shape;

    emb_vectors_tensor_shape = TensorShape(
        std::vector<int64>({static_cast<long long>(batch_size), emb_vec_size}));
    Tensor* emb_vectors_tensor = nullptr;
    // allocate output
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, emb_vectors_tensor_shape,
                                             &emb_vectors_tensor));

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
      int blocks = nnz % threads == 0 ? (nnz / threads) : (nnz / threads + 1);

      // calculate values offset
      CalcPerElementRowInBatchValuesOffset<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const int64_t*>(
              indices_tensor->flat<int64>().data()),
          values_offset_tensor->flat<int>().data(), nnz);
    }
    {
      const int blocks = int(batch_size);
      const int threads = int(emb_vec_size);
      if (combiner_ == "sqrtn") {
        EmbeddingLookUp<Sqrtn><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float*>(
                emb_variable_tensor->flat<float>().data()),
            reinterpret_cast<const int64_t*>(
                values_tensor->flat<int64>().data()),
            values_offset_tensor->flat<int>().data(),
            reinterpret_cast<float*>(emb_vectors_tensor->flat<float>().data()),
            max_norm_, int(emb_vec_size), batch_size, nnz);
      } else if (combiner_ == "mean") {
        EmbeddingLookUp<Mean><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float*>(
                emb_variable_tensor->flat<float>().data()),
            reinterpret_cast<const int64_t*>(
                values_tensor->flat<int64>().data()),
            values_offset_tensor->flat<int>().data(),
            reinterpret_cast<float*>(emb_vectors_tensor->flat<float>().data()),
            max_norm_, int(emb_vec_size), batch_size, nnz);
      } else {
        EmbeddingLookUp<Sum><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float*>(
                emb_variable_tensor->flat<float>().data()),
            reinterpret_cast<const int64_t*>(
                values_tensor->flat<int64>().data()),
            values_offset_tensor->flat<int>().data(),
            reinterpret_cast<float*>(emb_vectors_tensor->flat<float>().data()),
            max_norm_, int(emb_vec_size), batch_size, nnz);
      }
    }
  }

 private:
  std::string combiner_;
  float max_norm_;
};

REGISTER_KERNEL_BUILDER(Name("FusedEmbeddingLocalSparseLookUp")
                            .Device(DEVICE_GPU)
                            .HostMemory("sp_dense_shape"),
                        FusedEmbeddingLocalSparseLookUpGPU);

class FusedEmbeddingLocalSparseLookUpGradGPU : public OpKernel {
 public:
  explicit FusedEmbeddingLocalSparseLookUpGradGPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_norm", &max_norm_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto stream = ctx->eigen_device<GPUDevice>().stream();

    Tensor const* top_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("top_grad", &top_grad_tensor));

    Tensor const* emb_variable_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("emb_variable", &emb_variable_tensor));
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

    {
      const int blocks = int(batch_size);
      const int threads = int(emb_vec_size);

      if (combiner_ == "sqrtn") {
        DoEmbeddingGrad<Sqrtn><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float*>(
                top_grad_tensor->flat<float>().data()),
            reinterpret_cast<const float*>(
                emb_variable_tensor->flat<float>().data()),
            reinterpret_cast<const int64_t*>(
                values_tensor->flat<int64>().data()),
            values_offset_tensor->flat<int>().data(),
            reinterpret_cast<float*>(
                grad_emb_weight_sp_values_tensor->flat<float>().data()),
            max_norm_, emb_vec_size, batch_size, nnz);
      } else if (combiner_ == "mean") {
        DoEmbeddingGrad<Mean><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float*>(
                top_grad_tensor->flat<float>().data()),
            reinterpret_cast<const float*>(
                emb_variable_tensor->flat<float>().data()),
            reinterpret_cast<const int64_t*>(
                values_tensor->flat<int64>().data()),
            values_offset_tensor->flat<int>().data(),
            reinterpret_cast<float*>(
                grad_emb_weight_sp_values_tensor->flat<float>().data()),
            max_norm_, emb_vec_size, batch_size, nnz);
      } else {
        DoEmbeddingGrad<Sum><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float*>(
                top_grad_tensor->flat<float>().data()),
            reinterpret_cast<const float*>(
                emb_variable_tensor->flat<float>().data()),
            reinterpret_cast<const int64_t*>(
                values_tensor->flat<int64>().data()),
            values_offset_tensor->flat<int>().data(),
            reinterpret_cast<float*>(
                grad_emb_weight_sp_values_tensor->flat<float>().data()),
            max_norm_, emb_vec_size, batch_size, nnz);
      }
    }
  }

 private:
  float max_norm_;
  std::string combiner_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedEmbeddingLocalSparseLookUpGrad").Device(DEVICE_GPU),
    FusedEmbeddingLocalSparseLookUpGradGPU);

}  // namespace tensorflow
#endif  // GOOGLE_CUDA