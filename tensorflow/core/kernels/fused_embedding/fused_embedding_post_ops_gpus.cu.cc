#include <exception>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/fused_embedding/fused_embedding_common.cu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "cub/thread/thread_operators.cuh"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

namespace {
__global__ void SumUpEmbeddingShard(const float* emb_shard,
                                    const int64_t* partitioned_indice,
                                    float* emb_vectors, int* feature_nums,
                                    const float max_norm,
                                    const int emb_vec_size) {
  __shared__ float l2_sum[1];

  const int64_t row_in_batch = partitioned_indice[2 * blockIdx.x];
  float emb_element = emb_shard[blockIdx.x * emb_vec_size + threadIdx.x];
  if (max_norm >= 0.0f) {
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

  atomicAdd(emb_vectors + row_in_batch * emb_vec_size + threadIdx.x,
            emb_element);

  if (threadIdx.x == 0) {
    atomicAdd(feature_nums + row_in_batch, 1);
  }
}

template <Combiner combiner>
__global__ void ApplyCombiner(float* emb_vectors, const int* row_emptiness_flag,
                              const bool set_empty_row_zero,
                              const int* feature_nums) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (set_empty_row_zero) {
    if (row_emptiness_flag[blockIdx.x]) {
      emb_vectors[offset] = 0.0f;
      return;
    }
  }
  const int feature_num = feature_nums[blockIdx.x];
  const float emb_element = emb_vectors[offset];
  emb_vectors[offset] = Combine<combiner>(emb_element, feature_num);
}

template <Combiner combiner>
__global__ void DistributeGradToShard(
    const float* top_grad, const float* emb_shard,
    const int64_t* partitioned_indice, const int* feature_nums,
    const int* row_emptiness_flag, const bool set_empty_row_zero,
    float* grad_shard, const int64_t sub_nnz, const int64_t emb_vec_size,
    const float max_norm) {
  __shared__ int64_t row_in_batch_shared[1];
  __shared__ int feature_num_shared[1];
  __shared__ float l2_sum[1];
  int64_t row_in_batch;
  if (threadIdx.x == 0) {
    row_in_batch = partitioned_indice[2 * blockIdx.x];
    row_in_batch_shared[0] = row_in_batch;
    feature_num_shared[0] = feature_nums[row_in_batch];
  }
  __syncthreads();
  row_in_batch = row_in_batch_shared[0];
  const int feature_num = feature_num_shared[0];
  if (set_empty_row_zero) {
    if (row_emptiness_flag[row_in_batch]) {
      grad_shard[blockIdx.x * emb_vec_size + threadIdx.x] = 0.0f;
      return;
    }
  }
  float grad = top_grad[row_in_batch * emb_vec_size + threadIdx.x];
  grad = CombineGrad<combiner>(grad, feature_num);
  if (max_norm >= 0.0f) {
    const float emb_element =
        emb_shard[blockIdx.x * emb_vec_size + threadIdx.x];
    if (threadIdx.x == 0) {
      l2_sum[0] = 0.0f;
    }
    __syncthreads();
    atomicAdd(l2_sum, emb_element * emb_element);
    __syncthreads();
    float l2_norm = sqrtf(l2_sum[0]);
    if (l2_norm > max_norm) {
      grad *= max_norm / l2_norm;
    }
  }
  grad_shard[blockIdx.x * emb_vec_size + threadIdx.x] = grad;
}
}  // namespace

class FusedEmbeddingSparsePostLookUpGPU : public OpKernel {
 public:
  explicit FusedEmbeddingSparsePostLookUpGPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_axis", &partition_axis_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_norm", &max_norm_));
    int temp_default_id;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_id", &temp_default_id));
    default_id_ = int64_t(temp_default_id);
  }

  void Compute(OpKernelContext* ctx) override {
    auto stream = ctx->eigen_device<GPUDevice>().stream();

    OpInputList emb_shards;
    OP_REQUIRES_OK(ctx, ctx->input_list("emb_shards", &emb_shards));

    OpInputList partitioned_indices;
    OP_REQUIRES_OK(
        ctx, ctx->input_list("partitioned_indices", &partitioned_indices));

    Tensor const* dense_shape_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_dense_shape", &dense_shape_tensor));

    Tensor const* row_empty_and_invalid_flags = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("row_empty_and_invalid_flags",
                                   &row_empty_and_invalid_flags));

    const int64_t emb_vec_size = emb_shards[0].shape().dim_size(1);
    const int64_t batch_size = dense_shape_tensor->flat<int64>().data()[0];

    // 1. sum up emb values from different entries and dump into output
    Tensor* emb_vectors_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({batch_size, emb_vec_size}),
                                  &emb_vectors_tensor));
    // stream_executor::DeviceMemoryBase emb_vectors_wrapper(
    //    emb_vectors_tensor.flat<float>().data(),
    //    emb_vectors_tensor->NumElements() * sizeof(float));
    // stream->ThenMemZero(&emb_vectors_wrapper,
    //                    emb_vectors_tensor->NumElements() * sizeof(float));

    cudaMemsetAsync(emb_vectors_tensor->flat<float>().data(), 0x0,
                    sizeof(float) * emb_vectors_tensor->NumElements(), stream);

    Tensor* feature_nums;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({batch_size}), &feature_nums));
    // stream_executor::DeviceMemoryBase feature_nums_wrapper(
    //    feature_nums.flat<int>().data(),
    //    feature_nums.NumElements() * sizeof(int));
    // stream->ThenMemZero(&feature_nums_wrapper,
    //                    feature_nums.NumElements() * sizeof(int));
    cudaMemsetAsync(feature_nums->flat<int>().data(), 0x0,
                    sizeof(int) * feature_nums->NumElements(), stream);

    for (int i = 0; i < num_partitions_; i++) {
      const size_t sub_nnz = emb_shards[i].shape().dim_size(0);
      OP_REQUIRES(
          ctx, sub_nnz == partitioned_indices[i].shape().dim_size(0),
          errors::InvalidArgument(
              "emb_shard and partitioned_indice dosn't have the same length"));

      {
        const int blocks = sub_nnz;
        const int threads = emb_vec_size;
        SumUpEmbeddingShard<<<blocks, threads, 0, stream>>>(
            emb_shards[i].flat<float>().data(),
            reinterpret_cast<const int64_t*>(
                partitioned_indices[i].flat<int64>().data()),
            emb_vectors_tensor->flat<float>().data(),
            feature_nums->flat<int>().data(), max_norm_, emb_vec_size);
        CK_CUDA_THROW_(cudaGetLastError());
      }
    }

    const bool set_empty_row_zero = default_id_ >= 0;
    // 2. combiner
    {
      const int blocks = batch_size;
      const int threads = emb_vec_size;
      if (combiner_ == "sqrtn") {
        ApplyCombiner<Sqrtn><<<blocks, threads, 0, stream>>>(
            emb_vectors_tensor->flat<float>().data(),
            row_empty_and_invalid_flags->flat<int>().data(), set_empty_row_zero,
            feature_nums->flat<int>().data());
      } else if (combiner_ == "mean") {
        ApplyCombiner<Mean><<<blocks, threads, 0, stream>>>(
            emb_vectors_tensor->flat<float>().data(),
            row_empty_and_invalid_flags->flat<int>().data(), set_empty_row_zero,
            feature_nums->flat<int>().data());
      } else {
        ApplyCombiner<Sum><<<blocks, threads, 0, stream>>>(
            emb_vectors_tensor->flat<float>().data(),
            row_empty_and_invalid_flags->flat<int>().data(), set_empty_row_zero,
            feature_nums->flat<int>().data());
      }
      CK_CUDA_THROW_(cudaGetLastError());
    }
  }

 private:
  int num_partitions_;
  int partition_axis_;
  std::string combiner_;
  float max_norm_;
  int64_t default_id_;
};

REGISTER_KERNEL_BUILDER(Name("FusedEmbeddingSparsePostLookUp")
                            .Device(DEVICE_GPU)
                            .HostMemory("sp_dense_shape"),
                        FusedEmbeddingSparsePostLookUpGPU);

class FusedEmbeddingSparsePostLookUpGradGPU : public OpKernel {
 public:
  explicit FusedEmbeddingSparsePostLookUpGradGPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_axis", &partition_axis_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_norm", &max_norm_));
    int temp_default_id;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_id", &temp_default_id));
    default_id_ = int64_t(temp_default_id);
  }

  void Compute(OpKernelContext* ctx) override {
    auto stream = ctx->eigen_device<GPUDevice>().stream();

    Tensor const* top_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("top_grad", &top_grad_tensor));

    OpInputList emb_shards;
    OP_REQUIRES_OK(ctx, ctx->input_list("emb_shards", &emb_shards));

    OpInputList partitioned_indices;
    OP_REQUIRES_OK(
        ctx, ctx->input_list("partitioned_indices", &partitioned_indices));

    Tensor const* feature_nums = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("feature_nums", &feature_nums));

    Tensor const* row_empty_and_invalid_flags = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("row_empty_and_invalid_flags",
                                   &row_empty_and_invalid_flags));

    OpOutputList grad_shards;
    OP_REQUIRES_OK(ctx, ctx->output_list("grad_shards", &grad_shards));

    const int64_t batch_size = top_grad_tensor->shape().dim_size(0);
    const int64_t emb_vec_size = emb_shards[0].shape().dim_size(1);

    const bool set_empty_row_zero = default_id_ >= 0;

    for (int i = 0; i < num_partitions_; i++) {
      const int64_t sub_nnz = partitioned_indices[i].shape().dim_size(0);

      Tensor* grad_shard;
      OP_REQUIRES_OK(
          ctx, grad_shards.allocate(i, TensorShape({sub_nnz, emb_vec_size}),
                                    &grad_shard));

      {
        const int blocks = sub_nnz;
        const int threads = emb_vec_size;
        if (combiner_ == "sqrtn") {
          DistributeGradToShard<Sqrtn><<<blocks, threads, 0, stream>>>(
              top_grad_tensor->flat<float>().data(),
              emb_shards[i].flat<float>().data(),
              reinterpret_cast<const int64_t*>(
                  partitioned_indices[i].flat<int64>().data()),
              feature_nums->flat<int>().data(),
              row_empty_and_invalid_flags->flat<int>().data(),
              set_empty_row_zero, grad_shard->flat<float>().data(), sub_nnz,
              emb_vec_size, max_norm_);
        } else if (combiner_ == "mean") {
          DistributeGradToShard<Mean><<<blocks, threads, 0, stream>>>(
              top_grad_tensor->flat<float>().data(),
              emb_shards[i].flat<float>().data(),
              reinterpret_cast<const int64_t*>(
                  partitioned_indices[i].flat<int64>().data()),
              feature_nums->flat<int>().data(),
              row_empty_and_invalid_flags->flat<int>().data(),
              set_empty_row_zero, grad_shard->flat<float>().data(), sub_nnz,
              emb_vec_size, max_norm_);
        } else {
          DistributeGradToShard<Sum><<<blocks, threads, 0, stream>>>(
              top_grad_tensor->flat<float>().data(),
              emb_shards[i].flat<float>().data(),
              reinterpret_cast<const int64_t*>(
                  partitioned_indices[i].flat<int64>().data()),
              feature_nums->flat<int>().data(),
              row_empty_and_invalid_flags->flat<int>().data(),
              set_empty_row_zero, grad_shard->flat<float>().data(), sub_nnz,
              emb_vec_size, max_norm_);
        }
        CK_CUDA_THROW_(cudaGetLastError());
      }
    }
  }

 private:
  int num_partitions_;
  int partition_axis_;
  std::string combiner_;
  float max_norm_;
  int64_t default_id_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedEmbeddingSparsePostLookUpGrad").Device(DEVICE_GPU),
    FusedEmbeddingSparsePostLookUpGradGPU);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
