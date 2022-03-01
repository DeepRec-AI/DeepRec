#include <exception>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/fused_embedding/fused_embedding.cu.h"
#include "tensorflow/core/kernels/fused_embedding/gpu_functions/kernels.cu.h"
#include "tensorflow/core/profiler/nvtx_utils.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "third_party/cub/thread/thread_operators.cuh"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

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
    using namespace fused_embedding;
    auto device = ctx->eigen_device<GPUDevice>();

    OpInputList emb_shards;
    OP_REQUIRES_OK(ctx, ctx->input_list("emb_shards", &emb_shards));

    OpInputList partition_permutations;
    OP_REQUIRES_OK(ctx, ctx->input_list("partition_permutations",
                                        &partition_permutations));

    Tensor const* dense_shape_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_dense_shape", &dense_shape_tensor));

    Tensor const* row_empty_and_invalid_flags = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("row_empty_and_invalid_flags",
                                   &row_empty_and_invalid_flags));

    Tensor const* indices_before_unique = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->input("indices_before_unique", &indices_before_unique));

    Tensor const* unique_counts = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("unique_counts", &unique_counts));

    Tensor const* idx_of_input_to_unique = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->input("idx_of_input_to_unique", &idx_of_input_to_unique));

    Tensor const* unique_offsets = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("unique_offsets", &unique_offsets));

    nvtx::ScopedRangeIfEnabled<nvtx::CoreDomain> nvtx_range(this);

    const int64_t emb_vec_size = emb_shards[0].shape().dim_size(1);
    const int64_t batch_size = dense_shape_tensor->flat<int64>().data()[0];

    // = 1. sum up emb values from differententries and dump into output = //
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
                    sizeof(float) * emb_vectors_tensor->NumElements(),
                    device.stream());

    Tensor* feature_nums;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, TensorShape({batch_size}), &feature_nums));
    // stream_executor::DeviceMemoryBase feature_nums_wrapper(
    //    feature_nums.flat<int>().data(),
    //    feature_nums.NumElements() * sizeof(int));
    // stream->ThenMemZero(&feature_nums_wrapper,
    //                    feature_nums.NumElements() * sizeof(int));
    cudaMemsetAsync(feature_nums->flat<int>().data(), 0x0,
                    sizeof(int) * feature_nums->NumElements(), device.stream());

    for (int i = 0; i < num_partitions_; i++) {
      const size_t shard_len = emb_shards[i].shape().dim_size(0);
      OP_REQUIRES(ctx,
                  shard_len == partition_permutations[i].shape().dim_size(0),
                  errors::InvalidArgument(
                      "emb_shard and partition_permutations doesn't "
                      "have the same length"));
      SumUpEmbeddingShard(
          device, shard_len, data_p_with_type<const float>(emb_shards[i]),
          data_p_with_type<const int64_t>(partition_permutations[i]),
          data_p_with_type<const int64_t>(indices_before_unique),
          data_p_with_type<const int64_t>(unique_counts),
          data_p_with_type<const int64_t>(idx_of_input_to_unique),
          data_p_with_type<const int64_t>(unique_offsets), max_norm_,
          emb_vec_size, data_p_with_type<float>(emb_vectors_tensor),
          data_p_with_type<int>(feature_nums));
      CK_CUDA_THROW_(cudaGetLastError());
    }

    const bool set_empty_row_zero = default_id_ < 0;
    // ================================================================ //

    // ========================= 2. combiner ========================== //
    if (combiner_ == "sqrtn") {
      ApplyCombiner<Sqrtn>(
          device, batch_size, emb_vec_size,
          data_p_with_type<const int>(row_empty_and_invalid_flags),
          set_empty_row_zero, data_p_with_type<int>(feature_nums),
          data_p_with_type<float>(emb_vectors_tensor));
    } else if (combiner_ == "mean") {
      ApplyCombiner<Mean>(
          device, batch_size, emb_vec_size,
          data_p_with_type<const int>(row_empty_and_invalid_flags),
          set_empty_row_zero, data_p_with_type<int>(feature_nums),
          data_p_with_type<float>(emb_vectors_tensor));
    } else {
      ApplyCombiner<Sum>(
          device, batch_size, emb_vec_size,
          data_p_with_type<const int>(row_empty_and_invalid_flags),
          set_empty_row_zero, data_p_with_type<int>(feature_nums),
          data_p_with_type<float>(emb_vectors_tensor));
    }
    CK_CUDA_THROW_(cudaGetLastError());
    // ================================================================ //
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
    using namespace fused_embedding;
    auto device = ctx->eigen_device<GPUDevice>();

    Tensor const* top_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("top_grad", &top_grad_tensor));

    OpInputList emb_shards;
    OP_REQUIRES_OK(ctx, ctx->input_list("emb_shards", &emb_shards));

    OpInputList partition_permutations;
    OP_REQUIRES_OK(ctx, ctx->input_list("partition_permutations",
                                        &partition_permutations));

    Tensor const* feature_nums = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("feature_nums", &feature_nums));

    Tensor const* indices_before_unique = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->input("indices_before_unique", &indices_before_unique));

    Tensor const* unique_counts = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("unique_counts", &unique_counts));

    Tensor const* idx_of_input_to_unique = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->input("idx_of_input_to_unique", &idx_of_input_to_unique));

    Tensor const* unique_offsets = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("unique_offsets", &unique_offsets));

    Tensor const* row_empty_and_invalid_flags = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("row_empty_and_invalid_flags",
                                   &row_empty_and_invalid_flags));

    OpOutputList grad_shards;
    OP_REQUIRES_OK(ctx, ctx->output_list("grad_shards", &grad_shards));

    nvtx::ScopedRangeIfEnabled<nvtx::CoreDomain> nvtx_range(this);

    const int64_t batch_size = top_grad_tensor->shape().dim_size(0);
    const int64_t emb_vec_size = emb_shards[0].shape().dim_size(1);

    const bool set_empty_row_zero = default_id_ < 0;

    for (int i = 0; i < num_partitions_; i++) {
      const int64_t shard_len = partition_permutations[i].shape().dim_size(0);

      Tensor* grad_shard;
      OP_REQUIRES_OK(
          ctx, grad_shards.allocate(i, TensorShape({shard_len, emb_vec_size}),
                                    &grad_shard));

      if (combiner_ == "sqrtn") {
        DistributeGradToShard<Sqrtn>(
            device, data_p_with_type<const float>(top_grad_tensor),
            data_p_with_type<const float>(emb_shards[i]),
            data_p_with_type<const int64_t>(partition_permutations[i]),
            data_p_with_type<const int64_t>(indices_before_unique),
            data_p_with_type<const int64_t>(unique_counts),
            data_p_with_type<const int64_t>(idx_of_input_to_unique),
            data_p_with_type<const int64_t>(unique_offsets), shard_len,
            emb_vec_size, max_norm_, set_empty_row_zero,
            data_p_with_type<const int>(feature_nums),
            data_p_with_type<const int>(row_empty_and_invalid_flags),
            data_p_with_type<float>(grad_shard));
      } else if (combiner_ == "mean") {
        DistributeGradToShard<Mean>(
            device, data_p_with_type<const float>(top_grad_tensor),
            data_p_with_type<const float>(emb_shards[i]),
            data_p_with_type<const int64_t>(partition_permutations[i]),
            data_p_with_type<const int64_t>(indices_before_unique),
            data_p_with_type<const int64_t>(unique_counts),
            data_p_with_type<const int64_t>(idx_of_input_to_unique),
            data_p_with_type<const int64_t>(unique_offsets), shard_len,
            emb_vec_size, max_norm_, set_empty_row_zero,
            data_p_with_type<const int>(feature_nums),
            data_p_with_type<const int>(row_empty_and_invalid_flags),
            data_p_with_type<float>(grad_shard));
      } else {
        DistributeGradToShard<Sum>(
            device, data_p_with_type<const float>(top_grad_tensor),
            data_p_with_type<const float>(emb_shards[i]),
            data_p_with_type<const int64_t>(partition_permutations[i]),
            data_p_with_type<const int64_t>(indices_before_unique),
            data_p_with_type<const int64_t>(unique_counts),
            data_p_with_type<const int64_t>(idx_of_input_to_unique),
            data_p_with_type<const int64_t>(unique_offsets), shard_len,
            emb_vec_size, max_norm_, set_empty_row_zero,
            data_p_with_type<const int>(feature_nums),
            data_p_with_type<const int>(row_empty_and_invalid_flags),
            data_p_with_type<float>(grad_shard));
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

REGISTER_KERNEL_BUILDER(
    Name("FusedEmbeddingSparsePostLookUpGrad").Device(DEVICE_GPU),
    FusedEmbeddingSparsePostLookUpGradGPU);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA