#include <exception>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/fused_embedding/fused_embedding_common.cu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "third_party/cub/device/device_radix_sort.cuh"
#include "third_party/cub/device/device_reduce.cuh"
#include "third_party/cub/iterator/constant_input_iterator.cuh"
#include "third_party/cub/thread/thread_operators.cuh"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

namespace {

struct IndicePair {
  int64_t row_in_batch;
  int64_t entry_in_column;
};

__global__ void CalcElementsOffsetPerPartition(
    const int64_t* values_sorted, int64_t* partition_sizes,
    int64_t* elements_offset_per_partition, int nnz) {
  // dichotomy
  const int64_t target = partition_sizes[blockIdx.x];
  int pos = nnz / 2;
  while (1) {
    if (pos == 0) {
      pos = -1;
      break;
    } else if (pos == nnz - 1) {
      break;
    }
    int64_t value = values_sorted[pos];
    int64_t value_plus_1 = values_sorted[pos + 1];
    if (value < target && value_plus_1 >= target) {
      break;
    }
    if (value < target) {
      pos = (pos + nnz) / 2;
    } else {
      pos = pos / 2;
    }
  }
  elements_offset_per_partition[blockIdx.x] = int64_t(pos + 1);
}

__global__ void GatherAndConvertToSubPartition(
    const int64_t* values_sorted, int64_t* sub_partitioned_values,
    const int64_t partition_start_base, const int64_t partition_size) {
  const int t_offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_offset < partition_size) {
    int64_t value = values_sorted[t_offset];
    // rebase value to it's corresponding sub partition
    value = value - partition_start_base;
    sub_partitioned_values[t_offset] = value;
  }
}

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
__global__ void ApplyCombiner(float* emb_vectors, const int* feature_nums) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  const int feature_num = feature_nums[blockIdx.x];
  const float emb_element = emb_vectors[offset];
  emb_vectors[offset] = Combine<combiner>(emb_element, feature_num);
}

template <Combiner combiner>
__global__ void DistributeGradToShard(const float* top_grad,
                                      const float* emb_shard,
                                      const int64_t* partitioned_indice,
                                      const int* feature_nums,
                                      float* grad_shard, const int64_t sub_nnz,
                                      const int64_t emb_vec_size,
                                      const float max_norm) {
  __shared__ int64_t row_in_batch_shared[1];
  __shared__ int feature_num_shared[1];
  __shared__ float l2_sum[1];
  if (threadIdx.x == 0) {
    row_in_batch_shared[0] = partitioned_indice[2 * blockIdx.x];
  }
  __syncthreads();
  const int64_t row_in_batch = row_in_batch_shared[0];
  float grad = top_grad[row_in_batch * emb_vec_size + threadIdx.x];
  if (threadIdx.x == 0) {
    feature_num_shared[0] = feature_nums[row_in_batch];
  }
  __syncthreads();
  const int feature_num = feature_num_shared[0];
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
    grad *= max_norm / l2_norm;
  }
  grad_shard[blockIdx.x * emb_vec_size + threadIdx.x] = grad;
}

}  // namespace

class FusedEmbeddingDistributedSparsePreLookUpGPU : public OpKernel {
 public:
  explicit FusedEmbeddingDistributedSparsePreLookUpGPU(
      OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_axis", &partition_axis_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto stream = ctx->eigen_device<GPUDevice>().stream();

    // 1. bind inputs
    Tensor const* values_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_values", &values_tensor));
    const int64 nnz = values_tensor->shape().dim_size(0);

    Tensor const* indices_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_indices", &indices_tensor));

    OpInputList partition_shapes;
    OP_REQUIRES_OK(ctx, ctx->input_list("partition_shapes", &partition_shapes));

    for (const Tensor& shape : partition_shapes) {
      OP_REQUIRES(ctx, shape.dims() <= 2,
                  errors::InvalidArgument(
                      "input partition_shapes must all less than rank 2"));
    }

    // 2. sort the sp_values and indices
    Tensor values_sorted;
    ctx->allocate_temp(DT_INT64, values_tensor->shape(), &values_sorted);
    Tensor indices_sorted;
    ctx->allocate_temp(DT_INT64, indices_tensor->shape(), &indices_sorted);

    Tensor cub_temp_storage;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        NULL, temp_storage_bytes,
        reinterpret_cast<const int64_t*>(values_tensor->flat<int64>().data()),
        reinterpret_cast<int64_t*>(values_sorted.flat<int64>().data()),
        reinterpret_cast<const IndicePair*>(
            indices_tensor->flat<int64>().data()),
        reinterpret_cast<IndicePair*>(indices_sorted.flat<int64>().data()),
        int(nnz), 0, sizeof(int64) * 8, stream);

    ctx->allocate_temp(DT_INT8,
                       TensorShape({static_cast<int64>(temp_storage_bytes)}),
                       &cub_temp_storage);

    cub::DeviceRadixSort::SortPairs(
        cub_temp_storage.flat<int8>().data(), temp_storage_bytes,
        reinterpret_cast<const int64_t*>(values_tensor->flat<int64>().data()),
        reinterpret_cast<int64_t*>(values_sorted.flat<int64>().data()),
        reinterpret_cast<const IndicePair*>(
            indices_tensor->flat<int64>().data()),
        reinterpret_cast<IndicePair*>(indices_sorted.flat<int64>().data()),
        int(nnz), 0, sizeof(int64) * 8, stream);

    // 3. calculate how many elements for each partition
    Tensor partition_sizes;
    ctx->allocate_temp(DT_INT64,
                       TensorShape({static_cast<int64>(num_partitions_)}),
                       &partition_sizes);
    Tensor elements_offset_per_partition;
    ctx->allocate_temp(DT_INT64,
                       TensorShape({static_cast<int64>(num_partitions_)}),
                       &elements_offset_per_partition);
    {
      const int blocks = num_partitions_;
      const int threads = 1;
      CalcElementsOffsetPerPartition<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<int64_t*>(values_sorted.flat<int64>().data()),
          reinterpret_cast<int64_t*>(partition_sizes.flat<int64>().data()),
          reinterpret_cast<int64_t*>(
              elements_offset_per_partition.flat<int64>().data()),
          int(nnz));
    }
    elements_offset_per_partition_.clear();
    elements_offset_per_partition_.resize(num_partitions_);
    // stream_executor::DeviceMemoryBase elements_offset_per_partition_wrapped(
    //     elements_offset_per_partition.flat<int64>().data(), num_partitions_);
    // stream->ThenMemcpy(elements_offset_per_partition_.data(),
    //                    elements_offset_per_partition_wrapped,
    //                    num_partitions_ * sizeof(int64_t));
    // stream->BlockHostUntilDone();

    cudaMemcpyAsync(elements_offset_per_partition_.data(),
                    elements_offset_per_partition.flat<int64>().data(),
                    num_partitions_ * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);

    // 4. set output
    OpOutputList partitioned_values;
    OP_REQUIRES_OK(ctx,
                   ctx->output_list("partitioned_values", &partitioned_values));
    OpOutputList partitioned_indices;
    OP_REQUIRES_OK(
        ctx, ctx->output_list("partitioned_indices", &partitioned_indices));

    int64_t partition_start_base = 0;
    for (int i = 0; i < num_partitions_; i++) {
      int64_t size = elements_offset_per_partition_[i] - partition_start_base;

      Tensor* sub_partitioned_values;
      OP_REQUIRES_OK(ctx, partitioned_values.allocate(
                              i, TensorShape({static_cast<int64>(size)}),
                              &sub_partitioned_values));

      Tensor* sub_partitioned_indices;
      OP_REQUIRES_OK(ctx, partitioned_indices.allocate(
                              i, TensorShape({static_cast<int64>(size), 2}),
                              &sub_partitioned_indices));

      if (size > 0) {
        // some partition does not have any element that falls in it
        const int threads = 1024;
        int blocks =
            size % threads == 0 ? (size / threads) : (size / threads + 1);
        GatherAndConvertToSubPartition<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<int64_t*>(values_sorted.flat<int64>().data()),
            reinterpret_cast<int64_t*>(
                sub_partitioned_values->flat<int64>().data()),
            partition_start_base, size);

        // stream_executor::DeviceMemoryBase sub_indices_sorted_wrapped(
        //     reinterpret_cast<IndicePair*>(indices_sorted.flat<int64>().data())
        //     +
        //         partition_start_base,
        //     size * sizeof(IndicePair));
        // stream_executor::DeviceMemoryBase sub_indices_out_wrapped(
        //     reinterpret_cast<IndicePair*>(
        //         sub_partitioned_indices.flat<int64>().data()),
        //     size * sizeof(IndicePair));
        // stream->ThenMemcpy(&sub_indices_out_wrapped,
        //                    sub_indices_sorted_wrapped,
        //                    size * 2 * sizeof(int64_t));
        cudaMemcpyAsync(
            sub_partitioned_indices->flat<int64>().data(),
            indices_sorted.flat<int64>().data() + 2 * partition_start_base,
            size * 2 * sizeof(int64_t), cudaMemcpyDeviceToDevice);
      }
      partition_start_base = elements_offset_per_partition_[i];
    }
  }

 private:
  int num_partitions_;
  int partition_axis_;
  std::vector<int64_t> elements_offset_per_partition_;
};

REGISTER_KERNEL_BUILDER(Name("FusedEmbeddingDistributedSparsePreLookUp")
                            .Device(DEVICE_GPU)
                            .HostMemory("partition_shapes"),
                        FusedEmbeddingDistributedSparsePreLookUpGPU);

class FusedEmbeddingDistributedSparsePostLookUpGPU : public OpKernel {
 public:
  explicit FusedEmbeddingDistributedSparsePostLookUpGPU(
      OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_axis", &partition_axis_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_norm", &max_norm_));
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

    const size_t emb_vec_size = emb_shards[0].shape().dim_size(1);
    auto dense_shape = dense_shape_tensor->flat<int64>().data();
    const size_t batch_size = dense_shape[0];

    // 1. sum up emb values from different entries and dump into output
    Tensor* emb_vectors_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0,
                            TensorShape({static_cast<long long>(batch_size),
                                         static_cast<long long>(emb_vec_size)}),
                            &emb_vectors_tensor));
    // stream_executor::DeviceMemoryBase emb_vectors_wrapper(
    //    emb_vectors_tensor.flat<float>().data(),
    //    emb_vectors_tensor->NumElements() * sizeof(float));
    // stream->ThenMemZero(&emb_vectors_wrapper,
    //                    emb_vectors_tensor->NumElements() * sizeof(float));

    cudaMemsetAsync(emb_vectors_tensor->flat<float>().data(), 0x0,
                    sizeof(float) * emb_vectors_tensor->NumElements(), stream);

    Tensor* feature_nums;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(
                       1, TensorShape({static_cast<long long>(batch_size)}),
                       &feature_nums));
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
      }
    }

    // 2. do max_norm and combiner
    {
      const int blocks = batch_size;
      const int threads = emb_vec_size;
      if (combiner_ == "sqrtn") {
        ApplyCombiner<Sqrtn><<<blocks, threads, 0, stream>>>(
            emb_vectors_tensor->flat<float>().data(),
            feature_nums->flat<int>().data());
      } else if (combiner_ == "mean") {
        ApplyCombiner<Mean><<<blocks, threads, 0, stream>>>(
            emb_vectors_tensor->flat<float>().data(),
            feature_nums->flat<int>().data());
      } else {
        ApplyCombiner<Sum><<<blocks, threads, 0, stream>>>(
            emb_vectors_tensor->flat<float>().data(),
            feature_nums->flat<int>().data());
      }
    }
  }

 private:
  int num_partitions_;
  int partition_axis_;
  std::string combiner_;
  float max_norm_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedEmbeddingDistributedSparsePostLookUp").Device(DEVICE_GPU),
    FusedEmbeddingDistributedSparsePostLookUpGPU);

class FusedEmbeddingDistributedSparsePostLookUpGradGPU : public OpKernel {
 public:
  explicit FusedEmbeddingDistributedSparsePostLookUpGradGPU(
      OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_axis", &partition_axis_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_norm", &max_norm_));
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

    OpOutputList grad_shards;
    OP_REQUIRES_OK(ctx, ctx->output_list("grad_shards", &grad_shards));
    const size_t emb_vec_size = grad_shards[0]->shape().dim_size(1);

    for (int i = 0; i < num_partitions_; i++) {
      auto partitioned_indice = partitioned_indices[i];
      const size_t sub_nnz = partitioned_indice.shape().dim_size(0);

      Tensor* grad_shard;
      OP_REQUIRES_OK(ctx,
                     grad_shards.allocate(
                         i,
                         TensorShape({static_cast<long long>(sub_nnz),
                                      static_cast<long long>(emb_vec_size)}),
                         &grad_shard));

      {
        const int blocks = sub_nnz;
        const int threads = emb_vec_size;
        if (combiner_ == "sqrtn") {
          DistributeGradToShard<Sqrtn><<<blocks, threads, 0, stream>>>(
              top_grad_tensor->flat<float>().data(),
              emb_shards[i].flat<float>().data(),
              reinterpret_cast<int64_t*>(
                  partitioned_indice.flat<int64>().data()),
              feature_nums->flat<int>().data(),
              grad_shard->flat<float>().data(), sub_nnz, emb_vec_size,
              max_norm_);
        } else if (combiner_ == "mean") {
          DistributeGradToShard<Mean><<<blocks, threads, 0, stream>>>(
              top_grad_tensor->flat<float>().data(),
              emb_shards[i].flat<float>().data(),
              reinterpret_cast<int64_t*>(
                  partitioned_indice.flat<int64>().data()),
              feature_nums->flat<int>().data(),
              grad_shard->flat<float>().data(), sub_nnz, emb_vec_size,
              max_norm_);
        } else {
          DistributeGradToShard<Sum><<<blocks, threads, 0, stream>>>(
              top_grad_tensor->flat<float>().data(),
              emb_shards[i].flat<float>().data(),
              reinterpret_cast<int64_t*>(
                  partitioned_indice.flat<int64>().data()),
              feature_nums->flat<int>().data(),
              grad_shard->flat<float>().data(), sub_nnz, emb_vec_size,
              max_norm_);
        }
      }
    }
  }

 private:
  int num_partitions_;
  int partition_axis_;
  std::string combiner_;
  float max_norm_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedEmbeddingDistributedSparsePostLookUpGrad").Device(DEVICE_GPU),
    FusedEmbeddingDistributedSparsePostLookUpGradGPU);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA