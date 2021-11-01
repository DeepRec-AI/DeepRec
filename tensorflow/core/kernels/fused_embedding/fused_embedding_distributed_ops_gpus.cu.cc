#include <exception>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "third_party/cub/device/device_radix_sort.cuh"
#include "third_party/cub/device/device_reduce.cuh"
#include "third_party/cub/iterator/constant_input_iterator.cuh"
#include "third_party/cub/thread/thread_operators.cuh"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

namespace {

__global__ void CalcElementsOffsetPerPartition(
    const int64_t* values_sorted, int64_t* partition_sizes,
    int64_t* elements_offset_per_partition, int nnz) {
  // dichotomy
  const int64_t target = partition_sizes[blockIdx.x];
  const int pos = nnz / 2;
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
    const int64_t* values_sorted, const int64_t* sub_partitioned_values,
    const int64_t partition_start_base, const int64_t partition_size) {
  const int t_offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_offset < partition_size) {
    int64_t value = values_sorted[t_offset];
    // rebase value to it's corresponding sub partition
    value = value - partition_start_base;
    sub_partitioned_values[t_offset] = value;
  }
}

class FusedEmbeddingDistributedSparsePreLookUp : public OpKernel {
 public:
  explicit FusedEmbeddingDistributedSparsePreLookUp(OpKernelConstruction* ctx)
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
    OP_REQUIRES(ctx, num_partitions_ == partition_shapes.size(),
                errors::InvalidArgument(
                    "num_partitions does not match partition_shapes size"));

    partitioned_sizes_.clear();
    for (const Tensor& shape : partition_shapes) {
      OP_REQUIRES(ctx, shape.dims <= 2,
                  errors::InvalidArgument(
                      "input partition_shapes must all less than rank 2"));
      const int size = shape->flat<int>().data()[partition_axis_];
      partitioned_sizes_.push_back(size);
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
        reinterpret_cast<int64_t*>(values_tensor->flat<int64>().data()),
        reinterpret_cast<int64_t*>(values_sorted->flat<int64>().data()),
        reinterpret_cast<int4*>(indices_tensor->flat<int64>().data()),
        reinterpret_cast<int4*>(indices_sorted->flat<int64>().data()), int(nnz),
        0, sizeof(int64) * 8, stream);

    ctx->allocate_temp(DT_INT8,
                       TensorShape({static_cast<int64>(temp_storage_bytes)}),
                       &cub_temp_storage);

    cub::DeviceRadixSort::SortPairs(
        cub_temp_storage.flat<int8>().data(), temp_storage_bytes,
        reinterpret_cast<int64_t*>(values_tensor->flat<int64>().data()),
        reinterpret_cast<int64_t*>(values_sorted->flat<int64>().data()),
        reinterpret_cast<int4*>(indices_tensor->flat<int64>().data()),
        reinterpret_cast<int4*>(indices_sorted->flat<int64>().data()), int(nnz),
        0, sizeof(int64) * 8, stream);

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
      TF_CHECK_OK(GpuLaunchKernel(
          CalcElementsOffsetPerPartition, blocks, threads, 0, stream,
          reinterpret_cast<int64_t*>(values_sorted->flat<int64>().data()),
          reinterpret_cast<int64_t*>(partition_sizes->flat<int64>().data()),
          reinterpret_cast<int64_t*>(
              elements_offset_per_partition->flat<int64>().data()),
          int(nnz)));
    }
    elements_offset_per_partition_.clear();
    elements_offset_per_partition_.resize(num_partitions_);
    stream_executor::DeviceMemoryBase elements_offset_per_partition_wrapped(
        elements_offset_per_partition->flat<int64>().data(), num_partitions_);

    stream->ThenMemcpy(elements_offset_per_partition_.data(),
                       elements_offset_per_partition_wrapped,
                       num_partitions_ * sizeof(int64_t));
    stream->BlockHostUntilDone();

    // 4. set output
    OpOutputList partitioned_values;
    OP_REQUIRES_OK(ctx,
                   ctx->output_list("partitioned_values", partitioned_values));
    OpOutputList partitioned_indices;
    OP_REQUIRES_OK(
        ctx, ctx->output_list("partitioned_indices", partitioned_indices));

    int64_t partition_start_base = 0;
    for (int i = 0; i < num_partitions_; i++) {
      int64_t size = elements_offset_per_partition_[i] - partition_start_base;

      Tensor* sub_partitioned_values;
      OP_REQUIRES_OK(
          ctx, partitioned_values->allocate(
                   i, TensorShape({static_cast<int64>(partitioned_sizes_[i])}),
                   &sub_partitioned_values));

      Tensor* sub_partitioned_indices;
      OP_REQUIRES_OK(
          ctx,
          partitioned_indices->allocate(
              i, TensorShape({static_cast<int64>(partitioned_sizes_[i]), 2}),
              &sub_partitioned_indices));

      if (size > 0) {
        // some partition does not have any element that falls in it
        const int threads = 1024;
        int blocks =
            size % threads == 0 ? (size / threads) : (size / threads + 1);
        TF_CHECK_OK(GpuLaunchKernel(
            GatherAndConvertToSubPartition, blocks, threads, 0, stream,
            reinterpret_cast<int64_t*>(values_sorted->flat<int64>().data()),
            reinterpret_cast<int64_t*>(
                sub_partitioned_values->flat<int64>().data()),
            partition_start_base, size));

        stream_executor::DeviceMemoryBase sub_indices_sorted_wrapped(
            indices_sorted.flat<int64>().data() + 2 * partition_start_base,
            size * 2 * sizeof(int64_t));

        stream_executor::DeviceMemoryBase sub_indices_out_wrapped(
            sub_partitioned_indices.flat<int64>().data(),
            size * 2 * sizeof(int64_t));

        stream->ThenMemcpy(sub_indices_out_wrapped, sub_indices_sorted_wrapped,
                           size * 2 * sizeof(int64_t));
      }
      partition_start_base = elements_offset_per_partition_[i];
    }
  }

 private:
  int num_partitions_;
  int partition_axis_;
  std::vector<int64_t> elements_offset_per_partition_;
}

REGISTER_KERNEL_BUILDER(Name("FusedEmbeddingDistributedSparsePreLookUp")
                            .Device(DEVICE_GPU)
                            .HostMemory("partition_shapes"),
                        FusedEmbeddingDistributedSparsePreLookUp);

}  // namespace