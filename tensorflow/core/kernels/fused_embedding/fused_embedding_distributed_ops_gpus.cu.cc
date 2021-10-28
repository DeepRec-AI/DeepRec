#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <exception>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

namespace {

__global__ void CalcPartitionsOutputsDims(const int64_t* sp_values,
                                          const int* partitioned_offsets,
                                          const int* outputs_dims,
                                          const int num_partitions,
                                          const int64_t nnz) {
  const int t_offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_offset < nnz) {
    const int value = static_cast<int>(sp_values[t_offset]);
    for (int i = 0; i < num_partitions; i++) {
      if (value < partitioned_offsets[i]) {
        atomicAdd(outputs_dims[i]);
        return;
      }
    }
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
    Tensor const* values_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_values", &values_tensor));
    const int64 nnz = values_tensor->shape().dim_size(0);

    OpInputList partition_shapes;
    OP_REQUIRES_OK(ctx, ctx->input_list("partition_shapes", &partition_shapes));
    partitioned_offsets_.clear();
    OP_REQUIRES(ctx, num_partitions_ == partition_shapes.size(),
                errors::InvalidArgument(
                    "num_partitions does not match partition_shapes size"));
    for (const Tensor& shape : partition_shapes) {
      OP_REQUIRES(ctx, shape.dims <= 2,
                  errors::InvalidArgument(
                      "input partition_shapes must all less than rank 2"));
      const int dim = shape->flat<int>().data()[partition_axis_];
      if (partitioned_offsets_.size() == 0) {
        partitioned_offsets_.push_back(dim);
      } else {
        partitioned_offsets_.push_back(dim + partitioned_offsets_.back());
      }
    }
    Tensor partitioned_offsets;
    ctx->allocate_temp(DT_INT32, TensorShape{num_partitions_},
                       &partitioned_offsets);
    stream_executor::DeviceMemoryBase partitioned_offsets_wrapper(
        partitioned_offsets.flat<int>().data(),
        partitioned_offsets.TotalBytes());

    stream->ThenMemcpy((void*)partitioned_offsets_.data(),
                       partitioned_offsets_wrapper,
                       partitioned_offsets.TotalBytes());

    Tensor outputs_dims;
    ctx->allocate_temp(DT_INT32, TensorShape{num_partitions_}, &outputs_dims);
    stream_executor::DeviceMemoryBase outputs_dims_wrapper(
        outputs_dims.flat<int>().data(), outputs_dims.TotalBytes());
    stream->ThenMemset32(outputs_dims_wrapper, 0x0, outputs_dims.TotalBytes());

    {
      const int threads = 1024;
      int blocks = nnz / threads;
      blocks = nnz % threads == 0 ? blocks : blocks + 1;
      TF_CHECK_OK(GpuLaunchKernel(
          CalcPartitionsOutputsDims, blocks, threads, 0, stream,
          reinterpret_cast<const int64_t*>(values_tensor->flat<int64>().data()),
          partitioned_offsets.flat<int>.data(), outputs_dims.flat<int>.data(),
          num_partitions_, nnz));
    }

    OpOutputList outputs;
    for (int i = 0; i < num_partitions_; i++) {
      Tensor* out;
      OP_REQUIRES_OK(ctx, )
    }
  }

 private:
  int num_partitions_;
  int partition_axis_;
  std::vector<int> partitioned_offsets_;
}

REGISTER_KERNEL_BUILDER(Name("FusedEmbeddingDistributedSparsePreLookUp")
                            .Device(DEVICE_GPU)
                            .HostMemory("partition_shapes"),
                        FusedEmbeddingDistributedSparsePreLookUp);

}  // namespace