#include <exception>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/fused_embedding/gpu/common.cu.h"
#include "tensorflow/core/kernels/fused_embedding/gpu/functions/kernels.cu.h"
#include "tensorflow/core/kernels/fused_embedding/gpu/functions/partition_select.cu.h"
#include "tensorflow/core/profiler/nvtx_utils.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "third_party/cub/device/device_radix_sort.cuh"
#include "third_party/cub/device/device_select.cuh"
#include "third_party/cub/iterator/constant_input_iterator.cuh"
#include "third_party/cub/thread/thread_operators.cuh"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

class PartitionWithPermutationGPU : public OpKernel {
 public:
  explicit PartitionWithPermutationGPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_axis", &partition_axis_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("partition_strategy", &partition_strategy_));
  }

  void Compute(OpKernelContext* ctx) override {
    using namespace fused_embedding;
    auto device = ctx->eigen_device<GPUDevice>();

    nvtx::ScopedRangeIfEnabled<nvtx::CoreDomain> nvtx_range(this);

    OpOutputList partitioned_values;
    OP_REQUIRES_OK(ctx,
                   ctx->output_list("partitioned_values", &partitioned_values));
    OpOutputList partition_permutations;
    OP_REQUIRES_OK(ctx, ctx->output_list("partition_permutations",
                                         &partition_permutations));

    Tensor const* input = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input));
    const int64 input_size = input->NumElements();

    if (num_partitions_ == 1) {
      // Simply copy
      Tensor* partitioned_value;
      partitioned_values.allocate(0, TensorShape({input_size}),
                                  &partitioned_value);
      cudaMemcpyAsync(data_p_with_type<int64_t>(partitioned_values[0]),
                      data_p_with_type<int64_t>(input),
                      sizeof(int64_t) * input_size, cudaMemcpyDeviceToDevice,
                      device.stream());
      // Need to allocate_output here
      Tensor* partitioned_permutation;
      partition_permutations.allocate(0, TensorShape({input_size}),
                                      &partitioned_permutation);
      RangeInit(device, input_size,
                data_p_with_type<int64_t>(partitioned_permutation));
    } else {
      Tensor partition_permute_init;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DT_INT64, TensorShape({input_size}),
                                        &partition_permute_init));

      RangeInit(device, input_size,
                data_p_with_type<int64_t>(partition_permute_init));
      if (partition_strategy_ == "div") {
        OpInputList partition_shapes;
        OP_REQUIRES_OK(ctx,
                       ctx->input_list("partition_shapes", &partition_shapes));

        std::vector<int64_t> accu_div_host;
        accu_div_host.resize(num_partitions_);
        for (int i = 0; i < partition_shapes.size(); i++) {
          OP_REQUIRES(ctx, partition_shapes[i].dims() <= 2,
                      errors::InvalidArgument(
                          "input partition_shapes must all less than rank 2"));
          const int64_t div = partition_shapes[i].flat<int64>().data()[0];
          accu_div_host[i] = i == 0 ? div : accu_div_host[i - 1] + div;
        }

        Tensor accu_div;
        OP_REQUIRES_OK(
            ctx,
            ctx->allocate_temp(
                DT_INT64, TensorShape({static_cast<int64_t>(num_partitions_)}),
                &accu_div));
        cudaMemcpyAsync(data_p_with_type<int64>(accu_div), accu_div_host.data(),
                        num_partitions_ * sizeof(int64_t),
                        cudaMemcpyHostToDevice, device.stream());

        PartitionSelectDiv<int64, int64>(
            ctx, input, partition_permute_init, accu_div, num_partitions_,
            partitioned_values, partition_permutations);
      }
    }
  }

 private:
  int num_partitions_;
  int partition_axis_;
  std::string partition_strategy_;
  cudaEvent_t memcpy_event_;
};

REGISTER_KERNEL_BUILDER(Name("PartitionWithPermutation").Device(DEVICE_GPU),
                        PartitionWithPermutationGPU);
}  // namespace tensorflow

#endif  // GOOGLE_CUDA