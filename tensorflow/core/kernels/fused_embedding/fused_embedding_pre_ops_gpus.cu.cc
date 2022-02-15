#include <exception>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/fused_embedding/fused_embedding.cu.h"
#include "tensorflow/core/kernels/fused_embedding/gpu_functions/kernels.cu.h"
#include "tensorflow/core/kernels/fused_embedding/gpu_functions/partition_select.cu.h"
#include "tensorflow/core/kernels/fused_embedding/gpu_functions/unique.cu.h"
#include "tensorflow/core/profiler/nvtx_utils.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "third_party/cub/device/device_radix_sort.cuh"
#include "third_party/cub/device/device_select.cuh"
#include "third_party/cub/iterator/constant_input_iterator.cuh"
#include "third_party/cub/thread/thread_operators.cuh"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

class FusedEmbeddingSparsePreLookUpGPU : public OpKernel {
 public:
  explicit FusedEmbeddingSparsePreLookUpGPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_axis", &partition_axis_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_empty_row", &fill_empty_row_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("prune_invalid_id", &prune_invalid_id_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("partition_strategy", &partition_strategy_));
    int temp_default_id;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_id", &temp_default_id));
    default_id_ = int64_t(temp_default_id);
  }

  void Compute(OpKernelContext* ctx) override {
    using namespace fused_embedding;
    auto device = ctx->eigen_device<GPUDevice>();

    const int64_t default_id = default_id_ >= 0 ? default_id_ : 0;

    nvtx::ScopedRangeIfEnabled<nvtx::CoreDomain> nvtx_range(this);

    // ================ 1. bind inputs ================ //
    Tensor const* values_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_values", &values_tensor));
    const int64_t nnz = values_tensor->shape().dim_size(0);

    Tensor const* indices_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_indices", &indices_tensor));

    Tensor const* dense_shape = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_dense_shape", &dense_shape));
    const int64_t batch_size = dense_shape->flat<int64>().data()[0];

    cudaEvent_t memcpy_event;
    cudaEventCreateWithFlags(&memcpy_event, cudaEventDisableTiming);
    // =============================================== //

    // ========= 2. allocate cub tmp storage ========= //
    Tensor cub_temp_storage;
    size_t max_cub_bytes = 0;
    size_t temp_storage_bytes = 0;

    if (fill_empty_row_ || prune_invalid_id_) {
      cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, (int64_t*)nullptr,
                                 (int*)nullptr, (int64_t*)nullptr,
                                 (int*)nullptr, nnz, device.stream());

      max_cub_bytes = temp_storage_bytes > max_cub_bytes ? temp_storage_bytes
                                                         : max_cub_bytes;

      cub::DeviceSelect::Flagged((void*)nullptr, temp_storage_bytes,
                                 (IndicePair*)nullptr, (int*)nullptr,
                                 (IndicePair*)nullptr, (int*)nullptr, nnz,
                                 device.stream());

      max_cub_bytes = temp_storage_bytes > max_cub_bytes ? temp_storage_bytes
                                                         : max_cub_bytes;

      if (fill_empty_row_) {
        cub::DeviceSelect::Flagged((void*)nullptr, temp_storage_bytes,
                                   (IndicePair*)nullptr, (int*)nullptr,
                                   (IndicePair*)nullptr, (int*)nullptr,
                                   batch_size, device.stream());
        max_cub_bytes = temp_storage_bytes > max_cub_bytes ? temp_storage_bytes
                                                           : max_cub_bytes;
      }
    }

    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(
                 DT_INT8, TensorShape({static_cast<int64_t>(max_cub_bytes)}),
                 &cub_temp_storage));
    // =============================================== //

    // === 3. fill_empty_row, prune, if avaliable. === //
    Tensor values_extended;
    Tensor indices_extended;
    Tensor tmp_indices_buffer;
    Tensor* all_flags;
    Tensor selected_num_d;
    int new_nnz = nnz;

    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("row_empty_and_invalid_flags",
                                  TensorShape{batch_size + nnz}, &all_flags));

    if (fill_empty_row_ || prune_invalid_id_) {
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DT_INT64, TensorShape{nnz + batch_size},
                                        &values_extended));
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DT_INT64, TensorShape{2 * (nnz + batch_size)},
                                  &indices_extended));
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DT_INT64, TensorShape{2 * batch_size},
                                        &tmp_indices_buffer));
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DT_INT32, TensorShape{1}, &selected_num_d));

      InitFlagsToOneInt4(device, batch_size + nnz,
                         all_flags->flat<int>().data());
      CK_CUDA_THROW_(cudaGetLastError());

      // 3.1 set flags, init tmp_indices_buffer etc.
      if (fill_empty_row_) {
        FusedMultiFunctional(
            device, data_p_with_type<const IndicePair>(indices_tensor),
            data_p_with_type<const int64_t>(values_tensor), nnz, batch_size,
            prune_invalid_id_, default_id, data_p_with_type<int>(all_flags),
            data_p_with_type<int>(all_flags) + batch_size,
            data_p_with_type<IndicePair>(tmp_indices_buffer),
            data_p_with_type<int64_t>(values_extended));
        CK_CUDA_THROW_(cudaGetLastError());

      } else if (prune_invalid_id_) {
        DetectInvalid(device, data_p_with_type<const int64_t>(values_tensor),
                      nnz, data_p_with_type<int>(all_flags) + batch_size);
        CK_CUDA_THROW_(cudaGetLastError());
      }
      // 3.2 select copy valid id, select copy empty row indices

      cub::DeviceSelect::Flagged(
          data_p_with_type<int8>(cub_temp_storage), max_cub_bytes,
          data_p_with_type<const int64_t>(values_tensor),
          data_p_with_type<const int>(all_flags) + batch_size,
          data_p_with_type<int64_t>(values_extended),
          data_p_with_type<int>(selected_num_d), int(nnz), device.stream());
      CK_CUDA_THROW_(cudaGetLastError());

      cub::DeviceSelect::Flagged(
          data_p_with_type<int8>(cub_temp_storage), max_cub_bytes,
          data_p_with_type<const IndicePair>(indices_tensor),
          data_p_with_type<int>(all_flags) + batch_size,
          data_p_with_type<IndicePair>(indices_extended),
          data_p_with_type<int>(selected_num_d), nnz, device.stream());

      if (prune_invalid_id_) {
        int selected_num;
        cudaMemcpyAsync(&selected_num, data_p_with_type<int>(selected_num_d),
                        sizeof(int), cudaMemcpyDeviceToHost, device.stream());
        cudaEventRecord(memcpy_event, device.stream());
        cudaEventSynchronize(memcpy_event);
        new_nnz = selected_num;
      }

      if (fill_empty_row_) {
        cub::DeviceSelect::Flagged(
            data_p_with_type<int8>(cub_temp_storage), max_cub_bytes,
            data_p_with_type<const IndicePair>(tmp_indices_buffer),
            data_p_with_type<int>(all_flags),
            data_p_with_type<IndicePair>(indices_extended) + new_nnz,
            data_p_with_type<int>(selected_num_d), batch_size, device.stream());
        CK_CUDA_THROW_(cudaGetLastError());
        int selected_num;
        cudaMemcpyAsync(&selected_num, data_p_with_type<void>(selected_num_d),
                        sizeof(int), cudaMemcpyDeviceToHost, device.stream());
        cudaEventRecord(memcpy_event, device.stream());
        cudaEventSynchronize(memcpy_event);
        new_nnz += selected_num;
      }
    }
    // =============================================== //

    // 3.5 set the correct pointer
    const int64_t* values_in =
        (fill_empty_row_ || prune_invalid_id_)
            ? data_p_with_type<const int64_t>(values_extended)
            : data_p_with_type<const int64_t>(values_tensor);
    const IndicePair* indices_in =
        (fill_empty_row_ || prune_invalid_id_)
            ? data_p_with_type<const IndicePair>(indices_extended)
            : data_p_with_type<const IndicePair>(indices_tensor);

    // 4. unique
    Tensor unique_keys_out;
    // actually below 4 will be allocate_output inside UniqueWithCountsGPU.
    // So it's even okay to pass nullptr for these 4 to UniqueWithCountsGPU
    Tensor *unique_idxs_out, *unique_counts_out, *idx_of_input_to_unique_out,
        *unique_offsets_out;
    UniqueWithCountsGPU<int64, int64>(
        ctx,
        ((fill_empty_row_ || prune_invalid_id_) ? &values_extended
                                                : values_tensor),
        &unique_keys_out, unique_idxs_out, unique_counts_out,
        idx_of_input_to_unique_out, unique_offsets_out);

    int64 uniq_size = unique_keys_out.dim_size(0);

    // 5. partition select
    OpOutputList partitioned_values;
    OP_REQUIRES_OK(ctx,
                   ctx->output_list("partitioned_values", &partitioned_values));
    OpOutputList partition_permutations;
    OP_REQUIRES_OK(ctx, ctx->output_list("partition_permutations",
                                         &partition_permutations));

    if (num_partitions_ == 1) {
      // Simply copy
      Tensor* partitioned_value;
      partitioned_values.allocate(0, TensorShape({uniq_size}),
                                  &partitioned_value);
      cudaMemcpyAsync(data_p_with_type<int64_t>(partitioned_values[0]),
                      data_p_with_type<int64_t>(unique_keys_out),
                      sizeof(int64_t) * uniq_size, cudaMemcpyDeviceToDevice,
                      device.stream());
      // Need to allocate_output here
      Tensor* partitioned_permutation;
      partition_permutations.allocate(0, TensorShape({uniq_size}),
                                      &partitioned_permutation);
      RangeInit(device, uniq_size,
                data_p_with_type<int64_t>(partitioned_permutation));
    } else {
      Tensor partition_permute_init;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT64, TensorShape({uniq_size}),
                                             &partition_permute_init));

      RangeInit(device, uniq_size,
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
          accu_div_host[i] = i > 0 ? div : accu_div_host[i - 1] + div;
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
            ctx, unique_keys_out, partition_permute_init, accu_div,
            num_partitions_, partitioned_values, partition_permutations);
      }
    }
  }

 private:
  int num_partitions_;
  int partition_axis_;
  bool fill_empty_row_;
  bool prune_invalid_id_;
  std::string partition_strategy_;
  int64_t default_id_;
};

REGISTER_KERNEL_BUILDER(Name("FusedEmbeddingSparsePreLookUp")
                            .Device(DEVICE_GPU)
                            .HostMemory("partition_shapes")
                            .HostMemory("sp_dense_shape"),
                        FusedEmbeddingSparsePreLookUpGPU);
}  // namespace tensorflow

#endif  // GOOGLE_CUDA