#include <exception>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "tensorflow/core/kernels/fused_embedding/gpu/common.cu.h"
#include "tensorflow/core/kernels/fused_embedding/gpu/functions/kernels.cu.h"
#include "tensorflow/core/profiler/nvtx_utils.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "third_party/cub/device/device_radix_sort.cuh"
#include "third_party/cub/device/device_select.cuh"
#include "third_party/cub/iterator/constant_input_iterator.cuh"
#include "third_party/cub/thread/thread_operators.cuh"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

class PruneInvalidAndFillEmptyRowsGPU : public OpKernel {
 public:
  explicit PruneInvalidAndFillEmptyRowsGPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_empty_row", &fill_empty_row_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("prune_invalid_id", &prune_invalid_id_));
    int temp_default_id;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_id", &temp_default_id));
    default_id_ = int64_t(temp_default_id);
    cudaEventCreateWithFlags(&memcpy_event_, cudaEventDisableTiming);
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

    // =============================================== //

    // =============================================== //

    // === 2. fill_empty_row, prune, if avaliable. === //
    Tensor values_extended;
    Tensor indices_extended;
    Tensor tmp_indices_buffer;
    Tensor* row_empty_and_invalid_flags;
    Tensor selected_num_d;
    int new_nnz = nnz;

    Tensor cub_temp_storage;
    size_t max_cub_bytes = 0;
    size_t temp_storage_bytes = 0;

    OP_REQUIRES_OK(ctx, ctx->allocate_output("row_empty_and_invalid_flags",
                                             TensorShape{batch_size + nnz},
                                             &row_empty_and_invalid_flags));

    if (fill_empty_row_ || prune_invalid_id_) {
      // create cub temp storage

      cub::DeviceSelect::Flagged(
          nullptr, temp_storage_bytes,
          thrust::make_zip_iterator(thrust::make_tuple(
              data_p_with_type<const int64_t>(values_tensor),
              data_p_with_type<const IndicePair>(indices_tensor))),
          (int*)nullptr,
          thrust::make_zip_iterator(thrust::make_tuple(
              data_p_with_type<int64_t>(values_extended),
              data_p_with_type<IndicePair>(indices_extended))),
          (int*)nullptr, int(nnz), device.stream());

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

      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(
                   DT_INT8, TensorShape({static_cast<int64_t>(max_cub_bytes)}),
                   &cub_temp_storage));

      // allocate temp

      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DT_INT64, TensorShape{nnz + batch_size},
                                        &values_extended));
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DT_INT64, TensorShape{2 * (nnz + batch_size)},
                                  &indices_extended));
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DT_INT64, TensorShape{batch_size, 2},
                                        &tmp_indices_buffer));
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DT_INT32, TensorShape{1}, &selected_num_d));

      InitFlagsToOneInt4(device, batch_size + nnz,
                         row_empty_and_invalid_flags->flat<int>().data());

      // set flags, init tmp_indices_buffer etc.
      if (fill_empty_row_) {
        FusedMultiFunctional(
            device, data_p_with_type<const IndicePair>(indices_tensor),
            data_p_with_type<const int64_t>(values_tensor), nnz, batch_size,
            prune_invalid_id_, default_id,
            data_p_with_type<int>(row_empty_and_invalid_flags),
            data_p_with_type<int>(row_empty_and_invalid_flags) + batch_size,
            data_p_with_type<IndicePair>(tmp_indices_buffer),
            data_p_with_type<int64_t>(values_extended));

      } else if (prune_invalid_id_) {
        DetectInvalid(
            device, data_p_with_type<const int64_t>(values_tensor), nnz,
            data_p_with_type<int>(row_empty_and_invalid_flags) + batch_size);
      }
      // select copy valid id, select copy empty row indices

      cub::DeviceSelect::Flagged(
          data_p_with_type<int8>(cub_temp_storage), max_cub_bytes,
          thrust::make_zip_iterator(thrust::make_tuple(
              data_p_with_type<const int64_t>(values_tensor),
              data_p_with_type<const IndicePair>(indices_tensor))),
          data_p_with_type<const int>(row_empty_and_invalid_flags) + batch_size,
          thrust::make_zip_iterator(thrust::make_tuple(
              data_p_with_type<int64_t>(values_extended),
              data_p_with_type<IndicePair>(indices_extended))),
          data_p_with_type<int>(selected_num_d), int(nnz), device.stream());

      if (prune_invalid_id_) {
        int selected_num;
        CK_CUDA_THROW_(cudaMemcpyAsync(
            &selected_num, data_p_with_type<int>(selected_num_d), sizeof(int),
            cudaMemcpyDeviceToHost, device.stream()));
        CK_CUDA_THROW_(cudaEventRecord(memcpy_event_, device.stream()));
        CK_CUDA_THROW_(cudaEventSynchronize(memcpy_event_));
        new_nnz = selected_num;
      }

      if (fill_empty_row_) {
        cub::DeviceSelect::Flagged(
            data_p_with_type<int8>(cub_temp_storage), max_cub_bytes,
            data_p_with_type<const IndicePair>(tmp_indices_buffer),
            data_p_with_type<int>(row_empty_and_invalid_flags),
            data_p_with_type<IndicePair>(indices_extended) + new_nnz,
            data_p_with_type<int>(selected_num_d), batch_size, device.stream());
        int selected_num;
        CK_CUDA_THROW_(cudaMemcpyAsync(
            &selected_num, data_p_with_type<void>(selected_num_d), sizeof(int),
            cudaMemcpyDeviceToHost, device.stream()));
        CK_CUDA_THROW_(cudaEventRecord(memcpy_event_, device.stream()));
        CK_CUDA_THROW_(cudaEventSynchronize(memcpy_event_));
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

    Tensor* sp_values_out;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("sp_values_out", TensorShape{new_nnz},
                                        &sp_values_out));

    Tensor* sp_indices_out;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("sp_indices_out", TensorShape{new_nnz, 2},
                                  &sp_indices_out));

    CK_CUDA_THROW_(cudaMemcpyAsync(data_p_with_type<int64>(sp_values_out),
                                   values_in, sizeof(int64) * new_nnz,
                                   cudaMemcpyDeviceToDevice, device.stream()));

    CK_CUDA_THROW_(cudaMemcpyAsync(data_p_with_type<int64>(sp_indices_out),
                                   indices_in, sizeof(IndicePair) * new_nnz,
                                   cudaMemcpyDeviceToDevice, device.stream()));
  }

 private:
  bool fill_empty_row_;
  bool prune_invalid_id_;
  int64_t default_id_;
  cudaEvent_t memcpy_event_;
};

REGISTER_KERNEL_BUILDER(Name("PruneInvalidAndFillEmptyRows")
                            .Device(DEVICE_GPU)
                            .HostMemory("sp_dense_shape"),
                        PruneInvalidAndFillEmptyRowsGPU);
}  // namespace tensorflow

#endif  // GOOGLE_CUDA