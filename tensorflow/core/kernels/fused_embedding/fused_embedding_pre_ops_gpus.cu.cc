#include <exception>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/fused_embedding/fused_embedding.cu.h"
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
    int temp_default_id;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_id", &temp_default_id));
    default_id_ = int64_t(temp_default_id);
  }

  void Compute(OpKernelContext* ctx) override {
    using namespace fused_embedding;
    auto device = ctx->eigen_device<GPUDevice>();

    const int64_t default_id = default_id_ >= 0 ? default_id_ : 0;
    const int linear_mapping_threads = 128;

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

    OpInputList partition_shapes;
    OP_REQUIRES_OK(ctx, ctx->input_list("partition_shapes", &partition_shapes));

    partition_sizes_accumulate_.clear();
    for (const Tensor& shape : partition_shapes) {
      OP_REQUIRES(ctx, shape.dims() <= 2,
                  errors::InvalidArgument(
                      "input partition_shapes must all less than rank 2"));
      const int64_t accu = partition_sizes_accumulate_.empty()
                               ? shape.flat<int64>().data()[0]
                               : shape.flat<int64>().data()[0] +
                                     partition_sizes_accumulate_.back();
      partition_sizes_accumulate_.push_back(accu);
    }
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

    if (num_partitions_ > 1) {
      cub::DeviceRadixSort::SortPairs(
          (void*)nullptr, temp_storage_bytes, (int64_t*)nullptr,
          (int64_t*)nullptr, (IndicePair*)nullptr, (IndicePair*)nullptr,
          int(nnz + batch_size), 0, sizeof(int64_t) * 8, device.stream());
      max_cub_bytes = temp_storage_bytes > max_cub_bytes ? temp_storage_bytes
                                                         : max_cub_bytes;
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
        ctx, ctx->allocate_output(2 * num_partitions_,
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
            prune_invalid_id_, default_id,
            data_p_with_type<int>(all_flags),
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
        cudaStreamSynchronize(device.stream());
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
        cudaStreamSynchronize(device.stream());
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

    OpOutputList partitioned_values;
    OP_REQUIRES_OK(ctx,
                   ctx->output_list("partitioned_values", &partitioned_values));
    OpOutputList partitioned_indices;
    OP_REQUIRES_OK(
        ctx, ctx->output_list("partitioned_indices", &partitioned_indices));

    // 4. set output
    if (num_partitions_ == 1) {
      // single partition case, just directly copy
      Tensor* pv_out;
      OP_REQUIRES_OK(
          ctx, partitioned_values.allocate(
                   0, TensorShape({static_cast<int64_t>(new_nnz)}), &pv_out));
      Tensor* pi_out;
      OP_REQUIRES_OK(
          ctx,
          partitioned_indices.allocate(
              0, TensorShape({static_cast<int64_t>(new_nnz), 2}), &pi_out));

      cudaMemcpyAsync(data_p_with_type<int64>(pv_out), values_in,
                      sizeof(int64_t) * new_nnz, cudaMemcpyDeviceToDevice,
                      device.stream());
      cudaMemcpyAsync(data_p_with_type<int64>(pi_out), indices_in,
                      sizeof(IndicePair) * new_nnz, cudaMemcpyDeviceToDevice,
                      device.stream());

    } else {
      // multi-partitions case, calcaulate indices and split them.
      Tensor values_sorted;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT64, TensorShape{new_nnz},
                                             &values_sorted));
      Tensor indices_sorted;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT64, TensorShape{new_nnz, 2},
                                             &indices_sorted));

      cub::DeviceRadixSort::SortPairs(
          data_p_with_type<int8>(cub_temp_storage), max_cub_bytes, values_in,
          data_p_with_type<int64_t>(values_sorted), indices_in,
          data_p_with_type<IndicePair>(indices_sorted), int(new_nnz), 0,
          sizeof(int64_t) * 8, device.stream());
      CK_CUDA_THROW_(cudaGetLastError());

      // 4.1 calculate how many elements for each
      // partition
      Tensor partition_sizes_accumulate;
      OP_REQUIRES_OK(
          ctx,
          ctx->allocate_temp(
              DT_INT64, TensorShape({static_cast<int64_t>(num_partitions_)}),
              &partition_sizes_accumulate));
      cudaMemcpyAsync(data_p_with_type<int64>(partition_sizes_accumulate),
                      partition_sizes_accumulate_.data(),
                      num_partitions_ * sizeof(int64_t), cudaMemcpyHostToDevice,
                      device.stream());

      Tensor elements_offset_per_partition;
      OP_REQUIRES_OK(
          ctx,
          ctx->allocate_temp(
              DT_INT64, TensorShape({static_cast<int64_t>(num_partitions_)}),
              &elements_offset_per_partition));

      CalcElementsOffsetPerPartition(
          device, num_partitions_,
          data_p_with_type<const int64_t>(values_sorted),
          data_p_with_type<int64_t>(partition_sizes_accumulate),
          data_p_with_type<int64_t>(elements_offset_per_partition),
          int(new_nnz));
      CK_CUDA_THROW_(cudaGetLastError());

      elements_offset_per_partition_.clear();
      elements_offset_per_partition_.resize(num_partitions_);
      // device.stream()_executor::DeviceMemoryBase
      // elements_offset_per_partition_wrapped(
      //     elements_offset_per_partition.flat<int64>().data(),
      //     num_partitions_);
      // device.stream()->ThenMemcpy(elements_offset_per_partition_.data(),
      //                    elements_offset_per_partition_wrapped,
      //                    num_partitions_ *
      //                    sizeof(int64_t));
      // device.stream()->BlockHostUntilDone();

      cudaMemcpyAsync(elements_offset_per_partition_.data(),
                      data_p_with_type<int64>(elements_offset_per_partition),
                      num_partitions_ * sizeof(int64_t), cudaMemcpyDeviceToHost,
                      device.stream());
      cudaStreamSynchronize(device.stream());


      // 4.2 set output
      int64_t sub_start_offset = 0;
      for (int i = 0; i < num_partitions_; i++) {
        int64_t size = elements_offset_per_partition_[i] - sub_start_offset;

        Tensor* sub_partitioned_values;
        OP_REQUIRES_OK(ctx, partitioned_values.allocate(
                                i, TensorShape({static_cast<int64_t>(size)}),
                                &sub_partitioned_values));

        Tensor* sub_partitioned_indices;
        OP_REQUIRES_OK(ctx, partitioned_indices.allocate(
                                i, TensorShape({static_cast<int64_t>(size), 2}),
                                &sub_partitioned_indices));

        if (size > 0) {
          // some partition does not have any
          // element that falls in it
          const int partition_start_base =
              i == 0 ? 0 : partition_sizes_accumulate_[i - 1];

          GatherAndConvertToSubPartition(
              device,
              data_p_with_type<const int64_t>(values_sorted) + sub_start_offset,
              data_p_with_type<int64_t>(sub_partitioned_values),
              partition_start_base, size);

          CK_CUDA_THROW_(cudaGetLastError());

          // device.stream()_executor::DeviceMemoryBase
          // sub_indices_sorted_wrapped(
          //     reinterpret_cast<IndicePair*>(indices_sorted.flat<int64>().data())
          //     +
          //         partition_start_base,
          //     size * sizeof(IndicePair));
          // device.stream()_executor::DeviceMemoryBase
          // sub_indices_out_wrapped(
          //     reinterpret_cast<IndicePair*>(
          //         sub_partitioned_indices.flat<int64>().data()),
          //     size * sizeof(IndicePair));
          // device.stream()->ThenMemcpy(&sub_indices_out_wrapped,
          //                    sub_indices_sorted_wrapped,
          //                    size * 2 *
          //                    sizeof(int64_t));
          cudaMemcpyAsync(
              sub_partitioned_indices->flat<int64>().data(),
              indices_sorted.flat<int64>().data() + 2 * sub_start_offset,
              size * 2 * sizeof(int64_t), cudaMemcpyDeviceToDevice,
              device.stream());
        }
        sub_start_offset = elements_offset_per_partition_[i];
      }
    }
    // Op kernel execution done
  }

 private:
  int num_partitions_;
  int partition_axis_;
  bool fill_empty_row_;
  bool prune_invalid_id_;
  int64_t default_id_;
  std::vector<int64_t> partition_sizes_accumulate_;
  std::vector<int64_t> elements_offset_per_partition_;
};

REGISTER_KERNEL_BUILDER(Name("FusedEmbeddingSparsePreLookUp")
                            .Device(DEVICE_GPU)
                            .HostMemory("partition_shapes")
                            .HostMemory("sp_dense_shape"),
                        FusedEmbeddingSparsePreLookUpGPU);
}  // namespace tensorflow

#endif  // GOOGLE_CUDA