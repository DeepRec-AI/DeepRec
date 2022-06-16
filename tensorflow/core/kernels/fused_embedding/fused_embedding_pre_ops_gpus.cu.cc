#include <exception>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/fused_embedding/fused_embedding_common.cu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_select.cuh"
#include "cub/iterator/constant_input_iterator.cuh"
#include "cub/thread/thread_operators.cuh"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

namespace {

__global__ void InitFlagsToOneInt4(int length, int* flags) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (4 * offset + 3 < length) {
    *((int4*)(flags + 4 * offset)) = make_int4(1, 1, 1, 1);
  } else if (4 * offset < length) {
    for (int i = 0; i < length - 4 * offset; i++) {
      flags[4 * offset + i] = 1;
    }
  }
}

__global__ void FusedMultiFunctionalKernel(
    const IndicePair* indices, const int64_t* values, const int64_t nnz,
    const int64_t batch_size, const bool prune_invalid_id,
    const int64_t default_id, int* row_emptiness_flag, int* invalid_id_flag,
    IndicePair* tmp_indices_buffer, int64_t* values_extended) {
  // This kernel will do many things together
  // 1. The first part of threads will do job 1(DetectRowEmptiness), others will
  // do job2(InitBatchRowsBuffer)
  // 2. Do job3 (set values extended to default id)

  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < nnz) {
    // do DetectRowEmptiness
    if (prune_invalid_id) {
      const int64_t value = values[offset];
      if (value < 0) {
        // invalid, set invalid_id_flag
        atomicAnd(invalid_id_flag + offset, 0);
      } else {
        // valid, set row_emptiness_flag
        const int64_t row_in_batch = indices[offset].row_in_batch;
        atomicAnd(row_emptiness_flag + row_in_batch, 0);
      }
    } else {
      // set row_emptiness_flag
      const int64_t row_in_batch = indices[offset].row_in_batch;
      atomicAnd(row_emptiness_flag + row_in_batch, 0);
    }
  } else {
    // do InitBatchRowsBuffer
    const int other_offset = offset - nnz;
    if (other_offset < batch_size) {
      tmp_indices_buffer[other_offset].row_in_batch = other_offset;
      // always set entry id to 0;
      tmp_indices_buffer[other_offset].entry_in_column = 0;
    }
  }

  // set values extended to default id
  if (2 * offset + 1 < nnz + batch_size) {
    longlong2 l2 = make_longlong2(default_id, default_id);
    *((longlong2*)(values_extended + 2 * offset)) = l2;
  } else if (2 * offset < nnz + batch_size) {
    values_extended[2 * offset] = default_id;
  }
}

__global__ void DetectInvalid(const int64_t* values, const int64_t nnz,
                              int* invalid_id_flag) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < nnz) {
    const int64_t value = values[offset];
    if (value < 0) {
      atomicAnd(invalid_id_flag + offset, 0);
    }
  }
}

__global__ void CalcElementsOffsetPerPartition(
    const int64_t* values_sorted, int64_t* partition_sizes_accumulate,
    int64_t* elements_offset_per_partition, int nnz) {
  // dichotomy
  const int64_t target = partition_sizes_accumulate[blockIdx.x];
  int roof = nnz;
  int floor = 0;

  int pos = (roof + floor) / 2;
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
      floor = pos;
    } else {
      roof = pos;
    }
    pos = (roof + floor) / 2;
  }
  elements_offset_per_partition[blockIdx.x] = int64_t(pos + 1);
}

__global__ void GatherAndConvertToSubPartition(
    const int64_t* sub_values_sorted, int64_t* sub_partitioned_values,
    const int64_t partition_start_base, const int64_t partition_size) {
  const int t_offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (t_offset < partition_size) {
    int64_t value = sub_values_sorted[t_offset];
    // rebase value to it's corresponding sub partition
    value = value - partition_start_base;
    sub_partitioned_values[t_offset] = value;
  }
}

}  // namespace

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
    auto stream = ctx->eigen_device<GPUDevice>().stream();

    const int64_t default_id = default_id_ >= 0 ? default_id_ : 0;
    const int linear_mapping_threads = 128;

    // 1. bind inputs
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

    // 2. allocate cub tmp storage
    Tensor cub_temp_storage;
    size_t max_cub_bytes = 0;
    size_t temp_storage_bytes = 0;

    if (num_partitions_ > 1) {
      cub::DeviceRadixSort::SortPairs(
          (void*)nullptr, temp_storage_bytes, (int64_t*)nullptr,
          (int64_t*)nullptr, (IndicePair*)nullptr, (IndicePair*)nullptr,
          int(nnz + batch_size), 0, sizeof(int64_t) * 8, stream);
      max_cub_bytes = temp_storage_bytes > max_cub_bytes ? temp_storage_bytes
                                                         : max_cub_bytes;
    }

    if (fill_empty_row_ || prune_invalid_id_) {
      cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, (int64_t*)nullptr,
                                 (int*)nullptr, (int64_t*)nullptr,
                                 (int*)nullptr, nnz, stream);

      max_cub_bytes = temp_storage_bytes > max_cub_bytes ? temp_storage_bytes
                                                         : max_cub_bytes;

      cub::DeviceSelect::Flagged(
          (void*)nullptr, temp_storage_bytes, (IndicePair*)nullptr,
          (int*)nullptr, (IndicePair*)nullptr, (int*)nullptr, nnz, stream);

      max_cub_bytes = temp_storage_bytes > max_cub_bytes ? temp_storage_bytes
                                                         : max_cub_bytes;

      if (fill_empty_row_) {
        cub::DeviceSelect::Flagged((void*)nullptr, temp_storage_bytes,
                                   (IndicePair*)nullptr, (int*)nullptr,
                                   (IndicePair*)nullptr, (int*)nullptr,
                                   batch_size, stream);
        max_cub_bytes = temp_storage_bytes > max_cub_bytes ? temp_storage_bytes
                                                           : max_cub_bytes;
      }
    }

    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(
                 DT_INT8, TensorShape({static_cast<int64_t>(max_cub_bytes)}),
                 &cub_temp_storage));

    // 3. fill_empty_row, prune, if avaliable.
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

      {
        const int threads = linear_mapping_threads;
        const int blocks =
            CalcBlocksLinearMapping(batch_size + nnz, threads * 4);
        InitFlagsToOneInt4<<<blocks, threads, 0, stream>>>(
            batch_size + nnz, all_flags->flat<int>().data());
        CK_CUDA_THROW_(cudaGetLastError());
      }

      // 3.1 set flags, init tmp_indices_buffer etc.
      if (fill_empty_row_) {
        {
          const int threads = linear_mapping_threads;
          const int blocks = CalcBlocksLinearMapping(nnz + batch_size, threads);
          FusedMultiFunctionalKernel<<<blocks, threads, 0, stream>>>(
              reinterpret_cast<const IndicePair*>(
                  indices_tensor->flat<int64>().data()),
              reinterpret_cast<const int64_t*>(
                  values_tensor->flat<int64>().data()),
              nnz, batch_size, prune_invalid_id_, default_id,
              all_flags->flat<int>().data(),
              all_flags->flat<int>().data() + batch_size,
              reinterpret_cast<IndicePair*>(
                  tmp_indices_buffer.flat<int64>().data()),
              reinterpret_cast<int64_t*>(values_extended.flat<int64>().data()));
          CK_CUDA_THROW_(cudaGetLastError());
        }
      } else if (prune_invalid_id_) {
        {
          const int threads = linear_mapping_threads;
          const int blocks = CalcBlocksLinearMapping(nnz, threads);
          DetectInvalid<<<blocks, threads, 0, stream>>>(
              reinterpret_cast<const int64_t*>(
                  values_tensor->flat<int64>().data()),
              nnz, all_flags->flat<int>().data() + batch_size);
          CK_CUDA_THROW_(cudaGetLastError());
        }
      }
      // 3.2 select copy valid id, select copy empty row indices

      cudaError_t cuda_ret = cudaSuccess;
      cuda_ret = cub::DeviceSelect::Flagged(
          cub_temp_storage.flat<int8>().data(), max_cub_bytes,
          reinterpret_cast<const int64_t*>(values_tensor->flat<int64>().data()),
          (const int*)(all_flags->flat<int>().data() + batch_size),
          reinterpret_cast<int64_t*>(values_extended.flat<int64>().data()),
          selected_num_d.flat<int>().data(), int(nnz), stream);
      CK_CUDA_THROW_(cudaGetLastError());

      cub::DeviceSelect::Flagged(
          cub_temp_storage.flat<int8>().data(), max_cub_bytes,
          reinterpret_cast<const IndicePair*>(
              indices_tensor->flat<int64>().data()),
          all_flags->flat<int>().data() + batch_size,
          reinterpret_cast<IndicePair*>(indices_extended.flat<int64>().data()),
          selected_num_d.flat<int>().data(), nnz, stream);

      if (prune_invalid_id_) {
        int selected_num;
        cudaMemcpyAsync(&selected_num, selected_num_d.flat<int>().data(),
                        sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        new_nnz = selected_num;
      }

      if (fill_empty_row_) {
        cub::DeviceSelect::Flagged(
            cub_temp_storage.flat<int8>().data(), max_cub_bytes,
            reinterpret_cast<const IndicePair*>(
                tmp_indices_buffer.flat<int64>().data()),
            all_flags->flat<int>().data(),
            reinterpret_cast<IndicePair*>(
                indices_extended.flat<int64>().data()) +
                new_nnz,
            selected_num_d.flat<int>().data(), batch_size, stream);
        CK_CUDA_THROW_(cudaGetLastError());
        int selected_num;
        cudaMemcpyAsync(&selected_num, selected_num_d.flat<int>().data(),
                        sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        new_nnz += selected_num;
      }
    }

    // 3.5 set the correct pointer
    const int64_t* values_in = (fill_empty_row_ || prune_invalid_id_)
                                   ? reinterpret_cast<const int64_t*>(
                                         values_extended.flat<int64>().data())
                                   : reinterpret_cast<const int64_t*>(
                                         values_tensor->flat<int64>().data());
    const IndicePair* indices_in =
        (fill_empty_row_ || prune_invalid_id_)
            ? reinterpret_cast<const IndicePair*>(
                  indices_extended.flat<int64>().data())
            : reinterpret_cast<const IndicePair*>(
                  indices_tensor->flat<int64>().data());

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

      cudaMemcpyAsync(pv_out->flat<int64>().data(), values_in,
                      sizeof(int64_t) * new_nnz, cudaMemcpyDeviceToDevice,
                      stream);
      cudaMemcpyAsync(pi_out->flat<int64>().data(), indices_in,
                      sizeof(IndicePair) * new_nnz, cudaMemcpyDeviceToDevice,
                      stream);

    } else {
      // multi-partitions case, calcaulate indices and split them.
      Tensor values_sorted;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT64, TensorShape{new_nnz},
                                             &values_sorted));
      Tensor indices_sorted;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT64, TensorShape{new_nnz, 2},
                                             &indices_sorted));

      cub::DeviceRadixSort::SortPairs(
          cub_temp_storage.flat<int8>().data(), max_cub_bytes, values_in,
          reinterpret_cast<int64_t*>(values_sorted.flat<int64>().data()),
          indices_in,
          reinterpret_cast<IndicePair*>(indices_sorted.flat<int64>().data()),
          int(new_nnz), 0, sizeof(int64_t) * 8, stream);
      CK_CUDA_THROW_(cudaGetLastError());

      // 4.1 calculate how many elements for each
      // partition
      Tensor partition_sizes_accumulate;
      OP_REQUIRES_OK(
          ctx,
          ctx->allocate_temp(
              DT_INT64, TensorShape({static_cast<int64_t>(num_partitions_)}),
              &partition_sizes_accumulate));
      cudaMemcpyAsync(partition_sizes_accumulate.flat<int64>().data(),
                      partition_sizes_accumulate_.data(),
                      num_partitions_ * sizeof(int64_t), cudaMemcpyHostToDevice,
                      stream);

      Tensor elements_offset_per_partition;
      OP_REQUIRES_OK(
          ctx,
          ctx->allocate_temp(
              DT_INT64, TensorShape({static_cast<int64_t>(num_partitions_)}),
              &elements_offset_per_partition));

      {
        const int blocks = num_partitions_;
        const int threads = 1;
        CalcElementsOffsetPerPartition<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const int64_t*>(
                values_sorted.flat<int64>().data()),
            reinterpret_cast<int64_t*>(
                partition_sizes_accumulate.flat<int64>().data()),
            reinterpret_cast<int64_t*>(
                elements_offset_per_partition.flat<int64>().data()),
            int(new_nnz));
        CK_CUDA_THROW_(cudaGetLastError());
      }

      elements_offset_per_partition_.clear();
      elements_offset_per_partition_.resize(num_partitions_);
      // stream_executor::DeviceMemoryBase
      // elements_offset_per_partition_wrapped(
      //     elements_offset_per_partition.flat<int64>().data(),
      //     num_partitions_);
      // stream->ThenMemcpy(elements_offset_per_partition_.data(),
      //                    elements_offset_per_partition_wrapped,
      //                    num_partitions_ *
      //                    sizeof(int64_t));
      // stream->BlockHostUntilDone();

      cudaMemcpyAsync(elements_offset_per_partition_.data(),
                      elements_offset_per_partition.flat<int64>().data(),
                      num_partitions_ * sizeof(int64_t), cudaMemcpyDeviceToHost,
                      stream);
      cudaStreamSynchronize(stream);

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
          const int threads = linear_mapping_threads;
          int blocks = CalcBlocksLinearMapping(size, threads);

          const int partition_start_base =
              i == 0 ? 0 : partition_sizes_accumulate_[i - 1];
          GatherAndConvertToSubPartition<<<blocks, threads, 0, stream>>>(
              reinterpret_cast<const int64_t*>(
                  values_sorted.flat<int64>().data()) +
                  sub_start_offset,
              reinterpret_cast<int64_t*>(
                  sub_partitioned_values->flat<int64>().data()),
              partition_start_base, size);

          CK_CUDA_THROW_(cudaGetLastError());

          // stream_executor::DeviceMemoryBase
          // sub_indices_sorted_wrapped(
          //     reinterpret_cast<IndicePair*>(indices_sorted.flat<int64>().data())
          //     +
          //         partition_start_base,
          //     size * sizeof(IndicePair));
          // stream_executor::DeviceMemoryBase
          // sub_indices_out_wrapped(
          //     reinterpret_cast<IndicePair*>(
          //         sub_partitioned_indices.flat<int64>().data()),
          //     size * sizeof(IndicePair));
          // stream->ThenMemcpy(&sub_indices_out_wrapped,
          //                    sub_indices_sorted_wrapped,
          //                    size * 2 *
          //                    sizeof(int64_t));
          cudaMemcpyAsync(
              sub_partitioned_indices->flat<int64>().data(),
              indices_sorted.flat<int64>().data() + 2 * sub_start_offset,
              size * 2 * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream);
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
