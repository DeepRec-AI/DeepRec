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

inline int CalcBlocksLinearMapping(const int problem_size, const int threads) {
  return problem_size % threads == 0 ? (problem_size / problem_size)
                                     : (batch_size / threads + 1);
}

struct IndicePair {
  int64_t row_in_batch;
  int64_t entry_in_column;
};

_global__ void DetectInvalid(const int64_t* values, const int64_t nnz,
                             int* invalid_id_flag) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < nnz) {
    const int64_t value = values[offset];
    if (value < 0) {
      atomicAnd(invalid_id_flag + offset, 0x0);
    }
  }
}

__forceinline__ __device__ void DetectEmptyRow(
    const IndicePair* indices, const int64_t* values, const int64_t nnz,
    const int64_t batch_size, const int64_t default_id, int* row_emptiness_flag,
    IndicePair* batch_indices_buffer, int64_t* values_extended) {
  // This kernel will do many things together
  // 1. The first part of threads will do job 1(DetectRowEmptiness), others will
  // do job2(InitBatchRowsBuffer)
  // 2. Do job3 (set values extended to default id)

  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < nnz) {
    // do DetectRowEmptiness
    const int64_t row_in_batch = indices[offset].row_in_batch;
    atomicAnd(row_emptiness_flag + row_in_batch, 0x0);
  } else {
    // do InitBatchRowsBuffer
    const int other_offset = offset - nnz;
    if (other_offset < batch_size) {
      batch_indices_buffer[other_offset].row_in_batch = other_offset;
      // always set entry id to 0;
      batch_indices_buffer[other_offset].entry_in_column = 0;
    }

    // set values extended to default id
    if (offset + 2 < nnz + batch_size) {
      longlong2 l2 = make_longlong2(default_id, default_id);
      *((longlong2*)(values_extended + offset)) = l2;
    } else if (offset < nnz + batch_size) {
      values_extended[offset] = default_id;
    }
  }
}

__global__ void DetectEmptyRowKernelNNZHost(
    const IndicePair* indices, const int64_t* values, const int64_t nnz,
    const int64_t batch_size, const int64_t default_id, int* row_emptiness_flag,
    IndicePair* batch_indices_buffer, int64_t* values_extended) {
  MultiFunctional(indices, values, nnz, batch_size, default_id,
                  row_emptiness_flag, invalid_id_flag, batch_indices_buffer,
                  values_extended);
}

__global__ void DetectEmptyRowNNZDevice(
    const IndicePair* indices, const int64_t* values, const int64_t* nnz_p,
    const int64_t batch_size, const int64_t default_id, int* row_emptiness_flag,
    IndicePair* batch_indices_buffer, int64_t* values_extended) {
  __share__ int64_t nnz_shared[1];
  if (threadIdx.x == 0) {
    nnz_shared[0] = *(nnz_p)
  }
  __syncthreads();
  const int64_t nnz = nnz_shared[0];
  DetectEmptyRow(indices, values, nnz, batch_size, default_id,
                 row_emptiness_flag, batch_indices_buffer, values_extended);
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
  int64_t row_in_batch;
  if (threadIdx.x == 0) {
    row_in_batch = partitioned_indice[2 * blockIdx.x];
    row_in_batch_shared[0] = row_in_batch;
    feature_num_shared[0] = feature_nums[row_in_batch];
  }
  __syncthreads();
  row_in_batch = row_in_batch_shared[0];
  const int feature_num = feature_num_shared[0];
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

class FusedEmbeddingSparsePreLookUpGPU : public OpKernel {
 public:
  explicit FusedEmbeddingSparsePreLookUpGPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_axis", &partition_axis_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_empty_row", &fill_empty_row_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("prune_invalid_id", &prune_invalid_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_id", &default_id_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto stream = ctx->eigen_device<GPUDevice>().stream();

    const int64_t default_id = default_id_ >= 0 ? default_id_ : 0;

    // 1. bind inputs
    Tensor const* values_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_values", &values_tensor));
    const int64 nnz = values_tensor->shape().dim_size(0);

    Tensor const* indices_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_indices", &indices_tensor));

    Tensor const* dense_shape = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_dense_shape", &dense_shape));
    const int64 batch_size = dense_shape.flat<int64>().data()[0];

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
    size_t max_bytes = 0;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, NULL, NULL, NULL,
                                    NULL, int(nnz), 0, sizeof(int64) * 8,
                                    stream);
    max_cub_bytes =
        temp_storage_bytes > max_cub_bytes ? temp_storage_bytes : max_cub_bytes;

    if (fill_empty_row_) {
    }

    if (prune_invalid_id_) {
      cub::DeviceSelect::Flagged(NULL, temp_storage_bytes, NULL, NULL, NULL,
                                 NULL, nnz, stream);
      max_cub_bytes = temp_storage_bytes > max_cub_bytes ? temp_storage_bytes
                                                         : max_cub_bytes;
    }

    OP_REQUIRES_OK(
        ctx,
        ctx->allocate_temp(
            DT_INT8,
            TensorShape({static_cast<int64>(max_cub_bytes / sizeof(size_t))}),
            &max_cub_bytes));

    // 3. fill_empty_row, prune, if avaliable.
    Tensor values_extended;
    Tensor indices_extended;
    Tensor tmp_indices_buffer;
    Tensor* all_flags;
    Tensor selected_num;
    int new_nnz = nnz;

    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(2 * num_partitions_,
                                        TensorShape{batch_size + nnz}),
                   &all_flags);

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
      // pos 0 for prune_invalid, pos 1 for empty_row
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DT_INT32, TensorShape{2}, &selected_num));

      cudaMemsetAsync(all_flags->flat<int>().data(), 0x1,
                      sizeof(int) * (batch_size + nnz), stream);

      if (prune_invalid_id_) {
        // 3.1 detect invalid id
        {
          const int threads = 1024;
          const int blocks = CalcBlocksLinearMapping(nnz, threads);
          DetectInvalid<<<blocks, threads, 0, stream>>>(
              values_tensor->flat<int64>().data(), nnz,
              all_flags->flat<int>().data() + batch_size);
        }
        // only copy valid values and indices
        cub::DeviceSelect::Flagged(
            cub_temp_storage.flat<int8>().data(), max_cub_bytes,
            reinterpret_cast<const IndicePair*>(
                indices_tensor->flat<int64>().data()),
            all_flags->flat<int>().data() + batch_size,
            reinterpret_cast<IndicePair*>(
                indices_extended.flat<int64>().data()),
            selected_num_d.flat<int>().data(), nnz, stream);

        cub::DeviceSelect::Flagged(
            cub_temp_storage.flat<int8>().data(), max_cub_bytes,
            values_tensor->flat<int64>().data(),
            all_flags->flat<int>().data() + batch_size,
            values_extended.flat<int64>().data(),
            selected_num_d.flat<int>().data(), nnz, stream);
      }

      if (fill_empty_row_) {
        // 3.2 detect empty row and init other buffer which will be used later
        if (prune_invalid_id_) {
          // nnz may has changed, fetch it from device mem
          {
            const int threads = 1024;
            const int blocks =
                CalcBlocksLinearMapping(nnz + batch_size, threads);
            DetectEmptyRowNNZDevice<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<IndicePair*>(
                    indices_tensor->flat<int64>().data()),
                values_tensor->flat<int64>().data(),
                selected_num.flat<int>.data(), batch_size, default_id,
                all_flags->flat<int>().data(),
                reinterpret_cast<IndicePair*>(
                    tmp_indices_buffer.flat<int>().data()),
                values_extended.flat<int64>().data());
          }
        } else {
          // can fetch nnz directly from host args
          const int threads = 1024;
          const int blocks = CalcBlocksLinearMapping(nnz + batch_size, threads);
          DetectEmptyRowKernelNNZHost<<<blocks, threads, 0, stream>>>(
              reinterpret_cast<IndicePair*>(
                  indices_tensor->flat<int64>().data()),
              values_tensor->flat<int64>().data(), nnz, batch_size, default_id,
              all_flags->flat<int>().data(),
              reinterpret_cast<IndicePair*>(
                  tmp_indices_buffer.flat<int>().data()),
              values_extended.flat<int64>().data());
        }
        // 3.3 copy emtpy row indiecs
        cub::DeviceSelect::Flagged(
            cub_temp_storage.flat<int8>().data(), max_cub_bytes,
            reinterpret_cast<const IndicePair*>(
                tmp_indices_buffer.flat<int64>().data()),
            all_flags->flat<int>().data(),
            reinterpret_cast<IndicePair*>(
                indices_extended.flat<int64>().data()) +
                new_nnz,
            selected_num_d.flat<int>().data(), batch_size, stream);
      }

      // 3.1 set flags, init tmp_indices_buffer etc.
      if (fill_empty_row_) {
        {
          const int threads = 1024;
          const int blocks = CalcBlocksLinearMapping(nnz + batch_size, threads);

          FusedMultiFunctionalKernel<<<blocks, threads, 0, stream>>>(
              reinterpret_cast<IndicePair*>(
                  indices_tensor->flat<int64>().data()),
              values_tensor->flat<int64>().data(), nnz, batch_size,
              prune_invalid_id_, all_flags->flat<int>().data(),
              all_flags->flat<int>().data() + batch_size,
              reinterpret_cast<IndicePair*>(
                  tmp_indices_buffer.flat<int>().data()),
              values_extended.flat<int64>().data());
        }
      } else if (prune_invalid_id_) {
        const int threads = 1024;
        const int blocks = CalcBlocksLinearMapping(nnz, threads);
        DetectInvalid<<<blocks, threads, 0, stream>>>(
            values_tensor->flat<int64>().data(), nnz,
            all_flags->flat<int>().data() + batch_size);
      }
      // 3.2 select copy valid id, select copy empty row indices
      if (prune_invalid_id_) {
        cub::DeviceSelect::Flagged(
            cub_temp_storage.flat<int8>().data(), max_cub_bytes,
            reinterpret_cast<const IndicePair*>(
                indices_tensor->flat<int64>().data()),
            all_flags->flat<int>().data() + batch_size,
            reinterpret_cast<IndicePair*>(
                indices_extended.flat<int64>().data()),
            selected_num_d.flat<int>().data(), nnz, stream);

        cub::DeviceSelect::Flagged(
            cub_temp_storage.flat<int8>().data(), max_cub_bytes,
            values_tensor->flat<int64>().data(),
            all_flags->flat<int>().data() + batch_size,
            values_extended.flat<int64>().data(),
            selected_num_d.flat<int>().data(), nnz, stream);

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
        int selected_num;
        cudaMemcpyAsync(&selected_num, selected_num_d.flat<int>().data(),
                        sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        new_nnz += selected_num;
      }
    }

    // 3. sort the sp_values and indices
    Tensor values_sorted;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT64, TensorShape{new_nnz},
                                           &values_sorted));
    Tensor indices_sorted;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT64, TensorShape{new_nnz, 2},
                                           &indices_sorted));

    const int64_t* values_in = (fill_empty_row_ || prune_invalid_id_)
                                   ? values_extended.flat<int64>().data()
                                   : values_tensor->flat<int64>().data();
    const IndicePair* indices_in =
        (fill_empty_row_ || prune_invalid_id_)
            ? reinterpret_cast<const IndicePair*>(
                  indices_extended.flat<int64>().data())
            : reinterpret_cast<const IndicePair*>(
                  indices_tensor->flat<int64>().data());

    cub::DeviceRadixSort::SortPairs(
        cub_temp_storage.flat<int8>().data(), temp_storage_bytes, values_in,
        reinterpret_cast<int64_t*>(values_sorted.flat<int64>().data()),
        IndicePair,
        reinterpret_cast<IndicePair*>(indices_sorted.flat<int64>().data()),
        int(new_nnz), 0, sizeof(int64) * 8, stream);

    // 3. calculate how many elements for each partition
    Tensor partition_sizes_accumulate;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(
                 DT_INT64, TensorShape({static_cast<int64>(num_partitions_)}),
                 &partition_sizes_accumulate));
    cudaMemcpyAsync(partition_sizes_accumulate.flat<int64>().data(),
                    partition_sizes_accumulate_.data(),
                    num_partitions_ * sizeof(int64_t), cudaMemcpyHostToDevice,
                    stream);

    Tensor elements_offset_per_partition;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(
                 DT_INT64, TensorShape({static_cast<int64>(num_partitions_)}),
                 &elements_offset_per_partition));

    {
      const int blocks = num_partitions_;
      const int threads = 1;
      CalcElementsOffsetPerPartition<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<int64_t*>(values_sorted.flat<int64>().data()),
          reinterpret_cast<int64_t*>(
              partition_sizes_accumulate.flat<int64>().data()),
          reinterpret_cast<int64_t*>(
              elements_offset_per_partition.flat<int64>().data()),
          int(new_nnz));
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
                    num_partitions_ * sizeof(int64_t), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);

    // 4. set output
    OpOutputList partitioned_values;
    OP_REQUIRES_OK(ctx,
                   ctx->output_list("partitioned_values", &partitioned_values));
    OpOutputList partitioned_indices;
    OP_REQUIRES_OK(
        ctx, ctx->output_list("partitioned_indices", &partitioned_indices));

    int64_t sub_start_offset = 0;
    for (int i = 0; i < num_partitions_; i++) {
      int64_t size = elements_offset_per_partition_[i] - sub_start_offset;

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
        int blocks = CalcBlocksLinearMapping(size, threads);

        const int partition_start_base =
            i == 0 ? 0 : partition_sizes_accumulate_[i - 1];
        GatherAndConvertToSubPartition<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<int64_t*>(values_sorted.flat<int64>().data()) +
                sub_start_offset,
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
            indices_sorted.flat<int64>().data() + 2 * sub_start_offset,
            size * 2 * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream);
      }
      sub_start_offset = elements_offset_per_partition_[i];
    }
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
                            .HostMemory("partition_shapes"),
                        FusedEmbeddingSparsePreLookUpGPU);

class FusedEmbeddingSparsePostLookUpGPU : public OpKernel {
 public:
  explicit FusedEmbeddingSparsePostLookUpGPU(OpKernelConstruction* ctx)
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
    const size_t batch_size = dense_shape_tensor->flat<int64>().data()[0];

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

    const size_t emb_vec_size = emb_shards[0].shape().dim_size(1);

    for (int i = 0; i < num_partitions_; i++) {
      const size_t sub_nnz = partitioned_indices[i].shape().dim_size(0);

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
              reinterpret_cast<const int64_t*>(
                  partitioned_indices[i].flat<int64>().data()),
              feature_nums->flat<int>().data(),
              grad_shard->flat<float>().data(), sub_nnz, emb_vec_size,
              max_norm_);
        } else if (combiner_ == "mean") {
          DistributeGradToShard<Mean><<<blocks, threads, 0, stream>>>(
              top_grad_tensor->flat<float>().data(),
              emb_shards[i].flat<float>().data(),
              reinterpret_cast<const int64_t*>(
                  partitioned_indices[i].flat<int64>().data()),
              feature_nums->flat<int>().data(),
              grad_shard->flat<float>().data(), sub_nnz, emb_vec_size,
              max_norm_);
        } else {
          DistributeGradToShard<Sum><<<blocks, threads, 0, stream>>>(
              top_grad_tensor->flat<float>().data(),
              emb_shards[i].flat<float>().data(),
              reinterpret_cast<const int64_t*>(
                  partitioned_indices[i].flat<int64>().data()),
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
    Name("FusedEmbeddingSparsePostLookUpGrad").Device(DEVICE_GPU),
    FusedEmbeddingSparsePostLookUpGradGPU);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA