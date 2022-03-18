

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/fused_embedding/gpu/functions/kernels.cu.h"

#include "tensorflow/core/kernels/fused_embedding/gpu/common.cu.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

namespace fused_embedding {

using GPUDevice = Eigen::GpuDevice;

#define LINER_MAPPING_THREADS 128

inline int CalcBlocksLinearMapping(const int problem_size, const int threads) {
  return problem_size % threads == 0 ? (problem_size / threads)
                                     : (problem_size / threads + 1);
}

__global__ void InitFlagsToOneInt4Kernel(int length, int* flags) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (4 * offset + 3 < length) {
    *((int4*)(flags + 4 * offset)) = make_int4(1, 1, 1, 1);
  } else if (4 * offset < length) {
    for (int i = 0; i < length - 4 * offset; i++) {
      flags[4 * offset + i] = 1;
    }
  }
}

void InitFlagsToOneInt4(const GPUDevice& d, int length, int* flags) {
  const int threads = LINER_MAPPING_THREADS;
  const int blocks = CalcBlocksLinearMapping(length, threads * 4);
  TF_CHECK_OK(GpuLaunchKernel(InitFlagsToOneInt4Kernel, blocks, threads, 0,
                              d.stream(), length, flags));
}

__global__ void DetectInvalidKernel(const int64_t* values, const int64_t nnz,
                                    int* invalid_id_flag) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < nnz) {
    const int64_t value = values[offset];
    if (value < 0) {
      atomicAnd(invalid_id_flag + offset, 0);
    }
  }
}

void DetectInvalid(const GPUDevice& d, const int64_t* values, const int64_t nnz,
                   int* invalid_id_flag) {
  const int threads = LINER_MAPPING_THREADS;
  const int blocks = CalcBlocksLinearMapping(nnz, threads);
  TF_CHECK_OK(GpuLaunchKernel(DetectInvalidKernel, blocks, threads, 0,
                              d.stream(), values, nnz, invalid_id_flag));
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

void FusedMultiFunctional(const GPUDevice& d, const IndicePair* indices,
                          const int64_t* values, const int64_t nnz,
                          const int64_t batch_size, const bool prune_invalid_id,
                          const int64_t default_id, int* row_emptiness_flag,
                          int* invalid_id_flag, IndicePair* tmp_indices_buffer,
                          int64_t* values_extended) {
  const int threads = LINER_MAPPING_THREADS;
  const int blocks = CalcBlocksLinearMapping(nnz + batch_size, threads);
  TF_CHECK_OK(GpuLaunchKernel(
      FusedMultiFunctionalKernel, blocks, threads, 0, d.stream(), indices,
      values, nnz, batch_size, prune_invalid_id, default_id, row_emptiness_flag,
      invalid_id_flag, tmp_indices_buffer, values_extended));
}

template <typename T>
__global__ void RangeInitKernel(const int64_t length, T* out) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    out[idx] = T(idx);
  }
}

template <typename T>
void RangeInit(const GPUDevice& d, const int64_t length, T* out) {
  const int threads = LINER_MAPPING_THREADS;
  const int blocks = CalcBlocksLinearMapping(length, threads);
  TF_CHECK_OK(GpuLaunchKernel(RangeInitKernel<T>, blocks, threads, 0,
                              d.stream(), length, out));
}

template void RangeInit<int64_t>(const GPUDevice& d, const int64_t length,
                                 int64_t* out);

__global__ void SumUpEmbeddingShardSinglePartitionKernel(
    const float* emb_shard, const int64_t* indices_before_unique,
    const int* unique_idxs, const int nnz, const float max_norm,
    const int emb_vec_size, float* emb_vectors, int* feature_nums) {
  __shared__ float l2_sum[1];

  if (blockIdx.x < nnz) {
    const int64_t row_in_batch = indices_before_unique[2 * blockIdx.x];
    const int unique_id = unique_idxs[blockIdx.x];
    float emb_element = emb_shard[unique_id * emb_vec_size + threadIdx.x];
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
}

void SumUpEmbeddingShardSinglePartition(const GPUDevice& d,
                                        const float* emb_shard,
                                        const int64_t* indices_before_unique,
                                        const int* unique_idxs, const int nnz,
                                        const float max_norm,
                                        const int emb_vec_size,
                                        float* emb_vectors, int* feature_nums) {
  const int blocks = nnz;
  const int threads = emb_vec_size;
  TF_CHECK_OK(GpuLaunchKernel(SumUpEmbeddingShardSinglePartitionKernel, blocks,
                              threads, 0, d.stream(), emb_shard,
                              indices_before_unique, unique_idxs, nnz, max_norm,
                              emb_vec_size, emb_vectors, feature_nums));
}

__global__ void SumUpEmbeddingShardMultiPartitionKernel(
    const void* const* emb_shard_ptrs, const int* partition_permutation,
    const int64_t* indices_before_unique, const int* unique_idxs, const int nnz,
    const float max_norm, const int emb_vec_size, float* emb_vectors,
    int* feature_nums) {
  __shared__ float l2_sum[1];

  if (blockIdx.x < nnz) {
    const int64_t row_in_batch = indices_before_unique[2 * blockIdx.x];
    const int unique_id = unique_idxs[blockIdx.x];
    const int partition_id = partition_permutation[2 * unique_id];
    const int64_t offset_in_partition =
        partition_permutation[2 * unique_id + 1];

    float emb_element =
        ((const float*)(emb_shard_ptrs[partition_id]))[offset_in_partition *
                                                           emb_vec_size +
                                                       threadIdx.x];
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
}

void SumUpEmbeddingShardMultiPartition(
    const GPUDevice& d, const void* const* emb_shard_ptrs,
    const int* partition_permutation, const int64_t* indices_before_unique,
    const int* unique_idxs, const int nnz, const float max_norm,
    const int emb_vec_size, float* emb_vectors, int* feature_nums) {
  const int blocks = nnz;
  const int threads = emb_vec_size;
  TF_CHECK_OK(GpuLaunchKernel(
      SumUpEmbeddingShardMultiPartitionKernel, blocks, threads, 0, d.stream(),
      emb_shard_ptrs, partition_permutatio, indices_before_unique, unique_idxs,
      nnz, max_norm, emb_vec_size, emb_vectors, feature_nums));
}

template <Combiner combiner>
__global__ void ApplyCombinerKernel(const int* row_emptiness_flag,
                                    const bool set_empty_row_zero,
                                    int* feature_nums, float* emb_vectors) {
  const int offset = blockIdx.x * blockDim.x + threadIdx.x;
  const int feature_num = feature_nums[blockIdx.x];
  if (set_empty_row_zero) {
    if (row_emptiness_flag[blockIdx.x]) {
      feature_nums[blockIdx.x] = 0;
      emb_vectors[offset] = 0.0f;
      return;
    }
  }
  const float emb_element = emb_vectors[offset];
  emb_vectors[offset] = Combine<combiner>(emb_element, feature_num);
}

template <Combiner combiner>
void ApplyCombiner(const GPUDevice& d, const int batch_size,
                   const int emb_vec_size, const int* row_emptiness_flag,
                   const bool set_empty_row_zero, int* feature_nums,
                   float* emb_vectors) {
  const int blocks = batch_size;
  const int threads = emb_vec_size;
  TF_CHECK_OK(GpuLaunchKernel(ApplyCombinerKernel<combiner>, blocks, threads, 0,
                              d.stream(), row_emptiness_flag,
                              set_empty_row_zero, feature_nums, emb_vectors));
}

template void ApplyCombiner<Sqrtn>(const GPUDevice& d, const int batch_size,
                                   const int emb_vec_size,
                                   const int* row_emptiness_flag,
                                   const bool set_empty_row_zero,
                                   int* feature_nums, float* emb_vectors);
template void ApplyCombiner<Mean>(const GPUDevice& d, const int batch_size,
                                  const int emb_vec_size,
                                  const int* row_emptiness_flag,
                                  const bool set_empty_row_zero,
                                  int* feature_nums, float* emb_vectors);
template void ApplyCombiner<Sum>(const GPUDevice& d, const int batch_size,
                                 const int emb_vec_size,
                                 const int* row_emptiness_flag,
                                 const bool set_empty_row_zero,
                                 int* feature_nums, float* emb_vectors);

template <Combiner combiner>
__global__ void DistributeGradToShardSinglePartitionKernel(
    const float* top_grad, const float* emb_shard,
    const int64_t* indices_before_unique, const int64_t* unique_idxs,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const int* row_emptiness_flag, float* grad_shard) {
  __shared__ float l2_sum[1];
  float l2_norm = -1.0f;

  if (blockIdx.x < nnz) {
    const int64_t row_in_batch = indices_before_unique[2 * blockIdx.x];
    const int unique_id = unique_idxs[blockIdx.x];

    if (set_empty_row_zero && row_emptiness_flag[row_in_batch]) {
      return;
    }
    if (max_norm >= 0.0f) {
      const float emb_element =
          emb_shard[unique_id * emb_vec_size + threadIdx.x];
      if (threadIdx.x == 0) {
        l2_sum[0] = 0.0f;
      }
      __syncthreads();
      atomicAdd(l2_sum, emb_element * emb_element);
      __syncthreads();
      l2_norm = sqrtf(l2_sum[0]);
    }

    float grad = top_grad[row_in_batch * emb_vec_size + threadIdx.x];
    const int feature_num = feature_nums[row_in_batch];
    grad = CombineGrad<combiner>(grad, feature_num);
    if (max_norm >= 0.0f && l2_norm > max_norm) {
      grad *= max_norm / l2_norm;
    }

    grad_shard[unique_id * emb_vec_size + threadIdx.x] = grad;
  }
}

template <Combiner combiner>
void DistributeGradToShardSinglePartition(
    const GPUDevice& d, const float* top_grad, const float* emb_shard,
    const int64_t* indices_before_unique, const int64_t* unique_idxs,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const int* row_emptiness_flag, float* grad_shard) {
  const int blocks = nnz;
  const int threads = emb_vec_size;
  TF_CHECK_OK(GpuLaunchKernel(
      DistributeGradToShardSinglePartitionKernel<combiner>, blocks, threads, 0,
      d.stream(), top_grad, emb_shard, indices_before_unique, unique_idxs, nnz,
      emb_vec_size, max_norm, set_empty_row_zero, feature_nums,
      row_emptiness_flag, grad_shard));
}

template void DistributeGradToShardSinglePartition<Sqrtn>(
    const GPUDevice& d, const float* top_grad, const float* emb_shard,
    const int64_t* indices_before_unique, const int64_t* unique_idxs,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const int* row_emptiness_flag, float* grad_shard);

template void DistributeGradToShardSinglePartition<Mean>(
    const GPUDevice& d, const float* top_grad, const float* emb_shard,
    const int64_t* indices_before_unique, const int64_t* unique_idxs,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const int* row_emptiness_flag, float* grad_shard);

template void DistributeGradToShardSinglePartition<Sum>(
    const GPUDevice& d, const float* top_grad, const float* emb_shard,
    const int64_t* indices_before_unique, const int64_t* unique_idxs,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const int* row_emptiness_flag, float* grad_shard);

template <Combiner combiner>
__global__ void DistributeGradToShardMultiPartitionKernel(
    const float* top_grad, const void* const* emb_shard_ptrs,
    const int* partition_permutation, const int64_t* indices_before_unique,
    const int64_t* unique_idxs, const int nnz, const int emb_vec_size,
    const float max_norm, const bool set_empty_row_zero,
    const int* feature_nums, const int* row_emptiness_flag,
    void** grad_shard_ptrs) {
  __shared__ float l2_sum[1];
  float l2_norm = -1.0f;

  if (blockIdx.x < nnz) {
    const int64_t row_in_batch = indices_before_unique[2 * blockIdx.x];
    if (set_empty_row_zero && row_emptiness_flag[row_in_batch]) {
      return;
    }
    const int unique_id = unique_idxs[blockIdx.x];
    const int partition_id = partition_permutation[2 * unique_id];
    const int64_t offset_in_partition =
        partition_permutation[2 * unique_id + 1];

    if (max_norm >= 0.0f) {
      float emb_element =
          ((const float*)(emb_shard_ptrs[partition_id]))[offset_in_partition *
                                                             emb_vec_size +
                                                         threadIdx.x];
      if (threadIdx.x == 0) {
        l2_sum[0] = 0.0f;
      }
      __syncthreads();
      atomicAdd(l2_sum, emb_element * emb_element);
      __syncthreads();
      l2_norm = sqrtf(l2_sum[0]);
    }

    float grad = top_grad[row_in_batch * emb_vec_size + threadIdx.x];
    const int feature_num = feature_nums[row_in_batch];
    grad = CombineGrad<combiner>(grad, feature_num);
    if (max_norm >= 0.0f && l2_norm > max_norm) {
      grad *= max_norm / l2_norm;
    }

    atomicAdd(((float*)(grad_shard_ptrs[partition_id])) +
                  offset_in_partition * emb_vec_size + threadIdx.x,
              grad);
  }
}

template <Combiner combiner>
void DistributeGradToShardMultiPartition(
    const GPUDevice& d, const float* top_grad,
    const void* const* emb_shard_ptrs, const int* partition_permutation,
    const int64_t* indices_before_unique, const int64_t* unique_idxs,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const int* row_emptiness_flag, void** grad_shard_ptrs) {
  const int blocks = nnz;
  const int threads = emb_vec_size;
  TF_CHECK_OK(GpuLaunchKernel(
      DistributeGradToShardMultiPartitionKernel<combiner>, blocks, threads, 0,
      d.stream(), top_grad, emb_shard_ptrs, partition_permutation,
      indices_before_unique, unique_idxs, nnz, emb_vec_size, max_norm,
      set_empty_row_zero, feature_nums, row_emptiness_flag, grad_shard_ptrs));
}

template void DistributeGradToShardMultiPartition<Sum>(
    const GPUDevice& d, const float* top_grad,
    const void* const* emb_shard_ptrs, const int* partition_permutation,
    const int64_t* indices_before_unique, const int64_t* unique_idxs,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const int* row_emptiness_flag, void** grad_shard_ptrs);

template void DistributeGradToShardMultiPartition<Mean>(
    const GPUDevice& d, const float* top_grad,
    const void* const* emb_shard_ptrs, const int* partition_permutation,
    const int64_t* indices_before_unique, const int64_t* unique_idxs,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const int* row_emptiness_flag, void** grad_shard_ptrs);

template void DistributeGradToShardMultiPartition<Sqrt>(
    const GPUDevice& d, const float* top_grad,
    const void* const* emb_shard_ptrs, const int* partition_permutation,
    const int64_t* indices_before_unique, const int64_t* unique_idxs,
    const int nnz, const int emb_vec_size, const float max_norm,
    const bool set_empty_row_zero, const int* feature_nums,
    const int* row_emptiness_flag, void** grad_shard_ptrs);

}  // namespace fused_embedding

}  // namespace tensorflow

#endif  // GOOGLE_CUDA