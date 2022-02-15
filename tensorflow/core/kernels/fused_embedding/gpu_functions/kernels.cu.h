#ifndef TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_FUNCTIONS_KERNELS_CU_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_FUNCTIONS_KERNELS_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/fused_embedding/fused_embedding.cu.h"

namespace tensorflow {

namespace fused_embedding {

enum Combiner { Mean, Sum, Sqrtn };

template <Combiner combiner>
__forceinline__ __device__ float Combine(const float in, const int feature_num);

template <>
__forceinline__ __device__ float Combine<Sqrtn>(const float in,
                                                const int feature_num) {
  return in / sqrtf(feature_num);
}

template <>
__forceinline__ __device__ float Combine<Mean>(const float in,
                                               const int feature_num) {
  return in / feature_num;
}

template <>
__forceinline__ __device__ float Combine<Sum>(const float in,
                                              const int feature_num) {
  return in;
}

template <Combiner combiner>
__forceinline__ __device__ float CombineGrad(const float grad,
                                             const int feature_num);

template <>
__forceinline__ __device__ float CombineGrad<Sqrtn>(const float grad,
                                                    const int feature_num) {
  return grad / sqrtf(feature_num);
}

template <>
__forceinline__ __device__ float CombineGrad<Mean>(const float grad,
                                                   const int feature_num) {
  return grad / feature_num;
}

template <>
__forceinline__ __device__ float CombineGrad<Sum>(const float grad,
                                                  const int feature_num) {
  return grad;
}

using GPUDevice = Eigen::GpuDevice;

void InitFlagsToOneInt4(const GPUDevice& d, int length, int* flags);

void DetectInvalid(const GPUDevice& d, const int64_t* values, const int64_t nnz,
                   int* invalid_id_flag);

void FusedMultiFunctional(const GPUDevice& d, const IndicePair* indices,
                          const int64_t* values, const int64_t nnz,
                          const int64_t batch_size, const bool prune_invalid_id,
                          const int64_t default_id, int* row_emptiness_flag,
                          int* invalid_id_flag, IndicePair* tmp_indices_buffer,
                          int64_t* values_extended);

void CalcElementsOffsetPerPartition(const GPUDevice& d, int num_partitions,
                                    const int64_t* values_sorted,
                                    int64_t* partition_sizes_accumulate,
                                    int64_t* elements_offset_per_partition,
                                    int nnz);

void GatherAndConvertToSubPartition(const GPUDevice& d,
                                    const int64_t* sub_values_sorted,
                                    int64_t* sub_partitioned_values,
                                    const int64_t partition_start_base,
                                    const int64_t partition_size);

template <typename T>
void RangeInit(const GPUDevice& d, const int64_t length, T* out);

void SumUpEmbeddingShard(const GPUDevice& d, const size_t sub_nnz,
                         const float* emb_shard,
                         const int64_t* partitioned_indice, float* emb_vectors,
                         int* feature_nums, const float max_norm,
                         const int emb_vec_size);

template <Combiner combiner>
void ApplyCombiner(const GPUDevice& d, const int batch_size,
                   const int emb_vec_size, float* emb_vectors,
                   const int* row_emptiness_flag, const bool set_empty_row_zero,
                   const int* feature_nums);

template <Combiner combiner>
void DistributeGradToShard(const GPUDevice& d, const float* top_grad,
                           const float* emb_shard,
                           const int64_t* partitioned_indice,
                           const int* feature_nums,
                           const int* row_emptiness_flag,
                           const bool set_empty_row_zero, float* grad_shard,
                           const int64_t sub_nnz, const int64_t emb_vec_size,
                           const float max_norm);

}  // namespace fused_embedding

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_GPU_FUNCTIONS_KERNELS_CU_H_