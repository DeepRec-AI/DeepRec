#ifndef TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_FUSED_EMBEDDING_COMMON_CU_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_FUSED_EMBEDDING_COMMON_CU_H_

#if GOOGLE_CUDA

namespace tensorflow {

namespace {
enum Combiner { Mean, Sum, Sqrtn };

template <Combiner combiner>
__forceinline__ __device__ float Combine(const float in,
                                         const int feature_num);

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
}  // namespace

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_FUSED_EMBEDDING_COMMON_CU_H_