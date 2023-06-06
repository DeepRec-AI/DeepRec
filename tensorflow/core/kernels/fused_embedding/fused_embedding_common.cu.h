#ifndef TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_FUSED_EMBEDDING_COMMON_CU_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_FUSED_EMBEDDING_COMMON_CU_H_

#if GOOGLE_CUDA

#define CK_CUDA_THROW_(x)                                                      \
  do {                                                                         \
    cudaError_t retval = (x);                                                  \
    if (retval != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("Runtime error: ") +                \
                               (cudaGetErrorString(retval)) + " " + __FILE__ + \
                               ":" + std::to_string(__LINE__) + " \n");        \
    }                                                                          \
  } while (0)

namespace tensorflow {

namespace {

inline int CalcBlocksLinearMapping(const int problem_size, const int threads) {
  return problem_size % threads == 0 ? (problem_size / threads)
                                     : (problem_size / threads + 1);
}

struct IndicePair {
  int64_t row_in_batch;
  int64_t entry_in_column;
};

enum Combiner { Mean, Sum, Sqrtn };

template <Combiner combiner, typename T>
__forceinline__ __device__ float Combine(const float in, const T feature_num);

template <>
__forceinline__ __device__ float Combine<Sqrtn, int>(const float in,
                                                const int feature_num) {
  return in / sqrtf(feature_num);
}

template <>
__forceinline__ __device__ float Combine<Mean, int>(const float in,
                                               const int feature_num) {
  return in / feature_num;
}

template <>
__forceinline__ __device__ float Combine<Sum, int>(const float in,
                                              const int feature_num) {
  return in;
}

template <>
__forceinline__ __device__ float Combine<Sqrtn, float>(const float in,
                                                const float feature_num) {
  return in / sqrtf(feature_num);
}

template <>
__forceinline__ __device__ float Combine<Mean, float>(const float in,
                                               const float feature_num) {
  return in / feature_num;
}

template <>
__forceinline__ __device__ float Combine<Sum, float>(const float in,
                                              const float feature_num) {
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
