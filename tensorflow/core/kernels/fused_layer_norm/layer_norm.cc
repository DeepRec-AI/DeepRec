#define EIGEN_USE_THREADS
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include <algorithm>
#include <numeric>
#include <vector>
#include <time.h>
#include <sys/time.h>
#include <immintrin.h>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/work_sharder.h"
#include "absl/strings/str_join.h"

using namespace tensorflow;
using namespace std;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

REGISTER_OP("LayerNormalization")
    .Input("input: T")
    .Input("gamma: float")
    .Input("beta: float")
    .Input("epsilon: float")
    //.Input("perm: int32")
    .Output("values: T")
    .Attr("T: {float,bfloat16}");

typedef Eigen::ThreadPoolDevice CPUDevice;

inline double get_total_ms_time(Time::time_point& start_time)
{
    return std::chrono::duration_cast<ns>(Time::now() - start_time).count() * 0.000001;
};

inline double get_total_us_time(Time::time_point& start_time)
{
    return std::chrono::duration_cast<ns>(Time::now() - start_time).count() * 0.001;
};

template <typename Tperm>
Status PermutationHelper(const Tensor& perm, const int dims,
                         std::vector<int32>* permutation) {
  auto Vperm = perm.vec<Tperm>();
  if (dims != Vperm.size()) {
    return errors::InvalidArgument("transpose expects a vector of size ", dims,
                                   ". But input(1) is a vector of size ",
                                   Vperm.size());
  }
  // using volatile instead of SubtleMustCopy here so that the
  // asynchrony boundary is permutation.
  const volatile Tperm* perm_begin =
      reinterpret_cast<const volatile Tperm*>(Vperm.data());
  *permutation = std::vector<int32>(perm_begin, perm_begin + dims);

  return Status::OK();
}

template <typename T>
struct LayerNormFunctor {
  static EIGEN_ALWAYS_INLINE Status
  Compute(OpKernelContext* context, const Tensor& in, std::vector<int32>& perm,
                        const typename TTypes<const float, 1>::Tensor gamma, const typename TTypes<const float, 1>::Tensor beta,
                        const float epsilon, Tensor* out) {
    const CPUDevice& d = context->eigen_device<CPUDevice>();
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    auto s = in.dim_size(in.dims() - 1);
    auto task_num = in.NumElements() / s;
    const auto in_ptr = in.flat<T>().data();
    const auto gamma_ptr = gamma.data();
    const auto beta_ptr = beta.data();
    auto out_ptr = out->flat<T>().data();

    auto DoCompute = [&](int start, int limit) {
      for (int32_t i = start; i < limit; i++) {
        int64_t offset = ComputeOffset(context, in, i, perm, s);
        ComputeUnit(context, in_ptr + i * s, gamma_ptr, beta_ptr, epsilon, s, out_ptr + offset);
      }
    };

    const int64_t unit_cost = 12 * s + 4; //evalation cost
    Shard(worker_threads.num_threads, worker_threads.workers, task_num,
          unit_cost, DoCompute);
    return Status::OK();
  }

  static EIGEN_ALWAYS_INLINE int64_t
  ComputeOffset(OpKernelContext* context, const Tensor& in, int64_t index,
                std::vector<int32>& perm, const int32_t unit) {
    int64_t res = index * unit;
    /*int64_t s = perm.size();
    std::vector<int32_t> org(s);
    std::vector<int32_t> changed(s);
    std::vector<int32_t> pos(s - 1);
    std::vector<int32_t> final_pos(s - 1);

    for (int i = 0; i < s; i++) {
      org[i] = in.dim_size(i);
      changed[i] = in.dim_size(perm[i]);
    }
    for (int i = s - 2; i >= 0; i--) {
      pos[i] = index % org[i];
      index = index / org[i];
    }

    for (int i = 0; i < s - 1; i++) {
      final_pos[i] = pos[perm[i]];
    }

    int64_t gamma = 1;
    for (int i = s - 2; i >= 0; i--) {
      res += final_pos[i] * gamma;
      gamma *= changed[i];
    }
    res *= unit;*/

    return res;
  }

  static EIGEN_ALWAYS_INLINE
  // x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
  float sum8(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
  }

  static EIGEN_ALWAYS_INLINE Status
  ComputeUnitNormal(OpKernelContext* context, const T* in,
              const float* gamma, const float* beta, const float epsilon,
              int32_t unit_size, T* out) {
    float sum = 0;
    if (std::is_same<T, bfloat16>::value) {
      for (int32_t i = 0; i < unit_size; i++) {
        sum += (float)in[i];
      }
    } else {
      for (int32_t i = 0; i < unit_size; i++) {
        sum += (float)in[i];
      }
    }
    float avg1 = sum / unit_size;
    sum = 0;
    for (int32_t i = 0; i < unit_size; i++) {
      float temp = ((float)in[i] - avg1);
      sum += temp * temp;
    }
    float avg2 = sum / unit_size;
    avg2 += epsilon;
    float res = 1/sqrt(avg2);

    if (std::is_same<T, bfloat16>::value) {
      for (int32_t i = 0; i < unit_size; i++) {
        float o = ((float)in[i] - avg1) * gamma[i] * res +  beta[i];
        out[i] = (bfloat16)o;
      }
    } else {
      for (int32_t i = 0; i < unit_size; i++) {
        float o = ((float)in[i] - avg1) * gamma[i] * res +  beta[i];
        out[i] = (bfloat16)o;
      }
    }
    return Status::OK();
  }

#if __AVX__
  static EIGEN_STRONG_INLINE __m256 Bf16ToF32AVX2(const __m128i a) {
    __m256i extend = _mm256_cvtepu16_epi32(a);
    return _mm256_castsi256_ps(_mm256_slli_epi32(extend, 16));
  }

  static EIGEN_STRONG_INLINE __m128i F32ToBf16AVX2(const __m256 a) {
    __m128i r;

    __m256i input = _mm256_castps_si256(a);
    // uint32_t lsb = (input >> 16);
    __m256i t = _mm256_srli_epi32(input, 16);
    // uint32_t lsb = lsb & 1;
    t = _mm256_and_si256(t, _mm256_set1_epi32(1));
    // uint32_t rounding_bias = 0x7fff + lsb;
    t = _mm256_add_epi32(t, _mm256_set1_epi32(0x7fff));
    // input += rounding_bias;
    t = _mm256_add_epi32(t, input);
    // input = input >> 16;
    t = _mm256_srli_epi32(t, 16);
    // Check NaN before converting back to bf16
    __m256 mask = _mm256_cmp_ps(a, a, _CMP_ORD_Q);
    __m256i nan = _mm256_set1_epi32(0x7fc0);
    t = _mm256_blendv_epi8(nan, t, _mm256_castps_si256(mask));
    // output.value = static_cast<uint16_t>(input);
    return _mm_packus_epi32(_mm256_extractf128_si256(t, 0),
                         _mm256_extractf128_si256(t, 1));
  }
#endif
  static EIGEN_ALWAYS_INLINE Status
  ComputeUnitAvx2(OpKernelContext* context, const float* in,
              const float* gamma, const float* beta, const float epsilon,
              int32_t unit_size, float* out) {
#if __AVX__
    float sum = 0;
    float square_sum = 0;
    int32_t avx2_size = unit_size / 8;

    __m256 vsum = _mm256_set1_ps(0);
    __m256 vsqare = _mm256_set1_ps(0);

    for (int32_t i = 0; i < avx2_size; i++) {
      __m256 float_256_in = _mm256_loadu_ps(in + i * 8);
      vsum = _mm256_add_ps(vsum, float_256_in);

      __m256 tmp = _mm256_mul_ps(float_256_in, float_256_in);
      vsqare = _mm256_add_ps(vsqare, tmp);
    }

    sum = sum8(vsum);
    square_sum = sum8(vsqare);

    // Mean
    float avg1 = sum / unit_size;
    __m256 float_256_avg_1 = _mm256_set1_ps(avg1);

    // Variance
    float res = 1 / sqrt(square_sum / unit_size - avg1 * avg1 + epsilon);
    __m256 float_256_res = _mm256_set1_ps(res);

    for (int32_t i = 0; i < avx2_size; i++) {
      __m256 float_256_in = _mm256_loadu_ps(in + i * 8);
      __m256 float_256_gamma = _mm256_loadu_ps(gamma + i * 8);
      __m256 float_256_beta = _mm256_loadu_ps(beta + i * 8);
      __m256 float_256_out = (float_256_in - float_256_avg_1) * float_256_gamma * float_256_res + float_256_beta;
      _mm256_storeu_ps(out + i * 8, float_256_out);
    }
#endif
    return Status::OK();
  }

  static EIGEN_ALWAYS_INLINE Status
  ComputeUnitAvx2(OpKernelContext* context, const bfloat16* in,
              const float* gamma, const float* beta, const float epsilon,
              int32_t unit_size, bfloat16* out) {
#if __AVX__
    float sum = 0;
    float square_sum = 0;
    int32_t avx2_size = unit_size / 8;

    __m256 vsum = _mm256_set1_ps(0);
    __m256 vsqare = _mm256_set1_ps(0);

    for (int32_t i = 0; i < avx2_size; i++) {
      __m128i bf_i = _mm_loadu_si128((__m128i_u*)(in + i * 8));
      __m256 float_256_in = Bf16ToF32AVX2(bf_i);

      vsum = _mm256_add_ps(vsum, float_256_in);

      __m256 tmp = _mm256_mul_ps(float_256_in, float_256_in);
      vsqare = _mm256_add_ps(vsqare, tmp);
    }

    sum = sum8(vsum);
    square_sum = sum8(vsqare);

    // Mean
    float avg1 = sum / unit_size;
    __m256 float_256_avg_1 = _mm256_set1_ps(avg1);

    // Variance
    float res = 1 / sqrt(square_sum / unit_size - avg1 * avg1 + epsilon);
    __m256 float_256_res = _mm256_set1_ps(res);

    for (int32_t i = 0; i < avx2_size; i++) {
      __m128i bf_i = _mm_loadu_si128((__m128i_u*)(in + i * 8));
      __m256 float_256_in = Bf16ToF32AVX2(bf_i);
      __m256 float_256_gamma = _mm256_loadu_ps(gamma + i * 8);
      __m256 float_256_beta = _mm256_loadu_ps(beta + i * 8);
      __m256 float_256_out = (float_256_in - float_256_avg_1) * float_256_gamma * float_256_res + float_256_beta;
      __m128i o = F32ToBf16AVX2(float_256_out);
      _mm_storeu_si128((__m128i_u*)(out + i * 8), o);
    }
#endif
    return Status::OK();
  }

#if __AVX512F__
  static EIGEN_STRONG_INLINE __m512 Bf16ToF32AVX3(const __m256i a) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a), 16));
  }

  static EIGEN_STRONG_INLINE __m256i F32ToBf16AVX3(const __m512 a) {
    __m256i r;
#if EIGEN_GNUC_AT_LEAST(10, 1)
    // Since GCC 10.1 supports avx512bf16 and C style explicit cast
    // (C++ static_cast is not supported yet), do converion via intrinsic
    // and register path for performance.
    r = (__m256i)(_mm512_cvtneps_pbh(a));
#else
    __m512i t;
    __m512i input = _mm512_castps_si512(a);
    __m512i nan = _mm512_set1_epi32(0x7fc0);

    // uint32_t lsb = (input >> 16) & 1;
    t = _mm512_and_si512(_mm512_srli_epi32(input, 16), _mm512_set1_epi32(1));
    // uint32_t rounding_bias = 0x7fff + lsb;
    t = _mm512_add_epi32(t, _mm512_set1_epi32(0x7fff));
    // input += rounding_bias;
    t = _mm512_add_epi32(t, input);
    // input = input >> 16;
    t = _mm512_srli_epi32(t, 16);

    // Check NaN before converting back to bf16
    __mmask16 mask = _mm512_cmp_ps_mask(a, a, _CMP_ORD_Q);
    t = _mm512_mask_blend_epi32(mask, nan, t);

    // output.value = static_cast<uint16_t>(input);
    r = _mm512_cvtepi32_epi16(t);
#endif // EIGEN_VECTORIZE_AVX512BF16

    return r;
  }
#endif
  static EIGEN_ALWAYS_INLINE Status
  ComputeUnitAvx512(OpKernelContext* context, const float* in,
              const float* gamma, const float* beta, const float epsilon,
              int32_t unit_size, float* out) {
#if __AVX512F__
    float sum = 0;
    float square_sum = 0;
    int32_t avx3_size = unit_size / 16;

    __m512 vsum = _mm512_set1_ps(0);
    __m512 vsqare = _mm512_set1_ps(0);

    for (int32_t i = 0; i < avx3_size; i++) {
      __m512 float_512_in = _mm512_loadu_ps(in + i * 16);
      vsum = _mm512_add_ps(vsum, float_512_in);

      __m512 tmp = _mm512_mul_ps(float_512_in, float_512_in);
      vsqare = _mm512_add_ps(vsqare, tmp);
    }

    sum = _mm512_reduce_add_ps(vsum);
    square_sum = _mm512_reduce_add_ps(vsqare);

    // Mean
    float avg1 = sum / unit_size;
    __m512 float_512_avg_1 = _mm512_set1_ps(avg1);

    // Variance
    float res = 1 / sqrt(square_sum / unit_size - avg1 * avg1 + epsilon);
    __m512 float_512_res = _mm512_set1_ps(res);

    for (int32_t i = 0; i < avx3_size; i++) {
      __m512 float_512_in = _mm512_loadu_ps(in + i * 16);
      __m512 float_512_gamma = _mm512_loadu_ps(gamma + i * 16);
      __m512 float_512_beta = _mm512_loadu_ps(beta + i * 16);
      __m512 float_512_out = (float_512_in - float_512_avg_1) * float_512_gamma * float_512_res + float_512_beta;
      _mm512_storeu_ps(out + i * 16, float_512_out);
    }
#endif
    return Status::OK();
  }

  static EIGEN_ALWAYS_INLINE Status
  ComputeUnitAvx512(OpKernelContext* context, const bfloat16* in,
              const float* gamma, const float* beta, const float epsilon,
              int32_t unit_size, bfloat16* out) {
#if __AVX512F__
    float sum = 0;
    float square_sum = 0;
    int32_t avx3_size = unit_size / 16;

    __m512 vsum = _mm512_set1_ps(0);
    __m512 vsqare = _mm512_set1_ps(0);

    for (int32_t i = 0; i < avx3_size; i++) {
      __m256i bf_i = _mm256_loadu_si256((__m256i_u*)(in + i * 16));
      __m512 float_512_in = Bf16ToF32AVX3(bf_i);
      vsum = _mm512_add_ps(vsum, float_512_in);

      __m512 tmp = _mm512_mul_ps(float_512_in, float_512_in);
      vsqare = _mm512_add_ps(vsqare, tmp);
    }

    sum = _mm512_reduce_add_ps(vsum);
    square_sum = _mm512_reduce_add_ps(vsqare);

    // Mean
    float avg1 = sum / unit_size;
    __m512 float_512_avg_1 = _mm512_set1_ps(avg1);

    // Variance
    float res = 1 / sqrt(square_sum / unit_size - avg1 * avg1 + epsilon);
    __m512 float_512_res = _mm512_set1_ps(res);

    for (int32_t i = 0; i < avx3_size; i++) {
      __m256i bf_i = _mm256_loadu_si256((__m256i_u*)(in + i * 16));
      __m512 float_512_in = Bf16ToF32AVX3(bf_i);
      __m512 float_512_gamma = _mm512_loadu_ps(gamma + i * 16);
      __m512 float_512_beta = _mm512_loadu_ps(beta + i * 16);
      __m512 float_512_out = (float_512_in - float_512_avg_1) * float_512_gamma * float_512_res + float_512_beta;
      __m256i o = F32ToBf16AVX3(float_512_out);
      _mm256_storeu_si256((__m256i_u*)(out + i * 16), o);
    }
#endif
    return Status::OK();
  }

  static EIGEN_ALWAYS_INLINE Status
  ComputeUnit(OpKernelContext* context, const T* in,
              const float* gamma, const float* beta, const float epsilon,
              int32_t unit_size, T* out) {
#if __AVX512F__
    if (unit_size & 0b1111) {
      ComputeUnitNormal(context, in, gamma, beta, epsilon, unit_size, out);
    } else {
      ComputeUnitAvx512(context, in, gamma, beta, epsilon, unit_size, out);
    }
#elif __AVX__
    if (unit_size & 0b0111) {
      ComputeUnitNormal(context, in, gamma, beta, epsilon, unit_size, out);
    } else {
      ComputeUnitAvx2(context, in, gamma, beta, epsilon, unit_size, out);
    }
#else
    ComputeUnitNormal(context, in, gamma, beta, epsilon, unit_size, out);
#endif
    return Status::OK();
  }
};

template <typename T>
class LayerNorm : public OpKernel {
 public:
  explicit LayerNorm(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    float epsilon = 0;
    OP_REQUIRES(context, num_inputs() >= 4,
                errors::InvalidArgument("input number >= 4, got ",
                                        num_inputs()));
    const auto& gamma_in = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(gamma_in.shape()),
                  errors::InvalidArgument("gamma must be vector, got shape ",
                                          gamma_in.shape().DebugString()));
    const auto& beta_in = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(beta_in.shape()),
                  errors::InvalidArgument("beta must be vector, got shape ",
                                          beta_in.shape().DebugString()));
    const auto& epsilon_in = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(epsilon_in.shape()),
                  errors::InvalidArgument("epsilon must be scalar, got shape ",
                                          epsilon_in.shape().DebugString()));
    /*const auto& perm_in = context->input(4);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(perm_in.shape()),
                  errors::InvalidArgument("perm must be vector, got shape ",
                                          perm_in.shape().DebugString()));*/
    epsilon = epsilon_in.scalar<float>()();
    const auto& gamma = gamma_in.vec<float>();
    const auto& beta = beta_in.vec<float>();
    const auto& input = context->input(0);
    OP_REQUIRES(context, input.dims() >= 1,
                errors::InvalidArgument("input must be >= 1-D, got shape ",
                                        input.shape().DebugString()));

    const int64 last_dim = input.dim_size(input.dims() - 1);
    OP_REQUIRES(context, last_dim == gamma.size() && last_dim == beta.size(),
                errors::InvalidArgument("input last dim  must equal to gamma and beta's dim, got shape ",
                                        input.shape().DebugString(), gamma_in.shape().DebugString(),
                                        beta_in.shape().DebugString()));

    std::vector<int32> permutation;
    /*const int dims = input.dims();
    if (perm_in.dtype() == DT_INT32) {
      OP_REQUIRES_OK(context, PermutationHelper<int32>(perm_in, dims, &permutation));
    } else {
      OP_REQUIRES_OK(context, PermutationHelper<int64>(perm_in, dims, &permutation));
    }
    TensorShape shape;

    // Check whether permutation is a permutation of integers of [0 .. dims).
    gtl::InlinedVector<bool, 8> bits(dims);
    bool is_identity = true;
    for (int i = 0; i < dims; ++i) {
      const int32 d = permutation[i];
      OP_REQUIRES(
        context, 0 <= d && d < dims,
        errors::InvalidArgument(d, " is out of range [0 .. ", dims, ")"));
      bits[d] = true;
      const auto dim_size = input.dim_size(d);
      shape.AddDim(dim_size);
      if (d != i) {
        is_identity = false;
      }
    }
    for (int i = 0; i < dims; ++i) {
      OP_REQUIRES(context, bits[i],
                errors::InvalidArgument(i, " is missing from {",
                                        absl::StrJoin(permutation, ","), "}."));
    }

    OP_REQUIRES(context, permutation[permutation.size() - 1] == dims - 1,
                errors::InvalidArgument("transpose should not change the last dim of input."));
    */
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));
    Status s = LayerNormFunctor<T>::Compute(
        context, input, permutation, gamma, beta, epsilon, output);

    OP_REQUIRES_OK(context, s);
  }
};

REGISTER_KERNEL_BUILDER(Name("LayerNormalization").Device(DEVICE_CPU).TypeConstraint<float>("T"), LayerNorm<float>);
REGISTER_KERNEL_BUILDER(Name("LayerNormalization").Device(DEVICE_CPU).TypeConstraint<bfloat16>("T"), LayerNorm<bfloat16>);
