#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)

#ifndef TENSORFLOW_CORE_KERNELS_DICE_COMPILE_UTIL_OP_H_
#define TENSORFLOW_CORE_KERNELS_DICE_COMPILE_UTIL_OP_H_

#include <type_traits>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/threadpool.h"

using namespace tensorflow;

#define INDEX_(x, id, j) (x + (row + id) * cols + j)
#define INDEX_1D(x, j) (x + j)

// A class for forced loop unrolling at compile time
template <int i>
struct compile_time_for {
  template <typename Lambda, typename... Args>
  inline static void op(const Lambda& function, Args... args) {
    compile_time_for<i - 1>::op(function, args...);
    function(std::integral_constant<int, i - 1>{}, args...);
  }
};
template <>
struct compile_time_for<1> {
  template <typename Lambda, typename... Args>
  inline static void op(const Lambda& function, Args... args) {
    function(std::integral_constant<int, 0>{}, args...);
  }
};
template <>
struct compile_time_for<0> {
  // 0 loops, do nothing
  template <typename Lambda, typename... Args>
  inline static void op(const Lambda& function, Args... args) {}
};

inline __m512 avx512_exp(const __m512& _x) {
  const __m512 p16f_1 = _mm512_set1_ps(1.0f);
  const __m512 p16f_half = _mm512_set1_ps(0.5f);
  const __m512 p16f_127 = _mm512_set1_ps(127.f);
  const __m512 p16f_exp_hi = _mm512_set1_ps(88.3762626647950f);
  const __m512 p16f_exp_lo = _mm512_set1_ps(-88.3762626647949f);
  const __m512 p16f_cephes_LOG2EF = _mm512_set1_ps(1.44269504088896341f);
  const __m512 p16f_cephes_exp_p0 = _mm512_set1_ps(1.9875691500E-4f);
  const __m512 p16f_cephes_exp_p1 = _mm512_set1_ps(1.3981999507E-3f);
  const __m512 p16f_cephes_exp_p2 = _mm512_set1_ps(8.3334519073E-3f);
  const __m512 p16f_cephes_exp_p3 = _mm512_set1_ps(4.1665795894E-2f);
  const __m512 p16f_cephes_exp_p4 = _mm512_set1_ps(1.6666665459E-1f);
  const __m512 p16f_cephes_exp_p5 = _mm512_set1_ps(5.0000001201E-1f);
  // Clamp x.
  __m512 x = _mm512_max_ps(_mm512_min_ps(_x, p16f_exp_hi), p16f_exp_lo);
  // Express exp(x) as exp(m*ln(2) + r), start by extracting
  // m = floor(x/ln(2) + 0.5).
  __m512 m = _mm512_floor_ps(_mm512_fmadd_ps(x, p16f_cephes_LOG2EF, p16f_half));
  // Get r = x - m*ln(2). If no FMA instructions are available, m*ln(2) is
  // subtracted out in two parts, m*C1+m*C2 = m*ln(2),
  // to avoid accumulating truncation errors.
  // Note that we don't use the "pmadd" function here to
  // ensure that a precision-preserving FMA instruction is used.
  const __m512 p16f_nln2 = _mm512_set1_ps(-0.6931471805599453f);
  __m512 r = _mm512_fmadd_ps(m, p16f_nln2, x);
  __m512 r2 = _mm512_mul_ps(r, r);
  // TODO(gonnet): Split into odd/even polynomials and try to exploit
  //               instruction-level parallelism.
  __m512 y = p16f_cephes_exp_p0;
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p1);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p2);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p3);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p4);
  y = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p5);
  y = _mm512_fmadd_ps(y, r2, r);
  y = _mm512_add_ps(y, p16f_1);
  // Build emm0 = 2^m.
  __m512i emm0 = _mm512_cvttps_epi32(_mm512_add_ps(m, p16f_127));
  emm0 = _mm512_slli_epi32(emm0, 23);
  // Return 2^m * exp(r).
  return _mm512_max_ps(_mm512_mul_ps(y, _mm512_castsi512_ps(emm0)), _x);
}

inline void compute_exp(__m512& x, __m512& mean, __m512& rvar, __m512& exp_t) {
  exp_t = _mm512_sub_ps(x, mean);
  exp_t = _mm512_mul_ps(exp_t, rvar);
  exp_t = avx512_exp(exp_t);
}

inline __m512 compute_px(__m512& exp_t, __m512& gamma) {
  const __m512 p16f_1 = _mm512_set1_ps(1.0f);
  __m512 denominator, numerator;
  denominator = _mm512_add_ps(exp_t, gamma);
  numerator = _mm512_add_ps(exp_t, p16f_1);
  return _mm512_div_ps(denominator, numerator);
}

#endif  // TENSORFLOW_CORE_KERNELS_DICE_COMPILE_UTIL_OP_H_

#endif  // AVX512F