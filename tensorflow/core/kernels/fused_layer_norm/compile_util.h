#ifndef TENSORFLOW_CORE_KERNELS_FUSED_LAYER_NORMALIZE_COMPILE_UTIL_OP_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_LAYER_NORMALIZE_COMPILE_UTIL_OP_H_

#include <type_traits>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"

using namespace tensorflow;
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
#ifdef __AVX512F__

template <int BLOCK_NUM>
inline __m512 reduce_sum_block(const __m512* v) {
  __m512 block_sum = _mm512_setzero_ps();
  auto reduce_sum = [&](auto idx) {
    block_sum = _mm512_add_ps(block_sum, v[idx]);
  };
  compile_time_for<BLOCK_NUM>::op(reduce_sum);
  return block_sum;
}

inline __m512 reduce_sum_block_ps(const __m512* v, int64 BLOCK_NUM) {
  switch (BLOCK_NUM) {
    case 1:
      return v[0];
    case 2:
      return reduce_sum_block<2>(v);
    case 3:
      return reduce_sum_block<3>(v);
    case 4:
      return reduce_sum_block<4>(v);
    case 5:
      return reduce_sum_block<5>(v);
    case 6:
      return reduce_sum_block<6>(v);
    case 7:
      return reduce_sum_block<7>(v);
    case 8:
      return reduce_sum_block<8>(v);
  }
}

static inline float horizontal_add(__m512 src) {
  __m512 tmp = _mm512_add_ps(
      src, _mm512_shuffle_f32x4(src, src, _MM_SHUFFLE(1, 0, 3, 2)));
  __m128 r = _mm512_castps512_ps128(_mm512_add_ps(
      tmp, _mm512_shuffle_f32x4(tmp, tmp, _MM_SHUFFLE(2, 3, 0, 1))));
  r = _mm_hadd_ps(r, r);
  return _mm_cvtss_f32(_mm_hadd_ps(r, r));
}

void add_n(const float* src, float* dst, int rows, int64 cols) {
  int64 c = 0;
  for (; c + 15 < cols; c += 16) {
    __m512 sum = _mm512_set1_ps(0);
    auto offset = c;
    for (int r = 0; r < rows; ++r) {
      sum = _mm512_add_ps(_mm512_loadu_ps(src + offset), sum);
      offset += cols;
    }
    _mm512_storeu_ps(dst + c, sum);
  }
  // Remain data
  for (; c < cols; ++c) {
    float sum = 0;
    auto offset = c;
    for (int r = 0; r < rows; ++r) {
      sum += src[offset];
      offset += cols;
    }
    dst[c] = sum;
  }
}

#endif  //AVX512
#endif  // TENSORFLOW_CORE_KERNELS_FUSED_LAYER_NORMALIZE_COMPILE_UTIL_OP_H_