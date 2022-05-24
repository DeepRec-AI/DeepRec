/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_KERNELS_BIAS_OP_CPU_H_
#define TENSORFLOW_KERNELS_BIAS_OP_CPU_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

#ifdef INTEL_MKL
#include "dnnl.hpp"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace {
template <class T>
struct AccumulatorType {
  typedef T type;
};

// float is faster on the CPU than half, and also more precise,
// so use float for the temporary accumulators.
template <>
struct AccumulatorType<Eigen::half> {
  typedef float type;
};

// Sum each rows of an 2D array by (m * n) into a single row vector 'Y'.
// The parameter 'lda' specifies the first dimension of A in memory.
// If 'overwrite' is false, values in 'Y' would be summed toggether.
template <typename T>
void SumIntoOneRow(const T* A, int m, int n, int lda, T* Y,
                   bool overwrite = true) {
  typedef typename AccumulatorType<T>::type AccT;
  // Directly sum for (column == 1 or 2) cases.
  // NOTE(zycao): Acctually we expect these special cases could be auto
  // optimized by compilers.
  if (n == 1) {
    AccT sum = overwrite ? static_cast<AccT>(0) : static_cast<AccT>(Y[0]);
    for (int i = 0; i < m; ++i) {
      //_mm512_reduce_add_ps
      sum += static_cast<AccT>(A[i]);
    }
    Y[0] = static_cast<T>(sum);
    return;
  }
  if (n == 2) {
    AccT sum0 = overwrite ? static_cast<AccT>(0) : static_cast<AccT>(Y[0]);
    AccT sum1 = overwrite ? static_cast<AccT>(0) : static_cast<AccT>(Y[1]);
    for (int i = 0; i < m * 2; i += 2) {
      sum0 += static_cast<AccT>(A[i]);
      sum1 += static_cast<AccT>(A[i+1]);
    }
    Y[0] = static_cast<T>(sum0);
    Y[1] = static_cast<T>(sum1);
    return;
  }

  // Normal cases.
  if (overwrite) {
    for (int j = 0; j < n; ++j) {
      Y[j] = static_cast<T>(0);
    }
  }
  if (std::is_same<T, AccT>::value) {
    for (int i = 0; i < m; ++i) {
      int offset = i * lda;
      for (int j = 0; j < n; ++j) {
        Y[j] += A[offset + j];
      }
    }
    return;
  }

  // Eigen::half case.
  // NOTE(zycao): We have to use a temp array for Eigen::half.
  AccT* sum = new AccT[n];
  for (int j = 0; j < n; ++j) {
    sum[j] = static_cast<AccT>(Y[j]);
  }
  for (int i = 0; i < m; ++i) {
    int offset = i * lda;
    for (int j = 0; j < n; ++j) {
      sum[j] += static_cast<AccT>(A[offset + j]);
    }
  }
  for (int j = 0; j < n; ++j) {
    Y[j] = static_cast<T>(sum[j]);
  }
  delete[] sum;
}

constexpr int block_size_avx512 = 16;
constexpr int block_size_avx2 = 8;

void OneColumnWiseReduction(const float* A, int m, int n, int lda,
                            float* Y, bool overwrite = true) {
#if defined(__GNUC__) && (__GNUC__ >6)
#ifdef __AVX512F__
  if (m >= block_size_avx512) {
    int block_num = m / block_size_avx512;
    int remain_num = m % block_size_avx512;

    register float sum = overwrite ? 0 : Y[0];
    __m512 val;
    float* pos = const_cast<float*>(A);
    for (int i = 0; i < block_num; ++i) {
      val = _mm512_loadu_ps(pos);
      sum += _mm512_reduce_add_ps(val);
      pos += block_size_avx512;
    }

    if (remain_num != 0) {
      int block_end = block_num * block_size_avx512;
      __mmask16 mask = 0xffff >> (block_size_avx512 - remain_num);
      pos = const_cast<float*>(A + block_end);
      val = _mm512_loadu_ps(pos);
      sum += _mm512_mask_reduce_add_ps(mask, val);
    }
    Y[0] = sum;
    return;
  }
#endif
#endif
  register float sum = overwrite ? 0 : Y[0];
  for (int i = 0; i < m; ++i) {
    sum += A[i];
  }
  Y[0] = sum;
}

void TwoColumnWiseReduction(const float* A, int m, int n, int lda, float* Y,
                            bool overwrite = true) {
#if defined(__GNUC__) && (__GNUC__ >6)
#ifdef __AVX512F__
  int total = m * 2;
  if (total >= block_size_avx512) {
    int block_num = total / block_size_avx512;
    int remain_num = total % block_size_avx512;

    register float sum0 = overwrite ? 0 : Y[0];
    register float sum1 = overwrite ? 0 : Y[1];

    __m512 val;
    __mmask16 mask0 = 0x5555;
    __mmask16 mask1 = 0xaaaa;
    float* pos = const_cast<float*>(A);
    for (int i = 0; i < block_num; ++i) {
      val = _mm512_loadu_ps(pos);
      sum0 += _mm512_mask_reduce_add_ps(mask0, val);
      sum1 += _mm512_mask_reduce_add_ps(mask1, val);
      pos += block_size_avx512;
    }

    if (remain_num != 0) {
      int block_end = block_num * block_size_avx512;
      mask0 = 0x5555 >> (block_size_avx512 - remain_num);
      mask1 = 0xaaaa >> (block_size_avx512 - remain_num);
      pos = const_cast<float*>(A + block_end);
      val = _mm512_loadu_ps(pos);
      sum0 += _mm512_mask_reduce_add_ps(mask0, val);
      sum1 += _mm512_mask_reduce_add_ps(mask1, val);
    }
    Y[0] = sum0;
    Y[1] = sum1;
    return;
  }
#endif
#endif
  register float sum0 = overwrite ? 0 : Y[0];
  register float sum1 = overwrite ? 0 : Y[1];
  for (int i = 0; i < m * 2; i += 2) {
    sum0 += A[i];
    sum1 += A[i+1];
  }
  Y[0] = sum0;
  Y[1] = sum1;
}

void MultipleColumnWiseReduction(const float* A, int m, int n, int lda,
                                float* Y, bool overwrite = true) {
#ifdef __AVX512F__
  if (n >= block_size_avx512) {
    int block_num = n / block_size_avx512;
    int remain_num = n % block_size_avx512;
    __m512 zero = _mm512_setzero_ps();
    int offset_block_end = block_num * block_size_avx512;

    __mmask16 mask = 0xffff >> (block_size_avx512 - remain_num);
    if (overwrite) {
      float* pos = Y;
      for (int j = 0; j < block_num; ++j) {
        _mm512_storeu_ps(pos, zero);
        pos += block_size_avx512;
      }

      if (remain_num != 0) {
        _mm512_mask_storeu_ps(Y + offset_block_end, mask, zero);
      }
    }

    __m512 sum, val;
    register float* pos_A = const_cast<float*>(A);
    register float* pos_Y = const_cast<float*>(Y);
    for (int i = 0; i < m; ++i) {
      pos_Y = const_cast<float*>(Y);
      for (int j = 0; j < block_num; ++j) {
        val = _mm512_loadu_ps(pos_A);
        sum = _mm512_loadu_ps(pos_Y);
        sum = _mm512_add_ps(sum, val);
        _mm512_storeu_ps(pos_Y, sum);

        pos_A += block_size_avx512;
        pos_Y += block_size_avx512;
      }
      if (remain_num != 0) {
        val = _mm512_mask_loadu_ps(zero, mask, pos_A);
        sum = _mm512_mask_loadu_ps(zero, mask, pos_Y);
        sum = _mm512_mask_add_ps(zero, mask, sum, val);
        _mm512_mask_storeu_ps(pos_Y, mask, sum);

        pos_A += remain_num;
        pos_Y += remain_num;
      }
    }
    return;
  }
#endif
  // Normal cases.
  if (overwrite) {
    for (int j = 0; j < n; ++j) {
      Y[j] = 0;
    }
  }
  for (int i = 0; i < m; ++i) {
    int offset = i * lda;
    for (int j = 0; j < n; ++j) {
      Y[j] += A[offset + j];
    }
  }
  return;
}

void SumIntoOneRow(const float* A, int m, int n, int lda, float* Y,
                         bool overwrite = true) {
  if (n == 1) {
    OneColumnWiseReduction(A, m, n, lda, Y, overwrite);
  }
  else if (n == 2) {
    TwoColumnWiseReduction(A, m, n, lda, Y, overwrite);
  }
  else {
    MultipleColumnWiseReduction(A, m, n, lda, Y, overwrite);
  }
}

#ifdef __AVX512F__
void ColumnParallel_512(const CPUDevice& d, float* input_data, float* output_data,
    int sum_size, int channel) {
  int block_num = channel / block_size_avx512;
  int remain_num = channel % block_size_avx512;

  auto do_work = [input_data, output_data, sum_size, channel, block_size_avx512]
    (int64 start, int64 end) {
    __m512 sum, val;
    for (int j = start; j < end; ++j) {
      auto block_offset = j * block_size_avx512;
      float* row_pos = const_cast<float*>(input_data + block_offset);
      sum = _mm512_loadu_ps(row_pos);
      for (int i = 1; i < sum_size; ++i) {
        row_pos += channel;
        val = _mm512_loadu_ps(row_pos);
        sum = _mm512_add_ps(sum, val);
      }
      _mm512_storeu_ps(output_data + block_offset, sum);
    }
  };
  auto cost = Eigen::TensorOpCost(sizeof(float) * sum_size,
                                  sizeof(float),
                                  sum_size);
  d.parallelFor(block_num, cost, do_work);

  if (remain_num == 0) {
    return;
  }
  int offset_block_end = block_num * block_size_avx512;
  __m512 zero = _mm512_setzero_ps();
  __mmask16 mask = 0xffff >> (block_size_avx512 - remain_num);
  __m512 sum, val;
  float* row_pos = const_cast<float*>(input_data + offset_block_end);
  sum = _mm512_mask_loadu_ps(zero, mask, row_pos);
  for (int i = 1; i < sum_size; ++i) {
    row_pos += channel;
    val = _mm512_mask_loadu_ps(zero, mask, row_pos);
    sum = _mm512_mask_add_ps(zero, mask, sum, val);
  }
  _mm512_mask_storeu_ps(output_data + offset_block_end, mask, sum);
}

void ColumnParallel_256(const CPUDevice& d, float* input_data, float* output_data,
    int sum_size, int channel) {
  int block_num = channel / block_size_avx2;
  int remain_num = channel % block_size_avx2;

  auto do_work = [input_data, output_data, sum_size, channel, block_size_avx2]
    (int64 start, int64 end) {
    __m256 sum, val;
    for (int j = start; j < end; ++j) {
      auto block_offset = j * block_size_avx2;
      float* row_pos = const_cast<float*>(input_data + block_offset);
      sum = _mm256_loadu_ps(row_pos);
      for (int i = 1; i < sum_size; ++i) {
        row_pos += channel;
        val = _mm256_loadu_ps(row_pos);
        sum = _mm256_add_ps(sum, val);
      }
      _mm256_storeu_ps(output_data + block_offset, sum);
    }
  };

  auto cost = Eigen::TensorOpCost(sizeof(float) * sum_size,
                                  sizeof(float),
                                  sum_size);
  d.parallelFor(block_num, cost, do_work);
  if (remain_num == 0) {
    return;
  }
  int offset_block_end = block_num * block_size_avx2;
  __m256 zero = _mm256_setzero_ps();
  __mmask8 mask = 0xff >> (block_size_avx2 - remain_num);
  __m256 sum, val;
  float* row_pos = const_cast<float*>(input_data + offset_block_end);
  sum = _mm256_mask_loadu_ps(zero, mask, row_pos);
  for (int i = 1; i < sum_size; ++i) {
    row_pos += channel;
    val = _mm256_mask_loadu_ps(zero, mask, row_pos);
    sum = _mm256_mask_add_ps(zero, mask, sum, val);
  }
  _mm256_mask_storeu_ps(output_data + offset_block_end, mask, sum);
}
#endif

template <typename T>
void BiasGrad2DInternal(const CPUDevice& d, typename TTypes<T>::ConstFlat input,
                        Eigen::DSizes<int, 2>& two_dims,
                        typename TTypes<T>::Flat output) {
  const int sum_size = two_dims[0];
  const int channel = two_dims[1];
  // NOTE(zycao): This is only threads number of Eigen threadpool. In MKL
  // handeled cases, threads number for computing will be decided by OpenMP.
  const T* in = input.data();
  T* out = output.data();

  // NOTE(zycao): These conditions are based on modern CPU architechure
  // features and verified by batch of tests on CPU. They are expected to
  // make positive impact on most cases.
#define CPU_CACHE_LINE_SIZE 64
#define HALF_L1_CACHE_SIZE 16384

  const int num_threads = d.numThreads();

  // Small cases would be explicitly done by single thread.
  if (sizeof(T) * sum_size * channel <= HALF_L1_CACHE_SIZE ||
      sum_size / num_threads <= 2 || num_threads == 1) {
    SumIntoOneRow(in, sum_size, channel, channel, out);
    return;
  }
  // Seperate the array by rows and parallel sum into temp array,
  // then sum the temp array in to output vector.
  std::vector<std::vector<T>> sum_vec(num_threads + 1);

  auto work_on_rows = [&in, &sum_vec, d, channel]
      (int64 start, int64 end) {
    // If running in caller thread, currentThreadId would return -1.
    int tid = d.currentThreadId() + 1;
    std::vector<T>& vec = sum_vec[tid];
    if (vec.empty()) vec.resize(channel, static_cast<T>(0));
    SumIntoOneRow(&(in[start * channel]), end - start, channel, channel,
                  vec.data(), false);
  };
  auto cost = Eigen::TensorOpCost(sizeof(T) * channel, // ld bytes
                                  sizeof(T) * channel, // st bytes
                                  channel); // compute cycles
  d.parallelFor(sum_size, cost, work_on_rows);

  // Sum temp array to output vector.
  for (int j = 0; j < channel; ++j) {
    out[j] = static_cast<T>(0);
  }
  for (int i = 0; i < num_threads + 1; ++i) {
    std::vector<T>& vec = sum_vec[i];
    if (!vec.empty()) {
      for (int j = 0; j < channel; ++j) {
        out[j] += vec[j];
      }
    }
  }
#undef CPU_CACHE_LINE_SIZE
#undef HALF_L1_CACHE_SIZE
}

} // namespace

namespace functor {

// Functor used by BiasGradOp to do the computations.
template <typename Device, typename T>
struct BiasGrad2D {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat input,
                  Eigen::DSizes<int, 2>& two_dims,
                  typename TTypes<T>::Flat output) {
#ifdef EIGEN_HAS_INDEX_LIST
    Eigen::IndexList<Eigen::type2index<0> > reduction_axis;
#else
    Eigen::array<int, 1> reduction_axis = {0};
#endif
    output.device(d) = input.template cast<typename AccumulatorType<T>::type>()
                            .reshape(two_dims)
                            .sum(reduction_axis)
                            .template cast<T>();
  }
};

// Functor used by BiasGradOp to do the computations on CPU.
template <typename T>
struct BiasGrad2D<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstFlat input,
                  Eigen::DSizes<int, 2>& two_dims,
                  typename TTypes<T>::Flat output) {
    BiasGrad2DInternal<T>(d, input, two_dims, output);
  }
};

template<>
struct BiasGrad2D<CPUDevice, float> {
  void operator()(const CPUDevice& d, typename TTypes<float>::ConstFlat input,
                  Eigen::DSizes<int, 2>& two_dims,
                  typename TTypes<float>::Flat output) {
    auto thread_num = d.numThreads();
#ifdef __AVX512F__
    if (two_dims[1] >= block_size_avx512 * thread_num) {
      ColumnParallel_512(d, (float*)(input.data()), output.data(),
          two_dims[0], two_dims[1]);
      return;
    }

    if (two_dims[1] >= block_size_avx2 * thread_num) {
      ColumnParallel_256(d, (float*)(input.data()), output.data(),
          two_dims[0], two_dims[1]);
      return;
    }
#endif
    BiasGrad2DInternal<float>(d, input, two_dims, output);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_BIAS_OP_CPU_H_
