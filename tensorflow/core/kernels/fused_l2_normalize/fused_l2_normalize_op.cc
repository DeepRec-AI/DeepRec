#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/lib/core/threadpool.h"

#include "compile_util.h"

#include <cmath>

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
class FusedL2NormalizeOp : public OpKernel {
public:
  explicit FusedL2NormalizeOp(OpKernelConstruction* context)
        : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis));
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
  }

  ~FusedL2NormalizeOp() {
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input
    const Tensor *input_tensor = &context->input(0);
    const T *input = input_tensor->flat<T>().data();

    // To check the input
    OP_REQUIRES(context, (input_tensor->dims() >= 2),
                errors::InvalidArgument("Input dimension should be >= 2"));

    int64 cols = input_tensor->dim_size(input_tensor->dims() - 1);
    int64 rows = 1;
    for (int64 i = 0; i < input_tensor->dims() - 1; ++i) {
      rows *= input_tensor->dim_size(i);
    }

    // Create output tensors
    Tensor *output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor->shape(),
                                                        &output_tensor));
    T *output = output_tensor->flat<T>().data();

    // Let every thread compute 16 rows to avoid false sharing
    #define BLOCK_SIZE 16
    const int64 total_unit = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64 unit_cost = BLOCK_SIZE * cols * 50; // assume every element consumes 50 cycles

    auto &worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    thread::ThreadPool *thread_pool = worker_threads.workers;

#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
    // Do forward in 128(8*16) block, avx512 handles 16 floats one time
    int64 block_num = cols >> 7;  // 128-size block nums 
    int64 remainder_128 = cols & 0x7F;  // remainder of 128
    int64 remainder_16 = remainder_128 & 0x0F;  // remainder of 16
    int64 remainder_block_num = remainder_128 >> 4;  // 16-size block num in 128-remainder
    int64 remainder_block_num_total = remainder_block_num+ !!remainder_16;  // total 16-size block num in remainder

    thread_pool->ParallelFor(total_unit, unit_cost, [&input, &output, rows, cols, block_num, remainder_block_num,
         remainder_block_num_total, remainder_128, remainder_16, this](int64 begin_unit, int64 end_unit) {
          auto begin_row = begin_unit * BLOCK_SIZE;
          auto end_row = end_unit * BLOCK_SIZE;
          if (end_row > rows) {
            end_row = rows;
          }
          forward_avx512<8>(input, output, begin_row, end_row, cols, block_num, remainder_block_num,
                               remainder_block_num_total, remainder_128, remainder_16);
        });
#else
    thread_pool->ParallelFor(total_unit, unit_cost, 
         [&input, &output, rows, cols, this](int64 begin_unit, int64 end_unit) {
          auto begin_row = begin_unit * BLOCK_SIZE;
          auto end_row = end_unit * BLOCK_SIZE;
          if (end_row > rows) {
            end_row = rows;
          }
          forward<8>(input, output, begin_row, end_row, cols);
        });
#endif
  }

private:
  // temp = tf.math.square(inputs)
  // temp = tf.math.reduce_sum(temp, reduction_indices=axis, keepdims=True)
  // temp = tf.math.maximum(temp, epsilon)
  // temp = tf.math.rsqrt(temp)
  // outputs = tf.math.multiply(temp, inputs)
  template <int SUM_BLOCK_SIZE>
  void forward(const T* input, T* output, int64 begin_row, int64 end_row,
               int64 cols) {
    int64 remainder = cols % SUM_BLOCK_SIZE;
    for (int64 i = begin_row; i < end_row; ++i) {
      T row_sum = 0;
      // Sum of squares of the inputs
      for (int64 j = 0; j < cols - remainder; j += SUM_BLOCK_SIZE) {
        T data_0 = input[i * cols + j];
        T data_1 = input[i * cols + j + 1];
        T data_2 = input[i * cols + j + 2];
        T data_3 = input[i * cols + j + 3];
        T data_4 = input[i * cols + j + 4];
        T data_5 = input[i * cols + j + 5];
        T data_6 = input[i * cols + j + 6];
        T data_7 = input[i * cols + j + 7];
        row_sum += data_0 * data_0 + data_1 * data_1 + data_2 * data_2 +
                   data_3 * data_3 + data_4 * data_4 + data_5 * data_5 +
                   data_6 * data_6 + data_7 * data_7;
      }
      for (int64 j = cols - remainder; j < cols; j++) {
        T data_0 = input[i * cols + j];
        row_sum += data_0 * data_0;
      }

      // Square
      row_sum += epsilon;
      row_sum = 1.0 / std::sqrt(row_sum);

      // Mul
      for (int64 j = 0; j < cols; ++j) {
        output[i * cols + j] = input[i * cols + j] * row_sum;
      }
    }
  }

#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
  template <int SUM_BLOCK_SIZE>
  void forward_avx512(const T* input, T* output, int64 begin_row, int64 end_row, int64 cols, int64 block_num, int64 remainder_block_num,
                         int64 remainder_block_num_total, int64 remainder_128, int64 remainder_16) {
    for (int64 i = begin_row; i < end_row; ++i) {
      float row_sum = 0.0;
      // Sum of squares of the inputs
      for (int64 j = 0; j < block_num; ++j) {
        __m512 inputs[SUM_BLOCK_SIZE];
        auto load = [&](auto idx) {
          inputs[idx] = _mm512_loadu_ps(input + cols * i +
                                        16 * SUM_BLOCK_SIZE * j + 16 * idx);
          inputs[idx] = _mm512_mul_ps(inputs[idx], inputs[idx]);
        };
        functor::compile_time_for<SUM_BLOCK_SIZE>::op(load);
        __m512 block_sum = reduce_sum_block<8>(inputs);
        row_sum += _mm512_reduce_add_ps(block_sum);
      }
      if (remainder_block_num_total) {
        __m512 inputs[remainder_block_num_total];

        for (int64 idx = 0; idx < remainder_block_num; idx++){
          inputs[idx] = _mm512_loadu_ps(input + cols * i + cols -
                                          remainder_128 + 16 * idx);
          inputs[idx] = _mm512_mul_ps(inputs[idx], inputs[idx]);
        }
        if (remainder_16) {
          __mmask16 mask = 0xFFFF >> (16 - remainder_16);
          inputs[remainder_block_num] = _mm512_maskz_loadu_ps(
              mask, input + cols * i + cols - remainder_16);
          inputs[remainder_block_num] = _mm512_mul_ps(inputs[remainder_block_num], inputs[remainder_block_num]);
        }
        
        __m512 block_sum = reduce_sum_block_ps(inputs, remainder_block_num_total);
        row_sum += _mm512_reduce_add_ps(block_sum);
      }

      // Square root 
      row_sum += epsilon;
      row_sum = 1.0 / std::sqrt(row_sum);

      // Mul & store
      __m512 row_sums = _mm512_set1_ps(row_sum);
      for (int64 j = 0; j < cols - 15; j += 16) {
        __m512 inputs = _mm512_loadu_ps(input + cols * i + j);
        inputs = _mm512_mul_ps(inputs, row_sums);
        _mm512_storeu_ps(output + cols * i + j, inputs);
      }
      if (remainder_16) {
        __mmask16 mask = 0xFFFF >> (16 - remainder_16);
        __m512 inputs = _mm512_maskz_loadu_ps(
            mask, input + cols * i + cols - remainder_16);
        inputs = _mm512_mul_ps(inputs, row_sums);
        _mm512_mask_storeu_ps(output + cols * i + cols - remainder_16, mask,
                              inputs);
      }
    }
  }

  // data type: FP32, 16 FP32 per __m512
  //  v0: v0_0, v0_1, ..., v0_15
  //  v1: v1_0, v1_1, ..., v1_15
  //  ...
  //  v7: v7_0, v7_1, ..., v7_15
  // sum:  v_0,  v_1, ...,  v_15
  template <int BLOCK_NUM>
  inline __m512 reduce_sum_block(const __m512* v) {
    __m512 block_sum = _mm512_setzero_ps();
    auto reduce_sum = [&](auto idx) {
      block_sum = _mm512_add_ps(block_sum, v[idx]);
    };
    functor::compile_time_for<BLOCK_NUM>::op(reduce_sum);
    return block_sum;
  }
  
  inline __m512 reduce_sum_block_ps(const __m512* v, int64 BLOCK_NUM) {
    switch (BLOCK_NUM)
    {
    case 1:
      return reduce_sum_block<1>(v);
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
#endif

private:
  float epsilon;
  int32 axis;
};

REGISTER_KERNEL_BUILDER(Name("FusedL2Normalize")            \
                            .Device(DEVICE_CPU)             \
                            .TypeConstraint<float>("T"),    \
                        FusedL2NormalizeOp<float>);


template <typename T>
class FusedL2NormalizeGradOp : public OpKernel {
public:
  explicit FusedL2NormalizeGradOp(OpKernelConstruction* context)
           : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis));
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
  }

  ~FusedL2NormalizeGradOp() {}

  void Compute(OpKernelContext* context) override {
    // Grab the input
    const Tensor *y_grad_tensor = &context->input(0);
    const Tensor *x_tensor = &context->input(1);

    const T *y_grad = y_grad_tensor->flat<T>().data();
    const T *x = x_tensor->flat<T>().data();

    int64 cols = x_tensor->dim_size(x_tensor->dims() - 1);
    int64 rows = 1;
    for (int64 i = 0; i < x_tensor->dims() - 1; ++i) {
      rows *= x_tensor->dim_size(i);
    }

    // Create output tensors
    Tensor *x_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_tensor->shape(),
                                                        &x_grad_tensor));
    T *x_grad = x_grad_tensor->flat<T>().data();

    // Do it in parallel
    #define BLOCK_SIZE 16
    const int64 total_unit = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64 unit_cost = BLOCK_SIZE * cols * 50; // assume every element consumes 50 cycles

    auto &worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    thread::ThreadPool *thread_pool = worker_threads.workers;

#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
    // Do forward in 128(8*16) block, avx512 handles 16 floats one time
    int64 block_num = cols >> 7;  // 128-size block nums 
    int64 remainder_128 = cols & 0x7F;  // remainder of 128
    int64 remainder_16 = remainder_128 & 0x0F;  // remainder of 16
    int64 remainder_block_num = remainder_128 >> 4;  // 16-size block num in 128-remainder
    int64 remainder_block_num_total = remainder_block_num+ !!remainder_16;  // total 16-size block num in remainder

    thread_pool->ParallelFor(total_unit, unit_cost, [&y_grad, &x, &x_grad, rows, cols, block_num, remainder_block_num,
         remainder_block_num_total, remainder_128, remainder_16, this](int64 begin_unit, int64 end_unit) {
          auto begin_row = begin_unit * BLOCK_SIZE;
          auto end_row = end_unit * BLOCK_SIZE;
          if (end_row > rows) {
            end_row = rows;
          }
          backward_avx512<8>(y_grad, x, x_grad, begin_row, end_row, cols, block_num, 
                               remainder_block_num, remainder_block_num_total, remainder_128, remainder_16);
        });
#else
    thread_pool->ParallelFor(total_unit, unit_cost, 
         [&y_grad, &x, &x_grad, rows, cols, this](int64 begin_unit, int64 end_unit) {
          auto begin_row = begin_unit * BLOCK_SIZE;
          auto end_row = end_unit * BLOCK_SIZE;
          if (end_row > rows) {
            end_row = rows;
          }
          backward<8>(y_grad, x, x_grad, begin_row, end_row, cols);
        });
#endif
  }

private:
  // rvar = tf.math.rsqrt(tf.math.reduce_sum(x * x, reduction_indices=1,
  // keepdims=True) + 1e-12) # rsqrt quickly sum = tf.math.reduce_sum(y_grad *
  // x, reduction_indices=1, keepdims=True) grad_x = y_grad * rvar - x * ((sum *
  // rvar) * (rvar * rvar))
  template <int SUM_BLOCK_SIZE>
  void backward(const float* y_grad, const float* x, float* x_grad,
                int64 begin_row, int64 end_row, int64 cols) {
    int64 remainder = cols % SUM_BLOCK_SIZE;
    for (int64 i = begin_row; i < end_row; ++i) {
      T x_row_sum = 0.0;
      T y_grad_row_sum = 0.0;
      // sum of squares of x and sum of y_grad * x
      for (int64 j = 0; j < cols - remainder; j += SUM_BLOCK_SIZE) {
        T x_0 = x[i * cols + j];
        T x_1 = x[i * cols + j + 1];
        T x_2 = x[i * cols + j + 2];
        T x_3 = x[i * cols + j + 3];
        T x_4 = x[i * cols + j + 4];
        T x_5 = x[i * cols + j + 5];
        T x_6 = x[i * cols + j + 6];
        T x_7 = x[i * cols + j + 7];
        x_row_sum += x_0 * x_0 + x_1 * x_1 + x_2 * x_2 + x_3 * x_3 + x_4 * x_4 +
                     x_5 * x_5 + x_6 * x_6 + x_7 * x_7;

        T y_grad_0 = y_grad[i * cols + j];
        T y_grad_1 = y_grad[i * cols + j + 1];
        T y_grad_2 = y_grad[i * cols + j + 2];
        T y_grad_3 = y_grad[i * cols + j + 3];
        T y_grad_4 = y_grad[i * cols + j + 4];
        T y_grad_5 = y_grad[i * cols + j + 5];
        T y_grad_6 = y_grad[i * cols + j + 6];
        T y_grad_7 = y_grad[i * cols + j + 7];
        y_grad_row_sum += x_0 * y_grad_0 + x_1 * y_grad_1 + x_2 * y_grad_2 +
                          x_3 * y_grad_3 + x_4 * y_grad_4 + x_5 * y_grad_5 +
                          x_6 * y_grad_6 + x_7 * y_grad_7;
      }
      for (int64 j = cols - remainder; j < cols; j++) {
        T x_0 = x[i * cols + j];
        x_row_sum += x_0 * x_0;

        T y_grad_0 = y_grad[i * cols + j];
        y_grad_row_sum += x_0 * y_grad_0;
      }
      x_row_sum += epsilon;
      x_row_sum = 1.0 / std::sqrt(x_row_sum);  // rvar
      y_grad_row_sum = (y_grad_row_sum * x_row_sum) * (x_row_sum * x_row_sum);
      // Calculate x_grad = y_grad * rvar - x * ((sum * rvar) * (rvar * rvar))
      for (int64 j = 0; j < cols; ++j) {
        x_grad[i * cols + j] =
            y_grad[i * cols + j] * x_row_sum - x[i * cols + j] * y_grad_row_sum;
      }
    }
  }

#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
  template <int SUM_BLOCK_SIZE>
  void backward_avx512(const float* y_grad, const float* x, float* x_grad, int64 begin_row, int64 end_row, int64 cols,
                          int64 block_num, int64 remainder_block_num, int64 remainder_block_num_total, int64 remainder_128,
                          int64 remainder_16) {
    for (int64 i = begin_row; i < end_row; ++i) {
      T x_row_sum = 0.0;
      T y_grad_row_sum = 0.0;
      // sum of squares of x and sum of y_grad * x
      for (int64 j = 0; j < block_num; ++j) {
        __m512 xs[SUM_BLOCK_SIZE];
        auto x_load = [&](auto idx) {
          xs[idx] = _mm512_loadu_ps(x + cols * i + 16 * SUM_BLOCK_SIZE * j +
                                    16 * idx);
          xs[idx] = _mm512_mul_ps(xs[idx], xs[idx]);
        };
        functor::compile_time_for<SUM_BLOCK_SIZE>::op(x_load);
        __m512 x_block_sum = reduce_sum_block<8>(xs);
        x_row_sum += _mm512_reduce_add_ps(x_block_sum);

        __m512 y_grads[SUM_BLOCK_SIZE];
        auto y_grad_load = [&](auto idx) {
          y_grads[idx] = _mm512_loadu_ps(y_grad + cols * i +
                                         16 * SUM_BLOCK_SIZE * j + 16 * idx);
          xs[idx] = _mm512_loadu_ps(x + cols * i + 16 * SUM_BLOCK_SIZE * j +
                                    16 * idx);
          y_grads[idx] = _mm512_mul_ps(y_grads[idx], xs[idx]);
        };
        functor::compile_time_for<SUM_BLOCK_SIZE>::op(y_grad_load);
        __m512 y_grad_block_sum = reduce_sum_block<8>(y_grads);
        y_grad_row_sum += _mm512_reduce_add_ps(y_grad_block_sum);
      }
      if (remainder_block_num_total) {
        __m512 xs[remainder_block_num_total];
        for (int64 idx = 0; idx < remainder_block_num; idx++){
          xs[idx] = _mm512_loadu_ps(x + cols * i + cols - remainder_128 + 16 * idx);
          xs[idx] = _mm512_mul_ps(xs[idx], xs[idx]);
        }
        if (remainder_16) {
          __mmask16 mask = 0xFFFF >> (16 - remainder_16);
          xs[remainder_block_num] = _mm512_maskz_loadu_ps(mask, x + cols * i + cols - remainder_16);
          xs[remainder_block_num] = _mm512_mul_ps(xs[remainder_block_num], xs[remainder_block_num]);
        }
        __m512 x_block_sum = reduce_sum_block_ps(xs, remainder_block_num_total);
        x_row_sum += _mm512_reduce_add_ps(x_block_sum);

        __m512 y_grads[remainder_block_num_total];
        for (int64 idx = 0; idx < remainder_block_num; idx++){
          y_grads[idx] = _mm512_loadu_ps(y_grad + cols * i + cols - remainder_128 + 16 * idx);
          xs[idx] = _mm512_loadu_ps(x + cols * i + cols - remainder_128 + 16 * idx);
          y_grads[idx] = _mm512_mul_ps(y_grads[idx], xs[idx]);
        }
        if (remainder_16) {
          __mmask16 mask = 0xFFFF >> (16 - remainder_16);
          y_grads[remainder_block_num] = _mm512_maskz_loadu_ps(mask, y_grad + cols * i + cols - remainder_16);
          xs[remainder_block_num] = _mm512_maskz_loadu_ps(mask, x + cols * i + cols - remainder_16);
          y_grads[remainder_block_num] = _mm512_mul_ps(y_grads[remainder_block_num], xs[remainder_block_num]);
        }
        __m512 y_grad_block_sum = reduce_sum_block_ps(y_grads, remainder_block_num_total);
        y_grad_row_sum += _mm512_reduce_add_ps(y_grad_block_sum);
      }

      x_row_sum += epsilon;
      x_row_sum = 1.0 / std::sqrt(x_row_sum); // rvar
      y_grad_row_sum = (y_grad_row_sum * x_row_sum) * (x_row_sum * x_row_sum);
      // Calculate x_grad = y_grad * rvar - x * ((sum * rvar) * (rvar * rvar))
      __m512 x_row_sums = _mm512_set1_ps(x_row_sum);
      __m512 y_grad_row_sums = _mm512_set1_ps(y_grad_row_sum);
      for (int64 j = 0; j < cols - 15; j += 16) {
        __m512 y_grads = _mm512_loadu_ps(y_grad + cols * i + j);
        __m512 xs = _mm512_loadu_ps(x + cols * i + j);
        y_grads = _mm512_mul_ps(y_grads, x_row_sums);
        xs = _mm512_mul_ps(xs, y_grad_row_sums);
        y_grads = _mm512_sub_ps(y_grads, xs);
        _mm512_storeu_ps(x_grad + cols * i + j, y_grads);
      }
      if (remainder_16 > 0) {
        __mmask16 mask = 0xFFFF >> (16 - remainder_16);
        __m512 y_grads = _mm512_maskz_loadu_ps(mask, y_grad + cols * i + cols - remainder_16);
        __m512 xs = _mm512_maskz_loadu_ps(mask, x + cols * i + cols - remainder_16);
        y_grads = _mm512_mul_ps(y_grads, x_row_sums);
        xs = _mm512_mul_ps(xs, y_grad_row_sums);
        y_grads = _mm512_sub_ps(y_grads, xs);
        _mm512_mask_storeu_ps(x_grad + cols * i + cols - remainder_16, mask, y_grads);
      }
    }
  }

  template <int BLOCK_NUM>
  inline __m512 reduce_sum_block(const __m512* v) {
    __m512 block_sum = _mm512_setzero_ps();
    auto reduce_sum = [&](auto idx) {
      block_sum = _mm512_add_ps(block_sum, v[idx]);
    };
    functor::compile_time_for<BLOCK_NUM>::op(reduce_sum);
    return block_sum;
  }
  
  inline __m512 reduce_sum_block_ps(const __m512* v, int64 BLOCK_NUM) {
    switch (BLOCK_NUM)
    {
    case 1:
      return reduce_sum_block<1>(v);
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
#endif

private:
  float epsilon;
  int32 axis;
};

REGISTER_KERNEL_BUILDER(Name("FusedL2NormalizeGrad")        \
                            .Device(DEVICE_CPU)             \
                            .TypeConstraint<float>("T"),    \
                        FusedL2NormalizeGradOp<float>);

}  // namespace tensorflow
