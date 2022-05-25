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
      printf("RUN ~FusedL2NormalizeOp().\n");
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

    thread_pool->ParallelFor(total_unit, unit_cost,
        [&input, &output, rows, cols, this](int64 begin_unit, int64 end_unit) {
          auto begin_row = begin_unit * BLOCK_SIZE;
          auto end_row = end_unit * BLOCK_SIZE;
          if (end_row > rows) {
            end_row = rows;
          }
          forward<8>(input, output, begin_row, end_row, cols);
        });
  }

private:
    // temp = tf.math.square(inputs)
    // temp = tf.math.reduce_sum(temp, reduction_indices=axis, keepdims=True)
    // temp = tf.math.maximum(temp, epsilon)
    // temp = tf.math.rsqrt(temp)
    // outputs = tf.math.multiply(temp, inputs)
    template <int SUM_BLOCK_SIZE>
    void ref_forward(const T* input, T* output, int64 begin_row, int64 end_row, int64 cols) {
        for (int64 i = begin_row; i < end_row; ++i) {
            T row_sum = 0;
            // must be SUM_BLOCK_SIZE block !!!
            for (int64 j = 0; j < cols; j += SUM_BLOCK_SIZE) {
                T data_0 = input[i * cols + j];
                T data_1 = input[i * cols + j + 1];
                T data_2 = input[i * cols + j + 2];
                T data_3 = input[i * cols + j + 3];
                T data_4 = input[i * cols + j + 4];
                T data_5 = input[i * cols + j + 5];
                T data_6 = input[i * cols + j + 6];
                T data_7 = input[i * cols + j + 7];
                row_sum += data_0 * data_0 + data_1 * data_1
                                  + data_2 * data_2 + data_3 * data_3
                                  + data_4 * data_4 + data_5 * data_5
                                  + data_6 * data_6 + data_7 * data_7;
            }
            row_sum += epsilon;
            row_sum = 1.0 / std::sqrt(row_sum);
            for (int64 j = 0; j < cols; ++j) {
                output[i * cols + j] = input[i * cols + j] * row_sum;
            }
        }
    }

    template <int SUM_BLOCK_SIZE>
    void forward(const T* input, T* output, int64 begin_row, int64 end_row, int64 cols) {
        int64 avx3_block_num = cols >> 7; // cols / 128
        // handle remainder of 128
        int64 remainder = cols - (avx3_block_num << 7);
        // printf("cols: %d, avx3_block_num: %d, remainder %d\n", cols, avx3_block_num, remainder);
        for (int64 i = begin_row; i < end_row; ++i) {
            int64 tmp_remainder = remainder;
            float row_sum = 0.0;
            for (int64 j = 0; j < avx3_block_num; ++j) {
                __m512 inputs[SUM_BLOCK_SIZE];
                auto load = [&](auto idx) {
                    inputs[idx] = _mm512_loadu_ps(input + cols * i + 16 * SUM_BLOCK_SIZE * j + 16 * idx);
                    inputs[idx] = _mm512_mul_ps(inputs[idx], inputs[idx]);
                };
                functor::compile_time_for<SUM_BLOCK_SIZE>::op(load);
                __m512 block_sum = reduce_sum_block8_ps(inputs);
                row_sum += _mm512_reduce_add_ps(block_sum);
            }
            if (tmp_remainder > 0) {
                if (tmp_remainder >= 64) {
                    __m256 inputs[8];
                    auto load_256 = [&](auto idx) {
                        inputs[idx] = _mm256_loadu_ps(input + cols * i + cols - tmp_remainder + 8 * idx);
                        inputs[idx] = _mm256_mul_ps(inputs[idx], inputs[idx]);
                    };
                    functor::compile_time_for<8>::op(load_256);
                    __m256 block_sum_remainder = reduce_sum_block8_mm256_ps(inputs);
                    row_sum += _mm512_reduce_add_ps(_mm512_castps256_ps512(block_sum_remainder));
                    tmp_remainder -= 64;
                }
                if (tmp_remainder > 32) {
                    __m256 inputs[4];
                    auto load_256 = [&](auto idx) {
                        inputs[idx] = _mm256_loadu_ps(input + cols * i + cols - tmp_remainder + 8 * idx);
                        inputs[idx] = _mm256_mul_ps(inputs[idx], inputs[idx]);
                    };
                    functor::compile_time_for<4>::op(load_256);
                    __m256 block_sum_remainder = reduce_sum_block4_mm256_ps(inputs);
                    row_sum += _mm512_reduce_add_ps(_mm512_castps256_ps512(block_sum_remainder));
                    tmp_remainder -= 32;
                }
                if (tmp_remainder >= 16) {
                    __m512 inputs = _mm512_loadu_ps(input + cols * i + cols - tmp_remainder);
                    inputs = _mm512_mul_ps(inputs, inputs);
                    row_sum += _mm512_reduce_add_ps(inputs);
                    tmp_remainder -= 16;
                }
                if (tmp_remainder > 0) {
                    __mmask16 mask = 0xFFFF >> (16 - tmp_remainder);
                    __m512 inputs = _mm512_maskz_loadu_ps(mask, input + cols * i + cols - tmp_remainder);
                    inputs = _mm512_mul_ps(inputs, inputs);
                    row_sum += _mm512_reduce_add_ps(inputs);
                }
            }

            row_sum += epsilon;
            row_sum = 1.0 / std::sqrt(row_sum);
            __m512 row_sums = _mm512_set1_ps(row_sum);
            for (int64 j = 0; j < cols - 15; j += 16) {
                __m512 inputs = _mm512_loadu_ps(input + cols * i + j);
                inputs = _mm512_mul_ps(inputs, row_sums);
                _mm512_storeu_ps(output + cols * i + j, inputs);
            }
            if (tmp_remainder > 0){
                __mmask16 mask = 0xFFFF >> (16 - tmp_remainder);
                __m512 inputs = _mm512_maskz_loadu_ps(mask, input + cols * i + cols - tmp_remainder);
                inputs = _mm512_mul_ps(inputs, row_sums);
                _mm512_mask_storeu_ps(output + cols * i + cols - tmp_remainder, mask, inputs);
            }
        }
    }

    // data type: FP32, 16 FP32 per __m512
    //  v0: v0_0, v0_1, ..., v0_15
    //  v1: v1_0, v1_1, ..., v1_15
    //  ...
    //  v7: v7_0, v7_1, ..., v7_15
    // sum:  v_0,  v_1, ...,  v_15
    inline __m512 reduce_sum_block8_ps(const __m512 (&v)[8]) {
        __m512 block_sum = _mm512_add_ps(v[0], v[1]);
        block_sum = _mm512_add_ps(block_sum, v[2]);
        block_sum = _mm512_add_ps(block_sum, v[3]);
        block_sum = _mm512_add_ps(block_sum, v[4]);
        block_sum = _mm512_add_ps(block_sum, v[5]);
        block_sum = _mm512_add_ps(block_sum, v[6]);
        block_sum = _mm512_add_ps(block_sum, v[7]);
        return block_sum;
    }
    inline __m256 reduce_sum_block8_mm256_ps(const __m256 (&v)[8]) {
        __m256 block_sum = _mm256_add_ps(v[0], v[1]);
        block_sum = _mm256_add_ps(block_sum, v[2]);
        block_sum = _mm256_add_ps(block_sum, v[3]);
        block_sum = _mm256_add_ps(block_sum, v[4]);
        block_sum = _mm256_add_ps(block_sum, v[5]);
        block_sum = _mm256_add_ps(block_sum, v[6]);
        block_sum = _mm256_add_ps(block_sum, v[7]);
        return block_sum;
    }
    inline __m256 reduce_sum_block4_mm256_ps(const __m256 (&v)[4]) {
        __m256 block_sum = _mm256_add_ps(v[0], v[1]);
        block_sum = _mm256_add_ps(block_sum, v[2]);
        block_sum = _mm256_add_ps(block_sum, v[3]);
        return block_sum;
    }

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

    thread_pool->ParallelFor(total_unit, unit_cost,
        [&y_grad, &x, &x_grad, rows, cols, this](int64 begin_unit, int64 end_unit) {
          auto begin_row = begin_unit * BLOCK_SIZE;
          auto end_row = end_unit * BLOCK_SIZE;
          if (end_row > rows) {
            end_row = rows;
          }
          backward<8>(y_grad, x, x_grad, begin_row, end_row, cols);
        });
  }

private:
    // rvar = tf.math.rsqrt(tf.math.reduce_sum(x * x, reduction_indices=1, keepdims=True) + 1e-12) # rsqrt quickly
    // sum = tf.math.reduce_sum(y_grad * x, reduction_indices=1, keepdims=True)
    // grad_x = y_grad * rvar - x * ((sum * rvar) * (rvar * rvar))
    template <int SUM_BLOCK_SIZE>
    void ref_backward(const float *y_grad, const float *x, float *x_grad, int64 begin_row, int64 end_row, int64 cols) {
        for (int64 i = begin_row; i < end_row; ++i) {
            int64 new_row = i - begin_row;
            T x_row_sum = 0.0;
            T y_grad_row_sum = 0.0;
            for (int64 j = cols - 1; j > 0; j -= SUM_BLOCK_SIZE) {
                T x_0 = x[i * cols + j];
                T x_1 = x[i * cols + j - 1];
                T x_2 = x[i * cols + j - 2];
                T x_3 = x[i * cols + j - 3];
                T x_4 = x[i * cols + j - 4];
                T x_5 = x[i * cols + j - 5];
                T x_6 = x[i * cols + j - 6];
                T x_7 = x[i * cols + j - 7];
                x_row_sum += x_0 * x_0 + x_1 * x_1
                                    + x_2 * x_2 + x_3 * x_3
                                    + x_4 * x_4 + x_5 * x_5
                                    + x_6 * x_6 + x_7 * x_7;

                T y_grad_0 = y_grad[i * cols + j];
                T y_grad_1 = y_grad[i * cols + j - 1];
                T y_grad_2 = y_grad[i * cols + j - 2];
                T y_grad_3 = y_grad[i * cols + j - 3];
                T y_grad_4 = y_grad[i * cols + j - 4];
                T y_grad_5 = y_grad[i * cols + j - 5];
                T y_grad_6 = y_grad[i * cols + j - 6];
                T y_grad_7 = y_grad[i * cols + j - 7];
                y_grad_row_sum += x_0 * y_grad_0 + x_1 * y_grad_1
                                         + x_2 * y_grad_2 + x_3 * y_grad_3
                                         + x_4 * y_grad_4 + x_5 * y_grad_5
                                         + x_6 * y_grad_6 + x_7 * y_grad_7;
            }
            x_row_sum += epsilon;
            x_row_sum = 1.0 / std::sqrt(x_row_sum); // rvar
            y_grad_row_sum = (y_grad_row_sum * x_row_sum) * (x_row_sum * x_row_sum);
            for (int64 j = 0; j < cols; ++j) {
                x_grad[i * cols + j] = y_grad[i * cols + j] * x_row_sum
                                     - x[i * cols + j] * y_grad_row_sum;
            }
        }
    }

    template <int SUM_BLOCK_SIZE>
    void backward(const float *y_grad, const float *x, float *x_grad, int64 begin_row, int64 end_row, int64 cols) {
        int64 avx3_block_num = cols >> 7; // cols / 128
        // handle remainder of 128
        int64 remainder = cols - (avx3_block_num << 7);
        // printf("cols: %d, avx3_block_num: %d, remainder %d\n", cols, avx3_block_num, remainder);
        for (int64 i = begin_row; i < end_row; ++i) {
            T x_row_sum = 0.0;
            T y_grad_row_sum = 0.0;
            int64 tmp_remainder = remainder;
            for (int64 j = 0; j < avx3_block_num; ++j) {
                __m512 xs[SUM_BLOCK_SIZE];
                auto x_load = [&](auto idx) {
                    xs[idx] = _mm512_loadu_ps(x + cols * i + 16 * SUM_BLOCK_SIZE * j + 16 * idx);
                    xs[idx] = _mm512_mul_ps(xs[idx], xs[idx]);
                };
                functor::compile_time_for<SUM_BLOCK_SIZE>::op(x_load);
                __m512 x_block_sum = reduce_sum_block8_ps(xs);
                x_row_sum += _mm512_reduce_add_ps(x_block_sum);

                __m512 y_grads[SUM_BLOCK_SIZE];
                auto y_grad_load = [&](auto idx) {
                    y_grads[idx] = _mm512_loadu_ps(y_grad + cols * i + 16 * SUM_BLOCK_SIZE * j + 16 * idx);
                    xs[idx] = _mm512_loadu_ps(x + cols * i + 16 * SUM_BLOCK_SIZE * j + 16 * idx);
                    y_grads[idx] = _mm512_mul_ps(y_grads[idx], xs[idx]);
                };
                functor::compile_time_for<SUM_BLOCK_SIZE>::op(y_grad_load);
                __m512 y_grad_block_sum = reduce_sum_block8_ps(y_grads);
                y_grad_row_sum += _mm512_reduce_add_ps(y_grad_block_sum);
            }
            if (tmp_remainder > 0) {
                if (tmp_remainder >= 64) {
                    __m256 xs[8];
                    auto x_load_256 = [&](auto idx) {
                        xs[idx] = _mm256_loadu_ps(x + cols * i + cols - tmp_remainder + 8 * idx);
                        xs[idx] = _mm256_mul_ps(xs[idx], xs[idx]);
                    };
                    functor::compile_time_for<8>::op(x_load_256);
                    __m256 block_sum_remainder = reduce_sum_block8_mm256_ps(xs);
                    x_row_sum += _mm512_reduce_add_ps(_mm512_castps256_ps512(block_sum_remainder));

                    __m256 y_grads[8];
                    auto y_grad_load_256 = [&](auto idx) {
                        y_grads[idx] = _mm256_loadu_ps(y_grad + cols * i + cols - tmp_remainder + 8 * idx);
                        xs[idx] = _mm256_loadu_ps(x + cols * i + cols - tmp_remainder + 8 * idx);
                        y_grads[idx] = _mm256_mul_ps(y_grads[idx], xs[idx]);
                    };
                    functor::compile_time_for<8>::op(y_grad_load_256);
                    __m256 y_grad_block_sum_remainder = reduce_sum_block8_mm256_ps(y_grads);
                    y_grad_row_sum += _mm512_reduce_add_ps(_mm512_castps256_ps512(y_grad_block_sum_remainder));
                    tmp_remainder -= 64;
                }
                if (tmp_remainder > 32) {
                    __m256 xs[4];
                    auto x_load_256 = [&](auto idx) {
                        xs[idx] = _mm256_loadu_ps(x + cols * i + cols - tmp_remainder + 8 * idx);
                        xs[idx] = _mm256_mul_ps(xs[idx], xs[idx]);
                    };
                    functor::compile_time_for<4>::op(x_load_256);
                    __m256 block_sum_remainder = reduce_sum_block4_mm256_ps(xs);
                    x_row_sum += _mm512_reduce_add_ps(_mm512_castps256_ps512(block_sum_remainder));
                    
                    __m256 y_grads[4];
                    auto y_grad_load_256 = [&](auto idx) {
                        y_grads[idx] = _mm256_loadu_ps(y_grad + cols * i + cols - tmp_remainder + 8 * idx);
                        xs[idx] = _mm256_loadu_ps(x + cols * i + cols - tmp_remainder + 8 * idx);
                        y_grads[idx] = _mm256_mul_ps(y_grads[idx], xs[idx]);
                    };
                    functor::compile_time_for<4>::op(y_grad_load_256);
                    __m256 y_grad_block_sum_remainder = reduce_sum_block4_mm256_ps(y_grads);
                    y_grad_row_sum += _mm512_reduce_add_ps(_mm512_castps256_ps512(y_grad_block_sum_remainder));
                    tmp_remainder -= 32;
                }
                if (tmp_remainder >= 16) {
                    __m512 xs = _mm512_loadu_ps(x + cols * i + cols - tmp_remainder);
                    __m512 y_grads = _mm512_loadu_ps(y_grad + cols * i + cols - tmp_remainder);
                    x_row_sum += _mm512_reduce_add_ps(_mm512_mul_ps(xs, xs));
                    y_grad_row_sum += _mm512_reduce_add_ps(_mm512_mul_ps(y_grads, xs));
                    tmp_remainder -= 16;
                }
                if (tmp_remainder > 0) {
                    __mmask16 mask = 0xFFFF >> (16 - tmp_remainder);
                    __m512 xs = _mm512_maskz_loadu_ps(mask, x + cols * i + cols - tmp_remainder);
                    __m512 y_grads = _mm512_maskz_loadu_ps(mask, y_grad + cols * i + cols - tmp_remainder);
                    x_row_sum += _mm512_reduce_add_ps(_mm512_mul_ps(xs, xs));
                    y_grad_row_sum += _mm512_reduce_add_ps(_mm512_mul_ps(y_grads, xs));
                }
            }

            x_row_sum += epsilon;
            x_row_sum = 1.0 / std::sqrt(x_row_sum);
            y_grad_row_sum = (y_grad_row_sum * x_row_sum) * (x_row_sum * x_row_sum);
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
            if (tmp_remainder > 0){
                __mmask16 mask = 0xFFFF >> (16 - tmp_remainder);
                __m512 y_grads = _mm512_maskz_loadu_ps(mask, y_grad + cols * i + cols - tmp_remainder);
                __m512 xs = _mm512_maskz_loadu_ps(mask, x + cols * i +  cols - tmp_remainder);
                y_grads = _mm512_mul_ps(y_grads, x_row_sums);
                xs = _mm512_mul_ps(xs, y_grad_row_sums);
                y_grads = _mm512_sub_ps(y_grads, xs);
                _mm512_mask_storeu_ps(x_grad + cols * i + cols - tmp_remainder, mask, y_grads);
            }
        }
    }

    inline __m512 reduce_sum_block8_ps(const __m512 (&v)[8]) {
        __m512 block_sum = _mm512_add_ps(v[0], v[1]);
        block_sum = _mm512_add_ps(block_sum, v[2]);
        block_sum = _mm512_add_ps(block_sum, v[3]);
        block_sum = _mm512_add_ps(block_sum, v[4]);
        block_sum = _mm512_add_ps(block_sum, v[5]);
        block_sum = _mm512_add_ps(block_sum, v[6]);
        block_sum = _mm512_add_ps(block_sum, v[7]);
        return block_sum;
    }
    inline __m256 reduce_sum_block8_mm256_ps(const __m256 (&v)[8]) {
        __m256 block_sum = _mm256_add_ps(v[0], v[1]);
        block_sum = _mm256_add_ps(block_sum, v[2]);
        block_sum = _mm256_add_ps(block_sum, v[3]);
        block_sum = _mm256_add_ps(block_sum, v[4]);
        block_sum = _mm256_add_ps(block_sum, v[5]);
        block_sum = _mm256_add_ps(block_sum, v[6]);
        block_sum = _mm256_add_ps(block_sum, v[7]);
        return block_sum;
    }
    inline __m256 reduce_sum_block4_mm256_ps(const __m256 (&v)[4]) {
        __m256 block_sum = _mm256_add_ps(v[0], v[1]);
        block_sum = _mm256_add_ps(block_sum, v[2]);
        block_sum = _mm256_add_ps(block_sum, v[3]);
        return block_sum;
    }

private:
    float epsilon;
    int32 axis;
};

REGISTER_KERNEL_BUILDER(Name("FusedL2NormalizeGrad")        \
                            .Device(DEVICE_CPU)             \
                            .TypeConstraint<float>("T"),    \
                        FusedL2NormalizeGradOp<float>);

}  // namespace tensorflow
