#include "compile_util.h"

using namespace tensorflow;

template <typename T>
class FusedLayerNormOp : public OpKernel {
 private:
  float epsilon;

 public:
  explicit FusedLayerNormOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
  }

  ~FusedLayerNormOp() {}

  void Compute(OpKernelContext* context) override {
    // Grab the input
    const Tensor* input_tensor = &context->input(0);
    const Tensor* gamma_tensor = &context->input(1);
    const Tensor* beta_tensor = &context->input(2);

    const T* input = input_tensor->flat<T>().data();
    const float* gamma = gamma_tensor->flat<float>().data();
    const float* beta = beta_tensor->flat<float>().data();

    // To check the input
    OP_REQUIRES(context, (input_tensor->dims() >= 2),
                errors::InvalidArgument("Input dimension should be >= 2"));
    OP_REQUIRES(context, (gamma_tensor->dims() == 1),
                errors::InvalidArgument("dims(gamma) != 1"));
    OP_REQUIRES(context, (beta_tensor->dims() == 1),
                errors::InvalidArgument("dims(beta) != 1"));

    int64 cols = input_tensor->dim_size(input_tensor->dims() - 1);
    OP_REQUIRES(
        context, (gamma_tensor->dim_size(0) == cols),
        errors::InvalidArgument("size(gamma) != last_dim_size_of_input"));
    OP_REQUIRES(
        context, (beta_tensor->dim_size(0) == cols),
        errors::InvalidArgument("size(beta) != last_dim_size_of_input"));

    int64 rows = 1;
    TensorShape mean_var_shape;
    for (int i = 0; i < input_tensor->dims() - 1; ++i) {
      auto dim_size = input_tensor->dim_size(i);
      rows *= dim_size;
      mean_var_shape.AddDim(dim_size);
    }

    // Create output tensors
    Tensor* output_tensor = NULL;
    Tensor* mean_tensor = NULL;
    Tensor* rvariance_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor->shape(), &output_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, mean_var_shape, &mean_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, mean_var_shape, &rvariance_tensor));
    T* output = output_tensor->flat<T>().data();
    float* mean = mean_tensor->flat<float>().data();
    float* rvariance = rvariance_tensor->flat<float>().data();

    // Init
    memset(mean, 0, sizeof(float) * rows);
    memset(rvariance, 0, sizeof(float) * rows);

    // Do it
    // Let every thread compute 16 rows to avoid false sharing
    const int64 total_unit = (rows + 15) / 16;
    const int64 unit_cost = 16 * cols * 50;  // assume every element consumes 50 cycles

#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
    int64 block_num = cols >> 7;
    int64 remainder_128 = cols & 0x7F;
    int64 remainder_16 = remainder_128 & 0x0F;
    int64 remainder_block_num = remainder_128 >> 4;
    int64 remainder_block_num_total = remainder_block_num + !!remainder_16;
#endif  //AVX512F
    const float one_over_cols = 1.0f / cols;

    auto& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    thread::ThreadPool* thread_pool = worker_threads.workers;

    thread_pool->ParallelFor(total_unit, unit_cost,
        [&](int64 begin_unit, int64 end_unit) {
          auto begin_row = begin_unit * 16;
          auto end_row = end_unit * 16;
          if (end_row > rows) {
            end_row = rows;
          }
#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
          forward_avx512(input, gamma, beta, output, mean, rvariance, cols, begin_row, end_row, block_num, 
                         remainder_block_num, remainder_block_num_total, remainder_128, remainder_16, one_over_cols);
#else
          forward(input, gamma, beta, output, mean, rvariance, cols, begin_row, end_row, one_over_cols);
#endif  //AVX512F
        });
  }


 private:
// Compute the rows locate in the range of [begin_row, begin_row + ROWS)
  void forward(const float* input, const float* gamma, const float* beta, float* output, float* mean, 
                              float* rvariance, int64 cols, int64 begin_row, int64 end_row, const float one_over_cols) {
    for (int64 i = begin_row; i < end_row; i++){
      // Sum
      int64 j = 0;
      for (; j + 7 < cols; j += 8) {
        T data_0 = input[i * cols + j];
        T data_1 = input[i * cols + j + 1];
        T data_2 = input[i * cols + j + 2];
        T data_3 = input[i * cols + j + 3];
        T data_4 = input[i * cols + j + 4];
        T data_5 = input[i * cols + j + 5];
        T data_6 = input[i * cols + j + 6];
        T data_7 = input[i * cols + j + 7];
        mean[i] += data_0 + data_1 + data_2 + data_3 + 
                   data_4 + data_5 + data_6 + data_7;
      }
      for (; j < cols; j++) {
        mean[i] += input[i * cols + j];
      }
      // Mean
      mean[i] *= one_over_cols;

      // variance
      for (j = 0; j + 7 < cols; j += 8) {
        T data_0 = input[i * cols + j] - mean[i];
        T data_1 = input[i * cols + j + 1] - mean[i];
        T data_2 = input[i * cols + j + 2] - mean[i];
        T data_3 = input[i * cols + j + 3] - mean[i];
        T data_4 = input[i * cols + j + 4] - mean[i];
        T data_5 = input[i * cols + j + 5] - mean[i];
        T data_6 = input[i * cols + j + 6] - mean[i];
        T data_7 = input[i * cols + j + 7] - mean[i];
        rvariance[i] += data_0 * data_0 + data_1 * data_1 + data_2 * data_2 +
                   data_3 * data_3 + data_4 * data_4 + data_5 * data_5 +
                   data_6 * data_6 + data_7 * data_7;
      }
      for (; j < cols; j++) {
        T data = input[i * cols + j] - mean[i];
        rvariance[i] += data * data;
      }
      rvariance[i] *= one_over_cols;
      rvariance[i] += epsilon;
      rvariance[i] = 1.0f / sqrtf(rvariance[i]);

      for (j = 0; j + 7 < cols; j += 8) {
        T data_0 = (input[i * cols + j] - mean[i]) * rvariance[i];
        T data_1 = (input[i * cols + j + 1] - mean[i]) * rvariance[i];
        T data_2 = (input[i * cols + j + 2] - mean[i]) * rvariance[i];
        T data_3 = (input[i * cols + j + 3] - mean[i]) * rvariance[i];
        T data_4 = (input[i * cols + j + 4] - mean[i]) * rvariance[i];
        T data_5 = (input[i * cols + j + 5] - mean[i]) * rvariance[i];
        T data_6 = (input[i * cols + j + 6] - mean[i]) * rvariance[i];
        T data_7 = (input[i * cols + j + 7] - mean[i]) * rvariance[i];
        output[i * cols + j] = gamma[j] * data_0 + beta[j];
        output[i * cols + j + 1] =  gamma[j] * data_1 + beta[j];
        output[i * cols + j + 2] =  gamma[j] * data_2 + beta[j];
        output[i * cols + j + 3] =  gamma[j] * data_3 + beta[j];
        output[i * cols + j + 4] =  gamma[j] * data_4 + beta[j];
        output[i * cols + j + 5] =  gamma[j] * data_5 + beta[j];
        output[i * cols + j + 6] =  gamma[j] * data_6 + beta[j];
        output[i * cols + j + 7] =  gamma[j] * data_7 + beta[j];
      }
      for (; j < cols; j ++) {
        T data = (input[i * cols + j] - mean[i]) * rvariance[i];
        output[i * cols + j] = gamma[j] * data + beta[j];
      }
    }
  }

#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
  // AVX512 block size = 8; pack 8 * 16 = 128;
  inline void forward_avx512(const float* input, const float* gamma, const float* beta, float* output, 
                              float* mean, float* rvariance, int64 cols, int64 begin_row, int64 end_row,
                              int64 block_num, int64 remainder_block_num,int64 remainder_block_num_total, 
                              int64 remainder_128, int64 remainder_16, const float one_over_cols) {
    for (int64 i = begin_row; i < end_row; ++i) {
      // Sum
      for (int64 j = 0; j < block_num; ++j) {
      __m512 inputs[8];
      auto load = [&](auto idx) {
          inputs[idx] = _mm512_loadu_ps(input + cols * i + 128 * j + 16 * idx);
        };
      compile_time_for<8>::op(load);
      __m512 block_sum = reduce_sum_block<8>(inputs);
      mean[i] += _mm512_reduce_add_ps(block_sum);
      }
      if (remainder_block_num_total) { // remainder sum
        __m512 inputs[remainder_block_num_total];
        for (int64 idx = 0; idx < remainder_block_num; idx++){
          inputs[idx] = _mm512_loadu_ps(input + cols * i + cols - remainder_128 + 16 * idx);
        }
        if (remainder_16) {
          __mmask16 mask = 0xFFFF >> (16 - remainder_16);
          inputs[remainder_block_num] = _mm512_maskz_loadu_ps(
              mask, input + cols * i + cols - remainder_16);
        }
        __m512 block_sum = reduce_sum_block_ps(inputs, remainder_block_num_total);
        mean[i] += _mm512_reduce_add_ps(block_sum);
      }

      // Mean
      mean[i] *= one_over_cols;
      __m512 means = _mm512_set1_ps(mean[i]);

      // Variance
      for (int64 j = 0; j < block_num; ++j) {
        __m512 inputs[8];
        auto load_var = [&](auto idx) {
          inputs[idx] = _mm512_loadu_ps(input + cols * i + 128 * j + 16 * idx);
          inputs[idx] = _mm512_sub_ps(inputs[idx], means);
          inputs[idx] = _mm512_mul_ps(inputs[idx], inputs[idx]);
        };
        compile_time_for<8>::op(load_var);
        __m512 block_sum = reduce_sum_block<8>(inputs);
        rvariance[i] += _mm512_reduce_add_ps(block_sum);
      }
      if (remainder_block_num_total) { // remainder var
        __m512 inputs[remainder_block_num_total];
        for (int64 idx = 0; idx < remainder_block_num; idx++){
          inputs[idx] = _mm512_loadu_ps(input + cols * i + cols - remainder_128 + 16 * idx);
          inputs[idx] = _mm512_sub_ps(inputs[idx], means);
          inputs[idx] = _mm512_mul_ps(inputs[idx], inputs[idx]);
        }
        if (remainder_16) {
          __mmask16 mask = 0xFFFF >> (16 - remainder_16);
          inputs[remainder_block_num] = _mm512_maskz_loadu_ps(
              mask, input + cols * i + cols - remainder_16);
          inputs[remainder_block_num] = _mm512_maskz_sub_ps(mask, inputs[remainder_block_num], means);
          inputs[remainder_block_num] = _mm512_maskz_mul_ps(mask, inputs[remainder_block_num], inputs[remainder_block_num]);
        }
        __m512 block_sum = reduce_sum_block_ps(inputs, remainder_block_num_total);
        rvariance[i] += _mm512_reduce_add_ps(block_sum);
      }

      rvariance[i] *= one_over_cols;
      rvariance[i] += epsilon;
      rvariance[i] = 1.0f / sqrtf(rvariance[i]);
      __m512 rvariances = _mm512_set1_ps(rvariance[i]);
      // Normalize and store
      for (int64 j = 0; j < block_num; ++j) {
        __m512 inputs[8];
        __m512 nums[8]; // used to load gammas and betas 
        auto load_normalize = [&](auto idx) {
          // (x - mean) / sqrt(var + eps)
          inputs[idx] = _mm512_loadu_ps(input + cols * i + 128 * j + 16 * idx);
          inputs[idx] = _mm512_sub_ps(inputs[idx], means);
          inputs[idx] = _mm512_mul_ps(inputs[idx], rvariances);
          // Mul gamma
          nums[idx] = _mm512_loadu_ps(gamma + 128 * j + 16 * idx);
          inputs[idx] = _mm512_mul_ps(inputs[idx], nums[idx]);
          // Add beta
          nums[idx] = _mm512_loadu_ps(beta + 128 * j + 16 * idx);
          inputs[idx] = _mm512_add_ps(inputs[idx], nums[idx]);

          // Store
          _mm512_storeu_ps(output + cols * i + 128 * j + 16 * idx, inputs[idx]);
        };
        compile_time_for<8>::op(load_normalize);
      }
      if (remainder_block_num_total) { // remainder normalize and store
        __m512 inputs;
        __m512 nums; // used to load gammas and betas 
        for (int64 idx = 0; idx < remainder_block_num; idx++){ // remainder of 128
          // (x - mean) / sqrt(var + eps)
          inputs = _mm512_loadu_ps(input + cols * i + cols - remainder_128 + 16 * idx);
          inputs = _mm512_sub_ps(inputs, means);
          inputs = _mm512_mul_ps(inputs, rvariances);
          // Mul gamma
          nums = _mm512_loadu_ps(gamma + cols - remainder_128 + 16 * idx);
          inputs = _mm512_mul_ps(inputs, nums);
          // Add beta
          nums = _mm512_loadu_ps(beta + cols - remainder_128 + 16 * idx);
          inputs = _mm512_add_ps(inputs, nums);

          // Store
          _mm512_storeu_ps(output + cols * i + cols - remainder_128 + 16 * idx, inputs);
        }
        if (remainder_16) { // remainder of 16
          __mmask16 mask = 0xFFFF >> (16 - remainder_16);
          // (x - mean) / sqrt(var + eps)
          inputs = _mm512_maskz_loadu_ps(mask, input + cols * i + cols - remainder_16);
          inputs = _mm512_maskz_sub_ps(mask, inputs, means);
          inputs = _mm512_maskz_mul_ps(mask, inputs, rvariances);
          // Mul gamma
          nums = _mm512_maskz_loadu_ps(mask, gamma + cols - remainder_16);
          inputs = _mm512_maskz_mul_ps(mask, inputs, nums);
          // Add beta
          nums = _mm512_maskz_loadu_ps(mask, beta + cols - remainder_16);
          inputs = _mm512_maskz_add_ps(mask, inputs, nums);

          // Store
          _mm512_mask_storeu_ps(output + cols * i + cols - remainder_16, mask, inputs);
        }
      }
    }
  }

#endif // forward layer norm avx512 impl
};

REGISTER_KERNEL_BUILDER(Name("FusedLayerNorm")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        FusedLayerNormOp<float>);

template <typename T>
class FusedLayerNormGradOp : public OpKernel {
 public:
  explicit FusedLayerNormGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  ~FusedLayerNormGradOp() {}

  void Compute(OpKernelContext* context) override {
    // Grab the input
    const Tensor* y_grad_tensor = &context->input(0);
    const Tensor* x_tensor = &context->input(1);
    const Tensor* mean_tensor = &context->input(2);
    const Tensor* rvariance_tensor = &context->input(3);
    const Tensor* gamma_tensor = &context->input(4);

    const T* y_grad = y_grad_tensor->flat<T>().data();
    const T* x = x_tensor->flat<T>().data();
    const float* mean = mean_tensor->flat<float>().data();
    const float* rvariance = rvariance_tensor->flat<float>().data();
    const float* gamma = gamma_tensor->flat<float>().data();

    int64 cols = x_tensor->dim_size(x_tensor->dims() - 1);
    int64 rows = mean_tensor->NumElements();
    
    // Create output tensors
    Tensor* x_grad_tensor = NULL;
    Tensor* gamma_grad_tensor = NULL;
    Tensor* beta_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, x_tensor->shape(),
                                                     &x_grad_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, gamma_tensor->shape(),
                                                     &gamma_grad_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, gamma_tensor->shape(),
                                                     &beta_grad_tensor));
    T* x_grad = x_grad_tensor->flat<T>().data();
    float* gamma_grad = gamma_grad_tensor->flat<float>().data();
    float* beta_grad = beta_grad_tensor->flat<float>().data();

    // Init
    memset(gamma_grad, 0, sizeof(float) * cols);
    memset(beta_grad, 0, sizeof(float) * cols);

    auto& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    thread::ThreadPool* thread_pool = worker_threads.workers;

#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
    const int total_unit = (rows >= 128 ? 8 : (rows + 15) / 16);
    const int64 rows_per_unit = (rows + total_unit - 1) / total_unit; 
    const int64 unit_cost = rows_per_unit * cols * 100;
    thread_pool->ParallelFor(total_unit, unit_cost, 
        [&](int64 begin_unit, int64 end_unit) 
        {auto begin_row = begin_unit * rows_per_unit; 
            auto end_row = end_unit * rows_per_unit; 
            if (end_row > rows) 
            {end_row = rows;}
            backward(y_grad, x, mean, rvariance, gamma, x_grad, gamma_grad,
                     beta_grad, cols, begin_row, end_row); 
        }); 
#else
    const float one_over_cols = 1.0f / cols;
    const int64 total_unit = (rows + 15) / 16;
    const int64 unit_cost =
        16 * cols * 100;  // assume every element consumes 100 cycles

    thread_pool->ParallelFor(
        total_unit, unit_cost, [&](int64 begin_unit, int64 end_unit) {
          auto begin_row = begin_unit * 16;
          auto end_row = end_unit * 16;
          if (end_row > rows) {
            end_row = rows;
          }
          backward(y_grad, x, mean, rvariance, gamma, x_grad, gamma_grad,
                   beta_grad, begin_row, end_row, cols, one_over_cols);
        });
#endif // backward compute
  }

 private:
#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
  void backward(const float* diff, const float* x, const float* mean,
                   const float* rvariance, const float* gamma, float* x_diff,
                   float* gamma_diff, float* beta_diff, int64 cols,
                   int begin_row, int end_row) {
    int i = begin_row;
    for (; i + 3 < end_row; i += 4) {
      backward_avx512<4>(diff, x, mean, rvariance, gamma, x_diff, gamma_diff,
                       beta_diff, cols, i);
    }
    for (; i < end_row; ++i) {
      backward_avx512<1>(diff, x, mean, rvariance, gamma, x_diff, gamma_diff,
                       beta_diff, cols, i);
    }
  }
#else
  // For gradient of x, it comes from 3 parts: x-mean, mean, and rvariance
  //   grad from (x - mean): y_grad * gamma * [rvariance]
  //   grad from mean: - sum_row(y_grad * gamma * [rvariance]) / #cols
  //   grad from rvariance: sum_row(y_grad * gamma * (x - mean)) * (- [rvariance]^3) * (x - mean) / #cols
  // For gradient of gamma, grad = y_grad * (x - mean) * rvariance
  // For gradient of beta, grad = y_grad
  void backward(const float* y_grad, const float* x, const float* mean,
                    const float* rvariance, const float* gamma, float* x_grad,
                    float* gamma_grad, float* beta_grad, int64 begin_row,
                    int64 end_row, int64 cols, const float one_over_cols) {
    for (int64 i = begin_row; i < end_row; ++i) {
      int64 j = 0;
      float sum_m = 0;
      float sum_r = 0;
      // sum_m: sum_row(y_grad * gamma)
      // sum_r: sum_row(y_grad * gamma * (x - mean))
      for (; j + 7 < cols; j += 8) {
        T data_0 = y_grad[i * cols + j] * gamma[j];
        T data_1 = y_grad[i * cols + j + 1] * gamma[j + 1];
        T data_2 = y_grad[i * cols + j + 2] * gamma[j + 2];
        T data_3 = y_grad[i * cols + j + 3] * gamma[j + 3];
        T data_4 = y_grad[i * cols + j + 4] * gamma[j + 4];
        T data_5 = y_grad[i * cols + j + 5] * gamma[j + 5];
        T data_6 = y_grad[i * cols + j + 6] * gamma[j + 6];
        T data_7 = y_grad[i * cols + j + 7] * gamma[j + 7];
        sum_m += data_0 + data_1 + data_2 + data_3 + 
                    data_4 + data_5 + data_6 + data_7;
        
        data_0 = data_0 * (x[i * cols + j] - mean[i]);
        data_1 = data_1 * (x[i * cols + j + 1] - mean[i]);
        data_2 = data_2 * (x[i * cols + j + 2] - mean[i]);
        data_3 = data_3 * (x[i * cols + j + 3] - mean[i]);
        data_4 = data_4 * (x[i * cols + j + 4] - mean[i]);
        data_5 = data_5 * (x[i * cols + j + 5] - mean[i]);
        data_6 = data_6 * (x[i * cols + j + 6] - mean[i]);
        data_7 = data_7 * (x[i * cols + j + 7] - mean[i]);
        sum_r += data_0 + data_1 + data_2 + data_3 + 
                    data_4 + data_5 + data_6 + data_7;
      }
      for (; j < cols; ++j) { // remainder
        sum_m += y_grad[i * cols + j] * gamma[j];
        sum_r += y_grad[i * cols + j] * gamma[j] * (x[i * cols + j] - mean[i]);
      }
      sum_m *= one_over_cols;
      sum_r *= rvariance[i] * rvariance[i];
      sum_r *= one_over_cols;

      for (j = 0; j + 7 < cols; j += 8) {
        x_grad[i * cols + j] = y_grad[i * cols + j] * gamma[j];
        x_grad[i * cols + j + 1] = y_grad[i * cols + j + 1] * gamma[j + 1];
        x_grad[i * cols + j + 2] = y_grad[i * cols + j + 2] * gamma[j + 2];
        x_grad[i * cols + j + 3] = y_grad[i * cols + j + 3] * gamma[j + 3];
        x_grad[i * cols + j + 4] = y_grad[i * cols + j + 4] * gamma[j + 4];
        x_grad[i * cols + j + 5] = y_grad[i * cols + j + 5] * gamma[j + 5];
        x_grad[i * cols + j + 6] = y_grad[i * cols + j + 6] * gamma[j + 6];
        x_grad[i * cols + j + 7] = y_grad[i * cols + j + 7] * gamma[j + 7];

        x_grad[i * cols + j] -= sum_m + sum_r * (x[i * cols + j] - mean[i]);
        x_grad[i * cols + j + 1] -= sum_m + sum_r * (x[i * cols + j + 1] - mean[i]);
        x_grad[i * cols + j + 2] -= sum_m + sum_r * (x[i * cols + j + 2] - mean[i]);
        x_grad[i * cols + j + 3] -= sum_m + sum_r * (x[i * cols + j + 3] - mean[i]);
        x_grad[i * cols + j + 4] -= sum_m + sum_r * (x[i * cols + j + 4] - mean[i]);
        x_grad[i * cols + j + 5] -= sum_m + sum_r * (x[i * cols + j + 5] - mean[i]);
        x_grad[i * cols + j + 6] -= sum_m + sum_r * (x[i * cols + j + 6] - mean[i]);
        x_grad[i * cols + j + 7] -= sum_m + sum_r * (x[i * cols + j + 7] - mean[i]);

        x_grad[i * cols + j] *= rvariance[i];
        x_grad[i * cols + j + 1] *= rvariance[i];
        x_grad[i * cols + j + 2] *= rvariance[i];
        x_grad[i * cols + j + 3] *= rvariance[i];
        x_grad[i * cols + j + 4] *= rvariance[i];
        x_grad[i * cols + j + 5] *= rvariance[i];
        x_grad[i * cols + j + 6] *= rvariance[i];
        x_grad[i * cols + j + 7] *= rvariance[i];
      }
      for (; j < cols; ++j) { // remainder
        x_grad[i * cols + j] = y_grad[i * cols + j] * gamma[j];
        x_grad[i * cols + j] -= sum_m + sum_r * (x[i * cols + j] - mean[i]);
        x_grad[i * cols + j] *= rvariance[i];
      }

      // grad of gamma
      for (j = 0; j + 7 < cols; j += 8) {
        gamma_grad[j] += y_grad[i * cols + j] * (x[i * cols + j] - mean[i]) * rvariance[i];
        gamma_grad[j + 1] += y_grad[i * cols + j + 1] * (x[i * cols + j + 1] - mean[i]) * rvariance[i];
        gamma_grad[j + 2] += y_grad[i * cols + j + 2] * (x[i * cols + j + 2] - mean[i]) * rvariance[i];
        gamma_grad[j + 3] += y_grad[i * cols + j + 3] * (x[i * cols + j + 3] - mean[i]) * rvariance[i];
        gamma_grad[j + 4] += y_grad[i * cols + j + 4] * (x[i * cols + j + 4] - mean[i]) * rvariance[i];
        gamma_grad[j + 5] += y_grad[i * cols + j + 5] * (x[i * cols + j + 5] - mean[i]) * rvariance[i];
        gamma_grad[j + 6] += y_grad[i * cols + j + 6] * (x[i * cols + j + 6] - mean[i]) * rvariance[i];
        gamma_grad[j + 7] += y_grad[i * cols + j + 7] * (x[i * cols + j + 7] - mean[i]) * rvariance[i];
      }
      for (; j < cols; ++j) { // remainder
        gamma_grad[j] += y_grad[i * cols + j] * (x[i * cols + j] - mean[i]) * rvariance[i];
      }

      // grad of beta
      for (j = 0; j + 7 < cols; j += 8) {
        beta_grad[j] += y_grad[i * cols + j];
        beta_grad[j + 1] += y_grad[i * cols + j + 1];
        beta_grad[j + 2] += y_grad[i * cols + j + 2];
        beta_grad[j + 3] += y_grad[i * cols + j + 3];
        beta_grad[j + 4] += y_grad[i * cols + j + 4];
        beta_grad[j + 5] += y_grad[i * cols + j + 5];
        beta_grad[j + 6] += y_grad[i * cols + j + 6];
        beta_grad[j + 7] += y_grad[i * cols + j + 7];
      }
      for (; j < cols; ++j) { // remainder
        beta_grad[j] += y_grad[i * cols + j];
      }
    }
  }
#endif // backward define

#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
  template <int ROWS>
  inline void backward_avx512(const float* y_grad, const float* x,
                              const float* mean, const float* rvariance,
                              const float* gamma, float* x_grad,
                              float* gamma_grad, float* beta_grad, int64 cols,
                              int64 start_row) {
    float sum_m[ROWS], sum_r[ROWS];
    __m512 vsum_m[ROWS], vsum_r[ROWS], vmean[ROWS], vrvariance[ROWS];

    // Init
    auto setzero = [&](auto idx) {
      vsum_m[idx] = _mm512_setzero_ps();
      vsum_r[idx] = _mm512_setzero_ps();
      vmean[idx] = _mm512_set1_ps(mean[start_row + idx]);
      vrvariance[idx] = _mm512_set1_ps(rvariance[start_row + idx]);
    };
    compile_time_for<ROWS>::op(setzero);

    // Compute sum for y_grad * gamma and y_grad * gamma * (x - mean)
    int64 j = 0;
    for (; j + 15 < cols; j += 16) {
      auto compute_sum = [&](auto idx) {
        __m512 vy_grad = _mm512_loadu_ps(y_grad + (start_row + idx) * cols + j);
        __m512 vgamma = _mm512_loadu_ps(gamma + j);

        __m512 mul = _mm512_mul_ps(vy_grad, vgamma);
        vsum_m[idx] = _mm512_add_ps(mul, vsum_m[idx]);

        __m512 vx = _mm512_loadu_ps(x + (start_row + idx) * cols + j);
        __m512 x_minus_mean = _mm512_sub_ps(vx, vmean[idx]);
        vsum_r[idx] = _mm512_fmadd_ps(mul, x_minus_mean, vsum_r[idx]);
      };

      compile_time_for<ROWS>::op(compute_sum);
    }

    auto reduce_sum = [&](auto idx) {
      sum_m[idx] = horizontal_add(vsum_m[idx]);
      sum_r[idx] = horizontal_add(vsum_r[idx]);

      for (int64 c = j; c < cols; ++c) {
        const auto offset = (start_row + idx) * cols + c;
        sum_m[idx] += y_grad[offset] * gamma[c];
        sum_r[idx] +=
            y_grad[offset] * gamma[c] * (x[offset] - mean[start_row + idx]);
      }

      sum_m[idx] /= cols;
      sum_r[idx] *= rvariance[start_row + idx] * rvariance[start_row + idx];
      sum_r[idx] /= cols;

      vsum_m[idx] = _mm512_set1_ps(sum_m[idx]);
      vsum_r[idx] = _mm512_set1_ps(sum_r[idx]);
    };

    compile_time_for<ROWS>::op(reduce_sum);

    // Compute gradient for x, gamma, beta
    for (j = 0; j + 15 < cols; j += 16) {
      __m512 vgamma_grad = _mm512_loadu_ps(gamma_grad + j);
      __m512 vbeta_grad = _mm512_loadu_ps(beta_grad + j);

      auto compute_grad = [&](auto idx) {
        __m512 vy_grad = _mm512_loadu_ps(y_grad + (start_row + idx) * cols + j);
        __m512 vgamma = _mm512_loadu_ps(gamma + j);

        __m512 vx_grad = _mm512_mul_ps(vy_grad, vgamma);

        __m512 vx = _mm512_loadu_ps(x + (start_row + idx) * cols + j);
        __m512 x_minus_mean = _mm512_sub_ps(vx, vmean[idx]);

        vx_grad = _mm512_sub_ps(
            vx_grad, _mm512_fmadd_ps(vsum_r[idx], x_minus_mean, vsum_m[idx]));
        vx_grad = _mm512_mul_ps(vx_grad, vrvariance[idx]);

        // save gradient of x
        _mm512_storeu_ps(x_grad + (start_row + idx) * cols + j, vx_grad);

        // gradient for gamma and beta
        vgamma_grad = _mm512_fmadd_ps(_mm512_mul_ps(vy_grad, x_minus_mean),
                                      vrvariance[idx], vgamma_grad);
        vbeta_grad = _mm512_add_ps(vy_grad, vbeta_grad);
      };

      compile_time_for<ROWS>::op(compute_grad);

      // save gradient of gamma, beta
      _mm512_storeu_ps(gamma_grad + j, vgamma_grad);
      _mm512_storeu_ps(beta_grad + j, vbeta_grad);
    }

    // Deal with the remain data
    if (cols % 16 != 0) {
      int remain = cols % 16;
      auto remain_grad = [&](auto idx) {
        for (int64 c = j; c < cols; ++c) {
          const auto offset = (start_row + idx) * cols + c;
          float vx_grad = y_grad[offset] * gamma[c];
          float x_minus_mean = x[offset] - mean[start_row + idx];
          vx_grad -= sum_m[idx] + sum_r[idx] * x_minus_mean;
          vx_grad *= rvariance[start_row + idx];

          // save gradient of x
          x_grad[offset] = vx_grad;

          // gradient for gamma and beta
          gamma_grad[c] +=
              y_grad[offset] * x_minus_mean * rvariance[start_row + idx];
          beta_grad[c] += y_grad[offset];
        }
      };

      compile_time_for<ROWS>::op(remain_grad);
    }
  }
#endif // backward layer norm avx512 impl
};

REGISTER_KERNEL_BUILDER(Name("FusedLayerNormGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        FusedLayerNormGradOp<float>);
