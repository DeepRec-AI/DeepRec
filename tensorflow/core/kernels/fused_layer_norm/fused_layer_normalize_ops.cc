#include "compile_util.h"
#include "ln_util.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"

using namespace tensorflow;

template <typename T>
class FusedLayerNormOp : public OpKernel
{
private:
    float epsilon;

public:
    explicit FusedLayerNormOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    }

    ~FusedLayerNormOp()
    {
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input
        const Tensor *input_tensor = &context->input(0);
        const Tensor *gamma_tensor = &context->input(1);
        const Tensor *beta_tensor = &context->input(2);

        const T *input = input_tensor->flat<T>().data();
        const float *gamma = gamma_tensor->flat<float>().data();
        const float *beta = beta_tensor->flat<float>().data();

        // To check the input
        OP_REQUIRES(context, (input_tensor->dims() >= 2),
                    errors::InvalidArgument("Input dimension should be >= 2"));
        OP_REQUIRES(context, (gamma_tensor->dims() == 1),
                    errors::InvalidArgument("dims(gamma) != 1"));
        OP_REQUIRES(context, (beta_tensor->dims() == 1),
                    errors::InvalidArgument("dims(beta) != 1"));

        int64 cols = input_tensor->dim_size(input_tensor->dims() - 1);
        OP_REQUIRES(context, (gamma_tensor->dim_size(0) == cols),
                    errors::InvalidArgument("size(gamma) != last_dim_size_of_input"));
        OP_REQUIRES(context, (beta_tensor->dim_size(0) == cols),
                    errors::InvalidArgument("size(beta) != last_dim_size_of_input"));

        int64 rows = 1;
        TensorShape mean_var_shape;
        for (int i = 0; i < input_tensor->dims() - 1; ++i)
        {
            auto dim_size = input_tensor->dim_size(i);
            rows *= dim_size;
            mean_var_shape.AddDim(dim_size);
        }

        // Create output tensors
        Tensor *output_tensor = NULL;
        Tensor *mean_tensor = NULL;
        Tensor *rvariance_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor->shape(),
                                                         &output_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, mean_var_shape,
                                                         &mean_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(2, mean_var_shape,
                                                         &rvariance_tensor));
        T *output = output_tensor->flat<T>().data();
        float *mean = mean_tensor->flat<float>().data();
        float *rvariance = rvariance_tensor->flat<float>().data();

        // Do it
        // Let every thread compute 16 rows to avoid false sharing
        const int64 total_unit = (rows + 15) / 16;
        const int64 unit_cost = 16 * cols * 50; // assume every element consumes 50 cycles

        auto &worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
        thread::ThreadPool *thread_pool = worker_threads.workers;

        thread_pool->ParallelFor(
            total_unit, unit_cost,
            [&](int64 begin_unit, int64 end_unit)
            {
                auto begin_row = begin_unit * 16;
                auto end_row = end_unit * 16;
                if (end_row > rows)
                {
                    end_row = rows;
                }
                forward(input, gamma, beta,
                        output, mean, rvariance,
                        cols, begin_row, end_row);
            });
    }

private:
    // Compute the rows locate in the range of [begin_row, end_row)
    void forward(const float *input, const float *gamma, const float *beta,
                 float *output, float *mean, float *rvariance,
                 int64 cols, int begin_row, int end_row)
    {
        //printf("begin_row=%lld, end_row=%lld\n", begin_row, end_row);

        int i = begin_row;
        for (; i + 3 < end_row; i += 4)
        {
            forward_avx3<4>(input, gamma, beta, output, mean, rvariance, cols, i);
        }
        for (; i < end_row; ++i)
        {
            forward_avx3<1>(input, gamma, beta, output, mean, rvariance, cols, i);
        }
    }

    // Compute the rows locate in the range of [begin_row, begin_row + ROWS)
    template <int ROWS>
    inline void forward_avx3(const float *input, const float *gamma, const float *beta,
                             float *output, float *mean, float *rvariance,
                             int64 cols, int begin_row)
    {
        const float one_over_cols = 1.0f / cols;

        const float *px = input + begin_row * cols;
        float *py = output + begin_row * cols;

        float tmean[ROWS], tvar[ROWS]; // for temporary result
        __m512 vmean[ROWS], vvar[ROWS];

        // Init
        auto setzero = [&](auto idx)
        {
            vmean[idx] = _mm512_setzero_ps();
            vvar[idx] = _mm512_setzero_ps();
        };
        compile_time_for<ROWS>::op(setzero);

        // Sum
        int64 j = 0;
        for (; j + 15 < cols; j += 16)
        {
            auto compute_sum = [&](auto idx)
            {
                __m512 vx = _mm512_loadu_ps(px + idx * cols + j);
                vmean[idx] = _mm512_add_ps(vx, vmean[idx]);
            };
            compile_time_for<ROWS>::op(compute_sum);
        }

        // Mean: reduce the result and add remain elements, and average it
        auto reduce_mean = [&](auto idx)
        {
            tmean[idx] = LnUtil::horizontal_add(vmean[idx]);
            for (int r = j; r < cols; ++r)
            {
                tmean[idx] += *(px + idx * cols + r);
            }
            tmean[idx] *= one_over_cols;
            // save mean
            mean[begin_row + idx] = tmean[idx];
            vmean[idx] = _mm512_set1_ps(tmean[idx]);
        };
        compile_time_for<ROWS>::op(reduce_mean);

        // variance
        for (j = 0; j + 15 < cols; j += 16)
        {
            auto compute_variance = [&](auto idx)
            {
                __m512 vx = _mm512_loadu_ps(px + idx * cols + j);
                __m512 tmp = _mm512_sub_ps(vx, vmean[idx]);
                vvar[idx] = _mm512_fmadd_ps(tmp, tmp, vvar[idx]);
            };
            compile_time_for<ROWS>::op(compute_variance);
        }

        auto reduce_rvariance = [&](auto idx)
        {
            tvar[idx] = LnUtil::horizontal_add(vvar[idx]);
            for (int r = j; r < cols; ++r)
            {
                auto x = *(px + idx * cols + r);
                tvar[idx] += (x - tmean[idx]) * (x - tmean[idx]);
            }
            tvar[idx] *= one_over_cols;
            tvar[idx] += epsilon;
            tvar[idx] = 1.0f / sqrtf(tvar[idx]);
            // save rvariance
            rvariance[begin_row + idx] = tvar[idx];
            vvar[idx] = _mm512_set1_ps(tvar[idx]);
        };
        compile_time_for<ROWS>::op(reduce_rvariance);

        // Compute norm and save
        for (j = 0; j + 15 < cols; j += 16)
        {
            auto compute_norm = [&](auto idx)
            {
                // (x - mean) / variance
                __m512 vx = _mm512_loadu_ps(px + idx * cols + j);
                __m512 norm = _mm512_sub_ps(vx, vmean[idx]);
                norm = _mm512_mul_ps(norm, vvar[idx]);

                // * gamma
                __m512 vgamma = _mm512_loadu_ps(gamma + j);
                norm = _mm512_mul_ps(norm, vgamma);

                // + beta
                __m512 vbeta = _mm512_loadu_ps(beta + j);
                norm = _mm512_add_ps(norm, vbeta);

                _mm512_storeu_ps(py + idx * cols + j, norm);
            };
            compile_time_for<ROWS>::op(compute_norm);
        }
        auto remain_norm = [&](auto idx)
        {
            for (int r = j; r < cols; ++r)
            {
                auto x = *(px + idx * cols + r);
                //printf("x=%f, mean=%f, rvar=%f, gmma=%f, beta=%f\n", x, tmean[idx], tvar[idx], gamma[r], beta[r]);
                py[idx * cols + r] = (x - tmean[idx]) * tvar[idx] * gamma[r] + beta[r];
            }
        };
        compile_time_for<ROWS>::op(remain_norm);
    }
};

REGISTER_KERNEL_BUILDER(Name("FusedLayerNorm")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        FusedLayerNormOp<float>);


template <typename T>
class FusedLayerNormGradOp : public OpKernel
{
public:
    explicit FusedLayerNormGradOp(OpKernelConstruction *context) : OpKernel(context)
    {
    }

    ~FusedLayerNormGradOp()
    {
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input
        const Tensor *diff_tensor = &context->input(0);
        const Tensor *x_tensor = &context->input(1);
        const Tensor *mean_tensor = &context->input(2);
        const Tensor *rvariance_tensor = &context->input(3);
        const Tensor *gamma_tensor = &context->input(4);

        const T *diff = diff_tensor->flat<T>().data();
        const T *x = x_tensor->flat<T>().data();
        const float *mean = mean_tensor->flat<float>().data();
        const float *rvariance = rvariance_tensor->flat<float>().data();
        const float *gamma = gamma_tensor->flat<float>().data();

        int64 cols = x_tensor->dim_size(x_tensor->dims() - 1);
        int64 rows = mean_tensor->NumElements();

        // Create output tensors
        Tensor *x_diff_tensor = NULL;
        Tensor *gamma_diff_tensor = NULL;
        Tensor *beta_diff_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, x_tensor->shape(),
                                                         &x_diff_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, gamma_tensor->shape(),
                                                         &gamma_diff_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(2, gamma_tensor->shape(),
                                                         &beta_diff_tensor));
        T *x_diff = x_diff_tensor->flat<T>().data();
        float *gamma_diff = gamma_diff_tensor->flat<float>().data();
        float *beta_diff = beta_diff_tensor->flat<float>().data();

        // backward_ref(diff, x, mean, rvariance, gamma, x_diff, gamma_diff, beta_diff, rows, cols);
        // return;

        // Do it in parallel
        const int units = (rows >= 128 ? 8 : (rows + 15) / 16);
        const int64 rows_per_unit = (rows + units - 1) / units;
        const int64 unit_cost = rows_per_unit * cols * 100; // assume every element consumes 100 cycles

        auto &worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
        thread::ThreadPool *thread_pool = worker_threads.workers;

        // Allocate temporary buffer & set to 0 for accumulation
        float *t_gamma_diff = (float *)aligned_alloc(64, units * cols * sizeof(float));
        float *t_beta_diff = (float *)aligned_alloc(64, units * cols * sizeof(float));
        memset(t_gamma_diff, 0, units * cols * sizeof(float));
        memset(t_beta_diff, 0, units * cols * sizeof(float));

        thread_pool->ParallelFor(
            units, unit_cost,
            [&](int64 begin_unit, int64 end_unit)
            {
                auto begin_row = begin_unit * rows_per_unit;
                auto end_row = end_unit * rows_per_unit;
                if (end_row > rows)
                {
                    end_row = rows;
                }
                backward(diff, x, mean, rvariance, gamma,
                         x_diff, t_gamma_diff + begin_unit * cols, t_beta_diff + begin_unit * cols,
                         cols, begin_row, end_row);
            });

        // Reduce/sum N records into one
        add_n(t_gamma_diff, gamma_diff, units, cols);
        add_n(t_beta_diff, beta_diff, units, cols);

        free(t_gamma_diff);
        free(t_beta_diff);
    }

private:
    void backward(const float *diff, const float *x, const float *mean, const float *rvariance,
                  const float *gamma, float *x_diff, float *gamma_diff, float *beta_diff,
                  int64 cols, int begin_row, int end_row)
    {
        int i = begin_row;
        for (; i + 3 < end_row; i += 4)
        {
            backward_avx3<4>(diff, x, mean, rvariance, gamma, x_diff, gamma_diff, beta_diff, cols, i);
        }
        for (; i < end_row; ++i)
        {
            backward_avx3<1>(diff, x, mean, rvariance, gamma, x_diff, gamma_diff, beta_diff, cols, i);
        }
    }

    // look into backward_ref for more
    template <int ROWS>
    inline void backward_avx3(const float *diff, const float *x, const float *mean, const float *rvariance,
                              const float *gamma, float *x_diff, float *gamma_diff, float *beta_diff,
                              int64 cols, int64 start_row)
    {
        float sum_m[ROWS], sum_r[ROWS];
        __m512 vsum_m[ROWS], vsum_r[ROWS], vmean[ROWS], vrvariance[ROWS];

        // Init
        auto setzero = [&](auto idx)
        {
            vsum_m[idx] = _mm512_setzero_ps();
            vsum_r[idx] = _mm512_setzero_ps();
            vmean[idx] = _mm512_set1_ps(mean[start_row + idx]);
            vrvariance[idx] = _mm512_set1_ps(rvariance[start_row + idx]);
        };
        compile_time_for<ROWS>::op(setzero);

        // Compute sum for diff * gamma and diff * gamma * (x - mean)
        int64 j = 0;
        for (; j + 15 < cols; j += 16)
        {
            auto compute_sum = [&](auto idx)
            {
                __m512 vdiff = _mm512_loadu_ps(diff + (start_row + idx) * cols + j);
                __m512 vgamma = _mm512_loadu_ps(gamma + j);

                __m512 mul = _mm512_mul_ps(vdiff, vgamma);
                vsum_m[idx] = _mm512_add_ps(mul, vsum_m[idx]);

                __m512 vx = _mm512_loadu_ps(x + (start_row + idx) * cols + j);
                __m512 x_minus_mean = _mm512_sub_ps(vx, vmean[idx]);
                vsum_r[idx] = _mm512_fmadd_ps(mul, x_minus_mean, vsum_r[idx]);
            };

            compile_time_for<ROWS>::op(compute_sum);
        }

        auto reduce_sum = [&](auto idx)
        {
            sum_m[idx] = LnUtil::horizontal_add(vsum_m[idx]);
            sum_r[idx] = LnUtil::horizontal_add(vsum_r[idx]);

            for (int64 c = j; c < cols; ++c)
            {
                const auto offset = (start_row + idx) * cols + c;
                sum_m[idx] += diff[offset] * gamma[c];
                sum_r[idx] += diff[offset] * gamma[c] * (x[offset] - mean[start_row + idx]);
            }

            sum_m[idx] /= cols;
            sum_r[idx] *= rvariance[start_row + idx] * rvariance[start_row + idx];
            sum_r[idx] /= cols;

            vsum_m[idx] = _mm512_set1_ps(sum_m[idx]);
            vsum_r[idx] = _mm512_set1_ps(sum_r[idx]);
        };

        compile_time_for<ROWS>::op(reduce_sum);

        // Compute gradient for x, gamma, beta
        for (j = 0; j + 15 < cols; j += 16)
        {
            __m512 vgamma_diff = _mm512_loadu_ps(gamma_diff + j);
            __m512 vbeta_diff = _mm512_loadu_ps(beta_diff + j);

            auto compute_diff = [&](auto idx)
            {
                __m512 vdiff = _mm512_loadu_ps(diff + (start_row + idx) * cols + j);
                __m512 vgamma = _mm512_loadu_ps(gamma + j);

                __m512 v_diff_x = _mm512_mul_ps(vdiff, vgamma);

                __m512 vx = _mm512_loadu_ps(x + (start_row + idx) * cols + j);
                __m512 x_minus_mean = _mm512_sub_ps(vx, vmean[idx]);

                v_diff_x = _mm512_sub_ps(v_diff_x, _mm512_fmadd_ps(vsum_r[idx], x_minus_mean, vsum_m[idx]));
                v_diff_x = _mm512_mul_ps(v_diff_x, vrvariance[idx]);

                // save gradient of x
                _mm512_storeu_ps(x_diff + (start_row + idx) * cols + j, v_diff_x);

                // gradient for gamma and beta
                vgamma_diff = _mm512_fmadd_ps(_mm512_mul_ps(vdiff, x_minus_mean), vrvariance[idx], vgamma_diff);
                vbeta_diff = _mm512_add_ps(vdiff, vbeta_diff);
            };

            compile_time_for<ROWS>::op(compute_diff);

            // save gradient of gamma, beta
            _mm512_storeu_ps(gamma_diff + j, vgamma_diff);
            _mm512_storeu_ps(beta_diff + j, vbeta_diff);
        }

        // Deal with the remain data
        if (cols % 16 != 0)
        {
            int remain = cols % 16;
            //memset(gamma_diff + j, 0, remain * sizeof(float));
            //memset(beta_diff + j, 0, remain * sizeof(float));

            auto remain_diff = [&](auto idx)
            {
                for (int64 c = j; c < cols; ++c)
                {
                    const auto offset = (start_row + idx) * cols + c;
                    float v_diff_x = diff[offset] * gamma[c];
                    float x_minus_mean = x[offset] - mean[start_row + idx];
                    v_diff_x -= sum_m[idx] + sum_r[idx] * x_minus_mean;
                    v_diff_x *= rvariance[start_row + idx];

                    // save gradient of x
                    x_diff[offset] = v_diff_x;

                    // gradient for gamma and beta
                    gamma_diff[c] += diff[offset] * x_minus_mean * rvariance[start_row + idx];
                    beta_diff[c] += diff[offset];
                }
            };

            compile_time_for<ROWS>::op(remain_diff);
        }
    }

    void backward_ref(const float *diff, const float *x, const float *mean, const float *rvariance,
                      const float *gamma, float *x_diff, float *gamma_diff, float *beta_diff,
                      int64 rows, int64 cols)
    {
        //printf("in backward_ref\n");
        memset(gamma_diff, 0, cols * sizeof(float));
        memset(beta_diff, 0, cols * sizeof(float));

        // For gradient of x, it comes from 3 parts: x-mean, mean, and rvariance
        // grad from (x - mean): diff * gamma * rvariance
        // grad from mean: - sum_row(diff * gamma * rvariance) / #cols
        // grad from rvariance: sum_row(diff * gamma * (x - mean)) * (- rvariance^3) * (x - mean) / #cols
        for (int64 r = 0; r < rows; ++r)
        {
            float sum_m = 0, sum_r = 0;
            for (int64 c = 0; c < cols; ++c)
            {
                const auto idx = r * cols + c;
                sum_m += diff[idx] * gamma[c];
                sum_r += diff[idx] * gamma[c] * (x[idx] - mean[r]);
            }

            sum_m /= cols;
            sum_r *= rvariance[r] * rvariance[r];
            sum_r /= cols;

            for (int64 c = 0; c < cols; ++c)
            {
                const auto idx = r * cols + c;
                float v_diff_x = diff[idx] * gamma[c];
                v_diff_x -= sum_m + sum_r * (x[idx] - mean[r]);
                v_diff_x *= rvariance[r]; // rvariance is the common factor for 3 parts

                x_diff[idx] = v_diff_x;
            }

            // For gradient of gamma & beta
            for (int64 c = 0; c < cols; ++c)
            {
                const auto idx = r * cols + c;
                gamma_diff[c] += diff[idx] * (x[idx] - mean[r]) * rvariance[r];
                beta_diff[c] += diff[idx];
            }
        } // end for r
    }

    void add_n(const float *src, float *dst, int rows, int64 cols)
    {
        int64 c = 0;
        for (; c + 15 < cols; c += 16)
        {
            __m512 sum = _mm512_set1_ps(0);
            auto offset = c;
            for (int r = 0; r < rows; ++r)
            {
                sum = _mm512_add_ps(_mm512_loadu_ps(src + offset), sum);
                offset += cols;
            }
            _mm512_storeu_ps(dst + c, sum);
        }
        // Remain data
        for (; c < cols; ++c)
        {
            float sum = 0;
            auto offset = c;
            for (int r = 0; r < rows; ++r) {
                sum += src[offset];
                offset += cols;
            }
            dst[c] = sum;
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("FusedLayerNormGrad")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        FusedLayerNormGradOp<float>);
