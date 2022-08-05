#ifndef TENSORFLOW_CORE_KERNELS_FUSED_LAYER_NORMALIZE_COMPILE_UTIL_OP_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_LAYER_NORMALIZE_COMPILE_UTIL_OP_H_

#include <type_traits>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"


#include <cassert>
#include <cstdio>
#include <cmath>
#include <limits>
#include <immintrin.h>

using namespace tensorflow;
// A class for forced loop unrolling at compile time
template <int i>
struct compile_time_for {
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda& function, Args... args) {
        compile_time_for<i-1>::op(function, args...);
        function(std::integral_constant<int, i-1>{}, args...);
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
    inline static void op(const Lambda& function, Args... args) {
    }
};
#ifdef __AVX512F__
#define INDEX(x, y, ld) ((x) * (ld) + (y))
#define ADDRESS(p, x, y, ld) ((p) + (x) * (ld) + (y))

class LnUtil
{
public:
    // 4 AVX512 horizontal add, and 4 float value saved to a SSE register
    static inline __m128 horizontal_add(__m512 &v0, __m512 &v1, __m512 &v2, __m512 &v3)
    {
        __m128 r0, r1, r2, r3;

        {
            __m512 tmp = _mm512_shuffle_f32x4(v0, v1, _MM_SHUFFLE(1, 0, 3, 2));
            __m512 v0_hadd = _mm512_add_ps(v0, tmp);
            __m512 v1_hadd = _mm512_add_ps(v1, tmp);
            tmp = _mm512_shuffle_f32x4(v0_hadd, v1_hadd, _MM_SHUFFLE(2, 3, 0, 1));
            r0 = _mm512_castps512_ps128(_mm512_add_ps(v0_hadd, tmp));
            v1_hadd = _mm512_add_ps(v1_hadd, tmp);
            r1 = _mm512_extractf32x4_ps(v1_hadd, 2);
        }

        {
            __m512 tmp = _mm512_shuffle_f32x4(v2, v3, _MM_SHUFFLE(1, 0, 3, 2));
            __m512 v2_hadd = _mm512_add_ps(v2, tmp);
            __m512 v3_hadd = _mm512_add_ps(v3, tmp);
            tmp = _mm512_shuffle_f32x4(v2_hadd, v3_hadd, _MM_SHUFFLE(2, 3, 0, 1));
            r2 = _mm512_castps512_ps128(_mm512_add_ps(v2_hadd, tmp));
            v3_hadd = _mm512_add_ps(v3_hadd, tmp);
            r3 = _mm512_extractf32x4_ps(v3_hadd, 2);
        }

        r0 = _mm_hadd_ps(r0, r1);
        r2 = _mm_hadd_ps(r2, r3);
        r0 = _mm_hadd_ps(r0, r2);
        return r0;
    }

    static inline float horizontal_add(__m512 src)
    {
        __m512 tmp = _mm512_add_ps(src, _mm512_shuffle_f32x4(src, src, _MM_SHUFFLE(1, 0, 3, 2)));
        __m128 r = _mm512_castps512_ps128(_mm512_add_ps(tmp, _mm512_shuffle_f32x4(tmp, tmp, _MM_SHUFFLE(2, 3, 0, 1))));
        r = _mm_hadd_ps(r, r);
        return _mm_cvtss_f32(_mm_hadd_ps(r, r));
    }

    // AVX512 horizontal maximum
    static inline void horizontal_max(__m512 &v0)
    {
        // 512 bits -> 256 bits
        __m512 tmp = _mm512_shuffle_f32x4(v0, v0, _MM_SHUFFLE(1, 0, 3, 2));
        v0 = _mm512_max_ps(v0, tmp);

        // 256 -> 128 bits
        tmp = _mm512_shuffle_f32x4(v0, v0, _MM_SHUFFLE(2, 3, 0, 1));
        v0 = _mm512_max_ps(v0, tmp);

        // 128 bits -> 64 bits
        tmp = _mm512_shuffle_ps(v0, v0, _MM_SHUFFLE(1, 0, 3, 2));
        v0 = _mm512_max_ps(v0, tmp);

        // 64 bits -> 32 bits
        tmp = _mm512_shuffle_ps(v0, v0, _MM_SHUFFLE(2, 3, 0, 1));
        v0 = _mm512_max_ps(v0, tmp);
    }

    static __m512 exp_avx512(__m512 src)
    {
        // exp(x) =
        // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
        // = 2^n * exp(r) // simplify the exp(n*ln(2)) expression

        // General max number is 88.722839f, we lower the boudary to make sure 2^n is not NaN
        src = _mm512_min_ps(src, _mm512_set1_ps(88.376262f));
        src = _mm512_max_ps(src, _mm512_set1_ps(-87.336548f));
        __m512 aux1 = src;

        // fx = x * log2ef + 0.5
        src = _mm512_fmadd_ps(src, _mm512_set1_ps(1.442695f), _mm512_set1_ps(0.5f));

        // tmp = floorf(fx)
        __m512 aux2 = _mm512_floor_ps(src);

        // x = x - fx * ln2
        aux1 = _mm512_fmadd_ps(aux2, _mm512_set1_ps(-0.693147f), aux1);

        // compute 2^n
        __m512i aux3 = _mm512_add_epi32(_mm512_cvtps_epi32(aux2), _mm512_set1_epi32(127));
        aux3 = _mm512_slli_epi32(aux3, 23);

        // compute polynomial
        const float p1 = 0.999999701f;
        const float p2 = 0.499991506f;
        const float p3 = 0.166676521f;
        const float p4 = 0.0418978221f;
        const float p5 = 0.00828929059f;
        src = _mm512_set1_ps(p5);
        src = _mm512_fmadd_ps(src, aux1, _mm512_set1_ps(p4));
        src = _mm512_fmadd_ps(src, aux1, _mm512_set1_ps(p3));
        src = _mm512_fmadd_ps(src, aux1, _mm512_set1_ps(p2));
        src = _mm512_fmadd_ps(src, aux1, _mm512_set1_ps(p1));
        src = _mm512_fmadd_ps(src, aux1, _mm512_set1_ps(1));

        // y = y * 2^n
        src = _mm512_mul_ps(src, _mm512_castsi512_ps(aux3));

        return src;
    }

    static float exp_s(float src)
    {
        // exp(x) =
        // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
        // = 2^n * exp(r) // simplify the exp(n*ln(2)) expression
        src = std::fmin(src, 88.722839f); //88.376262
        src = std::fmax(src, -87.336548f);
        float aux1 = src;

        // calculate exp(x)
        // fx = x * log2ef + 0.5
        src = src * 1.442695f + 0.5;

        // tmp = floorf(fx)
        float aux2 = floorf(src);

        // keep vmm_src = fx for further computations
        src = aux2;

        // x = x - fx * ln2
        aux1 = aux1 - aux2 * 0.693147f;

        // We do not count 2^n here, because n can reach 128 and 2^128 is not
        // representable by fp32, so to get around this problem, instead of computing
        // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
        // and 2 are numbers representable in fp32.

        // compute 2^(n-1)
        src = src - 1;
        int aux3 = (int)src;

        aux3 = aux3 + 127;
        const int n_mantissa_bits = 23;
        aux3 = aux3 << n_mantissa_bits;

        // compute polynomial
        const float p1 = 0.999999701f;
        const float p2 = 0.499991506f;
        const float p3 = 0.166676521f;
        const float p4 = 0.0418978221f;
        const float p5 = 0.00828929059f;
        src = p5;
        src = src * aux1 + p4;
        src = src * aux1 + p3;
        src = src * aux1 + p2;
        src = src * aux1 + p1;
        src = src * aux1 + 1;

        // y = y * 2^n
        src = src * (*(float *)&aux3);
        src = src * 2;

        return src;
    }

    static void dump(const char *str, __m512 vec)
    {
        float v[16];
        _mm512_storeu_ps(v, vec);
        printf("%s: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
               str, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
               v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
    }
};


// Compute the rows locate in the range of [begin_row, begin_row + ROWS)
template <int ROWS>
inline void forward_avx3(const float *input, const float *gamma, const float *beta,
                            float *output, float *mean, float *rvariance,
                            int64 cols, int begin_row, float epsilon)
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
            py[idx * cols + r] = (x - tmean[idx]) * tvar[idx] * gamma[r] + beta[r];
        }
    };
    compile_time_for<ROWS>::op(remain_norm);
}

// Compute the rows locate in the range of [begin_row, end_row)
void forward_pj(const float *input, const float *gamma, const float *beta,
                float *output, float *mean, float *rvariance,
                int64 cols, int begin_row, int end_row, float epsilon)
{
    int i = begin_row;
    for (; i + 3 < end_row; i += 4)
    {
        forward_avx3<4>(input, gamma, beta, output, mean, rvariance, cols, i, epsilon);
    }
    for (; i < end_row; ++i)
    {
        forward_avx3<1>(input, gamma, beta, output, mean, rvariance, cols, i, epsilon);
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

void backward_pj(const float *diff, const float *x, const float *mean, const float *rvariance,
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

#define backwardpj  const int units = (rows >= 128 ? 8 : (rows + 15) / 16); \
        const int64 rows_per_unit = (rows + units - 1) / units; \
        float *t_gamma_diff = (float *)aligned_alloc(64, units * cols * sizeof(float)); \
        float *t_beta_diff = (float *)aligned_alloc(64, units * cols * sizeof(float)); \
        memset(t_gamma_diff, 0, units * cols * sizeof(float)); \
        memset(t_beta_diff, 0, units * cols * sizeof(float)); \
        thread_pool->ParallelFor(units, unit_cost, \
            [&](int64 begin_unit, int64 end_unit) \
            {auto begin_row = begin_unit * rows_per_unit; \
                auto end_row = end_unit * rows_per_unit; \
                if (end_row > rows) \
                {end_row = rows;} \
                backward_pj(y_grad, x, mean, rvariance, gamma, \
                            x_grad, t_gamma_diff + begin_unit * cols, t_beta_diff + begin_unit * cols, \
                            cols, begin_row, end_row); \
            }); \
        add_n(t_gamma_diff, gamma_grad, units, cols); \
        add_n(t_beta_diff, beta_grad, units, cols); \
        free(t_gamma_diff); \
        free(t_beta_diff); \


#endif
#endif  // TENSORFLOW_CORE_KERNELS_FUSED_LAYER_NORMALIZE_COMPILE_UTIL_OP_H_