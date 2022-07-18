#ifndef __FUSED_UTIL_H
#define __FUSED_UTIL_H
#include "compile_util.h"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <limits>
#include <immintrin.h>

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
#endif