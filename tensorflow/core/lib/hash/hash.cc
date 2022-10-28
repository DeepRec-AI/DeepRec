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

#include "tensorflow/core/lib/hash/hash.h"

#include "tensorflow/core/lib/core/raw_coding.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#if defined(__AVX512F__)
#include <immintrin.h>
#include <x86intrin.h>
#endif
#include <stdlib.h>
#include <iostream>
#include <string.h>

namespace tensorflow {

// 0xff is in case char is signed.
static inline uint32 ByteAs32(char c) { return static_cast<uint32>(c) & 0xff; }
static inline uint64 ByteAs64(char c) { return static_cast<uint64>(c) & 0xff; }

uint32 Hash32(const char* data, size_t n, uint32 seed) {
  // 'm' and 'r' are mixing constants generated offline.
  // They're not really 'magic', they just happen to work well.

  const uint32 m = 0x5bd1e995;
  const int r = 24;

  // Initialize the hash to a 'random' value
  uint32 h = seed ^ n;

  // Mix 4 bytes at a time into the hash
  while (n >= 4) {
    uint32 k = core::DecodeFixed32(data);

    k *= m;
    k ^= k >> r;
    k *= m;

    h *= m;
    h ^= k;

    data += 4;
    n -= 4;
  }

  // Handle the last few bytes of the input array

  switch (n) {
    case 3:
      h ^= ByteAs32(data[2]) << 16;
      TF_FALLTHROUGH_INTENDED;
    case 2:
      h ^= ByteAs32(data[1]) << 8;
      TF_FALLTHROUGH_INTENDED;
    case 1:
      h ^= ByteAs32(data[0]);
      h *= m;
  }

  // Do a few final mixes of the hash to ensure the last few
  // bytes are well-incorporated.

  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;

  return h;
}

uint64 SDBMHash64(const char *key, size_t len) {
  register uint64 hash_value = 0;
  constexpr size_t alignment = 8;
  register auto quotient = len / alignment;
  register auto remainder = len % alignment;
  register auto j = 0;
  for (size_t i = 0; i < quotient; ++i) {
    char c0 = key[j];
    char c1 = key[j + 1];
    char c2 = key[j + 2];
    char c3 = key[j + 3];
    char c4 = key[j + 4];
    char c5 = key[j + 5];
    char c6 = key[j + 6];
    char c7 = key[j + 7];
    hash_value = c0 + (hash_value << 6) + (hash_value << 16) - hash_value;
    hash_value = c1 + (hash_value << 6) + (hash_value << 16) - hash_value;
    hash_value = c2 + (hash_value << 6) + (hash_value << 16) - hash_value;
    hash_value = c3 + (hash_value << 6) + (hash_value << 16) - hash_value;
    hash_value = c4 + (hash_value << 6) + (hash_value << 16) - hash_value;
    hash_value = c5 + (hash_value << 6) + (hash_value << 16) - hash_value;
    hash_value = c6 + (hash_value << 6) + (hash_value << 16) - hash_value;
    hash_value = c7 + (hash_value << 6) + (hash_value << 16) - hash_value;
    j += alignment;
  }
  for (size_t i = 0; i < remainder; ++i) {
    char c = key[j];
    hash_value = c + (hash_value << 6) + (hash_value << 16) - hash_value;
    ++j;
  }
  return hash_value;
}

uint64 DJB2Hash64(const char *key, size_t len) {
  register uint64 hash_value = 5381;
  constexpr size_t alignment = 8;
  register auto quotient = len / alignment;
  register auto remainder = len % alignment;
  register auto j = 0;
  for (size_t i = 0; i < quotient; ++i) {
      char c0 = key[j];
      char c1 = key[j + 1];
      char c2 = key[j + 2];
      char c3 = key[j + 3];
      char c4 = key[j + 4];
      char c5 = key[j + 5];
      char c6 = key[j + 6];
      char c7 = key[j + 7];
      hash_value = ((hash_value << 5) + hash_value) + c0;
      hash_value = ((hash_value << 5) + hash_value) + c1;
      hash_value = ((hash_value << 5) + hash_value) + c2;
      hash_value = ((hash_value << 5) + hash_value) + c3;
      hash_value = ((hash_value << 5) + hash_value) + c4;
      hash_value = ((hash_value << 5) + hash_value) + c5;
      hash_value = ((hash_value << 5) + hash_value) + c6;
      hash_value = ((hash_value << 5) + hash_value) + c7;
      j += alignment;
  }
  for (size_t i = 0; i < remainder; ++i) {
      char c = key[j];
      hash_value = ((hash_value << 5) + hash_value) + c;
      ++j;
  }
  return hash_value;
}

uint64 Hash64(const char* data, size_t n, uint64 seed) {
  const uint64 m = 0xc6a4a7935bd1e995;
  const int r = 47;

  uint64 h = seed ^ (n * m);

  while (n >= 8) {
    uint64 k = core::DecodeFixed64(data);
    data += 8;
    n -= 8;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  switch (n) {
    case 7:
      h ^= ByteAs64(data[6]) << 48;
      TF_FALLTHROUGH_INTENDED;
    case 6:
      h ^= ByteAs64(data[5]) << 40;
      TF_FALLTHROUGH_INTENDED;
    case 5:
      h ^= ByteAs64(data[4]) << 32;
      TF_FALLTHROUGH_INTENDED;
    case 4:
      h ^= ByteAs64(data[3]) << 24;
      TF_FALLTHROUGH_INTENDED;
    case 3:
      h ^= ByteAs64(data[2]) << 16;
      TF_FALLTHROUGH_INTENDED;
    case 2:
      h ^= ByteAs64(data[1]) << 8;
      TF_FALLTHROUGH_INTENDED;
    case 1:
      h ^= ByteAs64(data[0]);
      h *= m;
  }

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

#if defined(__AVX512F__)

inline __m256i _mm256_mullo_epi64_hand(__m256i __A, __m256i __B) {
  return (__m256i)((__v4du)__A * (__v4du)__B);
}
inline __m128i _mm_mullo_epi64_hand (__m128i __A, __m128i __B)
{
  return (__m128i) ((__v2du) __A * (__v2du) __B);
}
inline __m512i _mm512_mullo_epi64_hand (__m512i __A, __m512i __B)
{
  return (__m512i) ((__v8du) __A * (__v8du) __B);
}

inline void swap_int64_256(__m256i& input) {
  constexpr uint8_t mask_data[] = {7,  6,  5,  4,  3,  2,  1,  0,  15, 14, 13,
                                   12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,
                                   1,  0,  15, 14, 13, 12, 11, 10, 9,  8};
  const __m256i swapped =
      _mm256_shuffle_epi8(input, _mm256_loadu_si256((const __m256i*)mask_data));
  _mm256_storeu_si256(&input, swapped);
}

inline void swap_int64_128(__m128i& input) {
  constexpr uint8_t mask_data[] = {7,  6,  5,  4,  3,  2,  1, 0,
                                   15, 14, 13, 12, 11, 10, 9, 8};
  const __m128i swapped =
      _mm_shuffle_epi8(input, _mm_loadu_si128((const __m128i*)mask_data));
  _mm_storeu_si128(&input, swapped);
}
    
inline void Hash64AVX_256_Impl(const char*& data, size_t& n, uint64& h,
                               const uint64& m, const int& r) {
  __m256i m_pack = _mm256_set1_epi64x(m);
  __m256i k;
  __m256i k_tmp;

  while (n >= 32) {
    // k packs 32 chars (bytes)
    k = _mm256_loadu_si256((const __m256i*)data);
    if (!port::kLittleEndian) {
      // change from BigEdian to LittleEdian
      swap_int64_256(k);
    }
    data += 32;
    n -= 32;

    k = _mm256_mullo_epi64_hand(k, m_pack);
    k_tmp = _mm256_srli_epi64(k, r);
    k = _mm256_xor_si256(k, k_tmp);
    k = _mm256_mullo_epi64_hand(k, m_pack);
    // reduction
    h ^= (uint64)_mm256_extract_epi64(k, 0);
    h *= m;
    h ^= (uint64)_mm256_extract_epi64(k, 1);
    h *= m;
    h ^= (uint64)_mm256_extract_epi64(k, 2);
    h *= m;
    h ^= (uint64)_mm256_extract_epi64(k, 3);
    h *= m;
  }
}

inline void Hash64AVX_128_Impl(const char*& data, size_t& n, uint64& h,
                               const uint64& m, const int& r) {
  __m128i m_pack = _mm_set_epi64x(m, m);
  __m128i k;
  __m128i k_tmp;

  // unrolling by using __m128i
  while (n >= 16) {
    k = _mm_loadu_si128((const __m128i*)data);
    if (!port::kLittleEndian) {
      // change from BigEdian to LittleEdian
      swap_int64_128(k);
    }
    data += 16;
    n -= 16;

    k = _mm_mullo_epi64_hand(k, m_pack);
    k_tmp = _mm_srli_epi64(k, r);
    k = _mm_xor_si128(k, k_tmp);
    k = _mm_mullo_epi64_hand(k, m_pack);
    h ^= (uint64)_mm_extract_epi64(k, 0);
    h *= m;
    h ^= (uint64)_mm_extract_epi64(k, 1);
    h *= m;
  }
}

// for batch-vectorized kernels in farmhash.h
// Some primes between 2^63 and 2^64 for various uses.
static const uint64_t k0 = 0xc3a5c85c97cb3127ULL;
static const uint64_t k1 = 0xb492b66fbe98f273ULL;
static const uint64_t k2 = 0x9ae16a3b2f90404fULL;
// Magic numbers for 32-bit hashing.  Copied from Murmur3.
static const uint32_t c1 = 0xcc9e2d51;
static const uint32_t c2 = 0x1b873593;

inline __m512i Fetch64Batch(const char** data, size_t offset) {
  return _mm512_set_epi64(core::DecodeFixed64(data[7]+offset), 
                          core::DecodeFixed64(data[6]+offset),
                          core::DecodeFixed64(data[5]+offset), 
                          core::DecodeFixed64(data[4]+offset),
                          core::DecodeFixed64(data[3]+offset), 
                          core::DecodeFixed64(data[2]+offset),
                          core::DecodeFixed64(data[1]+offset), 
                          core::DecodeFixed64(data[0]+offset));
}

inline __m512i Fetch32Batch(const char** data, size_t offset) {
  return _mm512_set_epi64((uint64_t)core::DecodeFixed32(data[7]+offset),
                          (uint64_t)core::DecodeFixed32(data[6]+offset),
                          (uint64_t)core::DecodeFixed32(data[5]+offset),
                          (uint64_t)core::DecodeFixed32(data[4]+offset),
                          (uint64_t)core::DecodeFixed32(data[3]+offset),
                          (uint64_t)core::DecodeFixed32(data[2]+offset),
                          (uint64_t)core::DecodeFixed32(data[1]+offset),
                          (uint64_t)core::DecodeFixed32(data[0]+offset));
}

inline __m512i Fetch8Batch(const char** data, size_t offset) {
  return _mm512_set_epi64((uint8_t)(data[7][offset]),
                          (uint8_t)(data[6][offset]),
                          (uint8_t)(data[5][offset]),
                          (uint8_t)(data[4][offset]),
                          (uint8_t)(data[3][offset]),
                          (uint8_t)(data[2][offset]),
                          (uint8_t)(data[1][offset]),
                          (uint8_t)(data[0][offset]));
}

inline __m512i Rotate64Batch(__m512i in, int shift) {
   if (shift != 0) {
     __m512i tmp1 = _mm512_srli_epi64(in, shift);
     __m512i tmp2 = _mm512_slli_epi64(in, 64-shift);
     return _mm512_or_si512(tmp1, tmp2); 
   } else {
     return in;
   } 
}

inline __m512i ShiftMixBatch(__m512i val) {
  __m512i tmp = _mm512_srli_epi64(val, 47);
  return _mm512_xor_si512(val, tmp);
}

inline std::pair<__m512i, __m512i> WeakHashLen32WithSeedsBatch(
    __m512i w, __m512i x, __m512i y, __m512i z, __m512i a, __m512i b) {
  a = _mm512_add_epi64(a, w);
  __m512i tmp = _mm512_add_epi64(b, a);
  tmp = _mm512_add_epi64(tmp, z);
  b = Rotate64Batch(tmp, 21);
  __m512i c = _mm512_srli_epi64(a, 0);
  a = _mm512_add_epi64(a, x);
  a = _mm512_add_epi64(a, y);
  b = _mm512_add_epi64(b, Rotate64Batch(a, 44));
  return std::make_pair(_mm512_add_epi64(a, z), _mm512_add_epi64(b, c));
}

inline __m512i HashLen16Batch(__m512i u, __m512i v, __m512i mul) {
  __m512i a = _mm512_xor_si512(u, v);
  a = _mm512_mullo_epi64_hand(a, mul);
  __m512i tmp = _mm512_srli_epi64(a, 47);
  a = _mm512_xor_si512(a, tmp);
  __m512i b = _mm512_xor_si512(v, a);
  b = _mm512_mullo_epi64_hand(b, mul);
  tmp = _mm512_srli_epi64(b, 47);
  b = _mm512_xor_si512(b, tmp);
  b = _mm512_mullo_epi64_hand(b, mul);
  return b;
}

inline void HashLen0to16Batch(const char** data, uint64_t* h_out, size_t len) { 
  __m512i k0_batch = _mm512_set1_epi64(k0);
  __m512i k1_batch = _mm512_set1_epi64(k1);
  __m512i k2_batch = _mm512_set1_epi64(k2);
  __m512i factor_2_batch = _mm512_set1_epi64(2);
  __m512i factor_len_batch = _mm512_set1_epi64(len);

  if (len >= 8) {
    __m512i mul = _mm512_mullo_epi64_hand(factor_len_batch, factor_2_batch);
    mul = _mm512_add_epi64(mul, k2_batch);
    __m512i a = Fetch64Batch(data, 0);
    a = _mm512_add_epi64(a, k2_batch);
    __m512i b = Fetch64Batch(data, len-8);
    __m512i c = Rotate64Batch(b, 37);
    c = _mm512_mullo_epi64_hand(c, mul);
    c = _mm512_add_epi64(c, a);
    __m512i d = Rotate64Batch(a, 25);
    d = _mm512_add_epi64(d, b);
    d = _mm512_mullo_epi64_hand(d, mul);
    __m512i ret = HashLen16Batch(c, d, mul);
    _mm512_storeu_si512((void*)h_out, ret);
    return;
  }
  if (len >= 4) {
    __m512i mul = _mm512_mullo_epi64_hand(factor_len_batch, factor_2_batch);
    mul = _mm512_add_epi64(mul, k2_batch);
    __m512i a = Fetch32Batch(data, 0);
    a = _mm512_slli_epi64(a, 3);
    a = _mm512_add_epi64(a, factor_len_batch);
    __m512i ret = HashLen16Batch(a, Fetch32Batch(data, len-4), mul);
    _mm512_storeu_si512((void*)h_out, ret);
    return;
  }
  if (len > 0) {
    __m512i a = Fetch8Batch(data, 0);
    __m512i b = Fetch8Batch(data, len >> 1);
    __m512i c = Fetch8Batch(data, len-1);
    __m512i y = _mm512_slli_epi64(b, 8);
    y = _mm512_add_epi64(y, a);
    __m512i z = _mm512_slli_epi64(c, 2);
    z = _mm512_add_epi64(z, factor_len_batch);
    y = _mm512_mullo_epi64_hand(y, k2_batch);
    z = _mm512_mullo_epi64_hand(z, k0_batch);
    y = _mm512_xor_si512(y, z);
    __m512i ret = _mm512_mullo_epi64_hand(ShiftMixBatch(y), k2_batch);
    _mm512_storeu_si512((void*)h_out, ret);
    return;
  }
  _mm512_storeu_si512((void*)h_out, k2_batch);
}

inline void HashLen17to32Batch(const char** data, uint64_t* h_out, size_t len) { 
  __m512i k2_batch = _mm512_set1_epi64(k2);
  __m512i k1_batch = _mm512_set1_epi64(k1);
  __m512i factor_2_batch = _mm512_set1_epi64(2);
  __m512i factor_len_batch = _mm512_set1_epi64(len);
  __m512i mul = _mm512_mullo_epi64_hand(factor_len_batch, factor_2_batch);
  mul = _mm512_add_epi64(mul, k2_batch);
  __m512i a = Fetch64Batch(data, 0);
  a = _mm512_mullo_epi64_hand(a, k1_batch);
  __m512i b = Fetch64Batch(data, 8);
  __m512i c = Fetch64Batch(data, len-8);
  c = _mm512_mullo_epi64_hand(c, mul);
  __m512i d = Fetch64Batch(data, len-16);
  d = _mm512_mullo_epi64_hand(d, k2_batch);
  __m512i r1 = Rotate64Batch(_mm512_add_epi64(a, b), 43);
  __m512i r2 = Rotate64Batch(c, 30);
  __m512i r3 = Rotate64Batch(_mm512_add_epi64(b, k2_batch), 18);
  r2 = _mm512_add_epi64(r2, d);
  r3 = _mm512_add_epi64(r3, a);
  r3 = _mm512_add_epi64(r3, c);
  __m512i ret = HashLen16Batch(_mm512_add_epi64(r1, r2), r3, mul);
  _mm512_storeu_si512((void*)h_out, ret);
}

inline void HashLen33to64Batch(const char** data, uint64_t* h_out, size_t len) { 
  __m512i k2_batch = _mm512_set1_epi64(k2);
  __m512i factor_2_batch = _mm512_set1_epi64(2);
  __m512i factor_len_batch = _mm512_set1_epi64(len);
  __m512i mul = _mm512_mullo_epi64_hand(factor_len_batch, factor_2_batch);
  mul = _mm512_add_epi64(mul, k2_batch);
  __m512i a = Fetch64Batch(data, 0);
  a = _mm512_mullo_epi64_hand(a, k2_batch);
  __m512i b = Fetch64Batch(data, 8);
  __m512i c = Fetch64Batch(data, len-8);
  c = _mm512_mullo_epi64_hand(c, mul);
  __m512i d = Fetch64Batch(data, len-16);
  d = _mm512_mullo_epi64_hand(d, k2_batch);
  __m512i r1 = Rotate64Batch(_mm512_add_epi64(a, b), 43);
  __m512i r2 = Rotate64Batch(c, 30);
  __m512i y = _mm512_add_epi64(r1, r2);
  y = _mm512_add_epi64(y, d);
  r2 = Rotate64Batch(_mm512_add_epi64(b, k2_batch), 18);
  r2 = _mm512_add_epi64(r2, a);
  r2 = _mm512_add_epi64(r2, c);
  __m512i z = HashLen16Batch(y, r2, mul);
  __m512i e = Fetch64Batch(data, 16);
  e = _mm512_mullo_epi64_hand(e, mul);
  __m512i f = Fetch64Batch(data, 24);
  __m512i g = Fetch64Batch(data, len - 32);
  g = _mm512_add_epi64(g, y);
  g = _mm512_mullo_epi64_hand(g, mul);
  __m512i h = Fetch64Batch(data, len - 24);
  h = _mm512_add_epi64(h, z);
  h = _mm512_mullo_epi64_hand(h, mul);

  r1 = Rotate64Batch(_mm512_add_epi64(e, f), 43);  
  r2 = Rotate64Batch(g, 30);  
  r1 = _mm512_add_epi64(r1, r2);
  r1 = _mm512_add_epi64(r1, h);
  r2 = Rotate64Batch(_mm512_add_epi64(f, a), 18);  
  r2 = _mm512_add_epi64(r2, e);
  r2 = _mm512_add_epi64(r2, g);

  __m512i ret = HashLen16Batch(r1, r2, mul);
  _mm512_storeu_si512((void*)h_out, ret);
}

inline void Hash64AVX_512_Impl(const char*& data, size_t& n, uint64& h,
                               const uint64& m, const int& r) {
  __m512i m_pack = _mm512_set1_epi64(m);
  __m512i k;
  __m512i k_tmp;
  __m256i k_low;
  __m256i k_high;

  while (n >= 64) {
    // k packs 32 chars (bytes)
    k = _mm512_loadu_si512((const void*)data);
    if (!port::kLittleEndian) {
      // change from BigEdian to LittleEdian
      // swap_int64_256(k);
    }
    data += 64;
    n -= 64;

    k = _mm512_mullo_epi64(k, m_pack);
    k_tmp = _mm512_srli_epi64(k, r);
    k = _mm512_xor_si512(k, k_tmp);
    k = _mm512_mullo_epi64(k, m_pack);
    // 1st reduction
    k_low = _mm512_extracti64x4_epi64(k, 0);
    k_high = _mm512_extracti64x4_epi64(k, 1);

    h ^= _mm256_extract_epi64(k_low, 0);
    h *= m;
    h ^= _mm256_extract_epi64(k_low, 1);
    h *= m;
    h ^= _mm256_extract_epi64(k_low, 2);
    h *= m;
    h ^= _mm256_extract_epi64(k_low, 3);
    h *= m;

    h ^= _mm256_extract_epi64(k_high, 0);
    h *= m;
    h ^= _mm256_extract_epi64(k_high, 1);
    h *= m;
    h ^= _mm256_extract_epi64(k_high, 2);
    h *= m;
    h ^= _mm256_extract_epi64(k_high, 3);
    h *= m;
  }
}

inline void Hash64AVX_64_Impl(const char*& data, size_t& n, uint64& h,
                              const uint64& m, const int& r) {
  while (n >= 8) {
    uint64 k = core::DecodeFixed64(data);
    data += 8;
    n -= 8;
    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }
}
#endif

uint64 Hash64V3(const char* data, size_t n, uint64 seed) {
#if defined(__AVX512F__)
  const uint64 m = 0xc6a4a7935bd1e995;
  const int r = 47;
  uint64 h = seed ^ (n * m);
  // n is number of byte in the char* array
  // unrolling y 64 (_mm512i)
  Hash64AVX_512_Impl(data, n, h, m, r);
  Hash64AVX_256_Impl(data, n, h, m, r);
  Hash64AVX_128_Impl(data, n, h, m, r);
  Hash64AVX_64_Impl(data, n, h, m, r);

  switch (n) {
    case 7:
      h ^= ByteAs64(data[6]) << 48;
      TF_FALLTHROUGH_INTENDED;
    case 6:
      h ^= ByteAs64(data[5]) << 40;
      TF_FALLTHROUGH_INTENDED;
    case 5:
      h ^= ByteAs64(data[4]) << 32;
      TF_FALLTHROUGH_INTENDED;
    case 4:
      h ^= ByteAs64(data[3]) << 24;
      TF_FALLTHROUGH_INTENDED;
    case 3:
      h ^= ByteAs64(data[2]) << 16;
      TF_FALLTHROUGH_INTENDED;
    case 2:
      h ^= ByteAs64(data[1]) << 8;
      TF_FALLTHROUGH_INTENDED;
    case 1:
      h ^= ByteAs64(data[0]);
      h *= m;
  }
  h ^= h >> r;
  h *= m;
  h ^= h >> r;
  return h;
#else
  return Hash64(data, n, seed);
#endif
}

#if defined(__AVX512F__)
void Hash64V3_Batch512(const char** data, uint64* h_out, 
                       size_t n, uint64 seed) {
  const uint64 m = 0xc6a4a7935bd1e995;
  const int r = 47;
  uint64 h = seed ^ (n * m);
  __m512i m_pack = _mm512_set1_epi64(m);
  __m512i h_pack = _mm512_set1_epi64(h);
  __m512i h_pack_tmp;
  __m512i k;
  __m512i k_tmp;

  while (n >= 8) {
    k = _mm512_set_epi64(core::DecodeFixed64(data[7]), 
                         core::DecodeFixed64(data[6]),
                         core::DecodeFixed64(data[5]), 
                         core::DecodeFixed64(data[4]),
                         core::DecodeFixed64(data[3]), 
                         core::DecodeFixed64(data[2]),
                         core::DecodeFixed64(data[1]), 
                         core::DecodeFixed64(data[0]));
    data[0] += 8;
    data[1] += 8;
    data[2] += 8;
    data[3] += 8;
    data[4] += 8;
    data[5] += 8;
    data[6] += 8;
    data[7] += 8;                   
    n -= 8;
    // computation
    k = _mm512_mullo_epi64_hand(k, m_pack);
    k_tmp = _mm512_srli_epi64(k, r);
    k = _mm512_xor_si512(k, k_tmp);
    k = _mm512_mullo_epi64_hand(k, m_pack);
    // reduction
    h_pack = _mm512_xor_si512(h_pack, k);
    h_pack = _mm512_mullo_epi64_hand(h_pack, m_pack);
  }
  // load the remained data
  if (n > 0) {
    switch (n) {
      case 7:
        k = _mm512_set_epi64( ByteAs64(data[7][6]), ByteAs64(data[6][6]), 
                              ByteAs64(data[5][6]), ByteAs64(data[4][6]), 
                              ByteAs64(data[3][6]), ByteAs64(data[2][6]), 
                              ByteAs64(data[1][6]), ByteAs64(data[0][6]));
        k_tmp = _mm512_slli_epi64(k, 48);
        h_pack = _mm512_xor_si512(h_pack, k_tmp);
        TF_FALLTHROUGH_INTENDED;
      case 6:
        k = _mm512_set_epi64( ByteAs64(data[7][5]), ByteAs64(data[6][5]), 
                              ByteAs64(data[5][5]), ByteAs64(data[4][5]), 
                              ByteAs64(data[3][5]), ByteAs64(data[2][5]), 
                              ByteAs64(data[1][5]), ByteAs64(data[0][5]));
        k_tmp = _mm512_slli_epi64(k, 40);
        h_pack = _mm512_xor_si512(h_pack, k_tmp);
        TF_FALLTHROUGH_INTENDED;
      case 5:
        k = _mm512_set_epi64( ByteAs64(data[7][4]), ByteAs64(data[6][4]), 
                              ByteAs64(data[5][4]), ByteAs64(data[4][4]), 
                              ByteAs64(data[3][4]), ByteAs64(data[2][4]), 
                              ByteAs64(data[1][4]), ByteAs64(data[0][4]));
        k_tmp = _mm512_slli_epi64(k, 32);
        h_pack = _mm512_xor_si512(h_pack, k_tmp);
        TF_FALLTHROUGH_INTENDED;
      case 4:
        k = _mm512_set_epi64( ByteAs64(data[7][3]), ByteAs64(data[6][3]), 
                              ByteAs64(data[5][3]), ByteAs64(data[4][3]), 
                              ByteAs64(data[3][3]), ByteAs64(data[2][3]), 
                              ByteAs64(data[1][3]), ByteAs64(data[0][3]));
        k_tmp = _mm512_slli_epi64(k, 24);
        h_pack = _mm512_xor_si512(h_pack, k_tmp);
        TF_FALLTHROUGH_INTENDED;
      case 3:
        k = _mm512_set_epi64( ByteAs64(data[7][2]), ByteAs64(data[6][2]), 
                              ByteAs64(data[5][2]), ByteAs64(data[4][2]), 
                              ByteAs64(data[3][2]), ByteAs64(data[2][2]), 
                              ByteAs64(data[1][2]), ByteAs64(data[0][2]));
        k_tmp = _mm512_slli_epi64(k, 16);
        h_pack = _mm512_xor_si512(h_pack, k_tmp);
        TF_FALLTHROUGH_INTENDED;
      case 2:
        k = _mm512_set_epi64( ByteAs64(data[7][1]), ByteAs64(data[6][1]), 
                              ByteAs64(data[5][1]), ByteAs64(data[4][1]), 
                              ByteAs64(data[3][1]), ByteAs64(data[2][1]), 
                              ByteAs64(data[1][1]), ByteAs64(data[0][1]));
        k_tmp = _mm512_slli_epi64(k, 8);
        h_pack = _mm512_xor_si512(h_pack, k_tmp);
        TF_FALLTHROUGH_INTENDED;
      case 1:
        k = _mm512_set_epi64( ByteAs64(data[7][0]), ByteAs64(data[6][0]), 
                              ByteAs64(data[5][0]), ByteAs64(data[4][0]), 
                              ByteAs64(data[3][0]), ByteAs64(data[2][0]), 
                              ByteAs64(data[1][0]), ByteAs64(data[0][0]));
        h_pack = _mm512_xor_si512(h_pack, k);
        h_pack = _mm512_mullo_epi64_hand(h_pack, m_pack);
    }
  }
  
  h_pack_tmp = _mm512_srli_epi64(h_pack, r);
  h_pack = _mm512_xor_si512(h_pack, h_pack_tmp);
  h_pack = _mm512_mullo_epi64_hand(h_pack, m_pack);
  h_pack_tmp = _mm512_srli_epi64(h_pack, r);
  h_pack = _mm512_xor_si512(h_pack, h_pack_tmp);
  // store data back to h
  _mm512_storeu_si512((void*)h_out, h_pack);
}

void Hash64Farm_Batch512(const char** data, uint64_t* h_out, size_t len) {
  const uint64_t seed = 81;
  if (len <= 32) {
    if (len <= 16) {
      HashLen0to16Batch(data, h_out, len);
      return;
    } else {
      HashLen17to32Batch(data, h_out, len);
      return;
    }
  } else if (len <= 64) {
    HashLen33to64Batch(data, h_out, len);
    return;
  }

  __m512i k2_batch = _mm512_set1_epi64(k2);
  __m512i k1_batch = _mm512_set1_epi64(k1);
  __m512i k0_batch = _mm512_set1_epi64(k0);
  __m512i factor_113_batch = _mm512_set1_epi64(113);
  __m512i factor_9_batch = _mm512_set1_epi64(9);
  __m512i factor_len_batch = _mm512_set1_epi64(len);
  __m512i x = _mm512_set1_epi64(seed); 
  __m512i seed_batch = _mm512_set1_epi64(seed); 

  __m512i y = _mm512_mullo_epi64_hand(seed_batch, k1_batch);
  y = _mm512_add_epi64(y, factor_113_batch);
  __m512i tmp = _mm512_mullo_epi64_hand(y, k2_batch);
  tmp = _mm512_add_epi64(tmp, factor_113_batch); 
  __m512i z = _mm512_mullo_epi64_hand(ShiftMixBatch(tmp), k2_batch); 

  std::pair<__m512i, __m512i> v = std::make_pair(_mm512_set1_epi64(0), 
                                                 _mm512_set1_epi64(0));
  std::pair<__m512i, __m512i> w = std::make_pair(_mm512_set1_epi64(0),
                                                 _mm512_set1_epi64(0));

  x = _mm512_mullo_epi64_hand(x, k2_batch);
  x = _mm512_add_epi64(x, Fetch64Batch(data, 0));

  size_t end_idx =  ((len - 1) / 64) * 64;
  size_t last64_idx = end_idx + ((len - 1) & 63) - 63;
  assert(len - 64 == last64_idx);

  size_t start_idx = 0;
  do {
    tmp = _mm512_add_epi64(x, y);
    tmp = _mm512_add_epi64(tmp, v.first);
    tmp = _mm512_add_epi64(tmp, Fetch64Batch(data, start_idx+8));
    x = Rotate64Batch(tmp, 37);
    x = _mm512_mullo_epi64_hand(x, k1_batch);

    tmp = _mm512_add_epi64(y, v.second);
    tmp = _mm512_add_epi64(tmp, Fetch64Batch(data, start_idx+48));
    y = Rotate64Batch(tmp, 42);
    y = _mm512_mullo_epi64_hand(y, k1_batch);

    x = _mm512_xor_si512(x, w.second);

    y = _mm512_add_epi64(y, v.first);
    y = _mm512_add_epi64(y, Fetch64Batch(data, start_idx+40));

    tmp = Rotate64Batch(_mm512_add_epi64(z, w.first), 33);
    z = _mm512_mullo_epi64_hand(tmp, k1_batch);

    v = WeakHashLen32WithSeedsBatch(Fetch64Batch(data, start_idx),
                                    Fetch64Batch(data, start_idx+8),
                                    Fetch64Batch(data, start_idx+16),
                                    Fetch64Batch(data, start_idx+24),
                                    _mm512_mullo_epi64_hand(v.second, k1_batch),
                                    _mm512_add_epi64(x, w.first));

    w = WeakHashLen32WithSeedsBatch(Fetch64Batch(data, start_idx+32),
                                    Fetch64Batch(data, start_idx+32+8),
                                    Fetch64Batch(data, start_idx+32+16),
                                    Fetch64Batch(data, start_idx+32+24),
                                    _mm512_add_epi64(z, w.second),
                                    _mm512_add_epi64(y, Fetch64Batch(data, 
                                    start_idx+16)));

    tmp = _mm512_srli_epi64(z, 0);
    z = _mm512_srli_epi64(x, 0);
    x = _mm512_srli_epi64(tmp, 0);
    start_idx += 64;
  } while (start_idx != end_idx);

  tmp = _mm512_and_si512(z, _mm512_set1_epi64(0xff));
  tmp = _mm512_slli_epi64(tmp, 1);
  __m512i mul = _mm512_add_epi64(k1_batch, tmp);

  start_idx = last64_idx;
  w.first = _mm512_add_epi64(w.first, _mm512_set1_epi64(((len - 1) & 63)));
  v.first = _mm512_add_epi64(v.first, w.first);
  w.first = _mm512_add_epi64(w.first, v.first);
  tmp =  _mm512_add_epi64(x, y);
  tmp =  _mm512_add_epi64(tmp, v.first);
  tmp =  _mm512_add_epi64(tmp, Fetch64Batch(data, start_idx+8));
  x = _mm512_mullo_epi64_hand(Rotate64Batch(tmp, 37), mul);

  tmp =  _mm512_add_epi64(y, v.second);
  tmp =  _mm512_add_epi64(tmp, Fetch64Batch(data, start_idx+48));
  y = _mm512_mullo_epi64_hand(Rotate64Batch(tmp, 42), mul);

  tmp = _mm512_mullo_epi64_hand(w.second, factor_9_batch); 
  x = _mm512_xor_si512(x, tmp);

  tmp = _mm512_mullo_epi64_hand(v.first, factor_9_batch);
  tmp = _mm512_add_epi64(tmp, Fetch64Batch(data, start_idx+40));
  y = _mm512_add_epi64(y, tmp);

  tmp = _mm512_add_epi64(z, w.first);
  z = _mm512_mullo_epi64_hand(Rotate64Batch(tmp, 33), mul);

  v = WeakHashLen32WithSeedsBatch(Fetch64Batch(data, start_idx),
                                    Fetch64Batch(data, start_idx+8),
                                    Fetch64Batch(data, start_idx+16),
                                    Fetch64Batch(data, start_idx+24),
                                    _mm512_mullo_epi64_hand(v.second, mul),
                                    _mm512_add_epi64(x, w.first));
  w = WeakHashLen32WithSeedsBatch(Fetch64Batch(data, start_idx+32),
                                    Fetch64Batch(data, start_idx+32+8),
                                    Fetch64Batch(data, start_idx+32+16),
                                    Fetch64Batch(data, start_idx+32+24),
                                    _mm512_add_epi64(z, w.second),
                                    _mm512_add_epi64(y, Fetch64Batch(data, 
                                    start_idx+16)));
  tmp = _mm512_srli_epi64(z, 0);
  z = _mm512_srli_epi64(x, 0);
  x = _mm512_srli_epi64(tmp, 0);

  __m512i h1 = HashLen16Batch(v.first, w.first, mul); 
  h1 = _mm512_add_epi64(h1, _mm512_mullo_epi64_hand(ShiftMixBatch(y), k0_batch));
  h1 = _mm512_add_epi64(h1, z);
  __m512i h2 = HashLen16Batch(v.second, w.second, mul);
  h2 = _mm512_add_epi64(h2, x);
  __m512i ret = HashLen16Batch(h1, h2, mul);
  _mm512_storeu_si512((void*)h_out, ret);

}
#endif
}  // namespace tensorflow
