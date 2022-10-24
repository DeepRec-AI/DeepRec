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

// Simple hash functions used for internal data structures

#ifndef TENSORFLOW_CORE_LIB_HASH_HASH_H_
#define TENSORFLOW_CORE_LIB_HASH_HASH_H_

#include <functional>
#include <stddef.h>
#include <stdint.h>
#include <string>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

extern uint32 Hash32(const char* data, size_t n, uint32 seed);
extern uint64 Hash64(const char* data, size_t n, uint64 seed);
extern uint64 SDBMHash64(const char* data, size_t len);
extern uint64 DJB2Hash64(const char* data, size_t len);

extern uint64 Hash64V3(const char* data, size_t n, uint64 seed);

#if defined(__AVX512F__)
extern inline __m256i _mm256_mullo_epi64_hand(__m256i __A, __m256i __B);
extern inline __m128i _mm_mullo_epi64_hand (__m128i __A, __m128i __B);
extern inline __m512i _mm512_mullo_epi64_hand (__m512i __A, __m512i __B);
extern inline void pack_mul_int64_256(__m256i& a, __m256i& b, __m256i& bswap, 
                                      __m256i& zero, __m256i& res);
extern inline void pack_mul_int64_128(__m128i& a, __m128i& b, __m128i& bswap, 
                                      __m128i& zero, __m128i& res);
extern inline void swap_int64_256(__m256i& input);
extern inline void swap_int64_128(__m128i& input);
extern inline void Hash64AVX_64_Impl(const char*& data, size_t& n, uint64& h, 
                                     const uint64& m, const int& r);
extern inline void Hash64AVX_512_Impl(const char*& data, size_t& n, uint64& h, 
                                      const uint64& m, const int& r);
extern inline void Hash64AVX_128_Impl(const char*& data, size_t& n, uint64& h, 
                                      const uint64& m, const int& r);
extern inline void Hash64AVX_256_Impl(const char*& data, size_t& n, uint64& h, 
                                      const uint64& m, const int& r);
// for farmhash batch-vectorized implementation
extern inline __m512i Fetch64Batch(const char** data, size_t offset);
extern inline __m512i Fetch32Batch(const char** data, size_t offset);
extern inline __m512i Fetch8Batch(const char** data, size_t offset);
extern inline __m512i Rotate64Batch(__m512i in, int shift);
extern inline __m512i ShiftMixBatch(__m512i val);
extern inline std::pair<__m512i, __m512i> WeakHashLen32WithSeedsBatch(
    __m512i w, __m512i x, __m512i y, __m512i z, __m512i a, __m512i b);
extern inline __m512i HashLen16Batch(__m512i u, __m512i v, __m512i mul);
extern inline void HashLen0to16Batch(const char** data, uint64_t* h_out, 
                                     size_t len);
extern inline void HashLen17to32Batch(const char** data, uint64_t* h_out, 
                                      size_t len);
extern inline void HashLen33to64Batch(const char** data, uint64_t* h_out, 
                                      size_t len);
extern void Hash64V3_Batch512(const char** data, uint64* h_out, 
                              size_t n, uint64 seed);
extern void Hash64Farm_Batch512(const char** data, uint64_t* h_out, size_t n);
#endif

inline uint64 Hash64(const char* data, size_t n) {
  return Hash64(data, n, 0xDECAFCAFFE);
}

inline uint64 Hash64(const string& str) {
  return Hash64(str.data(), str.size());
}

inline uint64 MurMurHash64(const char* data, size_t n) {
  return Hash64(data, n, 0);
}

inline uint64 MurMurHash64(const string& str) {
  return MurMurHash64(str.data(), str.size());
}

inline uint64 MurMurHash64(StringPiece sp) {
  return MurMurHash64(sp.data(), sp.size());
}

inline uint64 SDBMHash64(const string& str) {
  return SDBMHash64(str.c_str(), str.size());
}

inline uint64 SDBMHash64(StringPiece sp) {
  return SDBMHash64(sp.data(), sp.size());
}

inline uint64 DJB2Hash64(const string& str) {
  return DJB2Hash64(str.c_str(), str.size());
}

inline uint64 DJB2Hash64(StringPiece sp) {
  return DJB2Hash64(sp.data(), sp.size());
}

inline uint64 Hash64Combine(uint64 a, uint64 b) {
  return a ^ (b + 0x9e3779b97f4a7800ULL + (a << 10) + (a >> 4));
}

// Combine two hashes in an order-independent way. This operation should be
// associative and compute the same hash for a collection of elements
// independent of traversal order. Note that it is better to combine hashes
// symmetrically with addition rather than XOR, since (x^x) == 0 but (x+x) != 0.
inline uint64 Hash64CombineUnordered(uint64 a, uint64 b) { return a + b; }

// Hash functor suitable for use with power-of-two sized hashtables.  Use
// instead of std::hash<T>.
//
// In particular, tensorflow::hash is not the identity function for pointers.
// This is important for power-of-two sized hashtables like FlatMap and FlatSet,
// because otherwise they waste the majority of their hash buckets.
//
// The second type argument is only used for SFNIAE below.
template <typename T, typename = void>
struct hash {
  size_t operator()(const T& t) const { return std::hash<T>()(t); }
};

template <typename T>
struct hash<T, typename std::enable_if<std::is_enum<T>::value>::type> {
  size_t operator()(T value) const {
    // This works around a defect in the std::hash C++ spec that isn't fixed in
    // (at least) gcc 4.8.4:
    // http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2148
    //
    // We should be able to remove this and use the default
    // tensorflow::hash<EnumTy>() once we stop building with GCC versions old
    // enough to not have this defect fixed.
    return std::hash<uint64>()(static_cast<uint64>(value));
  }
};

template <typename T>
struct hash<T*> {
  size_t operator()(const T* t) const {
    // Hash pointers as integers, but bring more entropy to the lower bits.
    size_t k = static_cast<size_t>(reinterpret_cast<uintptr_t>(t));
    return k + (k >> 6);
  }
};

template <>
struct hash<string> {
  size_t operator()(const string& s) const {
    return static_cast<size_t>(Hash64(s));
  }
};

template <>
struct hash<StringPiece> {
  size_t operator()(StringPiece sp) const {
    return static_cast<size_t>(Hash64(sp.data(), sp.size()));
  }
};
using StringPieceHasher = ::tensorflow::hash<StringPiece>;

template <typename T, typename U>
struct hash<std::pair<T, U>> {
  size_t operator()(const std::pair<T, U>& p) const {
    return Hash64Combine(hash<T>()(p.first), hash<U>()(p.second));
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_HASH_HASH_H_
