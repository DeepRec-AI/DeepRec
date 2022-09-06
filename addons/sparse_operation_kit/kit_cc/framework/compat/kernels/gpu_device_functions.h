/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_GPU_DEVICE_FUNCTIONS_H_
#define TENSORFLOW_CORE_UTIL_GPU_DEVICE_FUNCTIONS_H_

#include <assert.h>

#include <type_traits>

/**
 * Wrappers and helpers for CUDA device code.
 *
 * Wraps the warp-cooperative intrinsics introduced in CUDA 9 to provide
 * backwards compatibility, see go/volta-porting for details.
 * Provides atomic operations on types that aren't natively supported.
 */

namespace tensorflow {

namespace detail {

// Helper for range-based for loop using 'delta' increments.
// Usage: see GpuGridRange?() functions below.
template <typename T>
class GpuGridRange {
  struct Iterator {
    __device__ Iterator(T index, T delta) : index_(index), delta_(delta) {}
    __device__ T operator*() const { return index_; }
    __device__ Iterator &operator++() {
      index_ += delta_;
      return *this;
    }
    __device__ bool operator!=(const Iterator &other) const {
      bool greater = index_ > other.index_;
      bool less = index_ < other.index_;
      // Anything past an end iterator (delta_ == 0) is equal.
      // In range-based for loops, this optimizes to 'return less'.
      if (!other.delta_) {
        return less;
      }
      if (!delta_) {
        return greater;
      }
      return less || greater;
    }

   private:
    T index_;
    const T delta_;
  };

 public:
  __device__ GpuGridRange(T begin, T delta, T end) : begin_(begin), delta_(delta), end_(end) {}

  __device__ Iterator begin() const { return Iterator{begin_, delta_}; }
  __device__ Iterator end() const { return Iterator{end_, 0}; }

 private:
  T begin_;
  T delta_;
  T end_;
};

template <typename From, typename To>
using ToTypeIfConvertible = typename std::enable_if<std::is_convertible<From, To>::value, To>::type;

template <typename T>
struct CudaSupportedTypeImpl {
  using type = T;
};

template <typename T>
using CudaSupportedType = typename CudaSupportedTypeImpl<T>::type;

template <typename T>
__device__ CudaSupportedType<T> *ToCudaSupportedPtr(T *ptr) {
  return reinterpret_cast<CudaSupportedType<T> *>(ptr);
}

}  // namespace detail

// Helper to visit indices in the range 0 <= i < count, using the x-coordinate
// of the global thread index. That is, each index i is visited by all threads
// with the same x-coordinate.
// Usage: for(int i : GpuGridRangeX(count)) { visit(i); }
template <typename T>
__device__ detail::GpuGridRange<T> GpuGridRangeX(T count) {
  return detail::GpuGridRange<T>(blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x,
                                 count);
}

// Helper to set all tensor entries to a specific value.
template <typename T>
__global__ void SetToValue(const int count, T *ptr, T value) {
  // Check that the grid is one dimensional and index doesn't overflow.
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  assert(blockDim.x * gridDim.x / blockDim.x == gridDim.x);
  for (int i : GpuGridRangeX(count)) {
    ptr[i] = value;
  }
}

template <typename T, typename U>
__device__ detail::ToTypeIfConvertible<U, T> GpuAtomicAdd(T *ptr, U value) {
  return atomicAdd(detail::ToCudaSupportedPtr(ptr), value);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_GPU_DEVICE_FUNCTIONS_H_
