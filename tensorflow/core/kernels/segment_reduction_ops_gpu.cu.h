/* Copyright 2023 The DeepRec Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_GPU_CU_H_
#define TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_GPU_CU_H_


#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/util/gpu_device_functions.h"

#include "tensorflow/core/kernels/segment_reduction_ops.h"

using GPUDevice = Eigen::GpuDevice;

namespace tensorflow {

class OpKernelContext;

namespace functor {

struct Sum {
  template <typename T>
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

struct Prod {
  template <typename T>
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return a * b;
  }
};

// Note that we don't use gpuprim::Min/Max because they use operator<, which is
// not implemented for AlignedVector types.
struct Min {
  template <typename T>
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return min(a, b);
  }
};

struct Max {
  template <typename T>
  __host__ __device__ T operator()(const T& a, const T& b) const {
    return max(a, b);
  }
};

// Non/Atomic reduction functors for the gpu.
#define DEFINE_REDUCE_UPDATE_OP_GPU(name, func)                             \
  struct name##OpGpu {                                                      \
    template <typename T>                                                   \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(T* dest,          \
                                                          const T& value) { \
      func;                                                                 \
    }                                                                       \
  };
DEFINE_REDUCE_UPDATE_OP_GPU(AtomicSum, GpuAtomicAdd(dest, value))
DEFINE_REDUCE_UPDATE_OP_GPU(AtomicProd, GpuAtomicMul(dest, value))
DEFINE_REDUCE_UPDATE_OP_GPU(AtomicMax, GpuAtomicMax(dest, value))
DEFINE_REDUCE_UPDATE_OP_GPU(AtomicMin, GpuAtomicMin(dest, value))
DEFINE_REDUCE_UPDATE_OP_GPU(NonAtomicSum, *dest += value)
DEFINE_REDUCE_UPDATE_OP_GPU(NonAtomicProd, *dest *= value)
DEFINE_REDUCE_UPDATE_OP_GPU(NonAtomicMax, *dest = max(*dest, value))
DEFINE_REDUCE_UPDATE_OP_GPU(NonAtomicMin, *dest = min(*dest, value))
#undef DEFINE_REDUCE_UPDATE_OP_GPU

template <typename ReduceOp>
struct ReduceUpdateOpFor {};

#define DEFINE_REDUCE_UPDATE_OP_FOR(reduce_op, atomic, nonatomic) \
  template <>                                                     \
  struct ReduceUpdateOpFor<reduce_op> {                           \
    using atomic_op = atomic;                                     \
    using nonatomic_op = nonatomic;                               \
  };
DEFINE_REDUCE_UPDATE_OP_FOR(functor::Sum, AtomicSumOpGpu, NonAtomicSumOpGpu)
DEFINE_REDUCE_UPDATE_OP_FOR(functor::Prod, AtomicProdOpGpu, NonAtomicProdOpGpu)
DEFINE_REDUCE_UPDATE_OP_FOR(functor::Max, AtomicMaxOpGpu, NonAtomicMaxOpGpu)
DEFINE_REDUCE_UPDATE_OP_FOR(functor::Min, AtomicMinOpGpu, NonAtomicMinOpGpu)
#undef DEFINE_REDUCE_UPDATE_OP_FOR

// Functor for SegmentReductionGPUOp.
// output_rows: the number of output segments (unique segment ids in
//                'segment_ids').
// segment_ids_shape: shape of 'segment_ids' tensor.
// segment_ids: unsorted map from input to output segment ids at which to
//                perform segment sum operation.
// data_size: size of input data tensor.
// data: input data tensor.
// output: output reshaped to {output_rows, output.size/output_rows}
template <typename T, typename Index, typename InitialValueF,
          typename EmptySegmentValueF, typename ReductionF>
struct SegmentReductionFunctor {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  const Index output_rows, const TensorShape& segment_ids_shape,
                  bool is_mean, typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output);
};

template <typename T, typename Index>
struct SparseSegmentReduceFunctor {
  void operator()(OpKernelContext* ctx,
                  const Tensor* input,
                  const Tensor* indices,
                  const Tensor* seg_ids,
                  Tensor* output,
                  const bool is_mean,
                  const bool is_sqrtn);
};

template <typename T, typename Index>
struct SparseSegmentReduceGradFunctor {
  void operator()(OpKernelContext* ctx,
                  const Tensor* input,
                  const Tensor* indices,
                  const Tensor* seg_ids,
                  Tensor* output,
                  const bool is_sqrtn);
};

template <typename Index>
struct FindMaxSegId {
  void operator()(OpKernelContext* ctx,
                  const Tensor* seg_ids,
                  Index& max_id);
};

template <typename T>
struct SetValueDefault {
  void operator()(OpKernelContext* ctx,
                  Tensor* target,
                  T default_value);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#endif  // TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_GPU_CU_H_
