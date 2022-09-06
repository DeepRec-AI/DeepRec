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

#if GOOGLE_CUDA //|| TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/platform/cuda.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

template <typename T, int NDIMS>
struct BCastSelectFunctor<GPUDevice, T, NDIMS> {
  void operator()(const GPUDevice& d,
                  typename TTypes<T, NDIMS>::Tensor output_tensor,
                  typename TTypes<bool, NDIMS>::ConstTensor cond_tensor,
                  typename TTypes<T, NDIMS>::ConstTensor then_tensor,
                  typename TTypes<T, NDIMS>::ConstTensor else_tensor,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> cond_bcast,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> then_bcast,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> else_bcast) {
    output_tensor.device(d) = cond_tensor.broadcast(cond_bcast)
                                  .select(then_tensor.broadcast(then_bcast),
                                          else_tensor.broadcast(else_bcast));
  }
};

template <typename T>
struct SelectFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstFlat cond_flat,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat) {
    To32Bit(out).device(d) =
        To32Bit(cond_flat).select(To32Bit(then_flat), To32Bit(else_flat));
  }
};

template <typename T>
struct SelectScalarFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstScalar cond,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat) {
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<int, 1> rank1{1};
#else
    Eigen::IndexList<Eigen::type2index<1> > rank1;
#endif
    const int size = then_flat.dimension(0);
    Eigen::array<int, 1> broadcast_dims{size};

    To32Bit(out).device(d) = cond.reshape(rank1)
                                 .broadcast(broadcast_dims)
                                 .select(then_flat, else_flat);
  }
};

template <typename T>
struct BatchSelectFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d,
                  typename TTypes<T>::Matrix output_flat_outer_dims,
                  TTypes<bool>::ConstVec cond_vec,
                  typename TTypes<T>::ConstMatrix then_flat_outer_dims,
                  typename TTypes<T>::ConstMatrix else_flat_outer_dims) {
    const int batch = cond_vec.size();
    const int all_but_batch = then_flat_outer_dims.dimension(1);

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<int, 2> broadcast_dims{{ 1, all_but_batch }};
    Eigen::Tensor<int, 2>::Dimensions reshape_dims{{ batch, 1 }};
#else
    Eigen::IndexList<Eigen::type2index<1>, int> broadcast_dims;
    broadcast_dims.set(1, all_but_batch);
    Eigen::IndexList<int, Eigen::type2index<1> > reshape_dims;
    reshape_dims.set(0, batch);
#endif

    // TODO(ebrevdo): Figure out why this leads to erroneous memory access.
    //
    // To32Bit(output_flat_outer_dims).device(d) =
    //     To32Bit(cond_vec)
    //         .reshape(reshape_dims)
    //         .broadcast(broadcast_dims)
    //         .select(To32Bit(then_flat_outer_dims),
    //                 To32Bit(else_flat_outer_dims));
    output_flat_outer_dims.device(d) =
        cond_vec.reshape(reshape_dims)
            .broadcast(broadcast_dims)
            .select(then_flat_outer_dims, else_flat_outer_dims);
  }
};

template <typename T>
__global__ void Select4ElementThenScalarFunctorKernel(
    const bool *c, const T *t, const T *e, size_t num, T *o) {
  GPU_1D_KERNEL_LOOP(i, num) {
    if (c[i]) {
      o[i] = t[0];
    } else {
      o[i] = e[i];
    }
  }
}

template <typename T>
__global__ void Select4ElementElseScalarFunctorKernel(
    const bool *c, const T *t, const T *e, size_t num, T *o) {
  GPU_1D_KERNEL_LOOP(i, num) {
    if (c[i]) {
      o[i] = t[i];
    } else {
      o[i] = e[0];
    }
  }
}

void calculateBlockAndThread(const int32_t processor_cnt, const int32_t max_thread_per_block,
    const size_t all_node,
    size_t& block_count, size_t& pysical_thread_per_block, size_t& thread_per_block) {
  block_count = processor_cnt;
  pysical_thread_per_block =  (all_node + block_count -1) / block_count;

  thread_per_block = (pysical_thread_per_block  + 31 ) / 32 * 32;
  thread_per_block = std::min((size_t)1024, thread_per_block);
  thread_per_block = std::min((size_t)max_thread_per_block, thread_per_block);
  thread_per_block = std::max((size_t)32, thread_per_block);
  return;
}

template <typename T>
struct Select4ElementScalarFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstFlat cond_flat,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat,
                  bool then_scalar) {
    const size_t num = cond_flat.size();
    const bool* c = cond_flat.data();
    const T* t = then_flat.data();
    const T* e = else_flat.data();
    T* o = out.data();

    size_t block_count, pysical_thread_per_block, thread_per_block;
    calculateBlockAndThread(d.getNumGpuMultiProcessors(), d.maxGpuThreadsPerBlock(),
        num, block_count, pysical_thread_per_block, thread_per_block);
    if (then_scalar) {
      Select4ElementThenScalarFunctorKernel<<<block_count, thread_per_block, 0, d.stream()>>>(
          c, t, e, num, o);
    } else {
      Select4ElementElseScalarFunctorKernel<<<block_count, thread_per_block, 0, d.stream()>>>(
          c, t, e, num, o);
    }
  }
};

template <typename T>
__global__ void BatchSelect4BroadcastingThenScalarFunctorKernel(
    const bool *c, const T *t, const T *e, size_t batch, size_t batch_size, T *o) {
  GPU_1D_KERNEL_LOOP(i, batch * batch_size) {
    size_t offset = i / batch_size;
    if (c[offset]) {
      o[i] = t[0];
    } else {
      o[i] = e[i];
    }
  }
}

template <typename T>
__global__ void BatchSelect4BroadcastingElseScalarFunctorKernel(
    const bool *c, const T *t, const T *e, size_t batch, size_t batch_size, T *o) {
  GPU_1D_KERNEL_LOOP(i, batch * batch_size) {
    size_t offset = i / batch_size;
    if (c[offset]) {
      o[i] = t[i];
    } else {
      o[i] = e[0];
    }
  }
}

template <typename T>
struct BatchSelect4BroadcastingScalarFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d,
                  typename TTypes<T>::Matrix output_flat_outer_dims,
                  TTypes<bool>::ConstVec cond_vec,
                  typename TTypes<T>::ConstMatrix then_flat_outer_dims,
                  typename TTypes<T>::ConstMatrix else_flat_outer_dims,
                  bool then_scalar) {
    const size_t batch = cond_vec.size();
    size_t batch_size = 0;
    if (then_scalar) {
      batch_size = else_flat_outer_dims.size() / batch;
    } else {
      batch_size = then_flat_outer_dims.size() / batch;
    }
    T* output = output_flat_outer_dims.data();
    const bool* c = cond_vec.data();
    const T* t = then_flat_outer_dims.data();
    const T* e = else_flat_outer_dims.data();

    size_t block_count, pysical_thread_per_block, thread_per_block;
    calculateBlockAndThread(d.getNumGpuMultiProcessors(), d.maxGpuThreadsPerBlock(),
        batch * batch_size, block_count, pysical_thread_per_block, thread_per_block);
    if (then_scalar) {
      BatchSelect4BroadcastingThenScalarFunctorKernel<<<block_count, thread_per_block, 0, d.stream()>>>(
          c, t, e, batch, batch_size, output);
    } else {
      BatchSelect4BroadcastingElseScalarFunctorKernel<<<block_count, thread_per_block, 0, d.stream()>>>(
          c, t, e, batch, batch_size, output);
    }
  }
};

#define SELECT_FUNCTOR(T)                              \
  template struct SelectFunctor<GPUDevice, T>;         \
  template struct SelectScalarFunctor<GPUDevice, T>;   \
  template struct Select4ElementScalarFunctor<GPUDevice, T>; \
  template struct BatchSelect4BroadcastingScalarFunctor<GPUDevice, T>; \
  template struct BatchSelectFunctor<GPUDevice, T>;    \
  template struct BCastSelectFunctor<GPUDevice, T, 1>; \
  template struct BCastSelectFunctor<GPUDevice, T, 2>; \
  template struct BCastSelectFunctor<GPUDevice, T, 3>; \
  template struct BCastSelectFunctor<GPUDevice, T, 4>; \
  template struct BCastSelectFunctor<GPUDevice, T, 5>; \
  template struct BCastSelectFunctor<GPUDevice, T, 6>; \
  template struct BCastSelectFunctor<GPUDevice, T, 7>; \
  template struct BCastSelectFunctor<GPUDevice, T, 8>;

SELECT_FUNCTOR(bool);
SELECT_FUNCTOR(Eigen::half);
SELECT_FUNCTOR(float);
SELECT_FUNCTOR(double);
SELECT_FUNCTOR(int32);
SELECT_FUNCTOR(int64);
SELECT_FUNCTOR(complex64);
SELECT_FUNCTOR(complex128);

#undef SELECT_FUNCTOR

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
