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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#define EIGEN_USE_GPU

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/cuda_solvers.h"  // For ScratchSpace
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/kernels/gpu_prim_helpers.h"
#include "tensorflow/core/kernels/sparse_slice_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "cub/iterator/counting_input_iterator.cuh"
#include "cub/iterator/transform_input_iterator.cuh"
using stream_executor::cuda::ScopedActivateExecutorContext;
namespace gpuprim = ::cub;
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/stream_executor/rocm/rocm_activation.h"
using stream_executor::rocm::ScopedActivateExecutorContext;
namespace gpuprim = ::hipcub;
#endif

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

struct SparseSliceSelectFunctor {
  SparseSliceSelectFunctor(int dims,
                           GpuDeviceArrayStruct<int64> input_start_data,
                           GpuDeviceArrayStruct<int64> input_size_data,
                           const int64* input_indices)
      : dims_(dims),
        input_start_data_(input_start_data),
        input_size_data_(input_size_data),
        input_indices_(input_indices) {}

  // Returns true iff input_indices[input_nz] is within the slice volume.
  __host__ __device__ bool operator()(int64 input_nz) const {
    // This is a workaround for GetGpuDeviceArrayOnDevice only accepting
    // a non-const pointer.
    auto* mutable_this = const_cast<SparseSliceSelectFunctor*>(this);
    const int64* __restrict__ input_start =
        GetGpuDeviceArrayOnDevice(&mutable_this->input_start_data_);
    const int64* __restrict__ input_size =
        GetGpuDeviceArrayOnDevice(&mutable_this->input_size_data_);
    for (int dim = 0; dim < dims_; ++dim) {
      int64 index = input_indices_[input_nz * dims_ + dim];
      int64 slice_start = input_start[dim];
      int64 slice_end = slice_start + input_size[dim];
      if (index < slice_start || index >= slice_end) {
        return false;
      }
    }
    return true;
  }

 private:
  int dims_;
  GpuDeviceArrayStruct<int64> input_start_data_;
  GpuDeviceArrayStruct<int64> input_size_data_;
  const int64* __restrict__ input_indices_;
};

// Gathers (and offsets) selected indices and values from input into output.
template <typename T>
__global__ void SparseSliceGatherKernel(
    int dims, int64 output_nnz,
    GpuDeviceArrayStruct<int64> input_start_data,
    const int64* __restrict__ input_indices,
    const T* __restrict__ input_values,
    const int64* __restrict__ selected_nonzeros,
    int64* __restrict__ output_indices, T* __restrict__ output_values) {
  const int64* __restrict__ input_start =
      GetGpuDeviceArrayOnDevice(&input_start_data);
  for (int64 output_nz : GpuGridRangeX<int64>(output_nnz)) {
    int64 input_nz = selected_nonzeros[output_nz];
    output_values[output_nz] = input_values[input_nz];
    for (int dim = 0; dim < dims; ++dim) {
      output_indices[output_nz * dims + dim] =
          input_indices[input_nz * dims + dim] - input_start[dim];
    }
  }
}

}  // End of anonymous namespace

namespace functor {

template <typename T>
struct SparseSliceFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_indices,
                  const Tensor& input_values, const Tensor& input_shape,
                  const Tensor& input_start, const Tensor& input_size,
                  typename AsyncOpKernel::DoneCallback done) const {
    const int dims = input_shape.NumElements();
    se::Stream* stream = context->op_device_context()->stream();

    // Note: This needs to be wrapped in shared_ptr so that it can be captured
    // in the lambda below.
    auto shared_input_start_gpu =
        std::make_shared<GpuDeviceArrayOnHost<int64>>(context, dims);
    GpuDeviceArrayOnHost<int64> input_size_gpu(context, dims);
    OP_REQUIRES_OK_ASYNC(context, shared_input_start_gpu->Init(), done);
    OP_REQUIRES_OK_ASYNC(context, input_size_gpu.Init(), done);

    // Allocate and compute output shape.
    Tensor* output_shape = nullptr;
    int64 output_volume = 1;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(2, {dims}, &output_shape), done);
    for (int dim = 0; dim < dims; ++dim) {
      int64 input_dimsize = input_shape.vec<int64>()(dim);
      int64 slice_start = input_start.vec<int64>()(dim);
      int64 slice_size = input_size.vec<int64>()(dim);
      shared_input_start_gpu->Set(dim, slice_start);
      input_size_gpu.Set(dim, slice_size);
      int64 output_size = std::max(
          std::min(slice_start + slice_size, input_dimsize) - slice_start,
          int64(0));
      output_shape->vec<int64>()(dim) = output_size;
      output_volume *= output_size;
    }
    OP_REQUIRES_OK_ASYNC(context, shared_input_start_gpu->Finalize(), done);
    OP_REQUIRES_OK_ASYNC(context, input_size_gpu.Finalize(), done);

    int64 input_nnz = input_indices.dim_size(0);

    // Early exit for empty input or output shape.
    if (input_nnz == 0 || output_volume == 0) {
      Tensor* output_indices = nullptr;
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(0, {0, dims}, &output_indices),
          done);
      Tensor* output_values = nullptr;
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(1, {0}, &output_values), done);
      done();
      return;
    }

    const int64* input_indices_ptr = input_indices.matrix<int64>().data();
    const T* input_values_ptr = input_values.vec<T>().data();

    gpuprim::CountingInputIterator<int64, int64> nonzeros(int64(0));
    SparseSliceSelectFunctor select_fn(dims, shared_input_start_gpu->data(),
                                       input_size_gpu.data(),
                                       input_indices_ptr);
    gpuprim::TransformInputIterator<bool, decltype(select_fn),
                                    decltype(nonzeros), int64>
        select_iterator(nonzeros, select_fn);

    Tensor selected_nonzeros;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_temp(DT_INT64, {input_nnz}, &selected_nonzeros),
        done);
    int64* selected_nonzeros_ptr = selected_nonzeros.vec<int64>().data();
    Tensor output_nnz_t;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_temp(DT_INT64, {1}, &output_nnz_t), done);
    int64* output_nnz_ptr = output_nnz_t.vec<int64>().data();

    // Select non-zeros that are inside the slice volume.
    OP_REQUIRES_OK_ASYNC(
        context,
        GpuSelectFlagged(context, input_nnz, nonzeros, select_iterator,
                         selected_nonzeros_ptr, output_nnz_ptr),
        done);

    // Copy the number of selected non-zeros to the host.
    ScratchSpace<int64> output_nnz_host(context, 1, /*on_host=*/true);
    OP_REQUIRES_ASYNC(
        context,
        stream
            ->ThenMemcpy(output_nnz_host.mutable_data(),
                         se::DeviceMemoryBase(output_nnz_ptr,
                                              sizeof(*output_nnz_host.data())),
                         sizeof(*output_nnz_host.data()))
            .ok(),
        errors::Internal("Failed to copy output_nnz to host"), done);

    // Asynchronously wait for the copy to complete before finishing.
    auto async_finish_computation =
        [context, dims, shared_input_start_gpu, input_indices,
         input_indices_ptr, input_values, input_values_ptr, output_nnz_t,
         output_nnz_host, selected_nonzeros, selected_nonzeros_ptr,
         done]() -> void {
      // Ensure that within the callback, the proper GPU settings are
      // configured.
      auto stream = context->op_device_context()->stream();
      ScopedActivateExecutorContext scoped_activation{stream->parent()};
      int64 output_nnz = *output_nnz_host.data();

      Tensor* output_indices = nullptr;
      OP_REQUIRES_OK_ASYNC(
          context,
          context->allocate_output(0, {output_nnz, dims}, &output_indices),
          done);
      int64* output_indices_ptr = output_indices->matrix<int64>().data();

      Tensor* output_values = nullptr;
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(1, {output_nnz}, &output_values),
          done);
      T* output_values_ptr = output_values->vec<T>().data();

      if (output_nnz == 0) {
        done();
        return;
      }

      // Gather (and offset) selected indices and values from input into output.
      const GPUDevice& device = context->eigen_device<GPUDevice>();
      auto config = GetGpuLaunchConfig(output_nnz, device);
      OP_REQUIRES_OK_ASYNC(
          context,
          GpuLaunchKernel(SparseSliceGatherKernel<T>, config.block_count,
                          config.thread_per_block, 0, device.stream(), dims,
                          output_nnz, shared_input_start_gpu->data(),
                          input_indices_ptr, input_values_ptr,
                          selected_nonzeros_ptr, output_indices_ptr,
                          output_values_ptr),
          done);
      done();
    };

    context->device()
        ->tensorflow_gpu_device_info()
        ->event_mgr->ThenExecute(stream, async_finish_computation);
  }
};

} // End of namespace functor

#define DEFINE_SPARSE_SLICE(T) \
  template struct functor::SparseSliceFunctor<GPUDevice, T>;
TF_CALL_POD_TYPES(DEFINE_SPARSE_SLICE);
#undef DEFINE_SPARSE_SLICE

}  // End of namespace tensorflow

#endif  // End of GOOGLE_CUDA || TENSORFLOW_USE_ROCM
