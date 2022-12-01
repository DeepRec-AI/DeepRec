/* Copyright 2022 The DeepRec Authors. All Rights Reserved.

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
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <bitset>
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_scan.cuh"
#include "cub/iterator/constant_input_iterator.cuh"
#include "cub/iterator/transform_input_iterator.cuh"
#include "cub/device/device_select.cuh"
#include "cub/iterator/counting_input_iterator.cuh"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/platform/cuda.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/core/kernels/cuda_solvers.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;

template <typename TIndex>
__global__ void RangeInitKernel(const TIndex start, const TIndex delta,
                                const int64 size, TIndex* out) {  
  GPU_1D_KERNEL_LOOP(i, size) { out[i] = start + i * delta; }
}
template <typename TIndex>
__global__ void MoveValuesKernel(const TIndex* keys, const TIndex* values,
                                 const int64 size, TIndex* out) {
  GPU_1D_KERNEL_LOOP(i, size) {
    TIndex key = ldg(keys + i);
    out[key] = ldg(values + i);
  }
}
template <typename TIndex>
__global__ void MoveValuesKernel(const TIndex* keys, const TIndex* values,
                                 const int64* size_ptr, TIndex* out) {
  int64 size = ldg(size_ptr);
  GPU_1D_KERNEL_LOOP(i, size) {
    TIndex key = ldg(keys + i);
    out[key] = ldg(values + i);
  }
}
template <typename T, typename TIndex>
__global__ void MoveSparseValuesKernel(const TIndex* keys, const TIndex* idxs,
                                       const T* values, const int64 size,
                                       T* out) {
  GPU_1D_KERNEL_LOOP(i, size) {
    TIndex key = ldg(keys + i);
    TIndex idx = ldg(idxs + i);
    out[key] = ldg(values + idx);
  }
}
template <typename T, typename TIndex>
__global__ void CompareAdjacentKernel(const T* in, const int64 size,
                                      TIndex* out) {
  GPU_1D_KERNEL_LOOP(i, size) {
    out[i] = (i == 0 || ldg(in + (i - 1)) == ldg(in + i)) ? 0 : 1;
  }
}
template <typename TIndex>
void RangeInit(const GPUDevice& d, const TIndex start, const TIndex delta,
               const int64 size, TIndex* out) {
  GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
  RangeInitKernel<TIndex>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          start, delta, size, out);  
}
template <typename TIndex>
void MoveValues(const GPUDevice& d, const TIndex* keys, const TIndex* values,
                const int64 size, TIndex* out) {
  GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
  MoveValuesKernel<<<config.block_count, config.thread_per_block, 0,
                     d.stream()>>>(keys, values, size, out);
}

template <typename TIndex>
void MoveValues(const GPUDevice& d, const TIndex* keys, const TIndex* values,
                const int64* size_ptr, const int64 size, TIndex* out) {
  GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
  MoveValuesKernel<<<config.block_count, config.thread_per_block, 0,
                     d.stream()>>>(keys, values, size_ptr, out);
}
template <typename T, typename TIndex>
void MoveSparseValues(const GPUDevice& d, const TIndex* keys, const TIndex* idxs,
                      const T* values, const int64 size, T* out) {
  GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
  MoveSparseValuesKernel<<<config.block_count, config.thread_per_block, 0,
                           d.stream()>>>(keys, idxs, values, size, out);
}
template <typename T, typename TIndex>
void CompareAdjacent(const GPUDevice& d, const T* in, const int64 size,
                     TIndex* out) {
  GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
  CompareAdjacentKernel<
      T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      in, size, out);
}
template <typename T, typename TIndex>
class UniqueAliV2GpuOp : public OpKernel {
 public:
  explicit UniqueAliV2GpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    const T* keys = input_tensor.flat<T>().data();
    
    int64 N = input_tensor.NumElements();
    const GPUDevice& device = ctx->eigen_device<GPUDevice>();
    const cudaStream_t& cu_stream = GetGpuStream(ctx);
    
    Tensor* output_tensor = nullptr;
    Tensor* idx_tensor = nullptr;
    Tensor* part_tensor = nullptr;
    auto allocate_output = [ctx, &output_tensor, &idx_tensor, &part_tensor, N,
                            &device, this](int64 N_out) {
      TF_RETURN_IF_ERROR(ctx->allocate_output(0, {N_out}, &output_tensor));
      TF_RETURN_IF_ERROR(ctx->allocate_output(1, {N}, &idx_tensor));
      return Status::OK();
    };
    if (N == 0) {
      OP_REQUIRES_OK(ctx, allocate_output(0));
      return;
    }
    
    Tensor keys_sort_tensor;
    Tensor indicies_sort_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, {N},
                                           &keys_sort_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<TIndex>::value, {N},
                                           &indicies_sort_tensor));
    T* keys_sort = keys_sort_tensor.flat<T>().data();
    TIndex* indicies_sort = indicies_sort_tensor.flat<TIndex>().data();
    
    Tensor indices_in_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<TIndex>::value, {N},
                                           &indices_in_tensor));
    TIndex* indices_in = indices_in_tensor.flat<TIndex>().data();
    RangeInit(device, (TIndex)0, (TIndex)1, N, indices_in);
    
    {
      const T* keys_in;
      Tensor keys_in_tensor;
      keys_in = keys;
      using U = typename std::make_unsigned<T>::type;
      const U* keys_u_in = reinterpret_cast<const U*>(keys_in);
      U* keys_u_sort = reinterpret_cast<U*>(keys_sort);
      
      Tensor cub_temp_storage;
      size_t temp_storage_bytes = 0;
      cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, keys_u_in,
                                      keys_u_sort, indices_in, indicies_sort, N,
                                      0, sizeof(T) * 8, cu_stream);
      OP_REQUIRES_OK(
          ctx,
          ctx->allocate_temp(
              DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
              &cub_temp_storage));
      cub::DeviceRadixSort::SortPairs(cub_temp_storage.flat<int8>().data(),
                                      temp_storage_bytes, keys_u_in,
                                      keys_u_sort, indices_in, indicies_sort, N,
                                      0, sizeof(T) * 8, cu_stream);
    }
    
    Tensor output_indices_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<TIndex>::value, {N},
                                           &output_indices_tensor));
    TIndex* output_indices = output_indices_tensor.flat<TIndex>().data();
    
    {
      Tensor output_indices_in_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<TIndex>::value, {N},
                                             &output_indices_in_tensor));      
      TIndex* output_indices_in = output_indices_in_tensor.flat<TIndex>().data();
      CompareAdjacent(device, keys_sort, N, output_indices_in);
      Tensor cub_temp_storage;
      size_t temp_storage_bytes = 0;
      cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes, output_indices_in,
                                    output_indices, N, cu_stream);
      OP_REQUIRES_OK(
          ctx,
          ctx->allocate_temp(
              DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
              &cub_temp_storage));
      cub::DeviceScan::InclusiveSum(cub_temp_storage.flat<int8>().data(),
                                    temp_storage_bytes, output_indices_in,
                                    output_indices, N, cu_stream);
    }
    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));
    TIndex N_out;
    se::DeviceMemoryBase wrapped_num_out(output_indices + (N - 1),
                                         sizeof(TIndex));
    TensorReference ref_output_indices(output_indices_tensor);
    OP_REQUIRES(ctx,
                stream->ThenMemcpy(&N_out, wrapped_num_out, sizeof(TIndex)).ok(),
                errors::Internal("Failed to launch copy from device to host."));
    ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
        stream, [ref_output_indices]() { ref_output_indices.Unref(); });
    stream->BlockHostUntilDone();
    N_out += 1;
    OP_REQUIRES_OK(ctx, allocate_output(N_out));
    T* output = output_tensor->flat<T>().data();
    TIndex* idx = idx_tensor->flat<TIndex>().data();
    MoveValues(device, indicies_sort, output_indices, N, idx);
    MoveSparseValues(device, output_indices, indicies_sort, keys, N, output);
  }
};

#define REGISTER_UNIQUE_ALI_V2_GPU_KERNEL(T, TIndex)			\
  REGISTER_KERNEL_BUILDER(Name("Unique")				\
			  .Device(DEVICE_GPU)				\
			  .TypeConstraint<T>("T")			\
			  .TypeConstraint<TIndex>("out_idx"),		\
			  UniqueAliV2GpuOp<T, TIndex>)
#define REGISTER_UNIQUE_ALI_V2_GPU(T)		\
  REGISTER_UNIQUE_ALI_V2_GPU_KERNEL(T, int32);	\
  REGISTER_UNIQUE_ALI_V2_GPU_KERNEL(T, int64)
  
TF_CALL_int32(REGISTER_UNIQUE_ALI_V2_GPU);
TF_CALL_int64(REGISTER_UNIQUE_ALI_V2_GPU);
TF_CALL_uint32(REGISTER_UNIQUE_ALI_V2_GPU);
TF_CALL_uint64(REGISTER_UNIQUE_ALI_V2_GPU);

#undef REGISTER_UNIQUE_ALI_V2_GPU
#undef REGISTER_UNIQUE_ALI_V2_GPU_KERNEL

}  // namespace tensorflow
#endif  // GOOGLE_CUDA
