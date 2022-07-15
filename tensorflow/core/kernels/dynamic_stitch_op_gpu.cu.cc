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

#if GOOGLE_CUDA //|| TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace {

template <typename T>
__global__ void DynamicStitchKernel(const int32 slice_size,
                                    const int32 output_size,
                                    GpuDeviceArrayStruct<int32> input_indices,
                                    GpuDeviceArrayStruct<const T*> input_ptrs,
                                    T* output) {
  int32* data_indices = GetGpuDeviceArrayOnDevice(&input_indices);
  const T** data_ptrs = GetGpuDeviceArrayOnDevice(&input_ptrs);
  GPU_1D_KERNEL_LOOP(output_index, output_size) {
    const int32 slice_id = output_index / slice_size;
    const int32 slice_offset = output_index % slice_size;
    const int32 input_index = data_indices[slice_id];
    if (input_index != -1) {
      output[output_index] = ldg(data_ptrs[input_index] + slice_offset);
    }
  }
}

template <typename T>
__global__ void DynamicStitchKernelV2(const int32 slice_size,
                                      const int32 output_size,
                                      const int32* input_indices,
                                      T** input_ptrs,
                                      T* output) {
  GPU_1D_KERNEL_LOOP(output_index, output_size) {
    const int32 slice_id = output_index / slice_size;
    const int32 slice_offset = output_index % slice_size;
    const int32 input_index = input_indices[slice_id];
    if (input_index != -1) {
      output[output_index] = ldg(input_ptrs[input_index] + slice_offset);
    }
  }
}

__global__ void InitializeIndicesFlatWork(int32* indices_flat_work,
                                          const int32 flat_work_size,
                                          const int32 val) {
  GPU_1D_KERNEL_LOOP(output_index, flat_work_size) {
    indices_flat_work[output_index] = val;
  }
}

template <typename T>
__global__ void DynamicStitchPrepKernel(const int32* indices_flat,
                                        int32* indices_flat_work,
                                        T** data_ptr_heads,
                                        const int32* data_ptr_num,
                                        T** data_ptr_all,
                                        const int32 data_partition_num,
                                        const int32 slice_size,
                                        const int32 output_size) {

  GPU_1D_KERNEL_LOOP(output_index, output_size) {
    // for indices
    indices_flat_work[indices_flat[output_index]] = output_index;
    // find the partition id
    int32 data_ptr_id = 0;
    int32 data_ptr_accu = data_ptr_num[data_ptr_id];
    while((data_ptr_accu < (output_index + 1)) &&
           data_ptr_id < data_partition_num) {
      data_ptr_id++;
      data_ptr_accu += data_ptr_num[data_ptr_id];
    }
    // find the offset
    int32 data_ptr_offset = output_index - data_ptr_accu +
                            data_ptr_num[data_ptr_id];
    data_ptr_all[output_index] = data_ptr_heads[data_ptr_id] +
                                 data_ptr_offset * slice_size;
  }
}
}  // namespace

template <typename T>
void DynamicStitchGPUImpl(const Eigen::GpuDevice& gpu_device,
                          const int32 slice_size, const int32 first_dim_size,
                          const GpuDeviceArrayStruct<int>& input_indices,
                          const GpuDeviceArrayStruct<const T*>& input_ptrs,
                          T* output) {
  const int32 output_size = first_dim_size * slice_size;
  auto config = GetGpuLaunchConfig(output_size, gpu_device);

  TF_CHECK_OK(GpuLaunchKernel(DynamicStitchKernel<T>, config.block_count,
                              config.thread_per_block, 0, gpu_device.stream(),
                              slice_size, output_size, input_indices,
                              input_ptrs, output));
}

template <typename T>
void DynamicStitchGPUImplV2(const Eigen::GpuDevice& gpu_device,
                            const int32 slice_size,
                            const int32 first_dim_size,
                            Tensor* input_indices,
                            Tensor* input_ptrs,
                            T* output) {
  const int32 output_size = first_dim_size * slice_size;
  auto config = GetGpuLaunchConfig(output_size, gpu_device);

  DynamicStitchKernelV2<T>
      <<<config.block_count, config.thread_per_block, 0, gpu_device.stream()>>>(
          slice_size, output_size,
          input_indices->flat<int32>().data(),
          reinterpret_cast<T**>(input_ptrs->flat<int8>().data()),
          output);
}

template <typename T>
void DynamicStitchGPUPrep(const Eigen::GpuDevice& gpu_device,
                          Tensor* indices_flat,
                          Tensor* indices_flat_work,
                          Tensor* data_ptr_heads,
                          Tensor* data_ptr_num,
                          T** data_ptr_all,
                          const int32 data_partition_num,
                          const int32 slice_size,
                          const int32 data_elements_size,
                          const int32 first_dim_size) {

  // initialize indices_flat_work by -1
  auto config = GetGpuLaunchConfig(first_dim_size, gpu_device);
  InitializeIndicesFlatWork
    <<<config.block_count, config.thread_per_block, 0, gpu_device.stream()>>>(
      indices_flat_work->flat<int32>().data(),
      first_dim_size, -1);

  config = GetGpuLaunchConfig(data_elements_size, gpu_device);
  DynamicStitchPrepKernel<T>
      <<<config.block_count, config.thread_per_block, 0, gpu_device.stream()>>>(
          indices_flat->flat<int32>().data(),
          indices_flat_work->flat<int32>().data(),
          reinterpret_cast<T**>(data_ptr_heads->flat<int8>().data()),
          data_ptr_num->flat<int32>().data(),
          data_ptr_all,
          data_partition_num, slice_size, data_elements_size);
}

void AggregateIndiceOnGpu(OpKernelContext* c,
                          OpInputList* indices_list,
                          Tensor* indice_flat) {
  auto dst = indice_flat->flat<int32>();
  int32 offset_size = 0;
  auto* stream = c->op_device_context()->stream();
  for (const Tensor& indices : *indices_list) {
    if (indices.NumElements() > 0) {
       auto src = indices.flat<int32>();
       se::DeviceMemoryBase dst_wrapped(dst.data() + offset_size,
                                        indices.NumElements() * sizeof(int32));
       stream->ThenMemcpy(&dst_wrapped,
                          static_cast<const void*>(src.data()),
                          indices.NumElements() * sizeof(int32));
       offset_size += indices.NumElements();
    }
  }
  stream->BlockHostUntilDone();
}

#define REGISTER_GPU(T)                                           \
  template void DynamicStitchGPUImpl(                             \
      const Eigen::GpuDevice& gpu_device, const int32 slice_size, \
      const int32 first_dim_size,                                 \
      const GpuDeviceArrayStruct<int32>& input_indices,           \
      const GpuDeviceArrayStruct<const T*>& input_ptrs, T* output);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_int32(REGISTER_GPU)

#undef REGISTER_GPU

#define REGISTER_GPU(T)                                           \
  template void DynamicStitchGPUPrep(                             \
      const Eigen::GpuDevice& gpu_device,                         \
      Tensor* indices_flat,                                       \
      Tensor* indices_flat_work,                                  \
      Tensor* data_ptr_heads,                                     \
      Tensor* data_ptr_num,                                       \
      T** data_ptr_all,                                           \
      const int32 data_partition_num,                             \
      const int32 slice_size,                                     \
      const int32 data_elements_size,                             \
      const int32 first_dim_size);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_int32(REGISTER_GPU)
#undef REGISTER_GPU

#define REGISTER_GPU(T)                                           \
  template void DynamicStitchGPUImplV2(                           \
      const Eigen::GpuDevice& gpu_device, const int32 slice_size, \
      const int32 first_dim_size,                                 \
      Tensor* input_indices,                                      \
      Tensor* input_ptrs,                                         \
      T* output);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_int32(REGISTER_GPU)
#undef REGISTER_GPU

}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
