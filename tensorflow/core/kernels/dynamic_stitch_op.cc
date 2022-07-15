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

// See docs in ../ops/data_flow_ops.cc.
#include <algorithm>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/gpu_device_array.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
typedef Eigen::GpuDevice GPUDevice;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <class T>
class DynamicStitchOpImplBase : public OpKernel {
 public:
  explicit DynamicStitchOpImplBase(OpKernelConstruction* c,
                                   const string& op_name)
      : OpKernel(c) {
    // Compute expected input signature
    const DataType dt = DataTypeToEnum<T>::v();
    const int n = c->num_inputs() / 2;
    DataTypeVector expected;
    for (int i = 0; i < n; i++) {
      expected.push_back(DT_INT32);
    }
    for (int i = 0; i < n; i++) {
      expected.push_back(dt);
    }
    OP_REQUIRES_OK(c, c->MatchSignature(expected, {dt}));
    OP_REQUIRES(c, c->num_inputs() > 0,
                errors::InvalidArgument(op_name + ": Must have some inputs"));
    OP_REQUIRES(c, c->num_inputs() % 2 == 0,
                errors::InvalidArgument(
                    op_name + ": Must have even number of arguments"));
  }

 protected:
  // Check if data0.shape[indices0.dims():] == data1.shape[indices1.dims():]
  static bool SameExtraShape(const Tensor& data0, const Tensor& indices0,
                             const Tensor& data1, const Tensor& indices1) {
    const int extra0 = data0.dims() - indices0.dims();
    const int extra1 = data1.dims() - indices1.dims();
    if (extra0 != extra1) return false;
    for (int i = 0; i < extra0; i++) {
      if (data0.dim_size(indices0.dims() + i) !=
          data1.dim_size(indices1.dims() + i)) {
        return false;
      }
    }
    return true;
  }

  void CheckArgsAndAllocateResult(OpKernelContext* c,
                                  OpInputList* indices_inputs,
                                  OpInputList* data_inputs, int* first_dim_size,
                                  int* data_elements_size,
                                  Tensor** result_ptr) {
    // Find maximum index in the indices vectors
    OP_REQUIRES_OK(c, c->input_list("indices", indices_inputs));

    int32 max_index = -1;
    if (data_elements_size) {
      *data_elements_size = 0;
    }
    for (const Tensor& indices : *indices_inputs) {
      if (indices.NumElements() > 0) {
        Eigen::Tensor<int32, 0, Eigen::RowMajor> m =
            indices.flat<int32>().maximum();
        max_index = std::max(m(), max_index);
      }
      if (data_elements_size) {
        *data_elements_size += indices.NumElements();
      }
    }

    *first_dim_size = max_index + 1;

    // Validate that data[i].shape = indices[i].shape + constant
    OP_REQUIRES_OK(c, c->input_list("data", data_inputs));
    const Tensor& data0 = (*data_inputs)[0];
    const Tensor& indices0 = (*indices_inputs)[0];
    for (int input_num = 0; input_num < indices_inputs->size(); input_num++) {
      const Tensor& indices = (*indices_inputs)[input_num];
      const Tensor& data = (*data_inputs)[input_num];
      OP_REQUIRES(
          c, TensorShapeUtils::StartsWith(data.shape(), indices.shape()),
          errors::InvalidArgument("data[", input_num,
                                  "].shape = ", data.shape().DebugString(),
                                  " does not start with indices[", input_num,
                                  "].shape = ", indices.shape().DebugString()));
      OP_REQUIRES(
          c, input_num == 0 || SameExtraShape(data0, indices0, data, indices),
          errors::InvalidArgument(
              "Need data[0].shape[", indices0.dims(), ":] = data[", input_num,
              "].shape[", indices.dims(),
              ":], got data[0].shape = ", data0.shape().DebugString(),
              ", data[", input_num, "].shape = ", data.shape().DebugString(),
              ", indices[0].shape = ", indices0.shape().DebugString(),
              ", indices[", input_num,
              "].shape = ", indices.shape().DebugString()));
    }

    // Allocate result tensor of shape
    //   [*first_dim_size] + data.shape[indices.dims:]
    TensorShape result_shape;
    result_shape.AddDim(*first_dim_size);
    for (int d = indices0.dims(); d < data0.dims(); d++) {
      result_shape.AddDim(data0.dim_size(d));
    }
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, result_ptr));
  }

  void CheckArgsAndAllocateResultV2(OpKernelContext* c,
                                    OpInputList* indices_inputs,
                                    OpInputList* data_inputs,
                                    int* first_dim_size,
                                    int* data_elements_size,
                                    Tensor** result_ptr) {
    // Find maximum index in the indices vectors
    int32 max_index = -1;
    if (data_elements_size) {
      *data_elements_size = 0;
    }
    // find the max indices for result
    for (const Tensor& indices : *indices_inputs) {
      if (indices.NumElements() > 0) {
         Eigen::Tensor<int32, 0, Eigen::RowMajor> m =
            indices.flat<int32>().maximum();
         max_index = std::max(static_cast<int32>(m()), max_index);
      }
      if (data_elements_size) {
        *data_elements_size += indices.NumElements();
      }
    }
    *first_dim_size = max_index + 1;
    const Tensor& data0 = (*data_inputs)[0];
    const Tensor& indices0 = (*indices_inputs)[0];
    for (int input_num = 0; input_num < indices_inputs->size(); input_num++) {
      const Tensor& indices = (*indices_inputs)[input_num];
      const Tensor& data = (*data_inputs)[input_num];
      OP_REQUIRES(
          c, TensorShapeUtils::StartsWith(data.shape(), indices.shape()),
          errors::InvalidArgument("data[", input_num,
                                  "].shape = ", data.shape().DebugString(),
                                  " does not start with indices[", input_num,
                                  "].shape = ", indices.shape().DebugString()));
      OP_REQUIRES(
          c, input_num == 0 || SameExtraShape(data0, indices0, data, indices),
          errors::InvalidArgument(
              "Need data[0].shape[", indices0.dims(), ":] = data[", input_num,
              "].shape[", indices.dims(),
              ":], got data[0].shape = ", data0.shape().DebugString(),
              ", data[", input_num, "].shape = ", data.shape().DebugString(),
              ", indices[0].shape = ", indices0.shape().DebugString(),
              ", indices[", input_num,
              "].shape = ", indices.shape().DebugString()));
    }

    // Allocate result tensor of shape
    TensorShape result_shape;
    result_shape.AddDim(*first_dim_size);
    for (int d = indices0.dims(); d < data0.dims(); d++) {
      result_shape.AddDim(data0.dim_size(d));
    }
    OP_REQUIRES_OK(c, c->allocate_output(0, result_shape, result_ptr));
  }
};

#if GOOGLE_CUDA //|| TENSORFLOW_USE_ROCM

template <typename T>
void DynamicStitchGPUImpl(const Eigen::GpuDevice& gpu_device,
                          const int32 slice_size, const int32 first_dim_size,
                          const GpuDeviceArrayStruct<int>& input_indices,
                          const GpuDeviceArrayStruct<const T*>& input_ptrs,
                          T* output);
#define REGISTER_GPU(T)                                           \
  extern template void DynamicStitchGPUImpl(                      \
      const Eigen::GpuDevice& gpu_device, const int32 slice_size, \
      const int32 first_dim_size,                                 \
      const GpuDeviceArrayStruct<int32>& input_indices,           \
      const GpuDeviceArrayStruct<const T*>& input_ptrs, T* output);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_int32(REGISTER_GPU);
#undef REGISTER_GPU

template <typename T>
void DynamicStitchGPUImplV2(const Eigen::GpuDevice& gpu_device,
                            const int32 slice_size, const int32 first_dim_size,
                            Tensor* input_indices,
                            Tensor* input_ptrs,
                            T* output);

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
                          const int32 first_dim_size);

void AggregateIndiceOnGpu(OpKernelContext* c,
                          OpInputList* indices_list,
                          Tensor* indice_flat);

template <class T>
class DynamicStitchOpGPU : public DynamicStitchOpImplBase<T> {
 public:
  explicit DynamicStitchOpGPU(OpKernelConstruction* c)
      : DynamicStitchOpImplBase<T>(c, "DynamicStitchOp") {}

  void Compute(OpKernelContext* c) override {
    OpInputList indices_inputs;
    OpInputList data_inputs;
    int first_dim_size;
    int data_elements_size;
    Tensor* merged = nullptr;
    this->CheckArgsAndAllocateResult(c, &indices_inputs, &data_inputs,
                                     &first_dim_size, &data_elements_size,
                                     &merged);
    if (!c->status().ok()) {
      // Avoid segmentation faults if merged cannot be allocated and an error is
      // passed back in the context.
      return;
    }

    // TODO(jeff): Currently we leave uninitialized any portions of
    // merged that aren't covered by an index in indices.  What should we do?
    if (first_dim_size > 0) {
      // because the collision requirements, we have to deal with
      // collision first before send data to gpu kernel.
      // TODO(ekelsen): Instead of doing a serial scan on the CPU to pick the
      // last of duplicated indices, it could instead be done of the GPU
      // implicitly using atomics to make sure the last index is the final
      // write.
      const int slice_size = merged->flat_outer_dims<T>().dimension(1);
      GpuDeviceArrayOnHost<int32> indices_flat(c, first_dim_size);
      GpuDeviceArrayOnHost<const T*> data_flat(c, data_elements_size);
      OP_REQUIRES_OK(c, indices_flat.Init());
      OP_REQUIRES_OK(c, data_flat.Init());
      // initialize the indices_flat (-1 represents missing indices)
      for (int i = 0; i < first_dim_size; ++i) {
        indices_flat.Set(i, -1);
      }

      // data_flat index
      int32 idx = 0;
      // sum of indices_inputs[i].NumElements() for compute indicies_flat value.
      int32 base_size = 0;
      for (int i = 0; i < indices_inputs.size(); ++i) {
        auto indices_vec = indices_inputs[i].flat<int32>();
        auto data_ptr_base = data_inputs[i].template flat<T>().data();
        for (int j = 0; j < indices_vec.size(); ++j) {
          // indices_flat's indices represent the indices of output.
          // indices_flat's values represent the indices of input_data where the
          // data located.
          indices_flat.Set(indices_vec(j), base_size + j);
          data_flat.Set(
              idx, const_cast<T*>(reinterpret_cast<const T*>(data_ptr_base) +
                                  j * slice_size));
          ++idx;
        }
        base_size += indices_vec.size();
      }
      OP_REQUIRES_OK(c, indices_flat.Finalize());
      OP_REQUIRES_OK(c, data_flat.Finalize());

      auto output = merged->template flat<T>().data();
      DynamicStitchGPUImpl<T>(c->eigen_gpu_device(), slice_size, first_dim_size,
                              indices_flat.data(), data_flat.data(), output);
    }
  }
};

template <class T>
class DynamicStitchOpGPUV2 : public DynamicStitchOpImplBase<T> {
 public:
  explicit DynamicStitchOpGPUV2(OpKernelConstruction* c)
      : DynamicStitchOpImplBase<T>(c, "DynamicStitchOp") {}
  // an alternative implementation to minimize the cpu-gpu memory copy
  void Compute(OpKernelContext* c) override {
    OpInputList indices_inputs;
    OpInputList data_inputs;
    int first_dim_size;
    int data_elements_size;
    int total_indices_num = 0;
    Tensor* merged = nullptr;
    // obtain the tensors/tensor lists
    OP_REQUIRES_OK(c, c->input_list("indices", &indices_inputs));
    OP_REQUIRES_OK(c, c->input_list("data", &data_inputs));
    for (int i = 0; i < indices_inputs.size(); ++i) {
      total_indices_num += indices_inputs[i].NumElements();
    }
    Tensor indices_flat;
    c->allocate_temp(DT_INT32, TensorShape{total_indices_num},
                     &indices_flat);
    if (!c->status().ok()) {
      LOG(ERROR) << c->status();
      return;
    }

    this->CheckArgsAndAllocateResultV2(c, &indices_inputs, &data_inputs,
                                       &first_dim_size, &data_elements_size,
                                       &merged);

    if (!c->status().ok()) {
      // Avoid segmentation faults if merged cannot be
      // allocated and an error is passed back in the context.
      LOG(ERROR) << c->status();
      return;
    }

    // device to device aggregation
    AggregateIndiceOnGpu(c, &indices_inputs, &indices_flat);
    // a pointer head array on gpu (copy data from cpu)
    AllocatorAttributes host_alloc_attr;
    host_alloc_attr.set_on_host(true);
    host_alloc_attr.set_gpu_compatible(true);
    const uint64 data_ptr_heads_bytes = data_inputs.size() * sizeof(T*);

    Tensor data_ptr_heads_cpu;
    c->allocate_temp(DT_INT8, TensorShape{data_ptr_heads_bytes},
                     &data_ptr_heads_cpu, host_alloc_attr);
    if (!c->status().ok()) {
      LOG(ERROR) << c->status();
      return;
    }
    Tensor data_ptr_heads;
    c->allocate_temp(DT_INT8, TensorShape{data_ptr_heads_bytes},
                     &data_ptr_heads);
    if (!c->status().ok()) {
      LOG(ERROR) << c->status();
      return;
    }
    // an array to store number of indices in each partition
    Tensor data_ptr_num_cpu;
    c->allocate_temp(DT_INT32, TensorShape{data_inputs.size()},
                     &data_ptr_num_cpu, host_alloc_attr);
    if (!c->status().ok()) {
      LOG(ERROR) << c->status();
      return;
    }
    Tensor data_ptr_num;
    c->allocate_temp(DT_INT32, TensorShape{data_inputs.size()},
                     &data_ptr_num);
    if (!c->status().ok()) {
      LOG(ERROR) << c->status();
      return;
    }

    // assign values to data_ptr_heads_cpu and indices
    T** data_ptr_heads_cpu_val = reinterpret_cast<T**>(
      data_ptr_heads_cpu.flat<int8>().data());
    for(int i = 0; i < data_inputs.size(); i++) {
      data_ptr_heads_cpu_val[i] =
        const_cast<T*>(data_inputs[i].flat<T>().data());
      data_ptr_num_cpu.flat<int32>()(i) =
        indices_inputs[i].NumElements();
    }

    auto* stream = c->op_device_context()->stream();
    // copy data ptr heads to gpu
    TensorReference tensor_ref_heads(data_ptr_heads_cpu);
    se::DeviceMemoryBase dst_wrapped_heads(
        data_ptr_heads.flat<int8>().data(), data_ptr_heads_bytes);
    stream->ThenMemcpy(&dst_wrapped_heads,
                       data_ptr_heads_cpu.flat<int8>().data(),
                       data_ptr_heads_bytes);
    c->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
        stream, [tensor_ref_heads]() { tensor_ref_heads.Unref(); });
    // copy data ptr number to gpu
    TensorReference tensor_ref_num(data_ptr_num_cpu);
    se::DeviceMemoryBase dst_wrapped_num(data_ptr_num.flat<int32>().data(),
                                         data_inputs.size() * sizeof(int32));
    stream->ThenMemcpy(&dst_wrapped_num,
                       data_ptr_num_cpu.flat<int32>().data(),
                       data_inputs.size() * sizeof(int32));
    c->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
        stream, [tensor_ref_num]() { tensor_ref_num.Unref(); });
    // a pointer array on gpu
    Tensor data_ptr_all;
    c->allocate_temp(DT_INT8, TensorShape{data_elements_size * sizeof(T*)},
                     &data_ptr_all);
    if (!c->status().ok()) {
      LOG(ERROR) << c->status();
      return;
    }
    // a working space for indices_flat
    Tensor indices_flat_work;
    c->allocate_temp(DT_INT32, TensorShape{first_dim_size},
                     &indices_flat_work);
    if (!c->status().ok()) {
      LOG(ERROR) << c->status();
      return;
    }

    if (first_dim_size > 0) {
      const int slice_size = merged->flat_outer_dims<T>().dimension(1);
      // create a kernel to prepare indices_flat_work and data_ptr_all
      DynamicStitchGPUPrep<T>(c->eigen_gpu_device(),
                              &indices_flat,
                              &indices_flat_work,
                              &data_ptr_heads,
                              &data_ptr_num,
                              reinterpret_cast<T**>(
                              data_ptr_all.flat<int8>().data()),
                              data_inputs.size(),
                              slice_size,
                              data_elements_size,
                              first_dim_size);
      auto output = merged->template flat<T>().data();
      DynamicStitchGPUImplV2<T>(c->eigen_gpu_device(), slice_size,
                                first_dim_size,
                                &indices_flat_work,
                                &data_ptr_all, output);
    }
  }
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <class T, bool Parallel>
class DynamicStitchOpImplCPU : public DynamicStitchOpImplBase<T> {
 public:
  explicit DynamicStitchOpImplCPU(OpKernelConstruction* c)
      : DynamicStitchOpImplBase<T>(
            c, (Parallel ? "ParallelDynamicStitchOp" : "DynamicStitchOp")) {}

  void Compute(OpKernelContext* c) override {
    OpInputList indices_inputs;
    OpInputList data_inputs;
    int first_dim_size;
    Tensor* merged = nullptr;
    this->CheckArgsAndAllocateResult(c, &indices_inputs, &data_inputs,
                                     &first_dim_size, nullptr, &merged);
    if (!c->status().ok()) {
      // Avoid segmentation faults if merged cannot be allocated and an error is
      // passed back in the context.
      return;
    }

    // TODO(jeff): Currently we leave uninitialized any portions of
    // merged that aren't covered by an index in indices.  What should we do?
    if (first_dim_size > 0) {
      auto merged_flat = merged->flat_outer_dims<T>();
      const int slice_size = merged_flat.dimension(1);
      const size_t slice_bytes = slice_size * sizeof(T);
      auto OnInputNumber = [&](int input_num) {
        const Tensor& indices = indices_inputs[input_num];
        auto indices_vec = indices.flat<int32>();
        const Tensor& data = data_inputs[input_num];
        auto data_flat =
            data.shaped<T, 2>({indices_vec.dimension(0), slice_size});

        if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
          T* merged_base = merged_flat.data();
          const T* data_base = data_flat.data();
          for (int i = 0; i < indices_vec.size(); i++) {
            int32 index = internal::SubtleMustCopy(indices_vec(i));
            OP_REQUIRES(
                c, FastBoundsCheck(index, first_dim_size),
                errors::InvalidArgument("indices[", i, "] is out of range"));
            memcpy(merged_base + index * slice_size, data_base + i * slice_size,
                   slice_bytes);
          }
        } else {
          Eigen::DSizes<Eigen::DenseIndex, 2> sizes(1, slice_size);
          for (int i = 0; i < indices_vec.size(); i++) {
            // Copy slice data[i] to merged[indices[i]]
            Eigen::DSizes<Eigen::DenseIndex, 2> data_indices(i, 0);
            int32 index = internal::SubtleMustCopy(indices_vec(i));
            OP_REQUIRES(
                c, FastBoundsCheck(index, first_dim_size),
                errors::InvalidArgument("indices[", i, "] is out of range"));
            Eigen::DSizes<Eigen::DenseIndex, 2> merged_indices(index, 0);
            merged_flat.slice(merged_indices, sizes) =
                data_flat.slice(data_indices, sizes);
          }
        }
      };
      if (Parallel) {
        auto thread_pool =
            c->device()->tensorflow_cpu_worker_threads()->workers;
        size_t total_indices_size = 0;
        for (int input_num = 0; input_num < indices_inputs.size();
             ++input_num) {
          total_indices_size += indices_inputs[input_num].NumElements();
        }
        const double avg_indices_size =
            static_cast<double>(total_indices_size) / indices_inputs.size();
        auto bytes_processed = slice_bytes * avg_indices_size;
        auto LoopBody = [&](int first, int last) {
          for (int input_num = first; input_num < last; ++input_num) {
            OnInputNumber(input_num);
          }
        };
        thread_pool->ParallelFor(indices_inputs.size(), bytes_processed,
                                 LoopBody);
      } else {
        for (int input_num = 0; input_num < indices_inputs.size();
             input_num++) {
          OnInputNumber(input_num);
        }
      }
    }
  }
};

// Using inheritance rather than a typedef so that these classes might have more
// functionality later.

template <typename T>
struct DynamicStitchOpCPU : DynamicStitchOpImplCPU<T, false> {
  using DynamicStitchOpImplCPU<T, false>::DynamicStitchOpImplCPU;
};

template <typename T>
struct ParallelDynamicStitchOpCPU : DynamicStitchOpImplCPU<T, true> {
  using DynamicStitchOpImplCPU<T, true>::DynamicStitchOpImplCPU;
};

#define REGISTER_DYNAMIC_STITCH(type)                    \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")          \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices"),    \
                          DynamicStitchOpCPU<type>)      \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitchFast")      \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices"),    \
                          ParallelDynamicStitchOpCPU<type>) \
  REGISTER_KERNEL_BUILDER(Name("ParallelDynamicStitch")  \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices"),    \
                          ParallelDynamicStitchOpCPU<type>)

TF_CALL_POD_STRING_TYPES(REGISTER_DYNAMIC_STITCH);
TF_CALL_variant(REGISTER_DYNAMIC_STITCH);
TF_CALL_QUANTIZED_TYPES(REGISTER_DYNAMIC_STITCH);
#undef REGISTER_DYNAMIC_STITCH

#if GOOGLE_CUDA //|| TENSORFLOW_USE_ROCM
#define REGISTER_DYNAMIC_STITCH_GPU(type)                \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")          \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices"),    \
                          DynamicStitchOpGPU<type>)      \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitchFast")      \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices"),    \
                          DynamicStitchOpGPUV2<type>)    \
  REGISTER_KERNEL_BUILDER(Name("ParallelDynamicStitch")  \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices"),    \
                          DynamicStitchOpGPUV2<type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_DYNAMIC_STITCH_GPU);
TF_CALL_complex64(REGISTER_DYNAMIC_STITCH_GPU);
TF_CALL_complex128(REGISTER_DYNAMIC_STITCH_GPU);
TF_CALL_int64(REGISTER_DYNAMIC_STITCH_GPU);
TF_CALL_int32(REGISTER_DYNAMIC_STITCH_GPU);
#undef REGISTER_DYNAMIC_STITCH_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_DYNAMIC_STITCH_SYCL(type)               \
  REGISTER_KERNEL_BUILDER(Name("DynamicStitch")          \
                              .Device(DEVICE_SYCL)       \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices")     \
                              .HostMemory("data")        \
                              .HostMemory("merged"),     \
                          DynamicStitchOpCPU<type>)      \
  REGISTER_KERNEL_BUILDER(Name("ParallelDynamicStitch")  \
                              .Device(DEVICE_SYCL)       \
                              .TypeConstraint<type>("T") \
                              .HostMemory("indices")     \
                              .HostMemory("data")        \
                              .HostMemory("merged"),     \
                          ParallelDynamicStitchOpCPU<type>)

TF_CALL_POD_STRING_TYPES(REGISTER_DYNAMIC_STITCH_SYCL);
#undef REGISTER_DYNAMIC_STITCH_SYCL
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
