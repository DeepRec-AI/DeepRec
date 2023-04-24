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
#ifdef GOOGLE_CUDA

#include <cstring>

#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

class TensorPackTransH2DOp : public OpKernel {
 public:
  explicit TensorPackTransH2DOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    std::vector<size_t> tensor_offset_vec;
    size_t total_bytes = 0;
    for (int i = 0; i < ctx->num_inputs(); i++) {
      const Tensor& input_tensor = ctx->input(i);
      tensor_offset_vec.emplace_back(total_bytes);
      total_bytes += AlignedBytes(input_tensor.TotalBytes());
    }

    // 1. pack input
    Tensor merged_tensor_cpu;
    TensorShape merged_tensor_shape;
    AllocatorAttributes cpu_alloc_attr;
    merged_tensor_shape.AddDim(total_bytes);
    cpu_alloc_attr.set_on_host(true);
    cpu_alloc_attr.set_gpu_compatible(true);
    // Alloc pin memory
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT8, merged_tensor_shape,
                                           &merged_tensor_cpu, cpu_alloc_attr));

    auto merge_cpu_tensor = [this, ctx, &tensor_offset_vec, &merged_tensor_cpu](
                                int64 start, int64 end) {
      unsigned char* base =
          reinterpret_cast<unsigned char*>(merged_tensor_cpu.data());
      for (int i = start; i < end; i++) {
	const Tensor& input_tensor = ctx->input(i);
	size_t tensor_bytes = input_tensor.TotalBytes();
        std::memcpy(base+tensor_offset_vec[i], input_tensor.data(),
                    tensor_bytes);
      }
    };

    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
    int cost = 1000;
    Shard(worker_threads->num_threads, worker_threads->workers,
          ctx->num_inputs(), cost, merge_cpu_tensor);

    // 2. MemcpyH2D
    auto* dev_context = ctx->op_device_context();
    se::Stream* compute_stream =
        static_cast<const GPUDeviceContext*>(dev_context)->stream();

    Tensor merged_tensor_gpu;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT8, merged_tensor_shape,
                                           &merged_tensor_gpu));
    se::DeviceMemoryBase wrapped_dst(merged_tensor_gpu.data(), total_bytes);
    compute_stream->ThenMemcpy(&wrapped_dst, merged_tensor_cpu.data(), total_bytes)
        .ok();
    compute_stream->BlockHostUntilDone();

    // 3. UnPack
    for(int i = 0; i < ctx->num_inputs(); i++) {
      size_t offset = tensor_offset_vec[i];
      Tensor tensor_slice =
          merged_tensor_gpu.Slice(offset, offset + ctx->input(i).TotalBytes());
      Tensor output(ctx->input(i).dtype());
      output.BitcastFrom(tensor_slice, ctx->input(i).dtype(), ctx->input(i).shape());
      ctx->set_output(i, output);
    }
  }

 private:
  size_t AlignedBytes(size_t s) {
#if EIGEN_MAX_ALIGN_BYTES == 0
    return s;
#else
    return std::ceil(s * 1.0 / EIGEN_MAX_ALIGN_BYTES) * EIGEN_MAX_ALIGN_BYTES;
#endif
  }
};

REGISTER_KERNEL_BUILDER(Name("_TensorPackTransH2D")
                            .Device(DEVICE_GPU)
                            .HostMemory("input_tensor_list"),
                        TensorPackTransH2DOp);

} // End of namespace tensorflow

#endif // End of GOOGLE_CUDA
