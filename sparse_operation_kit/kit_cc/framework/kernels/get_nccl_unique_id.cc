/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <exception>

#include "facade.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device>
class GetNcclUniqueIdOp : public OpKernel {
 public:
  explicit GetNcclUniqueIdOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    Tensor* nccl_unique_id_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {32}, &nccl_unique_id_tensor));
    try {
      SparseOperationKit::Facade::instance()->get_nccl_unique_id(
          nccl_unique_id_tensor->flat<int32_t>().data());
    } catch (const std::exception& error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("GetNcclUniqueId").Device(DEVICE_GPU).HostMemory("nccl_unique_id"),
                        GetNcclUniqueIdOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("GetNcclUniqueId").Device(DEVICE_CPU).HostMemory("nccl_unique_id"),
                        GetNcclUniqueIdOp<CPUDevice>);

}  // namespace tensorflow