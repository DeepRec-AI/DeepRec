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

#include "facade.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <exception>

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 

template <typename Device>
class PluginInitOp : public OpKernel {
public:
    explicit PluginInitOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("global_batch_size", &global_batch_size_));
        OP_REQUIRES(ctx, global_batch_size_ > 0, errors::Aborted(__FILE__, ":", __LINE__, " ",
                        "global_batch_size must be > 0."));
    }
    void Compute(OpKernelContext* ctx) override {
        const Tensor* global_replica_id_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("global_replica_id", &global_replica_id_tensor));
        const Tensor* num_replicas_in_sync_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("num_replicas_in_sync", &num_replicas_in_sync_tensor));
        const Tensor* nccl_unique_id_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("nccl_unique_id", &nccl_unique_id_tensor));
        const Tensor* global_seed_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("global_seed", &global_seed_tensor));

        try {
            int32_t global_replica_id = global_replica_id_tensor->scalar<int32_t>()(0);
            int32_t num_replicas_in_sync = num_replicas_in_sync_tensor->scalar<int32_t>()(0);
            const uint64_t global_seed = global_seed_tensor->scalar<int64_t>()(0);

            const cudaStream_t& tf_stream = ctx->eigen_device<Device>().stream();

            SparseOperationKit::Facade::instance()->init(global_replica_id, 
                                                      num_replicas_in_sync, 
                                                      nccl_unique_id_tensor->flat<int32_t>().data(),
                                                      global_seed, global_batch_size_, tf_stream);
        } catch (const std::exception& error) {
            ctx->SetStatus(errors::Aborted(error.what()));
            return;
        }

        Tensor* status_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &status_tensor));
        status_tensor->flat<tstring>()(0) = "OK";
    }
private:
    tensorflow::int64 global_batch_size_;
};

REGISTER_KERNEL_BUILDER(Name("PluginInit")
                        .Device(DEVICE_GPU)
                        .HostMemory("global_replica_id")
                        .HostMemory("num_replicas_in_sync")
                        .HostMemory("nccl_unique_id")
                        .HostMemory("global_seed")
                        .HostMemory("status"),
                        PluginInitOp<GPUDevice>);

} // namespace tensorflow