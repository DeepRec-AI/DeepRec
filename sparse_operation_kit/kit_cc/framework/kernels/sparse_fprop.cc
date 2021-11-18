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
class PluginSparseFpropOp : public OpKernel {
public:
    explicit PluginSparseFpropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("training", &training_));
    }
    void Compute(OpKernelContext* ctx) override {
        Tensor const *emb_handle_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("emb_handle", &emb_handle_tensor));
        Tensor const *values_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("values", &values_tensor));
        Tensor const *indices_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("indices", &indices_tensor));
        Tensor const *global_replica_id_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("global_replica_id", &global_replica_id_tensor));

        // get output shape for the first time
        if (0 == emb_vector_tensor_shape_.dims()) {
            try {
                SparseOperationKit::Facade::instance()->get_output_shape(emb_handle_tensor, 
                                                                      emb_vector_tensor_shape_);
            } catch (std::exception const &error) {
                ctx->SetStatus(errors::Aborted(error.what()));
                return;
            }
        } 

        // allocate output
        Tensor *emb_vector_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, emb_vector_tensor_shape_, &emb_vector_tensor));

        // do forward propagation
        try {
            // TODO: check values and indices shape

            SparseOperationKit::Facade::instance()->forward(emb_handle_tensor, 
                                                         values_tensor, indices_tensor, 
                                                         global_replica_id_tensor->scalar<int32_t>()(),
                                                         training_,
                                                         emb_vector_tensor);
        } catch (std::exception const &error) {
            ctx->SetStatus(errors::Aborted(error.what()));
            return;
        }
    }
private:
    bool training_;
    TensorShape emb_vector_tensor_shape_;
};

REGISTER_KERNEL_BUILDER(Name("PluginSparseFprop")
                        .Device(DEVICE_GPU)
                        .HostMemory("emb_handle")
                        .HostMemory("global_replica_id"),
                        PluginSparseFpropOp<GPUDevice>);

} // namespace tensorflow