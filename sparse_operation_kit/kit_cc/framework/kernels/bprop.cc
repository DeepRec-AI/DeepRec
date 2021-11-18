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
class PluginBpropOp : public OpKernel {
public:
    explicit PluginBpropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        Tensor const *emb_handle_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("emb_handle", &emb_handle_tensor));
        Tensor const *global_replica_id_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("global_replica_id", &global_replica_id_tensor));
        Tensor const *top_gradient_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("top_gradient", &top_gradient_tensor));

        // TODO: what should be returned from bprop to represent the gradients of embedding_variable.
        // In TensorFlow, the gradient for GatherOp is IndexedSlices tensor.
        // So we would better return such tensor to tensorflow.
        // For IndexedSlices, it contains: 
        // values: A tensor of any dtype with [D0, D1, ..., Dn]
        // indices: A 1-D integer tensor with shape [D0]
        // dense_shape: A 1-D integer tensor containing the shape of the corresponding dense tensor.
        
        try {
            // get grad shape
            TensorShape grad_shape;
            SparseOperationKit::Facade::instance()->get_grad_shape(global_replica_id_tensor->scalar<int32_t>()(),
                                                                emb_handle_tensor, 
                                                                grad_shape);
            Tensor* gradient_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, grad_shape, &gradient_tensor));
            Tensor* value_index_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {gradient_tensor->dim_size(0)}, &value_index_tensor));

            // do backward propagation
            SparseOperationKit::Facade::instance()->backward(emb_handle_tensor,
                                                          global_replica_id_tensor->scalar<int32_t>()(),
                                                          top_gradient_tensor,
                                                          gradient_tensor,
                                                          value_index_tensor);
        } catch (std::exception const &error) {
            ctx->SetStatus(errors::Aborted(error.what()));
            return;
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("PluginBprop")
                        .Device(DEVICE_GPU)
                        .HostMemory("emb_handle")
                        .HostMemory("global_replica_id"),
                        PluginBpropOp<GPUDevice>);


} // namespace tensorflow