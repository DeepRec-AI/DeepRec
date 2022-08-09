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

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice; 

template <typename Device>
class TestOp : public OpKernel {
public:
    explicit TestOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        const Tensor* x_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

        std::cout << "\n[INFO]: x_tensor.NumElements = " << x_tensor->NumElements() << "\n" << std::endl;

        Tensor* y_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x_tensor->shape(), &y_tensor));

        *y_tensor = *x_tensor;
    }
};

REGISTER_KERNEL_BUILDER(Name("Test").Device(DEVICE_GPU), 
                        TestOp<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("Test").Device(DEVICE_CPU),
                        TestOp<CPUDevice>);

} // tensorflow