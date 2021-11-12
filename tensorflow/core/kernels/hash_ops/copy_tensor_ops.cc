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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/hash_table/hash_table.h"

namespace tensorflow {

class CopyTensorOp : public OpKernel {
 public:
  explicit CopyTensorOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& src_tensor = context->input(0);
    context->set_output(0, tensor::DeepCopy(src_tensor));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("CopyTensor").Device(DEVICE_CPU), CopyTensorOp);

}  // namespace tensorflow
