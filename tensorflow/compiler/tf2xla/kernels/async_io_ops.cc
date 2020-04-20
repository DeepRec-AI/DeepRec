/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/kernels/async_io_rendezvous.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace {

class AsyncOutSendOp : public XlaOpKernel {
 public:
  explicit AsyncOutSendOp(OpKernelConstruction* ctx);
  void Compile(XlaOpKernelContext* ctx) override;

 private:
  string tensor_name_;
  string device_name_;
  string key_;

  TF_DISALLOW_COPY_AND_ASSIGN(AsyncOutSendOp);
};

AsyncOutSendOp::AsyncOutSendOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("device_name", &device_name_));
  key_ = AsyncIoRendezvous::GetRendezvousKey(device_name_, tensor_name_);
}

void AsyncOutSendOp::Compile(XlaOpKernelContext* ctx) {
  // Always restore to the TensorShape.
  const Tensor& input = ctx->op_kernel_context()->input(0);
  const TensorShape& shape = input.shape();
  const DataType dtype = input.dtype();
  xla::Shape xla_shape;
  OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));
  // Tensorflow always expects a row-major layout. So, always provide a shape
  // with row-major layout to AsyncOutSend. The XLA layout_assignment pass will
  // be constrained accordingly.
  xla::Shape shape_with_layout = xla::ShapeUtil::MakeShapeWithDescendingLayout(
      xla_shape.element_type(), xla_shape.dimensions());
  xla::AsyncOutSend(ctx->Input(0), shape_with_layout, key_);
}

REGISTER_XLA_OP(Name("_XlaAsyncOutSend"), AsyncOutSendOp);

}  // namespace
}  // namespace tensorflow
