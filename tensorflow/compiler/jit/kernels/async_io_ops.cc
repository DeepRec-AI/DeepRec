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

#include "tensorflow/compiler/jit/kernels/async_io_ops.h"

#include "tensorflow/compiler/jit/kernels/async_io_rendezvous.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {
AsyncIoRendezvous::DoneCallback make_recv_callback(
    OpKernelContext* ctx, AsyncOpKernel::DoneCallback done) {
  using namespace std::placeholders;
  return std::bind(
      [ctx](AsyncOpKernel::DoneCallback done,
            // Begin unbound arguments.
            const Status& s, const AsyncIoRendezvous::TensorPayload& val) {
        Tensor tensor;
        if (val.addr.is_null()) {
          VLOG(2) << "AsyncIoRendezvous::DoneCallback with Tensor size "
                  << val.tensor.TotalBytes();
          tensor = val.tensor;
        } else {
          VLOG(2) << "AsyncIoRendezvous::DoneCallback with payload size "
                  << val.addr.size() << " @" << val.addr.opaque();
          TensorShape tensor_shape;
          XLAShapeToTensorShape(val.shape, &tensor_shape);
          auto data_type =
              EncodePrimitiveTypeAsDataType(val.shape.element_type())
                  .ValueOrDie();
          tensor =
              XlaTensorBuffer::MakeTensor(data_type, tensor_shape, val.addr,
                                          ctx->device()->GetAllocator({}));
        }

        ctx->SetStatus(s);
        if (s.ok()) {
          // 'ctx' allocates the output tensor of the expected type.
          // The runtime checks whether the tensor received here is
          // the same type.
          ctx->set_output(0, tensor);
        }
        done();
      },
      std::move(done), _1, _2);
}
}  // anonymous

XlaAsyncOutSendOp::XlaAsyncOutSendOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  string device_name, tensor_name;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("device_name", &device_name));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name));
  key_ = AsyncIoRendezvous::GetRendezvousKey(device_name, tensor_name);
  key_hash_ = AsyncIoRendezvous::GetRendezvousKeyHash(key_);
}

void XlaAsyncOutSendOp::Compute(OpKernelContext* ctx) {
  VLOG(2) << "_XlaAsyncOutSend with key " << key_ << ", hash " << key_hash_;
  AsyncIoRendezvous::TensorPayload val;
  val.tensor = ctx->input(0);
  GetXlaAsyncIORendezvous()->Send(key_hash_, val);
}

XlaAsyncOutRecvOp::XlaAsyncOutRecvOp(OpKernelConstruction* ctx)
    : AsyncOpKernel(ctx) {
  string device_name, tensor_name;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("device_name", &device_name));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name));
  key_ = AsyncIoRendezvous::GetRendezvousKey(device_name, tensor_name);
  key_hash_ = AsyncIoRendezvous::GetRendezvousKeyHash(key_);
}

void XlaAsyncOutRecvOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  VLOG(2) << "_XlaAsyncOutRecv with key " << key_ << ", hash " << key_hash_;
  GetXlaAsyncIORendezvous()->RecvAsync(
      key_hash_, make_recv_callback(ctx, std::move(done)));
}

// Serves as a helper function to initialize XlaAsyncOutInitOp and
// XlaAsyncOutDoneOp.
void Initialize(OpKernelConstruction* ctx, string* device_name,
                std::vector<string>* tensor_names,
                std::vector<uint64>* key_hashes) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("device_name", device_name));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_names", tensor_names));
  for (const string& t : *tensor_names) {
    string k = AsyncIoRendezvous::GetRendezvousKey(*device_name, t);
    uint64 h = AsyncIoRendezvous::GetRendezvousKeyHash(k);
    key_hashes->push_back(h);
  }
}

XlaAsyncOutInitOp::XlaAsyncOutInitOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  Initialize(ctx, &device_name_, &tensor_names_, &key_hashes_);
}

void XlaAsyncOutInitOp::Compute(OpKernelContext* ctx) {
  for (uint64 h : key_hashes_) {
    GetXlaAsyncIORendezvous()->InitializeRendezvousQueue(h);
  }
}

XlaAsyncOutDoneOp::XlaAsyncOutDoneOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  Initialize(ctx, &device_name_, &tensor_names_, &key_hashes_);
}

void XlaAsyncOutDoneOp::Compute(OpKernelContext* ctx) {
  for (uint64 h : key_hashes_) {
    GetXlaAsyncIORendezvous()->FinalizeRendezvousQueue(h);
  }
}

REGISTER_KERNEL_BUILDER(Name("_XlaAsyncOutSend").Device(DEVICE_GPU),
                        XlaAsyncOutSendOp);
REGISTER_KERNEL_BUILDER(Name("_XlaAsyncOutRecv").Device(DEVICE_GPU),
                        XlaAsyncOutRecvOp);
REGISTER_KERNEL_BUILDER(Name("_XlaAsyncOutInit").Device(DEVICE_GPU),
                        XlaAsyncOutInitOp);
REGISTER_KERNEL_BUILDER(Name("_XlaAsyncOutDone").Device(DEVICE_GPU),
                        XlaAsyncOutDoneOp);

}  // end namespace tensorflow
