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

#ifndef TENSORFLOW_COMPILER_JIT_KERNELS_ASYNC_IO_OPS_H_
#define TENSORFLOW_COMPILER_JIT_KERNELS_ASYNC_IO_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// XlaAsyncOutSendOp sends a tensor to XlaAsyncOutRecvOp on the same device.
// Instead of using data edges to send data, XlaAsyncOutSendOp sends the Tensor
// through a Rendezvous mechanism. This operation is primarily created for XLA
// clusters to send out data without waiting for the end of cluster execution
// to trigger output data edges. No data copies are involved in the operation
// as the Send and Recv are required to run on the same device.
class XlaAsyncOutSendOp : public OpKernel {
 public:
  explicit XlaAsyncOutSendOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  string key_;
  uint64 key_hash_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaAsyncOutSendOp);
};

// XlaAsyncOutRecvOp receives a tensor from XlaAsyncOutSendOp on the same
// device. Instead of using data edges to receive data, XlaAsyncOutRecvOp
// receives the Tensor through a Rendezvous mechanism. This operation is
// primarily created to receive data from XLA clusters without waiting for
// the end of cluster execution to trigger output data edges. No data copies
// are involved in the operation as the Send and Recv are required to run on
// the same device.
class XlaAsyncOutRecvOp : public AsyncOpKernel {
 public:
  explicit XlaAsyncOutRecvOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  string key_;
  uint64 key_hash_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaAsyncOutRecvOp);
};

// XlaAsyncOutInitOp initializes needed resources for AsyncOutSendOp and
// AsyncOutRecvOp.
class XlaAsyncOutInitOp : public OpKernel {
 public:
  explicit XlaAsyncOutInitOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

  bool IsExpensive() override { return true; }

 private:
  string device_name_;
  std::vector<string> tensor_names_;
  std::vector<uint64> key_hashes_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaAsyncOutInitOp);
};

// XlaAsyncOutDoneOp finalizes used resources for AsyncOutSendOp and
// AsyncOutRecvOp.
class XlaAsyncOutDoneOp : public OpKernel {
 public:
  explicit XlaAsyncOutDoneOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

  bool IsExpensive() override { return true; }

 private:
  string device_name_;
  std::vector<string> tensor_names_;
  std::vector<uint64> key_hashes_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaAsyncOutDoneOp);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_KERNELS_ASYNC_IO_OPS_H_
