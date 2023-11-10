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

#ifndef TENSORFLOW_CORE_KERNELS_SLICE_SENDRECV_OPS_H_
#define TENSORFLOW_CORE_KERNELS_SLICE_SENDRECV_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class SliceSendOp : public OpKernel {
 public:
  explicit SliceSendOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  // Variables.
  string key_prefix_;
  bool hostmem_sendrecv_;
  int32 slice_size_;
  DataType dtype_;

  // Functions.
  Status SendTotalBytes(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                        const Tensor& input_t);

  Status SendShape(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                   const Tensor& input_t);
  Status SendString(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                    const Tensor& input_t);

  Status SendStringSlice(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                         const std::string& elem, int64 index);

  Status SendBasicType(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                       const Tensor& input_t);

  TF_DISALLOW_COPY_AND_ASSIGN(SliceSendOp);
};

class SliceRecvOp : public OpKernel {
 public:
  explicit SliceRecvOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  // Variable.
  string key_prefix_;
  bool hostmem_sendrecv_;
  int32 slice_size_;
  int64 timeout_ms_;
  DataType dtype_;

  // Fucntions.
  Status RecvTotalBytes(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                        bool& is_dead, int64& total_bytes);

  Status RecvShape(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                   TensorShape& shape);

  Status RecvString(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                    const TensorShape& shape, Tensor*& output_t);

  Status RecvStringSlice(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                         const int64 index, const int64 element_size,
                         TTypes<tstring>::Flat& output_flat);

  Status RecvBasicType(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                       const int64 total_bytes, Tensor*& output_t);

  TF_DISALLOW_COPY_AND_ASSIGN(SliceRecvOp);
};

} // End of namespace tensorflow

#endif // End of TENSORFLOW_CORE_KERNELS_SLICE_SENDRECV_OPS_H_
