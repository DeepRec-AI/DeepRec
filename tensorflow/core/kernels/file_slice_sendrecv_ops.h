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

#ifndef TENSORFLOW_CORE_KERNELS_FILE_SLICE_SENDRECV_OPS_H_
#define TENSORFLOW_CORE_KERNELS_FILE_SLICE_SENDRECV_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class FileSliceSendOp : public OpKernel {
 public:
  explicit FileSliceSendOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  // Variables.
  string key_prefix_;
  bool hostmem_sendrecv_;
  int32 slice_size_;

  // Functions.
  Status GetElementBytes(OpKernelContext* ctx, const Tensor& file_path_t,
                         uint64& element_bytes);

  Status SendUInt64MetaMsg(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                           const string& name, const uint64 val);

  Status SendTotalBytes(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                        const uint64 total_bytes);

  Status SendScalarShape(OpKernelContext* ctx, const FrameAndIter& frame_iter);

  Status SendElementBytes(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                         const uint64 element_bytes);

  Status SendFileSlice(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                       const Tensor& file_path_t, const uint64 element_bytes);

  Status ReadFileSlice(const std::unique_ptr<RandomAccessFile>& file,
                       const uint64 pos, const uint64 offset, Tensor& data_t);

  TF_DISALLOW_COPY_AND_ASSIGN(FileSliceSendOp);
};

class FileSliceRecvOp: public OpKernel {
 public:
  explicit FileSliceRecvOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  // Variables.
  string key_prefix_;
  bool hostmem_sendrecv_;
  string recv_dir_;
  int32 slice_size_;
  int64 timeout_ms_;

  // Functions.
  Status RecvUInt64MetaMsg(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                           const string& name, bool &is_dead, uint64& val);

  Status RecvTotalBytes(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                        bool& is_dead, uint64& total_bytes);

  string GenerateRecvFileName(const string& op_name);

  Status RecvFile(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                  const string& file_path);

  Status RecvShape(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                   TensorShape& shape);

  Status RecvElementBytes(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                          uint64& element_bytes);

  Status RecvFileSlice(OpKernelContext* ctx, const FrameAndIter& frame_iter,
                       const uint64 element_bytes, const string& file_path);

  TF_DISALLOW_COPY_AND_ASSIGN(FileSliceRecvOp);
};

}; // End of namespace tensorflow

#endif // End of macro TENSORFLOW_CORE_KERNELS_FILE_SLICE_SENDRECV_OPS_H_
