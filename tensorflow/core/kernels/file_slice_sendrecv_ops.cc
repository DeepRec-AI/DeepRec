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

#include "tensorflow/core/kernels/file_slice_sendrecv_ops.h"
#include "tensorflow/core/kernels/slice_sendrecv_utils.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

//------------------------------------------------------------------------------
// Functions of FileSliceSendOp.

FileSliceSendOp::FileSliceSendOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  string send_device;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device));
  string recv_device;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device));
  uint64 send_device_incarnation;
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr("send_device_incarnation",
                        reinterpret_cast<int64*>(&send_device_incarnation)));
  string tensor_name;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name));
  key_prefix_ = \
    slice_sendrecv::GetSliceRendezvousKeyPrefix(send_device,
                      recv_device, send_device_incarnation, tensor_name);

  if (!ctx->GetAttr("_hostmem_sendrecv", &hostmem_sendrecv_).ok()) {
    hostmem_sendrecv_ = false;
  }
  OP_REQUIRES_OK(ctx, ctx->GetAttr("slice_size", &slice_size_));
}

void FileSliceSendOp::Compute(OpKernelContext* ctx) {
  OP_REQUIRES(ctx, ctx->rendezvous() != nullptr,
    errors::Internal("Op kernel context needs to provide a rendezvous."));

  const Tensor& file_path_t = ctx->input(0);
  if (!ctx->is_input_dead()) {
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(file_path_t.shape()),
                errors::InvalidArgument("file_path is not a scalar: ",
                                        file_path_t.shape().DebugString()));
  }

  FrameAndIter frame_iter = \
    slice_sendrecv::GetFrameAndIter(ctx, hostmem_sendrecv_);

  // get element_bytes.
  uint64 element_bytes = 0;
  OP_REQUIRES_OK(ctx, GetElementBytes(ctx, file_path_t, element_bytes));

  // send total_bytes.
  // total_bytes is the TotalBytes of the Tensor that contains the contents of
  // the file. please refer Tensor::TotalBytes()
  uint64 total_bytes = element_bytes + sizeof(tstring);
  OP_REQUIRES_OK(ctx, SendTotalBytes(ctx, frame_iter, total_bytes));
  // if input is dead, only send total_bytes dead tensor.
  if (ctx->is_input_dead()) {
    return;
  }

  // if total bytes is smaller than slice size, send directly.
  if (total_bytes <= slice_size_) {
    Rendezvous::Args args;
    args.device_context = ctx->op_device_context();
    args.alloc_attrs = ctx->input_alloc_attr(0);

    Rendezvous::ParsedKey parsed_key;
    slice_sendrecv::GetSliceRendezvousKey(key_prefix_, "_transfer_data",
                                          frame_iter, &parsed_key.buf_);
    VLOG(2) << "FileSliceSend " << parsed_key.buf_;
    OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(parsed_key.buf_, &parsed_key));
    Tensor data_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_STRING, TensorShape({}), &data_t));
    if (element_bytes > 0) {
      OP_REQUIRES_OK(ctx, ReadFileToString(Env::Default(),
        file_path_t.scalar<tstring>()(), data_t.scalar<tstring>().data()));
    }
    OP_REQUIRES_OK(ctx, ctx->rendezvous()->Send(parsed_key,args, data_t,
                                                ctx->is_input_dead()));
    return;
  }

  // send shape, in order to match the behavior of 'SliceSend'.
  OP_REQUIRES_OK(ctx, SendScalarShape(ctx, frame_iter));

  // send element bytes, in order to match the behavior of 'SliceSend'.
  OP_REQUIRES_OK(ctx, SendElementBytes(ctx, frame_iter, element_bytes));

  // send data.
  OP_REQUIRES_OK(ctx, SendFileSlice(ctx, frame_iter, file_path_t, element_bytes));
}

Status FileSliceSendOp::GetElementBytes(OpKernelContext* ctx,
                                        const Tensor& file_path_t,
                                        uint64& element_bytes) {

  if (ctx->is_input_dead()) {
    element_bytes = 0;
    return Status::OK();
  }

  const string& file_path = file_path_t.scalar<tstring>()();
  Env* env = Env::Default();

  if (env->FileExists(file_path) != Status::OK()) {
    element_bytes = 0;
    return Status::OK();
  }

  return env->GetFileSize(file_path, &element_bytes);
}

Status FileSliceSendOp::SendUInt64MetaMsg(OpKernelContext* ctx,
                                          const FrameAndIter& frame_iter,
                                          const string& name,
                                          const uint64 val) {
  Rendezvous::Args args;
  args.device_context = ctx->op_device_context();
  args.alloc_attrs = AllocatorAttributes();

  Rendezvous::ParsedKey parsed_key;
  Tensor val_t;
  TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_UINT64, TensorShape({}), &val_t));
  val_t.scalar<uint64>()() = val;
  slice_sendrecv::GetSliceRendezvousKey(key_prefix_, name, frame_iter,
                                        &parsed_key.buf_);
  VLOG(2) << "FileSliceSend " << parsed_key.buf_;
  TF_RETURN_IF_ERROR(Rendezvous::ParseKey(parsed_key.buf_, &parsed_key));
  return ctx->rendezvous()->Send(parsed_key, args, val_t, ctx->is_input_dead());
}

Status FileSliceSendOp::SendTotalBytes(OpKernelContext* ctx,
                                       const FrameAndIter& frame_iter,
                                       const uint64 total_bytes) {
  return SendUInt64MetaMsg(ctx, frame_iter, "_slice_transfer_totalbytes",
                           total_bytes);
}

Status FileSliceSendOp::SendScalarShape(OpKernelContext* ctx,
                                        const FrameAndIter& frame_iter) {
  Rendezvous::Args args;
  args.device_context = ctx->op_device_context();
  args.alloc_attrs = AllocatorAttributes();
  Rendezvous::ParsedKey parsed_key;

  Tensor shape_t;
  TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT64, TensorShape({0}), &shape_t));
  slice_sendrecv::GetSliceRendezvousKey(key_prefix_,
                    "_slice_transfer_shape", frame_iter, &parsed_key.buf_);
  VLOG(2) << "FileSliceSend " << parsed_key.buf_;
  TF_RETURN_IF_ERROR(Rendezvous::ParseKey(parsed_key.buf_, &parsed_key));

  return ctx->rendezvous()->Send(parsed_key, args, shape_t,
                                 ctx->is_input_dead());
}

Status FileSliceSendOp::SendElementBytes(OpKernelContext* ctx,
                                         const FrameAndIter& frame_iter,
                                         const uint64 element_bytes) {
  return SendUInt64MetaMsg(ctx, frame_iter, "_slice_transfer_elements_bytes",
                           element_bytes);
}

Status FileSliceSendOp::SendFileSlice(OpKernelContext* ctx,
                                      const FrameAndIter& frame_iter,
                                      const Tensor& file_path_t,
                                      const uint64 element_bytes) {
  Rendezvous::Args args;
  args.device_context = ctx->op_device_context();
  args.alloc_attrs = AllocatorAttributes();
  Rendezvous::ParsedKey parsed_key;

  std::unique_ptr<RandomAccessFile> file;
  Env* env = Env::Default();
  const string& file_path = file_path_t.scalar<tstring>()();
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(file_path, &file));

  // Slice Send.
  int64 slice_num = element_bytes / slice_size_;
  if (element_bytes % slice_size_ != 0) {
    slice_num += 1;
  }
  Tensor data_t;
  for (int64 i = 0; i < slice_num; i++) {
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_STRING, TensorShape({}), &data_t));
    uint64 start = i * slice_size_;
    uint64 copy_size = slice_size_;
    if (start > element_bytes - slice_size_) {
      copy_size = element_bytes - start;
    }
    TF_RETURN_IF_ERROR(ReadFileSlice(file, start, copy_size, data_t));
    std::string tensor_name_suffix = \
      strings::StrCat("_slice_transfer_data_", std::to_string(0), "_",
                      std::to_string(i));
    slice_sendrecv::GetSliceRendezvousKey(key_prefix_, tensor_name_suffix,
                                          frame_iter, &parsed_key.buf_);
    VLOG(2) << "FileSliceSend " << parsed_key.buf_;
    TF_RETURN_IF_ERROR(Rendezvous::ParseKey(parsed_key.buf_, &parsed_key));
    TF_RETURN_IF_ERROR(ctx->rendezvous()->Send(parsed_key, args, data_t,
                                               ctx->is_input_dead()));
  }


  return Status::OK();
}

Status FileSliceSendOp::ReadFileSlice(
                          const std::unique_ptr<RandomAccessFile>& file,
                          const uint64 pos, const uint64 offset,
                          Tensor& data_t) {
  string* data_s = data_t.scalar<tstring>().data();
  gtl::STLStringResizeUninitialized(data_s, offset);
  char* data_p = gtl::string_as_array(data_s);
  StringPiece result;
  TF_RETURN_IF_ERROR(file->Read(pos, offset, &result, data_p));
  if (result.data() != data_p) {
    memmove(data_p, result.data(), result.size());
  }

  return Status::OK();
}

REGISTER_KERNEL_BUILDER(Name("_FileSliceSend").Device(DEVICE_CPU),
                        FileSliceSendOp);
REGISTER_KERNEL_BUILDER(Name("_FileSliceSend").Device(DEVICE_DEFAULT),
                        FileSliceSendOp);

//------------------------------------------------------------------------------
// Functions of FileSliceRecvOp.

FileSliceRecvOp::FileSliceRecvOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  string send_device;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device));
  string recv_device;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device));
  uint64 send_device_incarnation;
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr("send_device_incarnation",
                        reinterpret_cast<int64*>(&send_device_incarnation)));
  string tensor_name;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name));
  key_prefix_ = \
    slice_sendrecv::GetSliceRendezvousKeyPrefix(send_device,
                      recv_device, send_device_incarnation, tensor_name);
  if (!ctx->GetAttr("_hostmem_sendrecv", &hostmem_sendrecv_).ok()) {
    hostmem_sendrecv_ = false;
  }
  OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_dir", &recv_dir_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("slice_size", &slice_size_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("timeout_ms", &timeout_ms_));
}

void FileSliceRecvOp::Compute(OpKernelContext* ctx) {
  OP_REQUIRES(ctx, ctx->rendezvous() != nullptr,
    errors::Internal("Op kernel context needs to provide a rendezvous."));

  FrameAndIter frame_iter = \
    slice_sendrecv::GetFrameAndIter(ctx, hostmem_sendrecv_);

  bool is_dead = false;
  uint64 total_bytes = 0;
  OP_REQUIRES_OK(ctx, RecvTotalBytes(ctx, frame_iter, is_dead, total_bytes));
  if (is_dead) {
    return;
  }

  // Create file path output.
  Env* env = Env::Default();
  if (!env->FileExists(recv_dir_).ok()) {
    OP_REQUIRES_OK(ctx, env->RecursivelyCreateDir(recv_dir_));
  }
  const string &filename = GenerateRecvFileName(ctx->op_kernel().name());
  const string &file_path = io::JoinPath(recv_dir_, "tempfilerecv-"+filename);
  Tensor* file_path_t = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &file_path_t));
  file_path_t->scalar<tstring>()() = file_path;

  // if total bytes is smaller than slice size, recv directly.
  if (total_bytes <= slice_size_) {
    OP_REQUIRES_OK(ctx, RecvFile(ctx, frame_iter, file_path));
    return;
  }

  // recv shape, in order to match the behavior of 'SliceRecv'.
  TensorShape shape;
  OP_REQUIRES_OK(ctx, RecvShape(ctx, frame_iter, shape));
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(shape),
    errors::InvalidArgument(
      "FileSliceRecv only supports receiving a tensor with a scalar shape."));

  // recv element_bytes, in order to match the behavior of 'SliceRecv'.
  uint64 element_bytes = 0;
  OP_REQUIRES_OK(ctx, RecvElementBytes(ctx, frame_iter, element_bytes));

  // recv data.
  OP_REQUIRES_OK(ctx, RecvFileSlice(ctx, frame_iter, element_bytes, file_path));
}

Status FileSliceRecvOp::RecvUInt64MetaMsg(OpKernelContext* ctx,
                                          const FrameAndIter& frame_iter,
                                          const string& name, bool &is_dead,
                                          uint64& val) {
  Rendezvous::Args args;
  args.device_context = ctx->op_device_context();
  args.alloc_attrs = AllocatorAttributes();
  if (ctx->is_eager()) {
    // NOTE(fishx): Only set cancellation_manager in eager mode. Because in
    // Tensorflow 1.x, session (or graph_mgr) will abort the underlying
    // rendezvous if it encounters any error.
    args.cancellation_manager = ctx->cancellation_manager();
  }

  Rendezvous::ParsedKey parsed_key;
  Tensor val_t;
  slice_sendrecv::GetSliceRendezvousKey(key_prefix_, name, frame_iter,
                                        &parsed_key.buf_);
  VLOG(2) << "FileSliceRecv " << parsed_key.buf_;
  TF_RETURN_IF_ERROR(Rendezvous::ParseKey(parsed_key.buf_, &parsed_key));
  TF_RETURN_IF_ERROR(
    ctx->rendezvous()->Recv(parsed_key, args, &val_t, &is_dead, timeout_ms_));
  if (!is_dead) {
    val = val_t.scalar<uint64>()();
  }

  return Status::OK();
}

Status FileSliceRecvOp::RecvTotalBytes(OpKernelContext* ctx,
                                       const FrameAndIter& frame_iter,
                                       bool& is_dead, uint64& total_bytes) {
  return RecvUInt64MetaMsg(ctx, frame_iter, "_slice_transfer_totalbytes",
                           is_dead, total_bytes);
}

string FileSliceRecvOp::GenerateRecvFileName(const string& op_name) {
  const std::vector<string>& file_name_vec = absl::StrSplit(op_name, "/");
  return absl::StrJoin(file_name_vec, "_");
}

Status FileSliceRecvOp::RecvShape(OpKernelContext* ctx,
                                  const FrameAndIter& frame_iter,
                                  TensorShape& shape) {
  Rendezvous::Args args;
  args.device_context = ctx->op_device_context();
  args.alloc_attrs = AllocatorAttributes();
  if (ctx->is_eager()) {
    // NOTE(fishx): Only set cancellation_manager in eager mode. Because in
    // Tensorflow 1.x, session (or graph_mgr) will abort the underlying
    // rendezvous if it encounters any error.
    args.cancellation_manager = ctx->cancellation_manager();
  }

  Rendezvous::ParsedKey parsed_key;
  slice_sendrecv::GetSliceRendezvousKey(key_prefix_, "_slice_transfer_shape",
                                        frame_iter, &parsed_key.buf_);
  VLOG(2) << "FileSliceRecv " << parsed_key.buf_;
  TF_RETURN_IF_ERROR(Rendezvous::ParseKey(parsed_key.buf_, &parsed_key));

  Tensor shape_t;
  bool is_dead;
  TF_RETURN_IF_ERROR(ctx->rendezvous()->Recv(parsed_key, args, &shape_t,
                                             &is_dead, timeout_ms_));
  // This shouldn't be a dead tensor.
  CHECK_EQ(is_dead, false);
  auto shape_vec = shape_t.vec<int64>();
  const int64 num_elements = shape_t.NumElements();
  for (int64 i = 0; i < num_elements; i++) {
    shape.AddDim(shape_vec(i));
  }

  return Status::OK();
}

Status FileSliceRecvOp::RecvElementBytes(OpKernelContext* ctx,
                                        const FrameAndIter& frame_iter,
                                        uint64& element_bytes) {
  bool is_dead = false;
  Status s = \
    RecvUInt64MetaMsg(ctx, frame_iter, "_slice_transfer_elements_bytes", is_dead,
                      element_bytes);
  CHECK_EQ(is_dead, false);

  return s;
}

Status FileSliceRecvOp::RecvFile(OpKernelContext* ctx,
                                 const FrameAndIter& frame_iter,
                                 const string& file_path) {
  Rendezvous::Args args;
  args.device_context = ctx->op_device_context();
  args.alloc_attrs = ctx->output_alloc_attr(0);
  if (ctx->is_eager()) {
    // NOTE(fishx): Only set cancellation_manager in eager mode. Because in
    // Tensorflow 1.x, session (or graph_mgr) will abort the underlying
    // rendezvous if it encounters any error.
    args.cancellation_manager = ctx->cancellation_manager();
  }

  Rendezvous::ParsedKey parsed_key;
  slice_sendrecv::GetSliceRendezvousKey(key_prefix_, "_transfer_data",
                                        frame_iter, &parsed_key.buf_);
  VLOG(2) << "FileSliceRecv " << parsed_key.buf_;
  TF_RETURN_IF_ERROR(Rendezvous::ParseKey(parsed_key.buf_, &parsed_key));
  Tensor data_t;
  bool is_dead = false;
  TF_RETURN_IF_ERROR(ctx->rendezvous()->Recv(parsed_key, args, &data_t,
                                             &is_dead, timeout_ms_));

  // This shouldn't be a dead tensor.
  CHECK_EQ(is_dead, false);

  // Write data_t to file.
  Env* env = Env::Default();
  return WriteStringToFile(env, file_path, data_t.scalar<tstring>()());
}

Status FileSliceRecvOp::RecvFileSlice(OpKernelContext* ctx,
                                      const FrameAndIter& frame_iter,
                                      const uint64 element_bytes,
                                      const string& file_path) {
  // create file
  Env* env = Env::Default();
  std::unique_ptr<WritableFile> file_ptr;
  TF_RETURN_IF_ERROR(env->NewWritableFile(file_path, &file_ptr));

  Rendezvous::Args args;
  args.device_context = ctx->op_device_context();
  args.alloc_attrs = ctx->output_alloc_attr(0);
  if (ctx->is_eager()) {
    // NOTE(fishx): Only set cancellation_manager in eager mode. Because in
    // Tensorflow 1.x, session (or graph_mgr) will abort the underlying
    // rendezvous if it encounters any error.
    args.cancellation_manager = ctx->cancellation_manager();
  }
  Rendezvous::ParsedKey parsed_key;

  int64 slice_num = element_bytes / slice_size_;
  if (element_bytes % slice_size_ != 0) {
    slice_num += 1;
  }
  Tensor data_t;
  bool is_dead = false;
  for (int64 i = 0; i < slice_num; i++) {
    std::string tensor_name_suffix = \
      strings::StrCat("_slice_transfer_data_", std::to_string(0), "_",
                      std::to_string(i));
    slice_sendrecv::GetSliceRendezvousKey(key_prefix_, tensor_name_suffix,
                                          frame_iter, &parsed_key.buf_);
    VLOG(2) << "FileSliceRecv " << parsed_key.buf_;
    TF_RETURN_IF_ERROR(Rendezvous::ParseKey(parsed_key.buf_, &parsed_key));
    TF_RETURN_IF_ERROR(ctx->rendezvous()->Recv(parsed_key, args, &data_t,
                                               &is_dead, timeout_ms_));
    // This shouldn't be a dead tensor.
    CHECK_EQ(is_dead, false);
    file_ptr->Append(data_t.scalar<tstring>()());
  }

  return Status::OK();
}

REGISTER_KERNEL_BUILDER(Name("_FileSliceRecv").Device(DEVICE_CPU),
                        FileSliceRecvOp);
REGISTER_KERNEL_BUILDER(Name("_FileSliceRecv").Device(DEVICE_DEFAULT),
                        FileSliceRecvOp);

}; // End of namespace tensorflow
