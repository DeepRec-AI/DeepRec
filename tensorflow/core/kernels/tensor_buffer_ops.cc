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

#include "tensorflow/core/kernels/tensor_buffer_ops.h"

namespace tensorflow {

class TensorBufferOp : public OpKernel {
 public:
  explicit TensorBufferOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    auto rm = ctx->resource_manager();
    auto ndef = def();

    ContainerInfo cinfo;
    OP_REQUIRES_OK(ctx, cinfo.Init(rm, ndef, true /* use name() */));

    TensorBuf* buffer = nullptr;
    OP_REQUIRES_OK(ctx, rm->LookupOrCreate<TensorBuf>(
                            cinfo.container(), cinfo.name(), &buffer,
                            [&ndef](TensorBuf** pbuf) -> Status {
                              int64 capacity;
                              TF_RETURN_IF_ERROR(GetNodeAttr(
                                  ndef, "shared_capacity", &capacity));
                              *pbuf = new TensorBuf(capacity);
                              return Status::OK();
                            }));
    core::ScopedUnref scope(buffer);
    ComputeWithTensorBuf(ctx, buffer);
  }

 protected:
  virtual void ComputeWithTensorBuf(OpKernelContext* ctx, TensorBuf* buf) = 0;
};

class TensorBufferAsyncOp : public AsyncOpKernel {
 public:
  explicit TensorBufferAsyncOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_threads", &shared_threads_));
  }

  void ComputeAsync(OpKernelContext* ctx,
                    AsyncOpKernel::DoneCallback done) override {
    auto rm = ctx->resource_manager();
    NodeDef ndef(def());
    ContainerInfo cinfo;
    OP_REQUIRES_OK_ASYNC(ctx, cinfo.Init(rm, ndef, true /* use name() */),
                         done);
    TensorBuf* buffer = nullptr;
    OP_REQUIRES_OK_ASYNC(ctx, rm->LookupOrCreate<TensorBuf>(
                                  cinfo.container(), cinfo.name(), &buffer,
                                  [&ndef](TensorBuf** resource) {
                                    int64 capacity;
                                    TF_RETURN_IF_ERROR(GetNodeAttr(
                                        ndef, "shared_capacity", &capacity));
                                    *resource = new TensorBuf(capacity);
                                    return Status::OK();
                                  }),
                         done);
    core::ScopedUnref scoped_list(buffer);
    Schedule(buffer, [this, ctx, done, buffer]() {
      ComputeAsyncWithTensorBuf(ctx, done, buffer);
    });
  }

 protected:
  virtual void ComputeAsyncWithTensorBuf(OpKernelContext* ctx,
                                         AsyncOpKernel::DoneCallback done,
                                         TensorBuf* buffer) = 0;

 private:
  string shared_name_;
  int64 shared_threads_;

  void Schedule(TensorBuf* buffer, std::function<void()> fn) {
    buffer->Schedule(shared_name_, shared_threads_, fn);
  }
};

class TensorBufferPutOp : public TensorBufferOp {
 public:
  explicit TensorBufferPutOp(OpKernelConstruction* ctx) : TensorBufferOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("timeout_millis", &timeout_millis_));
  }

  void ComputeWithTensorBuf(OpKernelContext* ctx, TensorBuf* buf) override {
    std::vector<Tensor> record;
    record.reserve(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      record.push_back(ctx->input(i));
    }
    ctx->SetStatus(buf->Put(record, timeout_millis_));
  }

 private:
  int64 timeout_millis_;
};

REGISTER_KERNEL_BUILDER(Name("TensorBufferPut").Device(DEVICE_CPU),
                        TensorBufferPutOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("TensorBufferPut").Device(DEVICE_GPU),
                        TensorBufferPutOp);
#endif  // GOOGLE_CUDA
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("TensorBufferPut").Device(DEVICE_SYCL),
                        TensorBufferPutOp);
#endif  // TENSORFLOW_USE_SYCL

class TensorBufferTakeOp : public TensorBufferAsyncOp {
 public:
  explicit TensorBufferTakeOp(OpKernelConstruction* ctx)
      : TensorBufferAsyncOp(ctx) {}

  void ComputeAsyncWithTensorBuf(OpKernelContext* ctx,
                                 AsyncOpKernel::DoneCallback done,
                                 TensorBuf* buf) override {
    std::vector<Tensor> record;
    Status s = buf->Take(&record);
    if (TF_PREDICT_FALSE(!s.ok())) {
      ctx->SetStatus(s);
      done();
      return;
    }

    OP_REQUIRES_ASYNC(
        ctx, record.size() == (size_t)ctx->num_outputs(),
        errors::Internal(ctx->num_outputs(), " tensors required, but ",
                         record.size(), " tensors were taken."),
        done);

    for (size_t i = 0; i < record.size(); ++i) {
      ctx->set_output(i, record[i]);
    }
    done();
  }
};

REGISTER_KERNEL_BUILDER(Name("TensorBufferTake").Device(DEVICE_CPU),
                        TensorBufferTakeOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("TensorBufferTake").Device(DEVICE_GPU),
                        TensorBufferTakeOp);
#endif  // GOOGLE_CUDA
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("TensorBufferTake").Device(DEVICE_SYCL),
                        TensorBufferTakeOp);
#endif  // TENSORFLOW_USE_SYCL

class TensorBufferCancelOp : public TensorBufferOp {
 public:
  explicit TensorBufferCancelOp(OpKernelConstruction* ctx) : TensorBufferOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_cancelled", &is_cancelled_));
  }

  void ComputeWithTensorBuf(OpKernelContext* ctx, TensorBuf* buf) override {
    ctx->SetStatus(buf->Cancel(is_cancelled_));
  }

 private:
  bool is_cancelled_;
};

REGISTER_KERNEL_BUILDER(Name("TensorBufferCancel").Device(DEVICE_CPU),
                        TensorBufferCancelOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("TensorBufferCancel").Device(DEVICE_GPU),
                        TensorBufferCancelOp);
#endif  // GOOGLE_CUDA
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("TensorBufferCancel").Device(DEVICE_SYCL),
                        TensorBufferCancelOp);
#endif  // TENSORFLOW_USE_SYCL

class TensorBufferCloseOp : public TensorBufferOp {
 public:
  explicit TensorBufferCloseOp(OpKernelConstruction* ctx) : TensorBufferOp(ctx) {}

  void ComputeWithTensorBuf(OpKernelContext* ctx, TensorBuf* buf) override {
    ctx->SetStatus(buf->Close());
  }
};

REGISTER_KERNEL_BUILDER(Name("TensorBufferClose").Device(DEVICE_CPU),
                        TensorBufferCloseOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("TensorBufferClose").Device(DEVICE_GPU),
                        TensorBufferCloseOp);
#endif  // GOOGLE_CUDA
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("TensorBufferClose").Device(DEVICE_SYCL),
                        TensorBufferCloseOp);
#endif  // TENSORFLOW_USE_SYCL

class TensorBufferSizeOp : public TensorBufferOp {
 public:
  explicit TensorBufferSizeOp(OpKernelConstruction* ctx) : TensorBufferOp(ctx) {}

  void ComputeWithTensorBuf(OpKernelContext* ctx, TensorBuf* buf) override {
    Tensor* size = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &size));
    OP_REQUIRES_OK(ctx, buf->GetSize(size));
  }
};

REGISTER_KERNEL_BUILDER(Name("TensorBufferSize").Device(DEVICE_CPU),
                        TensorBufferSizeOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("TensorBufferSize").HostMemory("size").Device(DEVICE_GPU),
    TensorBufferSizeOp);
#endif  // GOOGLE_CUDA
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(
    Name("TensorBufferSize").HostMemory("size").Device(DEVICE_SYCL),
    TensorBufferSizeOp);
#endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
