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

#include "tensorflow/core/kernels/data_buffer_ops.h"

namespace tensorflow {

class DataBufferOp : public OpKernel {
 public:
  explicit DataBufferOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    auto rm = ctx->resource_manager();
    auto ndef = def();

    ContainerInfo cinfo;
    OP_REQUIRES_OK(ctx, cinfo.Init(rm, ndef, true /* use name() */));

    DataBuffer* buffer = nullptr;
    OP_REQUIRES_OK(ctx, rm->LookupOrCreate<DataBuffer>(
                            cinfo.container(), cinfo.name(), &buffer,
                            [&ndef](DataBuffer** pbuf) -> Status {
                              int64 capacity;
                              TF_RETURN_IF_ERROR(GetNodeAttr(
                                  ndef, "shared_capacity", &capacity));
                              *pbuf = new DataBuffer(capacity);
                              return Status::OK();
                            }));
    core::ScopedUnref scope(buffer);
    ComputeWithDataBuffer(ctx, buffer);
  }

 protected:
  virtual void ComputeWithDataBuffer(OpKernelContext* ctx, DataBuffer* buf) = 0;
};

class DataBufferAsyncOp : public AsyncOpKernel {
 public:
  explicit DataBufferAsyncOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
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
    DataBuffer* buffer = nullptr;
    OP_REQUIRES_OK_ASYNC(ctx, rm->LookupOrCreate<DataBuffer>(
                                  cinfo.container(), cinfo.name(), &buffer,
                                  [&ndef](DataBuffer** resource) {
                                    int64 capacity;
                                    TF_RETURN_IF_ERROR(GetNodeAttr(
                                        ndef, "shared_capacity", &capacity));
                                    *resource = new DataBuffer(capacity);
                                    return Status::OK();
                                  }),
                         done);
    core::ScopedUnref scoped_list(buffer);
    Schedule(buffer, [this, ctx, done, buffer]() {
      ComputeAsyncWithDataBuffer(ctx, done, buffer);
    });
  }

 protected:
  virtual void ComputeAsyncWithDataBuffer(OpKernelContext* ctx,
                                          AsyncOpKernel::DoneCallback done,
                                          DataBuffer* buffer) = 0;

 private:
  string shared_name_;
  int64 shared_threads_;

  void Schedule(DataBuffer* buffer, std::function<void()> fn) {
    buffer->Schedule(shared_name_, shared_threads_, fn);
  }
};

class DataBufferPutOp : public DataBufferOp {
 public:
  explicit DataBufferPutOp(OpKernelConstruction* ctx) : DataBufferOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("timeout_millis", &timeout_millis_));
  }

  void ComputeWithDataBuffer(OpKernelContext* ctx, DataBuffer* buf) override {
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

REGISTER_KERNEL_BUILDER(Name("DataBufferPut").Device(DEVICE_CPU),
                        DataBufferPutOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("DataBufferPut").Device(DEVICE_GPU),
                        DataBufferPutOp);
#endif  // GOOGLE_CUDA
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("DataBufferPut").Device(DEVICE_SYCL),
                        DataBufferPutOp);
#endif  // TENSORFLOW_USE_SYCL

class DataBufferTakeOp : public DataBufferAsyncOp {
 public:
  explicit DataBufferTakeOp(OpKernelConstruction* ctx)
      : DataBufferAsyncOp(ctx) {}

  void ComputeAsyncWithDataBuffer(OpKernelContext* ctx,
                                  AsyncOpKernel::DoneCallback done,
                                  DataBuffer* buf) override {
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

REGISTER_KERNEL_BUILDER(Name("DataBufferTake").Device(DEVICE_CPU),
                        DataBufferTakeOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("DataBufferTake").Device(DEVICE_GPU),
                        DataBufferTakeOp);
#endif  // GOOGLE_CUDA
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("DataBufferTake").Device(DEVICE_SYCL),
                        DataBufferTakeOp);
#endif  // TENSORFLOW_USE_SYCL

class DataBufferCancelOp : public DataBufferOp {
 public:
  explicit DataBufferCancelOp(OpKernelConstruction* ctx) : DataBufferOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_cancelled", &is_cancelled_));
  }

  void ComputeWithDataBuffer(OpKernelContext* ctx, DataBuffer* buf) override {
    ctx->SetStatus(buf->Cancel(is_cancelled_));
  }

 private:
  bool is_cancelled_;
};

REGISTER_KERNEL_BUILDER(Name("DataBufferCancel").Device(DEVICE_CPU),
                        DataBufferCancelOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("DataBufferCancel").Device(DEVICE_GPU),
                        DataBufferCancelOp);
#endif  // GOOGLE_CUDA
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("DataBufferCancel").Device(DEVICE_SYCL),
                        DataBufferCancelOp);
#endif  // TENSORFLOW_USE_SYCL

class DataBufferCloseOp : public DataBufferOp {
 public:
  explicit DataBufferCloseOp(OpKernelConstruction* ctx) : DataBufferOp(ctx) {}

  void ComputeWithDataBuffer(OpKernelContext* ctx, DataBuffer* buf) override {
    ctx->SetStatus(buf->Close());
  }
};

REGISTER_KERNEL_BUILDER(Name("DataBufferClose").Device(DEVICE_CPU),
                        DataBufferCloseOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("DataBufferClose").Device(DEVICE_GPU),
                        DataBufferCloseOp);
#endif  // GOOGLE_CUDA
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("DataBufferClose").Device(DEVICE_SYCL),
                        DataBufferCloseOp);
#endif  // TENSORFLOW_USE_SYCL

class DataBufferSizeOp : public DataBufferOp {
 public:
  explicit DataBufferSizeOp(OpKernelConstruction* ctx) : DataBufferOp(ctx) {}

  void ComputeWithDataBuffer(OpKernelContext* ctx, DataBuffer* buf) override {
    Tensor* size = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &size));
    OP_REQUIRES_OK(ctx, buf->GetSize(size));
  }
};

REGISTER_KERNEL_BUILDER(Name("DataBufferSize").Device(DEVICE_CPU),
                        DataBufferSizeOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("DataBufferSize").HostMemory("size").Device(DEVICE_GPU),
    DataBufferSizeOp);
#endif  // GOOGLE_CUDA
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(
    Name("DataBufferSize").HostMemory("size").Device(DEVICE_SYCL),
    DataBufferSizeOp);
#endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
