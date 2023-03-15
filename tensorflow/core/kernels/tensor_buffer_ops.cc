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
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/stream.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

class TensorBufferOp : public OpKernel {
 public:
  explicit TensorBufferOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    auto rm = ctx->resource_manager();
    auto ndef = def();

    ContainerInfo cinfo;
    OP_REQUIRES_OK(ctx, cinfo.Init(rm, ndef, true /* use name() */));

    TensorBuf *buffer = nullptr;
    OP_REQUIRES_OK(ctx, rm->LookupOrCreate<TensorBuf>(
                            cinfo.container(), cinfo.name(), &buffer,
                            [&ndef](TensorBuf **pbuf) -> Status {
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
  virtual void ComputeWithTensorBuf(OpKernelContext *ctx, TensorBuf *buf) = 0;
};

class TensorBufferAsyncOp : public AsyncOpKernel {
 public:
  explicit TensorBufferAsyncOp(OpKernelConstruction *ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_threads", &shared_threads_));
  }

  void ComputeAsync(OpKernelContext *ctx,
                    AsyncOpKernel::DoneCallback done) override {
    auto rm = ctx->resource_manager();
    NodeDef ndef(def());
    ContainerInfo cinfo;
    OP_REQUIRES_OK_ASYNC(ctx, cinfo.Init(rm, ndef, true /* use name() */),
                         done);
    TensorBuf *buffer = nullptr;
    OP_REQUIRES_OK_ASYNC(ctx,
                         rm->LookupOrCreate<TensorBuf>(
                             cinfo.container(), cinfo.name(), &buffer,
                             [&ndef](TensorBuf **resource) {
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
  virtual void ComputeAsyncWithTensorBuf(OpKernelContext *ctx,
                                         AsyncOpKernel::DoneCallback done,
                                         TensorBuf *buffer) = 0;

 private:
  string shared_name_;
  int64 shared_threads_;

  void Schedule(TensorBuf *buffer, std::function<void()> fn) {
    buffer->Schedule(shared_name_, shared_threads_, fn);
  }
};

#ifdef GOOGLE_CUDA
class TensorBufferPutGpuOp : public TensorBufferOp {
 public:
  explicit TensorBufferPutGpuOp(OpKernelConstruction *ctx)
      : TensorBufferOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("timeout_millis", &timeout_millis_));
  }

  inline size_t AlignBytes(size_t s) {
#if EIGEN_MAX_ALIGN_BYTES == 0
    return s;
#else
    return std::ceil(s * 1.0 / EIGEN_MAX_ALIGN_BYTES) * EIGEN_MAX_ALIGN_BYTES;
#endif
  }

  void ComputeWithTensorBuf(OpKernelContext *ctx, TensorBuf *buf) override {
    std::vector<int> input_offsets;
    int total_bytes = 0;
    int input_nums = ctx->num_inputs();
    for (int i = 0; i < input_nums; ++i) {
      auto &tensor_in = ctx->input(i);
      input_offsets.push_back(total_bytes);
      total_bytes += AlignBytes(tensor_in.TotalBytes());
    }

    Tensor fused_tensor;
    // Allocate Pinned memory
    AllocatorAttributes cpu_alloc_attr;
    cpu_alloc_attr.set_on_host(true);
    cpu_alloc_attr.set_gpu_compatible(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT8, {total_bytes},
                                           &fused_tensor, cpu_alloc_attr));
    auto fused_tensor_data =
        const_cast<char *>(fused_tensor.tensor_data().data());

    auto copy_task = [this, ctx, &input_offsets, &fused_tensor_data](
                         int64 start, int64 end) {
      for (auto i = start; i < end; ++i) {
        const Tensor &input_tensor = ctx->input(i);
        size_t tensor_bytes = input_tensor.TotalBytes();
        std::copy_n(input_tensor.tensor_data().data(), tensor_bytes,
                    fused_tensor_data + input_offsets[i]);
      }
    };
    static const int cost = 1000;
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, input_nums, cost,
          copy_task);

    auto *d_context =
        static_cast<const GPUDeviceContext *>(ctx->op_device_context());
    se::Stream *copy_stream = d_context->host_to_device_stream();
    se::Stream *compute_stream = d_context->stream();

    Tensor gpu_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_INT8, {total_bytes}, &gpu_tensor));

    copy_stream->ThenWaitFor(compute_stream);
    se::DeviceMemoryBase wrapped_dst(
        const_cast<char *>(gpu_tensor.tensor_data().data()),
        gpu_tensor.TotalBytes());
    copy_stream
        ->ThenMemcpy(&wrapped_dst, const_cast<char *>(fused_tensor_data),
                     gpu_tensor.TotalBytes())
        .ok();
    compute_stream->ThenWaitFor(copy_stream);

    std::vector<Tensor> record;
    record.reserve(input_nums);
    for (int i = 0; i < input_nums; ++i) {
      size_t bytes_tensor_offset = input_offsets[i];
      Tensor tensor_slice =
          gpu_tensor.Slice(bytes_tensor_offset,
                           bytes_tensor_offset + ctx->input(i).TotalBytes());
      Tensor output(ctx->input(i).dtype());
      OP_REQUIRES_OK(ctx,
                     output.BitcastFrom(tensor_slice, ctx->input(i).dtype(),
                                        ctx->input(i).shape()));
      record.emplace_back(output);
    }
    ctx->SetStatus(buf->Put(record, timeout_millis_));
  }

 private:
  int64 timeout_millis_;
};
#endif  // GOOGLE_CUDA

class TensorBufferPutCpuOp : public TensorBufferOp {
 public:
  explicit TensorBufferPutCpuOp(OpKernelConstruction *ctx)
      : TensorBufferOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("timeout_millis", &timeout_millis_));
  }

  void ComputeWithTensorBuf(OpKernelContext *ctx, TensorBuf *buf) override {
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
                        TensorBufferPutCpuOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("TensorBufferPut").Device(DEVICE_GPU).HostMemory("record"),
    TensorBufferPutGpuOp);
#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("TensorBufferPut").Device(DEVICE_SYCL),
                        TensorBufferPutOp);
#endif  // TENSORFLOW_USE_SYCL

class TensorBufferTakeOp : public TensorBufferAsyncOp {
 public:
  explicit TensorBufferTakeOp(OpKernelConstruction *ctx)
      : TensorBufferAsyncOp(ctx) {}

  void ComputeAsyncWithTensorBuf(OpKernelContext *ctx,
                                 AsyncOpKernel::DoneCallback done,
                                 TensorBuf *buf) override {
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
  explicit TensorBufferCancelOp(OpKernelConstruction *ctx)
      : TensorBufferOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_cancelled", &is_cancelled_));
  }

  void ComputeWithTensorBuf(OpKernelContext *ctx, TensorBuf *buf) override {
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
  explicit TensorBufferCloseOp(OpKernelConstruction *ctx)
      : TensorBufferOp(ctx) {}

  void ComputeWithTensorBuf(OpKernelContext *ctx, TensorBuf *buf) override {
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
  explicit TensorBufferSizeOp(OpKernelConstruction *ctx)
      : TensorBufferOp(ctx) {}

  void ComputeWithTensorBuf(OpKernelContext *ctx, TensorBuf *buf) override {
    Tensor *size = nullptr;
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
