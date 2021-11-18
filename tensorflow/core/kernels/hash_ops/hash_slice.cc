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
#include "tensorflow/core/framework/hash_table/status_collector.h"

namespace tensorflow {

static constexpr int kIdBlockSize = 4096;

class HashSlice : public AsyncOpKernel {
 public:
  explicit HashSlice(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("segment_size", &segment_size_));
    OP_REQUIRES(context, segment_size_ > 0,
                errors::InvalidArgument("Hash Slice segment size should more than 0"));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    Tensor slicer = ctx->input(0);
    Tensor keys = ctx->input(1);
    OP_REQUIRES_ASYNC(
        ctx, slicer.shape().dims() == 1,
        errors::InvalidArgument("Slicer should be rank-1"), done);
    int32 slicer_size = slicer.shape().dim_size(0);
    int32* slicer_beg = &slicer.flat<int32>()(0);
    int32* slicer_end = slicer_beg + slicer_size;
    OP_REQUIRES_ASYNC(
        ctx, slicer_size > 0,
        errors::InvalidArgument("Slicer size should more than 0"), done);
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, keys.shape(), &output), done);
    int64* pkeys = &keys.flat<int64>()(0);
    int32* pslice = &output->flat<int32>()(0);
    int64_t segment_size = segment_size_;
    auto fn = [slicer_beg, slicer_end, segment_size, pkeys, pslice]
    (int64_t offset, int64_t size) -> Status {
      int64* src = pkeys + offset;
      int32* dst = pslice + offset;
      for (int64_t i = 0; i < size; i++) {
        *dst = std::upper_bound(slicer_beg, slicer_end, (uint64_t)(*src) % segment_size) - slicer_beg - 1;
        dst++;
        src++;
      }
      return Status::OK();
    };
    auto done_fn = [done, ctx](Status st) {
      OP_REQUIRES_OK_ASYNC(ctx, st, done);
      done();
    };
    ParrellRun(keys.NumElements(), kIdBlockSize, *ctx->runner(), fn, done_fn);
  }

 private:
  int64 segment_size_;
};

REGISTER_KERNEL_BUILDER(
    Name("HashSlice").Device(DEVICE_CPU), HashSlice);

}  // namespace tensorflow

