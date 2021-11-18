/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

class CoalescedBucketizedEmbeddingEncodeOp : public OpKernel {
 public:
  explicit CoalescedBucketizedEmbeddingEncodeOp(OpKernelConstruction* context)
    : OpKernel(context) { }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& ids_t = ctx->input(0);
    const Tensor& local_offset_t = ctx->input(1);
    const Tensor& global_offset_t = ctx->input(2);
    OP_REQUIRES(ctx, local_offset_t.dims() == 1, errors::InvalidArgument(
            "local offset should be a tensor of rank 1, but rank=",
            local_offset_t.dims()));
    OP_REQUIRES(ctx, global_offset_t.dims() == 1, errors::InvalidArgument(
            "global offset should be a tensor of rank 1, but rank=",
            global_offset_t.dims()));
    OP_REQUIRES(
        ctx, local_offset_t.NumElements() == global_offset_t.NumElements(),
        errors::InvalidArgument("local offset size not equal to global"));

    Tensor* out_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, ids_t.shape(), &out_t));

    size_t local_size = local_offset_t.dim_size(0);
    int64* local_begin = reinterpret_cast<int64*>(const_cast<char*>(
            local_offset_t.tensor_data().data()));
    int64* local_end = local_begin + local_size;
    auto local_offset = local_offset_t.flat<int64>();
    auto global_offset = global_offset_t.flat<int64>();

    auto ids = ids_t.flat<int64>();
    auto out = out_t->flat<int64>();
    for (size_t i = 0; i < ids_t.NumElements(); ++i) {
      auto iter = std::lower_bound(local_begin, local_end, ids(i));
      int64 index = std::distance(local_begin, iter);
      if (index == local_size || *iter != ids(i)) {
        index -= 1;
      }
      out(i) = ids(i) - local_offset(index) + global_offset(index);
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("CoalescedBucketizedEmbeddingEncode").Device(DEVICE_CPU),
    CoalescedBucketizedEmbeddingEncodeOp);

}  // namespace tensorflow
