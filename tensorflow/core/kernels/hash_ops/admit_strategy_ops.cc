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

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/hash_table/bloom_filter_strategy.h"
#include "tensorflow/core/framework/hash_table/status_collector.h"
#include "tensorflow/core/framework/hash_table/hash_table.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/hash_ops/admit_strategy_op.h"

namespace tensorflow {

class BloomFilterAdmitStrategyOp : public HashTableAdmitStrategyOp {
 public:
  explicit BloomFilterAdmitStrategyOp(OpKernelConstruction* context)
    : HashTableAdmitStrategyOp(context) {
  }
  virtual Status CreateStrategy(
      OpKernelContext* ctx, HashTableAdmitStrategy** strategy) {
    *strategy = nullptr;
    return Status::OK();
  }
};

class BloomFilterInitializeOp : public OpKernel {
 public:
  explicit BloomFilterInitializeOp(OpKernelConstruction* context)
    : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("min_frequency", &min_frequency_));
    OP_REQUIRES_OK(context, context->GetAttr("num_hash_func", &num_hash_func_));
    OP_REQUIRES_OK(context, context->GetAttr("slice_offset", &slice_offset_));
    OP_REQUIRES_OK(context, context->GetAttr("max_slice_size", &max_slice_size_));
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
    OP_REQUIRES_OK(context, context->GetAttr("initialized", &initialized_));
    CHECK(shape_.dims() == 2) << "Invalid shape, must be 2-dimensional";
  }
  void Compute(OpKernelContext* ctx) override {
    HashTableAdmitStrategyResource* resource;
    OP_REQUIRES_OK(
        ctx,
        LookupOrCreateResource<HashTableAdmitStrategyResource>(
          ctx, HandleFromInput(ctx, 0), &resource,
          [this](HashTableAdmitStrategyResource** ptr) {
            *ptr = new HashTableAdmitStrategyResource;
            return Status::OK();
          }));
    core::ScopedUnref s(resource);
    resource->CreateInternal(
        new BloomFilterAdmitStrategy(
          min_frequency_, num_hash_func_, dtype_, shape_, 
          slice_offset_, max_slice_size_));
    resource->SetInitialized(initialized_);
  }
 private:
  int64 min_frequency_;
  int64 num_hash_func_;
  int64 slice_offset_;
  int64 max_slice_size_;
  DataType dtype_;
  TensorShape shape_;
  bool initialized_;
};

class BloomFilterIsInitializedOp : public OpKernel {
 public:
  explicit BloomFilterIsInitializedOp(OpKernelConstruction* context)
    : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    auto output_tensor = output->tensor<bool, 0>();
    HashTableAdmitStrategyResource* resource;
    Status s = LookupResource(ctx, HandleFromInput(ctx, 0), &resource);
    if (!s.ok()) {
      output_tensor() = false;
      return;
    }
    core::ScopedUnref su(resource);
    output_tensor() = resource->Initialized();
  }
};

class BloomFilterAdmitOp : public AsyncOpKernel {
 public:
  explicit BloomFilterAdmitOp(OpKernelConstruction* context)
    : AsyncOpKernel(context) {
  }
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    HashTableAdmitStrategyResource* resource;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &resource), done);
    core::ScopedUnref s(resource);
    HashTableAdmitStrategy* strategy = resource->Internal();
    CHECK(strategy != nullptr) 
        << "BloomFilterAdmitStrategy not initialized.";
    Tensor keys_tensor = ctx->input(1);
    Tensor freqs_tensor = ctx->input(2);
    CHECK(keys_tensor.NumElements() == freqs_tensor.NumElements())
        << "keys size not match to frequency size";
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(0, freqs_tensor.shape(), &output_tensor), done);
    auto keys = keys_tensor.flat<int64>();
    auto output = output_tensor->flat<bool>();
    for (size_t i = 0; i < freqs_tensor.NumElements(); ++i) {
      switch(freqs_tensor.dtype()) {
        case DT_UINT8:
          output(i) = strategy->Admit(keys(i), static_cast<int64>(
                  freqs_tensor.flat<uint8>()(i)));
          break;
        case DT_UINT16:
          output(i) = strategy->Admit(keys(i), static_cast<int64>(
                  freqs_tensor.flat<uint16>()(i)));
          break;
        case DT_UINT32:
          output(i) = strategy->Admit(keys(i), static_cast<int64>(
                  freqs_tensor.flat<uint32>()(i)));
          break;
        default:
          LOG(FATAL) << "Unknown data type " << freqs_tensor.dtype();
      }
    }
    done();
  }
};

REGISTER_KERNEL_BUILDER(Name("BloomFilterAdmitStrategyOp")                    \
                            .Device(DEVICE_CPU),                              \
                        BloomFilterAdmitStrategyOp);
REGISTER_KERNEL_BUILDER(Name("BloomFilterInitializeOp")                       \
                            .Device(DEVICE_CPU),                              \
                        BloomFilterInitializeOp);
REGISTER_KERNEL_BUILDER(Name("BloomFilterIsInitializedOp")                    \
                            .Device(DEVICE_CPU),                              \
                        BloomFilterIsInitializedOp);
REGISTER_KERNEL_BUILDER(Name("BloomFilterAdmitOp")                            \
                            .Device(DEVICE_CPU),                              \
                        BloomFilterAdmitOp);

}  // namespace tensorflow
