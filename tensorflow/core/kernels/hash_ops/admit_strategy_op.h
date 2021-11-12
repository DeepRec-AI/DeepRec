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
#include "tensorflow/core/framework/hash_table/status_collector.h"
#include "tensorflow/core/framework/hash_table/hash_table.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"

#ifndef TENSORFLOW_CORE_KERNELS_HASH_OPS_ADMIT_STRATEGY_OP_H_
#define TENSORFLOW_CORE_KERNELS_HASH_OPS_ADMIT_STRATEGY_OP_H_

namespace tensorflow {

class HashTableAdmitStrategyOp : public OpKernel {
 public:
  explicit HashTableAdmitStrategyOp(OpKernelConstruction* context)
    : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("container", &container_));
    OP_REQUIRES_OK(context, context->GetAttr("shared_name", &name_));
  }

  virtual Status CreateStrategy(
      OpKernelContext* ctx, HashTableAdmitStrategy** strategy) = 0;

  void Compute(OpKernelContext* ctx) override {
    if (!initialized_.load()) {
      mutex_lock ml(mutex_);
      // Checking again to see if another thread has initialized the resource.
      if (!initialized_.load()) {
        AllocatorAttributes attr;
        attr.set_on_host(true);
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}),
                                               &resource_, attr));
        resource_.scalar<ResourceHandle>()() =
            MakeResourceHandle<HashTableAdmitStrategyResource>(
                ctx, container_, name_);
        initialized_.store(true);
      }
    }
    HashTableAdmitStrategyResource* resource;
    OP_REQUIRES_OK(
        ctx,
        LookupOrCreateResource<HashTableAdmitStrategyResource>(
            ctx, resource_.scalar<ResourceHandle>()(), &resource,
            [this, ctx](HashTableAdmitStrategyResource** ptr) {
              HashTableAdmitStrategy* strategy;
              TF_RETURN_IF_ERROR(CreateStrategy(ctx, &strategy));
              *ptr = new HashTableAdmitStrategyResource(strategy);
              return Status::OK();
            }));
    resource->Unref();
    ctx->set_output(0, resource_);
  }

 private:
  string container_;
  string name_;
  mutex mutex_;
  Tensor resource_;
  std::atomic<bool> initialized_{false};
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_HASH_OPS_ADMIT_STRATEGY_OP_H_
