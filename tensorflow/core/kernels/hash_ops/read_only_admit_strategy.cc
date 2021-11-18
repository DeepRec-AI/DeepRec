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
#include "tensorflow/core/kernels/hash_ops/admit_strategy_op.h"

namespace tensorflow {

class ReadOnlyHashTableAdmitStrategy : public HashTableAdmitStrategy {
 public:
  virtual bool Admit(int64 key) {
    return false;
  }
};

class ReadOnlyHashTableAdmitStrategyOp : public HashTableAdmitStrategyOp {
 public:
  explicit ReadOnlyHashTableAdmitStrategyOp(OpKernelConstruction* context)
    : HashTableAdmitStrategyOp(context) {
  }
  virtual Status CreateStrategy(
      OpKernelContext* ctx, HashTableAdmitStrategy** strategy) {
    *strategy = new ReadOnlyHashTableAdmitStrategy;
    return Status::OK();
  }
};

REGISTER_KERNEL_BUILDER(
    Name("ReadOnlyHashTableAdmitStrategyOp").Device(DEVICE_CPU),
    ReadOnlyHashTableAdmitStrategyOp);

}  // namespace tensorflow
