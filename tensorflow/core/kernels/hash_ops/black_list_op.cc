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
#include "tensorflow/core/lib/io/inputbuffer.h"

#include <unordered_set>

namespace tensorflow {

class BlackListHashTableAdmitStrategy : public HashTableAdmitStrategy {
 public:
  virtual bool Admit(int64 key) {
    return set_.find(key) == set_.end();
  }
  std::unordered_set<int64>* Internal() {
    return &set_;
  }
 private:
  std::unordered_set<int64> set_;
};

class BlackListHashTableAdmitStrategyOp : public HashTableAdmitStrategyOp {
 public:
  explicit BlackListHashTableAdmitStrategyOp(OpKernelConstruction* context)
    : HashTableAdmitStrategyOp(context) {
  }
  virtual Status CreateStrategy(
      OpKernelContext* ctx, HashTableAdmitStrategy** strategy) {
    *strategy = new BlackListHashTableAdmitStrategy;
    return Status::OK();
  }
};

class InitBlackList : public OpKernel {
 private:
  struct BlackListType {
    std::string fea_name;
    int64 slice_beg, slice_end;
    double threshold;
    std::unordered_set<int64>* set;
  };
 public:
  explicit InitBlackList(OpKernelConstruction* context)
    : OpKernel(context) {
  }
  void Compute(OpKernelContext* ctx) override {
    Tensor strategies_tensor = ctx->input(0);
    Tensor fea_names_tensor = ctx->input(1);
    Tensor slices_tensor = ctx->input(2);
    Tensor thresholds_tensor = ctx->input(3);
    Tensor files_tensor = ctx->input(4);
    std::unordered_map<std::string, BlackListType> types;
    auto strategies = strategies_tensor.flat<ResourceHandle>();
    auto fea_names = fea_names_tensor.flat<string>();
    auto slices = slices_tensor.matrix<int64>();
    auto files = files_tensor.flat<string>();
    OP_REQUIRES(ctx, strategies.size() == fea_names.size(),
        errors::InvalidArgument("FEA NAME SIZE ERROR"));
    OP_REQUIRES(ctx, strategies.size() * 2 == slices.size(),
        errors::InvalidArgument("SLICES SIZE ERROR"));
    for (int i = 0; i < strategies.size(); i++) {
      HashTableAdmitStrategyResource* s;
      OP_REQUIRES_OK(ctx, LookupResource<HashTableAdmitStrategyResource>(
            ctx, strategies(i), &s));
      BlackListHashTableAdmitStrategy* ss = dynamic_cast<BlackListHashTableAdmitStrategy*>(s->Internal());
      OP_REQUIRES(ctx, ss != nullptr,
          errors::InvalidArgument("NOT BLACK LIST"));
      auto t = types[fea_names(i)];
      t.fea_name = fea_names(i);
      t.slice_beg = slices(i, 0);
      t.slice_end = slices(i, 1);
      t.set = ss->Internal();
    }
    for (int i = 0; i < files.size(); i++) {
      std::unique_ptr<RandomAccessFile> file;
      OP_REQUIRES_OK(ctx, ctx->env()->NewRandomAccessFile(files(i), &file));
      io::InputBuffer stream(file.get(), 64 << 20);
      auto r = [&] ()->Status {
        while (true) {
          int str_len;
          size_t bytes_to_read;
          int64 id;
          double d;
          std::string token;
          TF_RETURN_IF_ERROR(
              stream.ReadNBytes(4, (char*)(void*)&str_len, &bytes_to_read));
          TF_RETURN_IF_ERROR(stream.ReadNBytes(str_len, &token));
          TF_RETURN_IF_ERROR(
              stream.ReadNBytes(8, (char*)(void*)&id, &bytes_to_read));
          TF_RETURN_IF_ERROR(
              stream.ReadNBytes(8, (char*)(void*)&d, &bytes_to_read));
          auto iter = types.find(token);
          if (iter == types.end()) {
            continue;
          }
          if (d < iter->second.threshold &&
              (id & 0xFF) >= iter->second.slice_beg &&
              (id & 0xFF) < iter->second.slice_end) {
            iter->second.set->insert(id);
          }
        }
      };
      Status st = r();
      if (!errors::IsOutOfRange(st)) {
        OP_REQUIRES_OK(ctx, st);
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("BlackListHashTableAdmitStrategyOp").Device(DEVICE_CPU),
    BlackListHashTableAdmitStrategyOp);

REGISTER_KERNEL_BUILDER(
    Name("InitBlackList").Device(DEVICE_CPU),
    InitBlackList);

}  // namespace tensorflow

