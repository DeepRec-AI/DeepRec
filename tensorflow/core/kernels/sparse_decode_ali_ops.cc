/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <iostream>
#include <fstream>
#include <sys/time.h>

#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

class SparseDecodeOp : public OpKernel {
 public:
  explicit SparseDecodeOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
    }
  ~SparseDecodeOp() {
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    const Tensor* max_id_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("max_id", &max_id_tensor));
    int64 max_id = max_id_tensor->scalar<int64>()();
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<string>();
    const int64 batch_size = input_vec.dimension(0);

    int64 tensorLen = 0;
    int offset_[batch_size + 1];
    offset_[0] = 0;
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        Int64List int64_list;
        int64_list.ParseFromString(input_vec(batch_id));
        tensorLen += int64_list.value_size();
        offset_[batch_id + 1] = tensorLen;
    }

    Tensor indices(DT_INT64, TensorShape({tensorLen, 2}));
    Tensor values(DT_INT64, TensorShape({tensorLen}));
    Tensor dense_shape(DT_INT64, TensorShape({2}));
    auto indices_ = indices.matrix<int64>();
    auto values_ = values.vec<int64>();
    auto dense_shape_ = dense_shape.flat<int64>();

    auto doWork = [input_vec, &offset_, &indices_, &values_] (int64 start_i,
                                                              int64 limit_i) {
        for (int batch_id = start_i; batch_id < limit_i; ++batch_id) {
            Int64List int64_list;
            int64_list.ParseFromString(input_vec(batch_id));
            int64 col_len = int64_list.value_size();
            const int64 offset = offset_[batch_id];
            for(int index = 0; index < col_len; ++index) {
                const int64& value = int64_list.value(index);
                indices_(index + offset, 0) = batch_id;
                indices_(index + offset, 1) = value;
                values_(index + offset) = value;
            }
        }
    };
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    const int64 cost = 5000; //very unreliable estimate for cost per step.
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost,
          doWork);

    dense_shape_(0) = batch_size;
    dense_shape_(1) = max_id;
    ctx->set_output(0, indices);
    ctx->set_output(1, values);
    ctx->set_output(2, dense_shape);
  }
};

class KV2SparseDecodeOp : public OpKernel {
 public:
  explicit KV2SparseDecodeOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
    }
  ~KV2SparseDecodeOp() {
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    const Tensor* max_id_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("max_id", &max_id_tensor));
    int64 max_id = max_id_tensor->scalar<int64>()();
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<string>();
    const int64 batch_size = input_vec.dimension(0);

    int64 tensorLen = 0;
    int offset_[batch_size + 1];
    offset_[0] = 0;
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        KvList kv_list;
        kv_list.ParseFromString(input_vec(batch_id));
        tensorLen += kv_list.value_size();
        offset_[batch_id + 1] = tensorLen;
    }

    Tensor indices(DT_INT64, TensorShape({tensorLen, 2}));
    Tensor values(DT_FLOAT, TensorShape({tensorLen}));
    Tensor dense_shape(DT_INT64, TensorShape({2}));
    auto indices_ = indices.matrix<int64>();
    auto values_ = values.vec<float>();
    auto dense_shape_ = dense_shape.flat<int64>();

    auto doWork = [input_vec, &offset_, &indices_, &values_] (int64 start_i,
                                                              int64 limit_i) {
        for (int batch_id = start_i; batch_id < limit_i; ++batch_id) {
            KvList kv_list;
            kv_list.ParseFromString(input_vec(batch_id));
            int64 col_len = kv_list.value_size();
            const int64 offset = offset_[batch_id];
            for(int index = 0; index < col_len; ++index) {
                indices_(index + offset, 0) = batch_id;
                indices_(index + offset, 1) = kv_list.id(index);
                values_(index + offset) = kv_list.value(index);
            }
        }
    };
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    const int64 cost = 5000; //very unreliable estimate for cost per step.
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost,
          doWork);

    dense_shape_(0) = batch_size;
    dense_shape_(1) = max_id;
    ctx->set_output(0, indices);
    ctx->set_output(1, values);
    ctx->set_output(2, dense_shape);
  }
};

class DenseDecodeFloatOp : public OpKernel {
 public:
  explicit DenseDecodeFloatOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
    }
  ~DenseDecodeFloatOp() {
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    //const Tensor* max_id_tensor;
    //OP_REQUIRES_OK(ctx, ctx->input("max_id", &max_id_tensor));
    //int64 max_id = max_id_tensor->scalar<int64>()();
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<string>();
    const int64 batch_size = input_vec.dimension(0);

    FloatList float_list;
    float_list.ParseFromString(input_vec(0));
    int64 max_id = float_list.value_size();

    Tensor values(DT_FLOAT, TensorShape({max_id * batch_size}));
    auto values_ = values.vec<float>();

    auto doWork = [input_vec, max_id, &values_] (int64 start_i, int64 limit_i) {
        for (int batch_id = start_i; batch_id < limit_i; ++batch_id) {
            FloatList float_list;
            float_list.ParseFromString(input_vec(batch_id));
            const int64 offset = batch_id * max_id;
            for(int index = 0; index < max_id; ++index) {
                values_(index + offset) = float_list.value(index);;
            }
        }
    };
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    const int64 cost = 5000; //very unreliable estimate for cost per step.
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost,
          doWork);

    ctx->set_output(0, values);
  }
};

class DenseDecodeInt32Op : public OpKernel {
 public:
  explicit DenseDecodeInt32Op(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
    }
  ~DenseDecodeInt32Op() {
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    //const Tensor* max_id_tensor;
    //OP_REQUIRES_OK(ctx, ctx->input("max_id", &max_id_tensor));
    //int64 max_id = max_id_tensor->scalar<int64>()();
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<string>();
    const int64 batch_size = input_vec.dimension(0);

    Int32List int32_list;
    int32_list.ParseFromString(input_vec(0));
    int64 max_id = int32_list.value_size();

    Tensor values(DT_INT32, TensorShape({max_id * batch_size}));
    auto values_ = values.vec<int32>();

    auto doWork = [input_vec, max_id, &values_] (int64 start_i, int64 limit_i) {
        for (int batch_id = start_i; batch_id < limit_i; ++batch_id) {
            Int32List int32_list;
            int32_list.ParseFromString(input_vec(batch_id));
            const int64 offset = batch_id * max_id;
            for(int index = 0; index < max_id; ++index) {
                values_(index + offset) = int32_list.value(index);;
            }
        }
    };
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    const int64 cost = 5000; //very unreliable estimate for cost per step.
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost,
          doWork);

    ctx->set_output(0, values);
  }
};

class KV2DenseDecodeOp : public OpKernel {
 public:
  explicit KV2DenseDecodeOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
    }
  ~KV2DenseDecodeOp() {
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    const Tensor* max_id_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("max_id", &max_id_tensor));
    int64 max_id = max_id_tensor->scalar<int64>()();
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<string>();
    const int64 batch_size = input_vec.dimension(0);

    Tensor* values;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({batch_size * max_id}),
                   &values));
    auto values_ = values->vec<float>();

    auto doWork = [input_vec, max_id, &values_] (int64 start_i, int64 limit_i) {
        for (int batch_id = start_i; batch_id < limit_i; ++batch_id) {
            KvList kv_list;
            kv_list.ParseFromString(input_vec(batch_id));
            int64 col_len = kv_list.value_size();
            const int64 offset = batch_id * max_id;
            for (int index = 0; index < max_id; ++index) {
                values_(index + offset) = 0.0;
            }
            for (int index = 0; index < col_len; ++index) {
                const int64& key = kv_list.id(index);
                if (key < max_id) {
                    values_(key + offset) = kv_list.value(index);
                } else {
                    LOG(WARNING) << "KV's key(" << key << ") is larger than max_col("
                                 << max_id << ").";
                }
            }
        }
    };
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    const int64 cost = 5000; //very unreliable estimate for cost per step.
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost, doWork);
  }
};

REGISTER_KERNEL_BUILDER(Name("SparseDecode")
                            .Device(DEVICE_CPU),
                        SparseDecodeOp);
REGISTER_KERNEL_BUILDER(Name("DenseDecode")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        DenseDecodeFloatOp);
REGISTER_KERNEL_BUILDER(Name("DenseDecode")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("T"),
                        DenseDecodeInt32Op);
REGISTER_KERNEL_BUILDER(Name("KV2SparseDecode")
                            .Device(DEVICE_CPU),
                        KV2SparseDecodeOp);
REGISTER_KERNEL_BUILDER(Name("KV2DenseDecode")
                            .Device(DEVICE_CPU),
                        KV2DenseDecodeOp);

}  // namespace tensorflow
