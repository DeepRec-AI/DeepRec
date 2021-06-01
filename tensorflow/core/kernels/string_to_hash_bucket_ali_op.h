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

#ifndef TENSORFLOW_CORE_KERNELS_STRING_TO_HASH_BUCKET_ALI_OP_H_
#define TENSORFLOW_CORE_KERNELS_STRING_TO_HASH_BUCKET_ALI_OP_H_

#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

template <uint64 hash(StringPiece)>
class StringToHashBucketAliOp : public OpKernel {
 public:
  explicit StringToHashBucketAliOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_buckets", &num_buckets_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<string>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output", input_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<int64>();

    auto RunTask = [this, &input_flat, &output_flat](int64 start, int64 end) {
      typedef decltype(input_flat.size()) Index;
      for (Index i = start; i < end; ++i) {
        const uint64 input_hash = hash(input_flat(i));
        const uint64 bucket_id = input_hash % num_buckets_;
        // The number of buckets is always in the positive range of int64 so is
        // the resulting bucket_id. Casting the bucket_id from uint64 to int64
        // is safe.
        output_flat(i) = static_cast<int64>(bucket_id);
      }
    };

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    const int64 element_cost = 100;  // Estimated for 32 byte strings.
    // NOTE(zycao): Here we have to use 'num_threads - 1' to make sure no more
    // task fractions should be created. The cost is also a coarse estimation.
    Shard(worker_threads->num_threads - 1, worker_threads->workers,
          input_flat.size(), element_cost, RunTask);
  }

 private:
  int64 num_buckets_;

  TF_DISALLOW_COPY_AND_ASSIGN(StringToHashBucketAliOp);
};

template <uint64 hash(StringPiece)>
class StringToHashBucketBatchAliOp : public OpKernel {
 public:
  explicit StringToHashBucketBatchAliOp(OpKernelConstruction* ctx) : 
  OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_buckets", &num_buckets_));
  }

  void Compute(OpKernelContext* context) override {
    VLOG(2) << "StringToHashBucketBatchAliOp executed";
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<string>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output", input_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<int64>();

    auto RunTask = [this, &input_flat, &output_flat](int64 start, int64 end) {
      typedef decltype(input_flat.size()) Index;
      Index batch_end = end - (end - start)%8;
      Index i = start;
#if defined(__AVX512F__)
      const char* batch_ptr[8]; 
      uint64_t input_hash[8];
      bool enable_batch_hash = true;
      if (batch_end - start >= 8) {
        for(; i < batch_end; i+=8) {
          // first unrolling by 8 (for Hash64V3_Batch512)
          // double check whether all the 8 strings within 
          // a batch having the same string length.
          enable_batch_hash = true;
          Index size_0 = input_flat(i).size();
          batch_ptr[0] = input_flat(i).data();
          for(int j=1; j<8; j++) {
            if (input_flat(i+j).size() == size_0) {
              batch_ptr[j] = input_flat(i+j).data();
            } else {
              enable_batch_hash = false;
              break;
            }
          }
          if (enable_batch_hash) {
            Hash64Farm_Batch512(batch_ptr, &input_hash[0], size_0);
          } else {
            // roll back to normal Hash64 function
            for(int j=0; j<8; j++) {
                input_hash[j] = (uint64_t)hash(input_flat(i+j)); 
            }
          }
          // feed ids to output tensor
          for(int j=0; j<8; j++) {
            output_flat(i+j) = static_cast<int64>(input_hash[j]%num_buckets_);
          }
        }
      } 
#endif
      // for remained iterations
      for(; i < end; ++i) {
        const uint64 input_hash = hash(input_flat(i));
        const uint64 bucket_id = input_hash % num_buckets_;
        // The number of buckets is always in the positive range of int64 so is
        // the resulting bucket_id. Casting the bucket_id from uint64 to int64
        // is safe.
        output_flat(i) = static_cast<int64>(bucket_id);
      }
    };

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
#if defined (__AVX512F__)
    const int64 element_cost = 25;  // for AVX512 batch-vectorized impl.
#else
    const int64 element_cost = 100;  // Estimated for 32 byte strings.
#endif
    // NOTE(zycao): Here we have to use 'num_threads - 1' to make sure no more
    // task fractions should be created. The cost is also a coarse estimation.
    Shard(worker_threads->num_threads - 1, worker_threads->workers,
          input_flat.size(), element_cost, RunTask);
  }

 private:
  int64 num_buckets_;

  TF_DISALLOW_COPY_AND_ASSIGN(StringToHashBucketBatchAliOp);
};

template <uint64 hash(const uint64 (&)[2], const string&)>
class StringToKeyedHashBucketAliOp : public OpKernel {
 public:
  explicit StringToKeyedHashBucketAliOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_buckets", &num_buckets_));

    std::vector<int64> key;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key", &key));
    OP_REQUIRES(ctx, key.size() == 2,
                errors::InvalidArgument("Key must have 2 elements"));
    std::memcpy(key_, key.data(), sizeof(key_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<string>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output", input_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<int64>();

    auto RunTask = [this, &input_flat, &output_flat](int64 start, int64 end) {
      typedef decltype(input_flat.size()) Index;
      for (Index i = start; i < end; ++i) {
        const uint64 input_hash = hash(key_, input_flat(i));
        const uint64 bucket_id = input_hash % num_buckets_;
        // The number of buckets is always in the positive range of int64 so is
        // the resulting bucket_id. Casting the bucket_id from uint64 to int64
        // is safe.
        output_flat(i) = static_cast<int64>(bucket_id);
      }
    };

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    const int64 element_cost = 100;  // Estimated for 32 byte strings.
    // NOTE(zycao): Here we have to use 'num_threads - 1' to make sure no more
    // task fractions should be created. The cost is also a coarse estimation.
    Shard(worker_threads->num_threads - 1, worker_threads->workers,
          input_flat.size(), element_cost, RunTask);
  }

 private:
  int64 num_buckets_;
  uint64 key_[2];

  TF_DISALLOW_COPY_AND_ASSIGN(StringToKeyedHashBucketAliOp);
};

template <uint64 hash(StringPiece)>
class StringToHash64Op : public OpKernel {
 public:
  explicit StringToHash64Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<string>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output", input_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<int64>();

    auto RunTask = [this, &input_flat, &output_flat](int64 start, int64 end) {
      typedef decltype(input_flat.size()) Index;
      auto batch_end = end - (end - start)%8;
      auto i = start;
#if defined(__AVX512F__)
      const char* batch_ptr[8];
      uint64_t input_hash[8];
      bool enable_batch_hash = true;
      if (batch_end - start >= 8) {
        for(; i < batch_end; i+=8) {
          // first unrolling by 8 (for Hash64V3_Batch512)
          // double check whether all the 8 strings within
          // a batch having the same string length.
          enable_batch_hash = true;
          Index size_0 = input_flat(i).size();
          batch_ptr[0] = input_flat(i).data();
          for(int j=1; j<8; j++) {
            if (input_flat(i+j).size() == size_0) {
              batch_ptr[j] = input_flat(i+j).data();
            } else {
              enable_batch_hash = false;
              break;
            }
          }
          if (enable_batch_hash) {
            Hash64Farm_Batch512(batch_ptr, &input_hash[0], size_0);
          } else {
            // roll back to normal Hash64 function
            for(int j=0; j<8; j++) {
                input_hash[j] = (uint64_t)hash(input_flat(i+j));
            }
          }
          // feed ids to output tensor
          for(int j=0; j<8; j++) {
            output_flat(i+j) = input_hash[j] & kint64max;
          }
        }
      }
#endif
      // for remained iterations
      for(; i < end; ++i) {
        const uint64 input_hash = hash(input_flat(i));
        // The number of buckets is always in the positive range of int64 so is
        // the resulting bucket_id. Casting the bucket_id from uint64 to int64
        // is safe.
        output_flat(i) = input_hash & kint64max;
      }
    };

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
#if defined (__AVX512F__)
    const int64 element_cost = 25;  // for AVX512 batch-vectorized impl.
#else
    const int64 element_cost = 100;  // Estimated for 32 byte strings.
#endif

    // NOTE(zycao): Here we have to use 'num_threads - 1' to make sure no more
    // task fractions should be created. The cost is also a coarse estimation.
    Shard(worker_threads->num_threads - 1, worker_threads->workers,
          input_flat.size(), element_cost, RunTask);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(StringToHash64Op);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_STRING_TO_HASH_BUCKET_ALI_OP_H_
