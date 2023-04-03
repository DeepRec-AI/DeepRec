/* Copyright 2022 The DeepRec Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/task_runner.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

namespace {

void ParallelFor(const std::function<void(size_t)>& f, size_t n,
                 thread::ThreadPool* thread_pool) {
  if (n == 0) return;
  if (thread_pool == nullptr) {
    for (size_t i = 0; i < n; ++i) {
      f(i);
    }
  } else {
    BlockingCounter counter(n - 1);
    for (size_t i = 1; i < n; ++i) {
      thread_pool->Schedule([i, &f, &counter] {
        f(i);
        counter.DecrementCount();
      });
    }
    f(0);
    counter.Wait();
  }
}

template <typename TValue>
static void sparse_variable_grad_reduce(const TValue* grads, const int* offsets,
                                        const int batch_size,
                                        const int dimension, const int nnz,
                                        TValue* outputs) {
  for (int i = 0; i < batch_size; ++i) {
    int batch_offset = 0;
    if (i != 0) {
      batch_offset = offsets[i-1];
    }
    int feature_num = offsets[i] - batch_offset;
    for (int j = 0; j < feature_num; ++j) {
      memcpy(outputs + (batch_offset + j) * dimension, grads + i * dimension,
             sizeof(TValue) * dimension);
    }
  }
}

}  // namespace

template <typename TKey, typename TValue>
class GroupEmbeddingVarLookupGradCpuOp : public OpKernel {
 public:
  explicit GroupEmbeddingVarLookupGradCpuOp(OpKernelConstruction* c)
      : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(c, c->GetAttr("num_lookups", &num_lookups_));
    OP_REQUIRES_OK(c, c->GetAttr("dimension", &dimension_));
    OP_REQUIRES_OK(c, c->GetAttr("max_norm", &max_norm_));
    thread_pool_ = new thread::ThreadPool(Env::Default(), "GroupEmbeddingBackward", num_lookups_ / 2);
  }

  void Compute(OpKernelContext* ctx) override {
    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
    auto do_compute = [this, ctx, worker_threads](int i) {
      const Tensor grads_tensor = ctx->input(i);
      auto* grads = grads_tensor.flat<TValue>().data();
      const Tensor unique_keys_tensor = ctx->input(2 * num_lookups_ + i);
      auto* unique_keys = unique_keys_tensor.flat<TKey>().data();
      int unique_nnz = unique_keys_tensor.NumElements();
      
      const Tensor sp_indices_tensor = ctx->input(3 * num_lookups_ + i);
      auto* sp_indices = sp_indices_tensor.flat<int64>().data();
      Tensor* grads_sp_values_tensor;
      TensorShape grads_sp_values_tensor_shape =
          TensorShape(std::vector<int64>({unique_nnz, dimension_}));
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, grads_sp_values_tensor_shape,
                                               &grads_sp_values_tensor));
      auto* grads_sp_values = grads_sp_values_tensor->flat<TValue>().data();

      int slice_bytes = unique_nnz * dimension_ * 1000;
      if (combiner_ == "mean") {
        auto embedding_var_grad_combiner = [this, &grads_sp_values, sp_indices, grads](int64 start, int64 end) {
          for (int64 i = start; i < end; ++i) {
             int segment_id = sp_indices[i];
             memcpy(grads_sp_values + i * dimension_, grads + segment_id * dimension_,
                    sizeof(TValue) * dimension_);
          }
        };
        Shard(worker_threads->num_threads, worker_threads->workers,
              unique_nnz, slice_bytes /*cost*/,
              embedding_var_grad_combiner);  // Parallel on batch
      } else if (combiner_ == "sum") {
        auto embedding_var_grad_combiner = [this, &grads_sp_values, sp_indices, grads](int64 start, int64 end) {
          for (int64 i = start; i < end; ++i) {
             int segment_id = sp_indices[i];
             memcpy(grads_sp_values + i * dimension_, grads + segment_id * dimension_,
                    sizeof(TValue) * dimension_);
          }
        };
        Shard(worker_threads->num_threads, worker_threads->workers,
              unique_nnz, slice_bytes /*cost*/,
              embedding_var_grad_combiner);  // Parallel on batch
      } else {
      }
    };

    ParallelFor(do_compute, num_lookups_, thread_pool_);
  }

 private:
  thread::ThreadPool* thread_pool_{nullptr};
  std::string combiner_;
  float max_norm_;
  int num_lookups_;
  int dimension_;
};

#define REGISTER_CPU_KERNELS(key_type, value_type) \
  REGISTER_KERNEL_BUILDER(                         \
      Name("GroupEmbeddingVariableLookupGrad")     \
          .Device(DEVICE_CPU)                      \
          .TypeConstraint<key_type>("Tkeys")       \
          .TypeConstraint<value_type>("dtype"),    \
      GroupEmbeddingVarLookupGradCpuOp<key_type, value_type>)

REGISTER_CPU_KERNELS(int32, float);
REGISTER_CPU_KERNELS(int64, float);
#undef REGISTER_CPU_KERNELS

template <typename TKey, typename TValue>
class GroupVariableLookupGradCpuOp : public OpKernel {
 public:
  explicit GroupVariableLookupGradCpuOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("combiner", &combiner_));
    OP_REQUIRES_OK(c, c->GetAttr("num_lookups", &num_lookups_));
    OP_REQUIRES_OK(c, c->GetAttr("dimension", &dimension_));
    OP_REQUIRES_OK(c, c->GetAttr("max_norm", &max_norm_));
  }

  void Compute(OpKernelContext* ctx) override {
    for (int i = 0; i < num_lookups_; ++i) {
      const Tensor grads_tensor = ctx->input(i);
      auto* grads = grads_tensor.flat<TValue>().data();
      const Tensor emb_variables_tensor = ctx->input(num_lookups_ + i);
      const Tensor unique_key_tensor = ctx->input(2 * num_lookups_ + i);
      const Tensor unique_idx_tensor = ctx->input(3 * num_lookups_ + i);
      auto* unique_idx = unique_idx_tensor.flat<int>().data();
      // int batch_size = sp_values_offset_tensor.shape().dim_size(0);
      const int nnz = unique_key_tensor.NumElements();

      Tensor* grads_sp_values_tensor;
      TensorShape grads_sp_values_tensor_shape =
          TensorShape(std::vector<int64>({nnz, dimension_}));
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, grads_sp_values_tensor_shape,
                                               &grads_sp_values_tensor));
      TValue* grads_sp_values = grads_sp_values_tensor->flat<TValue>().data();
      // sparse_variable_grad_reduce(grads, sp_values_offset, batch_size,
      //                             dimension_, nnz, grads_sp_values);
    }

    // TaskRunner task_runner(do_compute, worker_threads->workers, num_lookups_);
    // task_runner.Run();
  }

 private:
  std::string combiner_;
  float max_norm_;
  int num_lookups_;
  int dimension_;
};

#define REGISTER_CPU_KERNELS(key_type, value_type)                  \
  REGISTER_KERNEL_BUILDER(Name("GroupVariableLookupGrad")           \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<key_type>("Tkeys")    \
                              .TypeConstraint<value_type>("dtype"), \
                          GroupVariableLookupGradCpuOp<key_type, value_type>)

REGISTER_CPU_KERNELS(int32, float);
REGISTER_CPU_KERNELS(int64, float);
#undef REGISTER_CPU_KERNELS

}  // namespace tensorflow
