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

#ifndef TENSORFLOW_CORE_KERNELS_TRAINING_ALI_OP_HELPERS_H_
#define TENSORFLOW_CORE_KERNELS_TRAINING_ALI_OP_HELPERS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {

// **********************************************************************
// TODO: candy.dc
// this code is duplicated from training_op_helpers.h
// Once this function and Class support template, this duplicated code
// should be removed
// **********************************************************************

// Returns a borrowed pointer to the mutex for the variable `input` in `ctx`.
//
// If `input` corresponds to a `DT_RESOURCE`-type variable input,
// `*maybe_resource` will be updated to contain the underlying resource, and the
// caller will be responsible for calling `Unref()` on that resource.
template<typename K, typename V>
mutex* GetTrainingEmbeddingVariableMutex(OpKernelContext* ctx, int input,
                                         EmbeddingVar<K, V>** maybe_resource) {
  *maybe_resource = nullptr;
  if (ctx->input_dtype(input) == DT_RESOURCE) {
    if (LookupResource(ctx, HandleFromInput(ctx, input), maybe_resource).ok()) {
      return (*maybe_resource)->mu();
    } else {
      ctx->CtxFailureWithWarning(
          errors::Internal("Invalid variable reference."));
      return nullptr;
    }
  }
  return ctx->input_ref_mutex(input);
}

// Utility structure that releases a sequence of borrowed mutexes when it is
// deleted.
template<typename K, typename V>
struct EmbeddingVariableInputLockHolder {
 public:
  EmbeddingVariableInputLockHolder(std::vector<EmbeddingVar<K, V>*> vars,
                          std::unique_ptr<std::vector<mutex_lock>> locks)
      : vars_(std::move(vars)), locks_(std::move(locks)) {}

  EmbeddingVariableInputLockHolder(EmbeddingVariableInputLockHolder&& other)
      : vars_(std::move(other.vars_)), locks_(std::move(other.locks_)) {}

  ~EmbeddingVariableInputLockHolder() {
    // Release the locks before unreffing the Vars, because each lock
    // is potentially borrowed from a Var in vars_.
    locks_.reset();
    for (EmbeddingVar<K, V>* var : vars_) {
      var->Unref();
    }
  }

 private:
  std::vector<EmbeddingVar<K, V>*> vars_;
  // NOTE: Use a `std::unique_ptr` instead of moving in a vector directly,
  // because a `std::vector<mutex_lock>` is not movable on all platforms.
  std::unique_ptr<std::vector<mutex_lock>> locks_;
};

template<typename K, typename V>
EmbeddingVariableInputLockHolder<K, V> MaybeLockEmbeddingVariableInputMutexesInOrder(
    OpKernelContext* ctx, bool do_lock, const std::vector<int>& input_ids) {
  if (!do_lock) {
    return EmbeddingVariableInputLockHolder<K, V>({}, {});
  }
  std::vector<EmbeddingVar<K, V>*> vars;
  std::vector<mutex*> mutexes;
  std::vector<int> acquire_order;
  for (auto input : input_ids) {
    EmbeddingVar<K, V>* var;
    mutex* mutex = GetTrainingEmbeddingVariableMutex(ctx, input, &var);
    if (var) vars.push_back(var);
    // Only lock each mutex once if duplicates exist (n^2 but n is 2 or 3).
    if (std::find(mutexes.begin(), mutexes.end(), mutex) == mutexes.end()) {
      acquire_order.push_back(mutexes.size());
      mutexes.push_back(mutex);
    }
  }
  std::sort(acquire_order.begin(), acquire_order.end(),
            [&mutexes](int a, int b) { return mutexes[a] < mutexes[b]; });

  std::unique_ptr<std::vector<mutex_lock>> locks =
      MakeUnique<std::vector<mutex_lock>>();
  locks->reserve(acquire_order.size());

  for (auto input : acquire_order) {
    EmbeddingVar<K, V>* var;
    mutex* mu = GetTrainingEmbeddingVariableMutex(ctx, input, &var);
    core::ScopedUnref scoped_unref(var);
    if (mu != nullptr) {
      locks->emplace_back(*mu);
    }
  }
  return EmbeddingVariableInputLockHolder<K, V>(std::move(vars), std::move(locks));
}

template<class K, class V, class Tstep>
void LookupKeyAndSetVersion(
    OpKernelContext* ctx, EmbeddingVar<K, V>* var,
    void** value_ptrs, Tstep gs, const K* indices,
    int64 task_size, bool indices_as_pointer,
    int counts_index) {
  EmbeddingVarContext<Eigen::GpuDevice> ev_ctx(ctx);
  int64* indices_counts = nullptr;
  std::function<int64(int64*, int64)> get_count_fn = 0;
  if (counts_index != -1) {
    const Tensor& counts_tensor = ctx->input(counts_index);
    indices_counts = (int64*)counts_tensor.data();
  }
  var->LookupOrCreateKey(ev_ctx, indices, value_ptrs,
                         task_size, indices_counts,
                         indices_as_pointer);

  auto update_version_fn = [var, value_ptrs, gs]
      (int64 start, int64 limit) {
    for (int i = start; i < limit; i++) {
      var->UpdateVersion(value_ptrs[i], gs);
    }
  };
  const int64 unit_cost = 1000; //very unreliable estimate for cost per step.
  auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
  Shard(worker_threads->num_threads,
        worker_threads->workers, task_size, unit_cost,
        update_version_fn);
}

template<class K, class V>
void LookupEmbedding(
    OpKernelContext* ctx,
    std::vector<std::pair<EmbeddingVar<K, V>*, V**>>& vars,
    void** value_ptrs,
    const K* indices,
    int64 num_of_keys) {
  for (auto it: vars) {
    EmbeddingVar<K, V>* var = it.first;
    V** var_ptr = it.second;
    auto lookup_emb_fn = [var, var_ptr, value_ptrs]
        (int64 start, int64 limit) {
      for (int i = start; i < limit; i++) {
        var_ptr[i] = var->GetValuePtr(value_ptrs[i]);
      }
    };
    const int64 unit_cost = 1000; //very unreliable estimate for cost per step.
    auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads->num_threads,
        worker_threads->workers, num_of_keys, unit_cost,
        lookup_emb_fn);
  }
}

template<class K, class V, class Tstep>
void GetEmbeddingPointers(
    OpKernelContext* ctx,
    std::vector<std::pair<EmbeddingVar<K, V>*, V**>>& vars,
    const K* indices, Tstep gs, bool indices_as_pointer,
    int counts_index, int64 num_of_keys,
    IntraThreadCopyIdAllocator* thread_copy_id_alloc) {
  std::vector<void*> value_ptrs(num_of_keys);
  LookupKeyAndSetVersion(ctx, vars[0].first, value_ptrs.data(),
                         gs, indices, num_of_keys,
                         indices_as_pointer, counts_index);
  LookupEmbedding(ctx, vars, value_ptrs.data(),
                  indices, num_of_keys);
}
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TRAINING_ALI_OP_HELPERS_H_
