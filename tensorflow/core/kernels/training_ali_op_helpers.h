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

// Allocate a copy id for each thread
class ThreadCopyIdAllocator {
 public:
  ThreadCopyIdAllocator(int num_threads): num_worker_threads_(num_threads) {
    is_occupy_flag_ = new bool[num_worker_threads_];
    memset(is_occupy_flag_, 0, sizeof(bool) * num_worker_threads_);
  }

  ~ThreadCopyIdAllocator() {
    delete[] is_occupy_flag_;
  }

  int64 GetCopyIdOfThread(uint64 main_thread_id) {
    uint64 thread_id = Env::Default()->GetCurrentThreadId();
    if (thread_id == main_thread_id) {
      return num_worker_threads_;
    } else {
      int64 copy_id = -1;
      {
        spin_rd_lock l(mu_);
        auto iter = hash_map_.find(thread_id);
        if (iter != hash_map_.end()) {
          copy_id = iter->second;
          return copy_id;
        }
      }
      if (copy_id == -1) {
        // bind a new thread to a local cursor_list
        copy_id = thread_id % num_worker_threads_;
        while (!__sync_bool_compare_and_swap(
            &(is_occupy_flag_[copy_id]), false, true)) {
          copy_id = (copy_id + 1) % num_worker_threads_;
        }
        {
          spin_wr_lock l(mu_);
          hash_map_.insert(std::pair<uint64, int64>(thread_id, copy_id));
        }
        return copy_id;
      }
    }
  }

 private:
  int num_worker_threads_;
  bool* is_occupy_flag_ = nullptr;
  std::map<uint64, int64> hash_map_;
  mutable easy_spinrwlock_t mu_ = EASY_SPINRWLOCK_INITIALIZER;
};

template<class K, class V, class Tstep>
void LookupKeyAndSetVersion(
    OpKernelContext* ctx, EmbeddingVar<K, V>* var,
    ValuePtr<V>** value_ptrs, Tstep gs, const K* indices,
    const int64 task_size, bool indices_as_pointer) {
  auto lookup_key_and_set_version_fn = [var, value_ptrs, gs,
                  indices, indices_as_pointer] (int64 start, int64 limit) {
    ValuePtr<V>* value_ptr = nullptr;
    for (int i = start; i < limit; i++) {
      bool is_filter = false;
      var->LookupOrCreateKey(indices[i], &value_ptr, &is_filter, indices_as_pointer);
      value_ptrs[i] = value_ptr;
      var->UpdateVersion(value_ptr, gs);
    }
  };
  const int64 unit_cost = 1000; //very unreliable estimate for cost per step.
  auto worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
  Shard(worker_threads->num_threads,
        worker_threads->workers, task_size, unit_cost,
        lookup_key_and_set_version_fn);
}
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TRAINING_ALI_OP_HELPERS_H_
