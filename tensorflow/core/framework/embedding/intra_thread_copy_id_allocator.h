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
======================================================================*/
#ifndef TENSORFLOW_CORE_FRAMEWORK_INTRA_THREAD_COPY_ID_ALLOCATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_INTRA_THREAD_COPY_ID_ALLOCATOR_H_

#include "tensorflow/core/lib/core/spin_rw_lock.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include <iostream>
#include <map>
#include <memory>
namespace tensorflow{

// Allocate a copy id for each thread
class IntraThreadCopyIdAllocator {
 public:
  IntraThreadCopyIdAllocator(int num_threads): num_worker_threads_(num_threads) {
    is_occupy_flag_.reset(new bool[num_worker_threads_]);
    memset(is_occupy_flag_.get(), 0, sizeof(bool) * num_worker_threads_);
  }

  int64 GetCopyIdOfThread(uint64 main_thread_id) {
    uint64 thread_id = Env::Default()->GetCurrentThreadId();
    if (thread_id == main_thread_id) {
      return num_worker_threads_;
    } else {
      int copy_id = -1;
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
  std::unique_ptr<bool[]> is_occupy_flag_;
  std::map<uint64, int64> hash_map_;
  mutable easy_spinrwlock_t mu_ = EASY_SPINRWLOCK_INITIALIZER;
};
} //namespace tensorflow
#endif //TENSORFLOW_CORE_FRAMEWORK_INTRA_THREAD_COPY_ID_ALLOCATOR_H_
