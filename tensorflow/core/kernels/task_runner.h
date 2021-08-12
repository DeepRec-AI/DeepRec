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

#ifndef TENSORFLOW_CORE_KERNELS_TASK_RUNNER_H_
#define TENSORFLOW_CORE_KERNELS_TASK_RUNNER_H_

#include <functional>

#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

// TaskRunner schedules tasks(function f) to ThreadPool
// and wait until all finished
class TaskRunner {
 public:
  explicit TaskRunner(const std::function<void(int32, int32)>& f,
                      thread::ThreadPool* tp, int32 n)
      : func_(f), thread_pool_(tp), num_tasks_(n) {}

  void Run() {
    if (num_tasks_ <= 0) return;
    BlockingCounter bc(num_tasks_ - 1);

    // Sending (num_tasks - 1) tasks to threadpool for scheduling
    for (int32 i = 0; i < num_tasks_ - 1; ++i) {
      thread_pool_->Schedule([this, &bc, i]() {
        func_(i, num_tasks_);
        bc.DecrementCount();
      });
    }
    // Run the last task in current thread.
    func_(num_tasks_ - 1, num_tasks_);
    bc.Wait();
  }

 private:
  std::function<void(int32 task_id, int32 num_tasks)> func_;
  thread::ThreadPool* thread_pool_;
  const int32 num_tasks_;
};

// add more types of SummaryUpdater
// for more types of summary or more ways of summary aggregation
class StatusSummaryUpdater {
 public:
  static void UpdateSummary(Status* mine, const Status& ret) {
    mine->Update(ret);
  }
};

class Int64SumSummaryUpdater {
 public:
  static void UpdateSummary(int64_t* mine, const int64_t& ret) {
    *mine += ret;
  }
};

// SummaryTaskRunner schedules tasks and summary their return values.
// S is the type of return values.
// SUpdater is the class for aggregating the return values.
template<typename S, typename SUpdater>
class SummaryTaskRunner {
 public:
  explicit SummaryTaskRunner(const std::function<S(int32, int32)>& f,
                      const S& init_summary, thread::ThreadPool* tp, int32 n)
      : func_(f), summary_(init_summary), thread_pool_(tp), num_tasks_(n) {}

  void Run() {
    if (num_tasks_ <= 0) return;
    BlockingCounter bc(num_tasks_ - 1);

    // Sending (num_tasks - 1) tasks to threadpool for scheduling
    for (int32 i = 0; i < num_tasks_ - 1; ++i) {
      thread_pool_->Schedule([this, &bc, i]() {
        const S& ret = func_(i, num_tasks_);
        UpdateSummaryUnlocked(ret);
        bc.DecrementCount();
      });
    }
    // Run the last task in current thread.
    const S& ret = func_(num_tasks_ - 1, num_tasks_);
    UpdateSummaryUnlocked(ret);
    bc.Wait();
  }

  S summary() {
    return summary_;
  }

 private:
  void UpdateSummaryUnlocked(const S& ret) {
    mutex_lock lock(mu_);
    SUpdater::UpdateSummary(&summary_, ret);
  }

  mutex mu_;
  std::function<S(int32 task_id, int32 num_tasks)> func_;
  S summary_;
  thread::ThreadPool* thread_pool_;
  const int32 num_tasks_;
};

}  // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_TASK_RUNNER_H_
