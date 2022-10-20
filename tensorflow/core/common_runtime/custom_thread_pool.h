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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_CUSTOM_THREAD_POOL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_CUSTOM_THREAD_POOL_H_

#include "tensorflow/core/lib/core/threadpool_interface.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
class CustomThreadPoolImpl : public thread::ThreadPoolInterface {
 public:
  explicit CustomThreadPoolImpl(std::string name, int num_threads) {
    underlying_threadpool_.reset(new thread::ThreadPool(
				 tensorflow::Env::Default(), name, num_threads));
    num_schedule_called_ = 0;
  }

  explicit CustomThreadPoolImpl(const SessionOptions& options, std::string name,
                               int num_threads) {
    underlying_threadpool_.reset(new thread::ThreadPool(
        options.env, ThreadOptions(), name, num_threads,
        !options.config.experimental().disable_thread_spinning(),
        /*allocator=*/nullptr));
    num_schedule_called_ = 0;
  }

  void Schedule(std::function<void()> fn) override {
    num_schedule_called_ += 1;
    underlying_threadpool_->Schedule(std::move(fn));
  }

  void ScheduleWithHint(std::function<void()> fn, int start, int end) override {
    num_schedule_called_ += 1;
    underlying_threadpool_->ScheduleWithHint(std::move(fn), start, end);
  }

  void Cancel() override {}

  int NumThreads() const override {
    return underlying_threadpool_->NumThreads();
  }

  int CurrentThreadId() const override {
    return underlying_threadpool_->CurrentThreadId();
  }

  int GetNumScheduleCalled() {return num_schedule_called_; }

  thread::ThreadPool *get_threadpool() { return underlying_threadpool_.get(); }

 private:
  int num_schedule_called_;
  std::unique_ptr<tensorflow::thread::ThreadPool> underlying_threadpool_;
};

} // end of namespace tensorflow

#endif // endof TENSORFLOW_CORE_COMMON_RUNTIME_CUSTOM_THREAD_POOL_H_
