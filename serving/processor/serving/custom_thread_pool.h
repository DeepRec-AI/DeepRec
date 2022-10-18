#ifndef SERVING_PROCESSOR_SERVING_CUSTOM_THREAD_POOL_H
#define SERVING_PROCESSOR_SERVING_CUSTOM_THREAD_POOL_H

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/threadpool_options.h"

namespace tensorflow {
namespace processor {

class CustomThreadPoolImpl : public thread::ThreadPoolInterface {
 public:
  explicit CustomThreadPoolImpl(int num_threads,
                                const std::string& name) {
    underlying_threadpool_.reset(new thread::ThreadPool(
        tensorflow::Env::Default(), name, num_threads));
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

  int GetNumScheduleCalled() { return num_schedule_called_; }

 private:
  int num_schedule_called_;
  std::unique_ptr<tensorflow::thread::ThreadPool> underlying_threadpool_;
};

} // namespace processor
} // namespace tensorflow

#endif // SERVING_PROCESSOR_SERVING_CUSTOM_THREAD_POOL_H
