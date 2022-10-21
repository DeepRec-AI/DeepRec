#include "tensorflow/core/common_runtime/kernel_stat.h"
#include "tensorflow/core/common_runtime/virtual_threadpool.h"

namespace tensorflow {

VirtualThreadpool::VirtualThreadpool(Eigen::ThreadPoolInterface* underlying_threadpool, 
    const NodeItem* item, ExecutorInternal::KernelStats* kernel_stats)
    : underlying_threadpool_(underlying_threadpool), 
      item_(item), 
      kernel_stats_(kernel_stats) {}

// Submits a closure to be run by a thread in the pool.
void VirtualThreadpool::Schedule(std::function<void()> fn) {
  kernel_stats_->OpScheduleTask(item_);
  underlying_threadpool_->Schedule(fn);
}

void VirtualThreadpool::ScheduleWithHint(std::function<void()> fn, int start, int end) {
  kernel_stats_->OpScheduleTask(item_);
  underlying_threadpool_->ScheduleWithHint(fn, start, end);
}

// Returns the number of threads in the pool.
int VirtualThreadpool::NumThreads() const {
  return underlying_threadpool_->NumThreads();
}

// Returns a logical thread index between 0 and NumThreads() - 1 if called
// from one of the threads in the pool. Returns -1 otherwise.
int VirtualThreadpool::CurrentThreadId() const {
  return underlying_threadpool_->CurrentThreadId();
}

} // namespace tensorflow
