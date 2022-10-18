#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_VIRTUAL_THREADPOOL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_VIRTUAL_THREADPOOL_H_

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/threadpool_interface.h"

namespace tensorflow {
class NodeItem;

namespace ExecutorInternal {
class KernelStats;
} // namespace ExecutorInternal

class VirtualThreadpool : public thread::ThreadPoolInterface {
 public:
  VirtualThreadpool(Eigen::ThreadPoolInterface* underlying_threadpool, 
      const NodeItem* item, ExecutorInternal::KernelStats* kernel_stats);
  virtual ~VirtualThreadpool() {}

  // Submits a closure to be run by a thread in the pool.
  void Schedule(std::function<void()> fn) override;

  void ScheduleWithHint(std::function<void()> fn, int start, int end);

  // Returns the number of threads in the pool.
  int NumThreads() const override;

  // Returns a logical thread index between 0 and NumThreads() - 1 if called
  // from one of the threads in the pool. Returns -1 otherwise.
  int CurrentThreadId() const override;

 private:
  Eigen::ThreadPoolInterface* underlying_threadpool_;
  const NodeItem* item_;
  ExecutorInternal::KernelStats* kernel_stats_;
};

} // namespace tensorflow

#endif // TENSORFLOW_CORE_COMMON_RUNTIME_VIRTUAL_THREADPOOL_H_
