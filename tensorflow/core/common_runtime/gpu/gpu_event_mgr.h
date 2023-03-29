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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_

#include <deque>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace stream_executor {
class Event;
class Stream;
class StreamExecutor;
}  // namespace stream_executor

namespace tensorflow {

class GPUOptions;

// The callback provided to EventMgr::ThenExecute must not block or take a long
// time.  If it does, performance may be impacted and GPU memory may be
// exhausted.  This macro is for checking that an EventMgr thread is not
// accidentally entering blocking parts of the code, e.g. the RPC subsystem.
//
// Intended use is something like
//
//   void RespondToAnRPC(Params* params) {
//      WARN_IF_IN_EVENT_MGR_THREAD;
//      if (params->status.ok()) { ...
//
namespace gpu_event_mgr {
// Logs a stack trace if current execution thread belongs to this EventMgr
// object.  If f is not nullptr, executes instead of  logging the stack trace.
// trace.
void WarnIfInCallback(std::function<void()> f);
}  // namespace gpu_event_mgr
#define WARN_IF_IN_EVENT_MGR_THREAD gpu_event_mgr::WarnIfInCallback(nullptr)

// An object to keep track of pending Events in the StreamExecutor streams
// and associated Tensors that cannot safely be deleted until the associated
// Events are recorded.
class EventMgr {
 public:
  virtual ~EventMgr();

  // Execute func when all pending stream actions have completed.
  // func must be brief and non-blocking since it executes in the one
  // thread used for all such callbacks and also buffer deletions.
  inline void ThenExecute(se::Stream* stream, std::function<void()> func) {
    mutex_lock l(mu_);
    QueueFunc(stream, std::move(func));
    PollEvents(stream);
  }

 private:
  friend class TEST_EventMgr;
  friend class TEST_EventMgrHelper;
  friend class EventMgrFactory;
  se::StreamExecutor* const exec_;
  const int32 polling_active_delay_usecs_;
  mutex mu_;
  condition_variable events_pending_ GUARDED_BY(mu_);

  EventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options);

  void SchduleCallBack(std::function<void()> func) {
    // The function must be called in another thread.
    CHECK(func != nullptr);
    threadpool_.Schedule(std::move(func));
  }

  void QueueFunc(se::Stream* stream, std::function<void()> func)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // This function should be called at roughly the same tempo as
  // QueueTensors() to check whether pending events have recorded,
  // and then retire them.  It appends InUse elements that need cleanup
  // to "*to_free".  The caller should call FreeMemory(to_free)
  // when this returns.
  void PollEvents(se::Stream* stream = nullptr)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // An internal polling loop that runs at a low frequency to clear
  // straggler Events.
  void PollLoop();

  // Setup/Teardown functions for the polling loop.
  void StartPollingLoop();
  void StopPollingLoop();

  // A stack of unused events
  std::vector<se::Event*> free_events_ GUARDED_BY(mu_);

  // Callbacks waiting on their events to complete.
  absl::flat_hash_map<
      se::Stream*,
      std::deque<std::pair<se::Event*, std::function<void()>>>>
      callbacks_ GUARDED_BY(mu_);

  bool stop_polling_ GUARDED_BY(mu_);
  std::unique_ptr<Notification> polling_stopped_;

  // The main PollLoop for the event manager runs in this threadpool.
  thread::ThreadPool threadpool_;
};

// Manages all the EventMgr instances.
class EventMgrFactory {
 public:
  static EventMgrFactory* Singleton();

  EventMgr* GetEventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options);

 private:
  mutex mu_;

  // Maintain one EventMgr per physical device (StreamExecutor is
  // per-physical-device).
  std::map<se::StreamExecutor*, EventMgr*> event_mgr_map_ GUARDED_BY(mu_);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_
