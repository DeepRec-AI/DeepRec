/* Copyright 2023 The DeepRec Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_TRAINING_PREFETCH_RUNNER_H_
#define TENSORFLOW_CC_TRAINING_PREFETCH_RUNNER_H_

#include <map>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/cc/training/coordinator.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

/// PrefetchRunner class is responsible for prefetching tensor by repeating
/// running given ops.
class PrefetchRunner : public RunnerInterface {
 public:
  PrefetchRunner(std::string graph_key, std::string runner_name,
                 Session* sess, Coordinator* coord,
                 PrefetchRunnerOptions& params);

  /// The destructor would join all the threads.
  ~PrefetchRunner();

  std::string name() const;

  std::string graph_key() const;

  /// Returns true iff the runner is running.
  bool IsRunning() const override;

  /// Starts the prefetch runner with the given session.
  void Start();

  /// Requests to stop and runs the cancel op. It would be called in a separate
  /// thread when coordinator is set. If there is no coordinator it should be
  /// called before calling Join.
  void Stop();

  /// Joins all the threads. Returns okay if all threads run successfully;
  /// otherwise returns the first captured failure status.
  Status Join() final;

 private:
  std::string graph_key_;
  std::string name_;
  /// Parameters required for the session to run.
  PrefetchRunnerOptions params_;
  Session *sess_; // not owned
  Coordinator* coord_; // not owned

  mutex mu_;
  size_t thread_nums_ GUARDED_BY(mu_);
  std::atomic<bool> is_running_;
  std::atomic<bool> force_stop_;
  std::vector<std::unique_ptr<std::thread>> thread_pool_;
  std::unique_ptr<std::thread> cancel_thread_;

  /// Run prefetch subgraph.
  void Run(size_t index);

  /// Check the return status of Session run, return `true` if execution can
  /// continue, otherwise return `false`.
  bool CheckRunErrorStatus(Status &s);

  void DealWithCancelledError();

  void DealWithClosedError();

  void DealWithIgnoredError(Status& s);

  void DealWithCancelledUnexpectedError(Status& s);
};

/// PrefetchRunnerMgr class is used to managed PrefetcRunner.
class PrefetchRunnerMgr {
 public:
  static PrefetchRunnerMgr* singleton();
  /// Add a new PrefetRunner to PrefetchRunnerMgr.
  Status RegisterPrefetchRunner(std::string graph_key, std::string runner_name,
                                PrefetchRunnerOptions& params);

  /// Start all PrefetchRunners
  Status StartRunners(std::string graph_key, Session* sess);

  /// Stop all PrefetchRunners
  Status StopRunners(std::string graph_key, Session* sess);

 private:
  PrefetchRunnerMgr();
  ~PrefetchRunnerMgr();

  // map<runner_name, PrefetchOptions>
  typedef std::unordered_map<std::string, PrefetchRunnerOptions>
      name_runner_options_map_t;
  // map<graph_key, runner_options_t>
  std::unordered_map<std::string, name_runner_options_map_t>
      register_runner_options_;

  // map<Session*, coord>
  std::map<Session*, std::unique_ptr<Coordinator>> coords_;
  // map<Session*, runner_name_map_t>. PrefetchRunner* is not owned.
  std::map<Session*, std::set<PrefetchRunner*>> prefetch_runners_;
};

} // end of namespace tensorflow

#endif // TENSORFLOW_CC_TRAINING_PREFETCH_RUNNER_H_
