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

#include "tensorflow/cc/training/prefetch_runner.h"
#include "tensorflow/core/platform/default/logging.h"

namespace tensorflow {
/*----------------------------- PrefetchRunner -------------------------------*/

PrefetchRunner::PrefetchRunner(std::string graph_key, std::string runner_name,
                               Session* sess, Coordinator* coord,
                               PrefetchRunnerOptions& params)
    : graph_key_(graph_key),
      name_(runner_name),
      sess_(sess),
      coord_(coord),
      params_(params),
      thread_nums_(0),
      is_running_(false) {}

PrefetchRunner::~PrefetchRunner() {
  Join();
}

std::string PrefetchRunner::graph_key() const {
  return graph_key_;
}

std::string PrefetchRunner::name() const {
  return name_;
}

bool PrefetchRunner::IsRunning() const {
  return is_running_;
}

void PrefetchRunner::Start() {
  {
    mutex_lock l(mu_);
    thread_nums_ = params_.fetch_ops().size();
  }

  thread_pool_.resize(thread_nums_);
  for (size_t i = 0; i < thread_nums_; i++)
    thread_pool_[i].reset(new std::thread(&PrefetchRunner::Run, this, i));

  if (coord_)
    cancel_thread_.reset(new std::thread(&PrefetchRunner::Stop, this));

  is_running_ = true;
}

void PrefetchRunner::Stop() {
  if (coord_)
    coord_->WaitForStop();

  sess_->Run({}, {}, {params_.cancel_op()}, nullptr);
  is_running_ = false;
}

Status PrefetchRunner::Join() {
  for (size_t i = 0; i < thread_pool_.size(); i++) {
    if (thread_pool_[i] != nullptr && thread_pool_[i]->joinable())
      thread_pool_[i]->join();
  }

  if (cancel_thread_ != nullptr && cancel_thread_->joinable()) {
    cancel_thread_->join();
  }

  return Status::OK();
}

void PrefetchRunner::Run(size_t index) {
  Status status;

  // initialize tensor buffer.
  status = sess_->Run({}, {}, {params_.resume_op()}, nullptr);
  if (status != Status::OK()) {
    DealWithCancelledUnexpectedError(status);
    return;
  }

  // get value generator
  auto feed_in_tensors = params_.named_feed_input_tensors();
  std::vector<std::string> val_consume_ops;
  std::vector<std::string> val_gen_tensors;
  val_consume_ops.reserve(feed_in_tensors.size());
  val_gen_tensors.reserve(feed_in_tensors.size());
  for (auto iter = feed_in_tensors.begin(); iter != feed_in_tensors.end(); ++iter) {
    val_consume_ops.emplace_back(iter->first);
    val_gen_tensors.emplace_back(iter->second);
  }

  std::vector<Tensor> gen_val;
  std::vector<std::pair<std::string, Tensor>> gen_inputs;
  gen_val.reserve(feed_in_tensors.size());
  gen_inputs.reserve(feed_in_tensors.size());
  while (true) {
    if (TF_PREDICT_FALSE(coord_ && coord_->ShouldStop())) {
      mutex_lock l (mu_);
      thread_nums_--;
      return;
    }

    // generate feed tensor
    if (!val_gen_tensors.empty()) {
      gen_val.clear();
      gen_inputs.clear();
      status = sess_->Run(params_.run_options(), {}, val_gen_tensors, {},
                         &gen_val, nullptr);
      if (TF_PREDICT_FALSE(status != Status::OK())) {
        if (!CheckRunErrorStatus(status))
          return;
      }

      for (size_t i = 0; i < val_gen_tensors.size(); i++) {
        gen_inputs.emplace_back(val_consume_ops[i], gen_val[i]);
      }
    }

    // run prefetch subgraph.
    status = sess_->Run(params_.run_options(), gen_inputs, {},
                       {params_.fetch_ops(index)}, nullptr, nullptr);
    if (TF_PREDICT_FALSE(status != Status::OK())) {
      if (!CheckRunErrorStatus(status))
        return;
    }
  }
}

/// Only Status Code is in `ignored_exceptions`, return `true`,
/// otherwise return `false`.
bool PrefetchRunner::CheckRunErrorStatus(Status& s) {
  auto& closed_e = params_.closed_exceptions();
  auto& ignored_e = params_.ignored_exceptions();
  if (error::CANCELLED == s.code()) {
    DealWithCancelledError();
    return false;
  } else if (std::count(closed_e.begin(), closed_e.end(), s.code()) != 0) {
    DealWithClosedError();
    return false;
  } else if (std::count(ignored_e.begin(), ignored_e.end(), s.code()) != 0) {
    DealWithIgnoredError(s);
    return true;
  }

  DealWithCancelledUnexpectedError(s);
  return false;
}

void PrefetchRunner::DealWithCancelledError() {
  LOG(INFO) << "PrefetchRunner <" << name_ << "> Prefetching was cancelled.";
  {
    mutex_lock l (mu_);
    thread_nums_--;
  }
}

void PrefetchRunner::DealWithClosedError() {
  LOG(INFO) << "PrefetchRunner <" << name_ << "> Prefetching was closed.";
  {
    mutex_lock l(mu_);
    thread_nums_--;
    if (thread_nums_ == 0)
      sess_->Run({}, {}, {params_.close_op()}, nullptr);
  }
}

void PrefetchRunner::DealWithIgnoredError(Status& s) {
  LOG(WARNING) << "PrefetchRunner <" << name_
               << "> Corrupted inputs were ignored in prefetching: "
               << s.error_message();
}

void PrefetchRunner::DealWithCancelledUnexpectedError(Status& s) {
  LOG(ERROR) << "PrefetchRunner <" << name_
             << "> Prefetching was cancelled unexpectedly: "
             << s.error_message();
  if (coord_)
    coord_->RequestStop();

  {
    mutex_lock l (mu_);
    thread_nums_--;
  }
}

/*----------------------------- PrefetchRunnerMgr ----------------------------*/

/*static*/ PrefetchRunnerMgr* PrefetchRunnerMgr::singleton() {
  static PrefetchRunnerMgr* instance = new PrefetchRunnerMgr;
  return instance;
}

PrefetchRunnerMgr::PrefetchRunnerMgr() {}

PrefetchRunnerMgr::~PrefetchRunnerMgr() {
  register_runner_options_.clear();
  prefetch_runners_.clear();
  coords_.clear();
}

Status PrefetchRunnerMgr::RegisterPrefetchRunner(
    std::string graph_key, std::string runner_name,
    PrefetchRunnerOptions& params) {
  if (register_runner_options_[graph_key].count(runner_name))
    return Status(errors::AlreadyExists("PrefetchRunner <" + runner_name +
                                        "> has already existed in graph <" +
                                        graph_key + ">."));

  register_runner_options_[graph_key][runner_name] = params;
  return Status::OK();
}

Status PrefetchRunnerMgr::StartRunners(std::string graph_key, Session* sess) {
  if (register_runner_options_.count(graph_key) == 0)
    return Status(
        errors::NotFound("graph <" + graph_key + "> has no PrefetchRunner"));

  if (prefetch_runners_.count(sess) != 0)
    return Status(errors::AlreadyExists(
        "PrefetchRunners has already started in Session."));

  // Create and Start the PrefetchRunners.
  coords_[sess].reset(new Coordinator());

  for (auto option : register_runner_options_[graph_key]) {
    std::unique_ptr<PrefetchRunner> runner(new PrefetchRunner(
        graph_key, option.first, sess, coords_[sess].get(), option.second));
    prefetch_runners_[sess].insert(runner.get());
    coords_[sess]->RegisterRunner(std::move(runner));
  }

  for (auto runner : prefetch_runners_[sess])
    runner->Start();

  return Status::OK();
}

Status PrefetchRunnerMgr::StopRunners(std::string graph_key, Session* sess) {
  if (prefetch_runners_.count(sess) == 0)
    return Status(errors::NotFound("No PrefechRunners run in Session"));

  coords_[sess]->RequestStop();
  for (auto runner : prefetch_runners_[sess])
    runner->Join();

  coords_.erase(sess);
  prefetch_runners_.erase(sess);

  return Status::OK();
}

} // end of namespace tensorflow
