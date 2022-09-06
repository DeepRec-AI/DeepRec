/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/direct_session_group.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {

DirectSessionGroup::DirectSessionGroup()
    : cpu_shared_resource_mgr_(nullptr),
      gpu_shared_resource_mgr_(nullptr) {}

DirectSessionGroup::DirectSessionGroup(ResourceMgr* cpu_mgr, ResourceMgr* gpu_mgr)
    : cpu_shared_resource_mgr_(cpu_mgr),
      gpu_shared_resource_mgr_(gpu_mgr) {}

DirectSessionGroup::~DirectSessionGroup() {
  if (cpu_shared_resource_mgr_) {
    delete cpu_shared_resource_mgr_;
  }

  if (gpu_shared_resource_mgr_) {
    delete gpu_shared_resource_mgr_;
  }
}

Status DirectSessionGroup::Close() {
  for (int32_t i = 1; i < session_num_; ++i) {
    if (sessions_[i]) {
      sessions_[i]->Close().IgnoreError();
    }
  }
  if (session_num_ > 0 && sessions_[0]) {
    sessions_[0]->Close().IgnoreError();
  }

  return Status::OK();
}

int32_t DirectSessionGroup::GetSessionNum() const {
  return session_num_;
}

Status DirectSessionGroup::CreateLeaderSession(Session* leader_session) {
  if (session_num_ > 0) {
    return errors::AlreadyExists("Leader session is already existed.");
  }
  std::unique_ptr<Session> tmp;
  tmp.reset(leader_session);
  sessions_.emplace_back(std::move(tmp));
  ++session_num_;
  return Status::OK();
}

Status DirectSessionGroup::CreateFollowerSession(Session* follower_session) {
  if (session_num_ < 1) {
    return errors::NotFound(
        "Leader session is not created, please create it firstly.");
  }
  std::unique_ptr<Session> tmp;
  tmp.reset(follower_session);
  sessions_.emplace_back(std::move(tmp));
  ++session_num_;
  return Status::OK();
}

Session* DirectSessionGroup::GetLeaderSession() {
  return sessions_[0].get();
}

Status DirectSessionGroup::Create(const GraphDef& graph) {
  for (auto& sess : sessions_) {
    Status s = sess->Create(graph);
    if (!s.ok()) return s;
  }
  return Status::OK();
}

Status DirectSessionGroup::Run(
    const std::vector<std::pair<string, Tensor> >& inputs,
    const std::vector<string>& output_tensor_names,
    const std::vector<string>& target_node_names,
    std::vector<Tensor>* outputs, int32_t session_id) {
  int32_t id = 0;
  Status s = GetServingSessionId(&id, session_id);
  if (!s.ok()) return s;
  return sessions_[id]->Run(inputs, output_tensor_names,
                            target_node_names, outputs);
}

Status DirectSessionGroup::Run(
    const RunOptions& run_options,
    const std::vector<std::pair<string, Tensor> >& inputs,
    const std::vector<string>& output_tensor_names,
    const std::vector<string>& target_node_names,
    std::vector<Tensor>* outputs, RunMetadata* run_metadata,
    int32_t session_id) {
  int32_t id = 0;
  Status s = GetServingSessionId(&id, session_id);
  if (!s.ok()) return s;
  return sessions_[id]->Run(run_options, inputs, output_tensor_names,
                            target_node_names, outputs, run_metadata);
}

Session* DirectSessionGroup::GetSession(int32_t hint_id) {
  int32_t id = 0;
  Status s = GetServingSessionId(&id, hint_id);
  if (!s.ok()) {
    LOG(ERROR) << "Get serving session error, use default session[0]";
    return sessions_[0].get();
  }
  return sessions_[id].get();
}

std::unique_ptr<Session>* DirectSessionGroup::GetSessionPtr(int id) {
  if (id < 0 || id >= session_num_) {
    LOG(ERROR) << "session num in current sess_group is " << session_num_
               << ", can not get session[" << id << "].";
    return nullptr;
  }
  return &(sessions_[id]);
}

}  // end namespace tensorflow

