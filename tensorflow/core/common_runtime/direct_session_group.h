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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DIRECT_SESSION_GROUP_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DIRECT_SESSION_GROUP_H_

#include <atomic>
#include <string>
#include <vector>

#include "tensorflow/core/public/session.h"

namespace tensorflow {
class ResourceMgr;

class DirectSessionGroup : public SessionGroup {
 public:
  DirectSessionGroup();
  DirectSessionGroup(ResourceMgr* cpu_mgr, ResourceMgr* gpu_mgr);
  virtual ~DirectSessionGroup();
  virtual Status Close() override;
  virtual int32_t GetSessionNum() const override;
  virtual Status CreateLeaderSession(Session* leader_session) override;
  virtual Status CreateFollowerSession(Session* follower_session) override;
  virtual Session* GetLeaderSession() override;
  virtual Status Create(const GraphDef& graph) override;
  virtual Status Run(
      const std::vector<std::pair<string, Tensor> >& inputs,
      const std::vector<string>& output_tensor_names,
      const std::vector<string>& target_node_names,
      std::vector<Tensor>* outputs, int32_t session_id = -1) override;
  virtual Status Run(
      const RunOptions& run_options,
      const std::vector<std::pair<string, Tensor> >& inputs,
      const std::vector<string>& output_tensor_names,
      const std::vector<string>& target_node_names,
      std::vector<Tensor>* outputs, RunMetadata* run_metadata,
      int32_t session_id = -1) override;
  virtual Session* GetSession(int32_t hint_id = -1) override;
  virtual std::unique_ptr<Session>* GetSessionPtr(int id) override;

 private:
  // sessions_[0] is leader session which own resource,
  // and others are follower sessions who
  // will reuse leader's resource.
  std::vector<std::unique_ptr<Session>> sessions_;
  int32_t session_num_ = 0;
  std::atomic<int64_t> serving_index_{0};
  ResourceMgr* cpu_shared_resource_mgr_ = nullptr;
  ResourceMgr* gpu_shared_resource_mgr_ = nullptr;

  Status GetServingSessionId(int32_t* serving_id, int32_t hint_id = -1) {
    if (session_num_ < 1) {
      return errors::InvalidArgument("Not existed a session object in SessionGroup.");
    } else if (session_num_ == 1) {
      *serving_id = 0;
    } else {
      if (hint_id >= 0) {
        *serving_id = hint_id % session_num_;
      } else {
        *serving_id = serving_index_.fetch_add(1);
        *serving_id %= session_num_;
      }
    }
    return Status::OK();
  }
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DIRECT_SESSION_GROUP_H_
