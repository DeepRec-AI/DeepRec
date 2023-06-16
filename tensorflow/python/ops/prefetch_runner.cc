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
#include "tensorflow/python/ops/prefetch_runner.h"

namespace tensorflow {

void TF_RegisterPrefetchRunner(const char* graph_key, const char* runner_name,
                             const void* proto, size_t proto_len,
                             TF_Status* status) {
  tensorflow::PrefetchRunnerOptions options;
  if (!options.ParseFromArray(proto, proto_len)) {
    status->status =
        errors::InvalidArgument("Unparseable PrefetchRunnerOptions");
    return;
  }

  auto prefetch_runner_mgr = tensorflow::PrefetchRunnerMgr::singleton();
  status->status = prefetch_runner_mgr->RegisterPrefetchRunner(
      graph_key, runner_name, options);
}

void TF_StartPrefetchRunners(const char* graph_key, TF_Session* session,
                             TF_Status* status) {
  auto prefetch_runner_mgr = tensorflow::PrefetchRunnerMgr::singleton();
  status->status =
      prefetch_runner_mgr->StartRunners(graph_key, session->session);
}

void TF_StopPrefetchRunners(const char* graph_key, TF_Session* session,
                            TF_Status* status) {
  auto prefetch_runner_mgr = tensorflow::PrefetchRunnerMgr::singleton();
  status->status =
      prefetch_runner_mgr->StopRunners(graph_key, session->session);
}

} // end of namespace tensorflow
