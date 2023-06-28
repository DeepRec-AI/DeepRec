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
#ifndef TENSORFLOW_PYTHON_OPS_PREFETCH_RUNNER_H_
#define TENSORFLOW_PYTHON_OPS_PREFETCH_RUNNER_H_

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {

// Register a PrefetchRunner to PrefetchRunnerMgr.
void TF_RegisterPrefetchRunner(const char* graph_key,
                             const char* runner_name,
                             const void* proto, size_t proto_len,
                             TF_Status* status);

// Start PrefetchRunners managed by PrefetchRunnerMgr.
void TF_StartPrefetchRunners(const char* graph_key, TF_Session* session,
                             TF_Status* status);

// Stop PrefetchRunners managed by PrefetchRunnerMgr.
void TF_StopPrefetchRunners(const char* graph_key, TF_Session* session,
                            TF_Status* status);

} // end of namespace tensorflow

#endif // End of TENSORFLOW_PYTHON_OPS_PREFETCH_RUNNER_H_
