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

#ifndef TENSORFLOW_PUBLIC_SESSION_OPTIONS_H_
#define TENSORFLOW_PUBLIC_SESSION_OPTIONS_H_

#include <string>
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class Env;

const string CUDA_GRAPH_MODE_TARGET_NAME = "CudaGraphModeSession";
/// Configuration information for a Session.
struct SessionOptions {
  /// The environment to use.
  Env* env;

  /// \brief The TensorFlow runtime to connect to.
  ///
  /// If 'target' is empty or unspecified, the local TensorFlow runtime
  /// implementation will be used.  Otherwise, the TensorFlow engine
  /// defined by 'target' will be used to perform all computations.
  ///
  /// "target" can be either a single entry or a comma separated list
  /// of entries. Each entry is a resolvable address of the
  /// following format:
  ///   local
  ///   ip:port
  ///   host:port
  ///   ... other system-specific formats to identify tasks and jobs ...
  ///
  /// NOTE: at the moment 'local' maps to an in-process service-based
  /// runtime.
  ///
  /// Upon creation, a single session affines itself to one of the
  /// remote processes, with possible load balancing choices when the
  /// "target" resolves to a list of possible processes.
  ///
  /// If the session disconnects from the remote process during its
  /// lifetime, session calls may fail immediately.
  string target;

  /// Configuration options.
  ConfigProto config;

  SessionOptions();
};

struct CudaGraphModeSessionOptions: SessionOptions {
  int batch_size = 0;
  CudaGraphModeSessionOptions();
};

struct SessionGroupMetadata {
  // default 1
  // CPU job: total session count of this session_group.
  // GPU job: session count on each GPU device,
  //          total session count determined by gpu_ids size.
  int session_count = 1;
  // default 0
  int model_id = 0;
  // Now session_group support multi-gpus.
  // gpu_ids: TF gpu ids which current session_group run on.
  // By default session_group will use all visiable gpus.
  // e.g. VisibleDeviceCount=4,
  // tf_gpu_ids is a non-empty subset of {0,1,2,3}.
  // {0}, {1} ... {1,3} ... {0,1,2,3}
  //
  // Notice: Each gpu will create session_count streams/sessions.
  // total_session_count = gpu_ids.size() * session_count
  std::vector<size_t> gpu_ids;
  // Pin cpu cores
  // "1,2;3,4;5,6"
  // Pin core 1,2 to session0
  // Pin core 3,4 to session1
  // Pin core 5,6 to session2
  std::string cpusets;
};

struct SessionGroupOptions {
  SessionGroupMetadata metadata;
  Env* env;
  string target;
  ConfigProto config;
  SessionGroupOptions();
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_PUBLIC_SESSION_OPTIONS_H_
