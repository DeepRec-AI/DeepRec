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
  int session_num = 1;
  // default 0
  int model_id = 0;
  // Multi-stream: streams vector, [2, 4, ....]
  // gpu0: 2 streams
  // gpu1: 4 streams
  // ....
  std::vector<int> streams_vec;
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
