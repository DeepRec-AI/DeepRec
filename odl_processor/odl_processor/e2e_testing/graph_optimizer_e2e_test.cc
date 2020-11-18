/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <fstream>
#include <sstream>
#include <iostream>

#include "odl_processor/core/util/utils.h"
#include "odl_processor/core/graph_optimizer.h"
#include "odl_processor/core/util/utils.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"

// bazel  build //odl_processor/e2e_testing:graph_optimizer_e2e_test 

namespace tensorflow {
namespace processor {

namespace {

// Only for testing
std::string ReadFileIntoString(
    const std::string& filename) {
  std::ifstream ifile(filename);
  std::ostringstream buf;
  char ch;
  while(buf && ifile.get(ch)) {
    buf.put(ch);
  }

  return buf.str();
}

} // namespace

extern "C" int main(int argc, char** argv) {
  // TODO: For testing
  // Please download saved_model.pbtxt from
  //   http://tfsmoke1.cn-hangzhou.oss.aliyun-inc.com/jktest%2Fmm%2Fsaved_model.pbtxt
  // then copy the saved_model.pbtxt to /tmp/saved_model.pbtxt
  // at last, compile and run it!
  std::string saved_model_dir(
      "/tmp/saved_model.pbtxt");
  SavedModel saved_model;
  if (!tensorflow::protobuf::TextFormat::ParseFromString(ReadFileIntoString(saved_model_dir), &saved_model)) {
    LOG(FATAL) << "Can not parse saved model from text.";
  }
  std::string tag("serve");
  ClusteredGraphInfo cgi = ClusteringGraphDef(tag, saved_model);

  //LOG(INFO) << cgi.tf_saved_model.DebugString() << "\n";
  //LOG(INFO) << cgi.iree_saved_model.DebugString() << "\n";

  return 0;
}


} // namespace processor
} // namespace tensorflow
