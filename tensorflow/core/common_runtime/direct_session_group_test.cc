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

#include "tensorflow/core/common_runtime/direct_session.h"

#include <map>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {
namespace {

CallableOptions MakeCallableOptions(gtl::ArraySlice<string> feeds,
                                    gtl::ArraySlice<string> fetches,
                                    gtl::ArraySlice<string> targets) {
  CallableOptions ret;
  for (const string& feed : feeds) {
    ret.add_feed(feed);
  }
  for (const string& fetch : fetches) {
    ret.add_fetch(fetch);
  }
  for (const string& target : targets) {
    ret.add_target(target);
  }
  return ret;
}

SessionOptions DefaultSessionOptions() {
  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  return options;
}

std::unique_ptr<SessionGroup> CreateSessionGroup() {
  SessionGroup* sg = nullptr;
  SessionGroupMetadata metadata; 
  NewSessionGroup(DefaultSessionOptions(), &sg, metadata);
  return std::unique_ptr<SessionGroup>(sg);
}

std::unique_ptr<Session> CreateSession() {
  return std::unique_ptr<Session>(NewSession(DefaultSessionOptions()));
}

class DirectSessionMinusAXTest : public ::testing::Test {
 public:
  void Initialize(std::initializer_list<float> a_values) {
    Graph graph(OpRegistry::Global());

    Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
    test::FillValues<float>(&a_tensor, a_values);
    Node* a = test::graph::Constant(&graph, a_tensor);
    a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");
    a_ = a->name();

    Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
    test::FillValues<float>(&x_tensor, {1, 1});
    Node* x = test::graph::Constant(&graph, x_tensor);
    x->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");
    x_ = x->name();

    // y = A * x
    Node* y = test::graph::Matmul(&graph, a, x, false, false);
    y->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");
    y_ = y->name();

    Node* y_neg = test::graph::Unary(&graph, "Neg", y);
    y_neg_ = y_neg->name();
    y_neg->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");

    Node* z = test::graph::Unary(&graph, "Identity", y_neg);
    z_ = z->name();
    z->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");

    graph.ToGraphDef(&def_);
  }

  string a_;
  string x_;
  string y_;
  string y_neg_;
  string z_;
  GraphDef def_;
};

TEST_F(DirectSessionMinusAXTest, TestDirectSessionGroup) {
  for (int i = 0; i < 1000; ++i) {
    Initialize({3, 2, -1, 0});
    auto sg = CreateSessionGroup();
    ASSERT_TRUE(sg != nullptr);
    TF_ASSERT_OK(sg->Create(def_));
    std::vector<std::pair<string, Tensor>> inputs;

    // Request two targets: one fetch output and one non-fetched output.
    std::vector<string> output_names = {y_ + ":0"};
    std::vector<string> target_nodes = {y_neg_};
    std::vector<Tensor> outputs;
    Status s = sg->Run(inputs, output_names, target_nodes, &outputs);
    TF_ASSERT_OK(s);

    ASSERT_EQ(1, outputs.size());
    // The first output should be initialized and have the correct
    // output.
    auto mat = outputs[0].matrix<float>();
    ASSERT_TRUE(outputs[0].IsInitialized());
    EXPECT_FLOAT_EQ(5.0, mat(0, 0));

    //usleep(10000);
  }
}

TEST_F(DirectSessionMinusAXTest, TestDirectSession) {
  for (int i = 0; i < 1000; ++i) {
    Initialize({3, 2, -1, 0});
    auto sg = CreateSession();
    ASSERT_TRUE(sg != nullptr);
    TF_ASSERT_OK(sg->Create(def_));
    std::vector<std::pair<string, Tensor>> inputs;

    // Request two targets: one fetch output and one non-fetched output.
    std::vector<string> output_names = {y_ + ":0"};
    std::vector<string> target_nodes = {y_neg_};
    std::vector<Tensor> outputs;
    Status s = sg->Run(inputs, output_names, target_nodes, &outputs);
    TF_ASSERT_OK(s);

    ASSERT_EQ(1, outputs.size());
    // The first output should be initialized and have the correct
    // output.
    auto mat = outputs[0].matrix<float>();
    ASSERT_TRUE(outputs[0].IsInitialized());
    EXPECT_FLOAT_EQ(5.0, mat(0, 0));

    //usleep(10000);
  }
}

}
}  // namespace tensorflow
