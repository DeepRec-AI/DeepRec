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

#include "tensorflow/compiler/jit/async_io_conversion_pass.h"

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/graph_def_builder_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {
namespace {

const char* kTestDeviceName = "/job:worker/replica:0/task:0/device:GPU:0";

Status ConvertToAsyncOut(std::unique_ptr<Graph>* graph,
                         FunctionLibraryDefinition* flib_def) {
  FixupSourceAndSinkEdges(graph->get());

  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(OptimizerOptions::ON_2);

  GraphOptimizationPassOptions opt_options;
  opt_options.session_options = &session_options;
  opt_options.graph = graph;
  opt_options.flib_def = flib_def;

  VLOG(2) << "Before AsyncIoConversionPass:";
  VLOG(2) << (*graph)->ToGraphDefDebug().DebugString();

  AsyncIoConversionPass pass(/*async_io_level=*/2);
  TF_RETURN_IF_ERROR(pass.Run(opt_options));

  VLOG(2) << "After AsyncIoConversionPass:";
  VLOG(2) << (*graph)->ToGraphDefDebug().DebugString();

  return Status::OK();
}

TEST(AsyncOutConversionTest, 1In1OutCluster) {
  const string cluster_name = "cluster_1in1out";
  const string tensor_name = "identity_in_cluster:0";
  const string name_prefix = "identity_in_cluster_0";

  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  FunctionDefLibrary fdef_lib;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
    Node* input =
        ops::SourceOp("Placeholder", builder.opts()
                                         .WithName("placeholder")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("shape", TensorShape({})));
    Node* identity0 = ops::UnaryOp(
        "Identity", input, builder.opts()
                               .WithName("identity_in_cluster")
                               .WithDevice(kTestDeviceName)
                               .WithAttr("T", DT_FLOAT)
                               .WithAttr(kXlaClusterAttr, cluster_name));
    Node* identity1 = ops::UnaryOp("Identity", identity0,
                                   builder.opts()
                                       .WithName("identity_outside_cluster")
                                       .WithDevice(kTestDeviceName)
                                       .WithAttr("T", DT_FLOAT));

    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(ConvertToAsyncOut(&graph, &flib_def));
  GraphDef graphdef;
  graph->ToGraphDef(&graphdef);

  // Verify the results.
  GraphDef graphdef_expected;
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
    Node* init = ops::SourceOp(
        "_XlaAsyncOutInit",
        builder.opts()
            .WithName(absl::StrCat(cluster_name, "_XlaAsyncOutInit"))
            .WithDevice(kTestDeviceName)
            .WithAttr("device_name", kTestDeviceName)
            .WithAttr("tensor_names", {tensor_name}));
    Node* input =
        ops::SourceOp("Placeholder", builder.opts()
                                         .WithName("placeholder")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("shape", TensorShape({})));
    Node* identity_in_cluster = ops::UnaryOp(
        "Identity", input, builder.opts()
                               .WithName("identity_in_cluster")
                               .WithDevice(kTestDeviceName)
                               .WithAttr("T", DT_FLOAT)
                               .WithAttr(kXlaClusterAttr, cluster_name));
    Node* send = ops::UnaryOp(
        "_XlaAsyncOutSend", identity_in_cluster,
        builder.opts()
            .WithName(absl::StrCat(name_prefix, "_XlaAsyncOutSend"))
            .WithControlInput(init)
            .WithDevice(kTestDeviceName)
            .WithAttr("T", DT_FLOAT)
            .WithAttr("device_name", kTestDeviceName)
            .WithAttr("tensor_name", tensor_name)
            .WithAttr(kXlaClusterAttr, cluster_name));
    Node* recv = ops::SourceOp(
        "_XlaAsyncOutRecv",
        builder.opts()
            .WithName(absl::StrCat(name_prefix, "_XlaAsyncOutRecv"))
            .WithControlInput(init)
            .WithDevice(kTestDeviceName)
            .WithAttr("T", DT_FLOAT)
            .WithAttr("device_name", kTestDeviceName)
            .WithAttr("tensor_name", tensor_name));
    Node* identity_ouside_cluster =
        ops::UnaryOp("Identity", recv, builder.opts()
                                           .WithName("identity_outside_cluster")
                                           .WithDevice(kTestDeviceName)
                                           .WithAttr("T", DT_FLOAT));
    Node* done = ops::SourceOp(
        "_XlaAsyncOutDone",
        builder.opts()
            .WithName(absl::StrCat(cluster_name, "_XlaAsyncOutDone"))
            .WithDevice(kTestDeviceName)
            .WithControlInputs({send, recv})
            .WithAttr("device_name", kTestDeviceName)
            .WithAttr("tensor_names", {tensor_name}));
    TF_EXPECT_OK(builder.ToGraphDef(&graphdef_expected));
  }

  TF_EXPECT_GRAPH_EQ(graphdef_expected, graphdef);
}

// Tests that AsyncIoConversionPass does not replace either Resource
// or Const output edges.
TEST(AsyncIoConversionTest, ResourceAndConstRetval) {
  const string cluster_name = "cluster_var_and_const";

  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  FunctionDefLibrary fdef_lib;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately, &flib_def);
    Node* res_in =
        ops::SourceOp("VarHandleOp", builder.opts()
                                         .WithName("varhandle")
                                         .WithAttr("dtype", DT_FLOAT)
                                         .WithAttr("shape", TensorShape({})));
    Node* const_in =
        ops::SourceOp("Const", builder.opts()
                                   .WithName("")
                                   .WithAttr("dtype", DT_FLOAT)
                                   .WithAttr("value", Tensor())
                                   .WithAttr(kXlaClusterAttr, cluster_name));
    Node* identity0 = ops::UnaryOp(
        "Identity", res_in, builder.opts()
                                .WithName("identity0_in_cluster")
                                .WithDevice(kTestDeviceName)
                                .WithAttr("T", DT_RESOURCE)
                                .WithAttr(kXlaClusterAttr, cluster_name));
    Node* identity1 = ops::UnaryOp("Identity", identity0,
                                   builder.opts()
                                       .WithName("identity1_outside_cluster")
                                       .WithDevice(kTestDeviceName)
                                       .WithAttr("T", DT_RESOURCE));
    Node* identity2 = ops::UnaryOp("Identity", const_in,
                                   builder.opts()
                                       .WithName("identity2_outside_cluster")
                                       .WithDevice(kTestDeviceName)
                                       .WithAttr("T", DT_FLOAT));

    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(ConvertToAsyncOut(&graph, &flib_def));

  // Verify that no AsyncOut is inserted.
  for (const Node* n : graph->op_nodes()) {
    EXPECT_TRUE(!absl::StartsWith(n->type_string(), "_XlaAsyncOut"));
  }
}

}  // namespace
}  // namespace tensorflow
