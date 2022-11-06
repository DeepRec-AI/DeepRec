/* Copyright 2022 The DeepRec Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/cuda_graph_mode_cluster_util.h"
#include "tensorflow/compiler/jit/mark_for_cuda_graph_mode_pass_test_helper.h"

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/list_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/node_matchers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/graph_def_builder_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

using ::tensorflow::testing::FindNodeByName;

namespace tensorflow {
namespace {

REGISTER_OP("UncompilableNullary").Output("o: float");
REGISTER_OP("UncompilableUnary").Input("a: float").Output("o: float");

std::unordered_map<string, string> GetClusters(const Graph& graph) {
  std::unordered_map<string, string> ids;
  for (Node* node : graph.nodes()) {
    string cluster;
    if (TryGetNodeAttr(node->attrs(), kCGModeClusterAttr, &cluster)) {
      CHECK(!cluster.empty());
      ids[node->name()] = cluster;
    }
  }

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Clusters:";
    for (const auto& p : ids) {
      VLOG(2) << " " << p.first << " -> " << p.second;
    }
  }
  return ids;
}

absl::flat_hash_map<string, std::vector<string>> GetClusterSets(
    const Graph& g, std::vector<string>* cluster_names = nullptr) {
  CHECK(cluster_names == nullptr || cluster_names->empty());
  absl::flat_hash_map<string, std::vector<string>> cluster_sets;
  for (const auto& p : GetClusters(g)) {
    cluster_sets[p.second].push_back(p.first);
  }
  for (auto& p : cluster_sets) {
    if (cluster_names != nullptr) {
      cluster_names->push_back(p.first);
    }
    std::sort(p.second.begin(), p.second.end());
  }
  if (cluster_names != nullptr) {
    std::sort(cluster_names->begin(), cluster_names->end());
  }
  return cluster_sets;
}

TEST(CudaGraphModeCompilationTest, Chains) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* a =
        ops::SourceOp("UncompilableNullary", builder.opts().WithName("A"));
    Node* b = ops::UnaryOp("Relu", a, builder.opts().WithName("B"));
    Node* c = ops::UnaryOp("Relu", b, builder.opts().WithName("C"));
    Node* d =
        ops::UnaryOp("UncompilableUnary", c, builder.opts().WithName("D"));
    Node* e = ops::UnaryOp("Relu", d, builder.opts().WithName("E"));
    ops::UnaryOp("Relu", e, builder.opts().WithName("F"));
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(MarkForCudaGraphModePassTestHelper::MarkForCompilation(&graph));
  auto clusters = GetClusters(*graph);
  EXPECT_EQ(6, clusters.size());
  EXPECT_EQ(clusters["B"], clusters["C"]);
  EXPECT_EQ(clusters["E"], clusters["F"]);
  EXPECT_EQ(clusters["B"], clusters["E"]);
}
}  // namespace
}  // namespace tensorflow
