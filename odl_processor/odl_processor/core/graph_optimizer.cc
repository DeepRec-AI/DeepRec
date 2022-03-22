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

#include <queue>
#include <unordered_set>

#include "odl_processor/core/graph_optimizer.h"
#include "odl_processor/core/util/utils.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

namespace tensorflow {
namespace odl_processor {

namespace {

std::unordered_set<std::string> GetBlackOpsSet() {
  std::unordered_set<std::string> s = {
    "Unique",
    "SparseSegmentMean",
    "SparseSegmentMeanGrad",
    "SparseSegmentSum",
    "SparseSegmentSumGrad",
    "SparseSegmentSqrt",
    "SparseSegmentSqrtNGrad",
    "UnsortedSegmentSum",
    "UnsortedSegmentSumGrad",
    "UnsortedSegmentMax",
    "UnsortedSegmentMaxGrad",
    "HashTableV2",
    "LookupTableFindV2",
    "ParseExample",
    "Bucketize"
    // Add other ops
  };

  return s;
}

} // namespace

MetaGraphDef GetMetaGraphDefFromSavedModel(
    const std::string& saved_model_str) {
  GraphDef gdef;
  SavedModel saved_model;
  if (!saved_model.ParseFromString(saved_model_str)) {
    LOG(FATAL) << "Can not parse saved model from string.";
  }

  // TODO: How about many meta graphs?
  MetaGraphDef meta_gdef = saved_model.meta_graphs()[0];

  return meta_gdef;
}


void StaticShapeCluteringStrategy::Run(
    const GraphDef& gdef,
    std::vector<std::string>& inputs,
    std::vector<std::string>& outputs,
    ClusteredGraphInfo* clustered_graph_info) {
  std::unordered_map<std::string, bool> dynamic_ops_map =
      GetNodesHasDynamicShapeMap(gdef);

  std::unordered_map<std::string, bool> has_control_flow_input =
      GetNodesHasControlFlowInputs(gdef);

  std::unordered_map<std::string, const NodeDef*> nodes;
  for (const NodeDef& node : gdef.node()) {
    nodes[node.name()] = &node;
  }

  GraphDef& dynamic_graph_def = clustered_graph_info->tf_subgraph;
  GraphDef& static_graph_def = clustered_graph_info->iree_subgraph;

  std::unordered_set<const NodeDef*> static_nodes;
  std::unordered_set<const NodeDef*> visited;
  std::queue<const NodeDef*> q;
  for (auto output : outputs) {
    q.push(nodes[output]);
    visited.insert(nodes[output]);
  }

  std::unordered_set<std::string> black_ops = GetBlackOpsSet();
  while (!q.empty()) {
    const NodeDef* curr_node = q.front();
    // 1) no control edge
    // 2) no dynamic shape
    // 3) no blacklist ops
    if (!has_control_flow_input[curr_node->name()] &&
        !dynamic_ops_map[curr_node->name()] &&
        black_ops.find(curr_node->op()) == black_ops.end()) {
      // Add op into static_graph_def
      NodeDef* new_node = static_graph_def.add_node();
      new_node->CopyFrom(*curr_node);
      static_nodes.insert(curr_node);

      for (auto in_name : curr_node->input()) {
        size_t offset = in_name.find(":");
        in_name = in_name.substr(0, offset);
        if (visited.find(nodes[in_name]) == visited.end()) {
          q.push(nodes[in_name]);
          visited.insert(nodes[in_name]);
        }
      }
    }

    q.pop();
  }

  for (const NodeDef& node : gdef.node()) {
    if (static_nodes.find(&node) == static_nodes.end()) {
      NodeDef* new_node = dynamic_graph_def.add_node();
      new_node->CopyFrom(node);
    }
  }

  // TODO: tf_signature ???
}

void StaticShapeCluteringStrategy::Run(
    const MetaGraphDef& mgdef,
    ClusteredGraphInfo* clustered_graph_info) {
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;

  // TODO: FIXME consider several signature_defs here
  // Now only consider one signature_def
  for (auto sdef : mgdef.signature_def()) {
    for (auto input : sdef.second.inputs()) {
      // convert "input_example_tensor:0" to "input_example_tensor"
      std::string name = input.second.name();
      size_t offset = name.find(":");
      inputs.push_back(name.substr(0, offset));
    }

    for (auto output : sdef.second.outputs()) {
      // convert "Reshape_2:0" to "Reshape_2"
      std::string name = output.second.name();
      size_t offset = name.find(":");
      outputs.push_back(name.substr(0, offset));
    }

    // TODO: FIXME
    break;
  }

  Run(mgdef.graph_def(), inputs, outputs,
      clustered_graph_info);
}

ClusteredGraphInfo ClusteringGraphDef(
    const MetaGraphDef& mgdef,
    CluteringStrategy* cluster_strategy) {
  static StaticShapeCluteringStrategy static_strategy;
  if (cluster_strategy == nullptr) {
    cluster_strategy = &static_strategy;
  }

  ClusteredGraphInfo info;
  cluster_strategy->Run(mgdef, &info);

  return info;
}

} // namespace odl_processor
} // namespace tensorflow

