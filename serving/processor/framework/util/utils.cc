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

#include "serving/processor/framework/util/utils.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace processor {

std::unordered_map<std::string, bool>
GetNodesHasControlFlowInputs(const GraphDef& gdef) {
  std::unordered_map<std::string, bool> has_control_flow_input;
  for (const NodeDef& node : gdef.node()) {
    has_control_flow_input[node.name()] = false;
    for (const std::string& input : node.input()) {
      if (input[0] == '^') {
        has_control_flow_input[node.name()] = true;
        break;
      }
    }
  }

  return has_control_flow_input;
}

bool HasDynamicShapeOutput(NodeDef* node_def) {
  AttrValue attr_value = (*node_def->mutable_attr())["_output_shapes"];

  Status s = AttrValueHasType(attr_value, "list(shape)");
  if (!s.ok()) {
    s = AttrValueHasType(attr_value, "shape");
    if (!s.ok()) return true;
    for (auto d : attr_value.shape().dim()) {
      if (d.size() == -1) return true;
    }
    return false;
  }

  for (const auto& v : attr_value.list().shape()) {
    for (auto d : v.dim()) {
      if (d.size() == -1) return true;
    }
  }

  return false;
}

std::unordered_map<std::string, bool> GetNodesHasDynamicShapeMap(const GraphDef& gdef) {
  // NOTE(jiankeng.pt): should be optimized via topological sort algorithm later
  std::unordered_map<std::string, bool> output_shapes;
  for (const NodeDef& node : gdef.node()) {
    output_shapes[node.name()] = HasDynamicShapeOutput(const_cast<NodeDef*>(&node));
  }

  // True when node has dynamic input or output, false else.
  std::unordered_map<std::string, bool> result;
  for (const NodeDef& node : gdef.node()) {
    // output shapes
    if (output_shapes[node.name()]) {
      result[node.name()] = true;
    } else {
      // input shapes
      result[node.name()] = false;
      for (std::string in_name : node.input()) {
        // control flow edge
        if (in_name[0] == '^') {
          in_name = in_name.substr(1);
        }
        auto offset = in_name.find(":");
        if (offset != std::string::npos) {
          in_name = in_name.substr(0, offset);
        }
        if (!output_shapes[in_name]) continue;
        result[node.name()] = true;
        break;
      }
    }
  }

  return result;
}

// Sets any parameters not specified in a node to their defaults.
Status AddDefaultAttributes(const GraphDef& input_graph_def,
                            GraphDef* output_graph_def) {
  // Find all of the ops that are currently defined.
  std::unique_ptr<FunctionLibraryDefinition> flib_def(
      new FunctionLibraryDefinition(OpRegistry::Global(),
                                    input_graph_def.library()));
  // Works in-place, so copy over the original graph.
  *output_graph_def = input_graph_def;
  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(output_graph_def, *flib_def, 0));
  return Status::OK();
}

} // namespace processor
} // namespace tensorflow

