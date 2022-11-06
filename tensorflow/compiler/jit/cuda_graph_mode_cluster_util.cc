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

#include <unordered_map>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/xla_config_registry.h"

namespace tensorflow {

extern const char* const kCGModeClusterAttr;

namespace {
// Returns a string describing how an edge from src to dst would
// create a cycle.
string DescribeCycle(const GraphCycles* cycles, const Graph& graph, int src,
                     int dst) {
  int32 max_path_size = graph.num_node_ids() + 1;
  std::vector<int32> path(max_path_size);
  int32 path_size = cycles->FindPath(dst, src, max_path_size, path.data());
  if (path_size == 0) {
    return "";
  }

  auto node_name = [&graph](int node_id) {
    if (!FastBoundsCheck(node_id, graph.num_node_ids())) {
      return string("(null)");
    }
    auto* node = graph.FindNodeId(node_id);
    if (node == nullptr) {
      return string("(null)");
    }
    return node->name();
  };

  string description;
  absl::StrAppend(&description, "Edge from ", node_name(src), " to ",
                  node_name(dst), " would create a cycle.\n");
  path.resize(path_size);
  for (int32 node_id : path) {
    string ascii_art;
    if (node_id == dst) {
      ascii_art = "+-> ";
    } else if (node_id != src) {
      ascii_art = "|   ";
    } else {
      ascii_art = "+-- ";
    }
    absl::StrAppend(&description, ascii_art, node_name(node_id), "\n");
  }
  return description;
}

bool AlwaysForwardsRefInput(const Node& node) { return node.IsIdentity(); }

}  // namespace


absl::optional<absl::string_view> GetCGModeClusterForNode(const Node& node) {
  const AttrValue* attr_value = node.attrs().Find(kCGModeClusterAttr);
  if (attr_value == nullptr) {
    return absl::nullopt;
  }
  Status s = AttrValueHasType(*attr_value, "string");
  if (!s.ok()) {
    return absl::nullopt;
  }
  return attr_value->s();
}

void RemoveFromCGModeCluster(NodeDef* node_def) {
  node_def->mutable_attr()->erase(kCGModeClusterAttr);
}

void RemoveFromCGModeCluster(Node* node) { node->ClearAttr(kCGModeClusterAttr); }

}  // namespace tensorflow
