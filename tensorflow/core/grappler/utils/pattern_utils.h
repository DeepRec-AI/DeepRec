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

#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_PATTERN_HELPER_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_PATTERN_HELPER_H_

#include "tensorflow/core/grappler/utils/graph_view.h"

namespace tensorflow {
namespace grappler {
namespace utils {

// Pattern matcher recursively matches child subpatterns. The direction
// for children could be toward node's input (fanins) or outputs (fanouts).
enum class MatchingDirection { kFollowInputs, kFollowOutputs };

// Action for each node in the set of matched nodes for a given pattern.
enum class NodeStatus { kRemain, kRemove, kReplace };

// A pattern is a DAG of op types having a single root. Child subgraphs of the
// root are subpatterns. For example, the following pattern syntax describes a
// pattern for _FusedConv2D. Note that "*" means any type of op.
//
//  {"Relu",
//    {
//      "BiasAdd",
//      {
//        {"Conv2D"},
//        {"*"}
//      }
//    }
//  }
//
// In order to do better handling, we augment the aforementioned DAG
// with labels and status. Labels are strings
// to give a name (arbitrary) to each of the matched node. This enables
// identifying each matched node by name, which is convenient than node indices
// of the global graph. If the matched nodes need to be distinct (typically in
// matched tree) in the global graph, distinct label needs to be given in the
// pattern syntax. If a matched node belongs to two parents, same label has to
// be mentioned (typically when matched subgraph is not a tree) in the pattern
// syntax. In this regard, lables help matching a subgprah which is a DAG but
// not a tree. Status helps Subgraph pattern matcher to make sure that node
// marked for removal does not have a dependent outside the matched subgraph.
// Here is more concrete example of pattern syntax that can be found in residual
// blocks in deep learning models. Note the same name "my_residual_input" is
// used to tell that it is a child of both "AddV2" and "Conv2D".
//
//  {"AddV2", "my_add", NodeStatus::kReplace,
//    {
//      {"*", "my_residual_input", NodeStatus::kRemain},
//      {"BiasAdd", "my_bias_add", NodeStatus::kRemove,
//        {
//          {"Conv2D", "my_conv", NodeStatus::kRemove,
//            {
//              {"*", "my_residual_input", NodeStatus::kRemain},
//              {"*", "my_filter", NodeStatus::Remain}
//            }
//          },
//          {"*", my_bias", NodeStatus::kRemain}
//        }
//      }
//    }
//  }
//
// TODO (intel-tf): Support multiple roots by making them children of a single
// virtual root.
struct OpTypePattern {
  string op;
  string label;
  NodeStatus node_status;
  std::vector<OpTypePattern> children;

  string DebugString() const {
    string result = "{(op: " + op + ", " + "label: " + label + "), {";
    for (const OpTypePattern& child : children) {
      result += child.DebugString() + ",";
    }
    result += "}}";
    return result;
  }
};

// This is a helpful recursive structure that keeps one-to-one mapping of
// pattern syntax to the matched nodes. User can call DebugString to see what
// has been matched so far and where is the failing point.
struct NodeViewMatch {
  MutableNodeView* node_view = nullptr;
  std::vector<NodeViewMatch> children;

  string DebugString() const {
    string result = "{";
    if (node_view == nullptr) {
      result += "Non-Matched-Node}";
      return result;
    } else {
      result += node_view->node()->DebugString();
      result += ", {";
      for (const NodeViewMatch& child : children) {
        result += child.DebugString() + ",";
      }
      result += "}}";
      return result;
    }
  }

  void Clear() {
    for (NodeViewMatch& child : children) {
      child.Clear();  // child is an object.
    }
    children.clear();  // children is a vector.
    if (node_view != nullptr) {
      node_view = nullptr;
    }
  }
};

template <MatchingDirection DIRECTION = MatchingDirection::kFollowInputs>
class SubGraphMatcher {
 public:
  SubGraphMatcher(MutableGraphView* graph_view) : graph_view_(graph_view){};

  // If a given pattern is matched, this function returns true as well as the
  // matched node and remove node info is populated.
  bool GetMatchedNodes(const OpTypePattern& pattern, MutableNodeView* node_view,
                       std::map<string, int>* matched_nodes_map,
                       std::set<int>* remove_node_indices);

 private:
  MutableGraphView* graph_view_;
  std::map<string, int> node_label_to_index_;
  std::set<int> matched_node_indices_;
  std::set<int> remove_node_indices_;
  std::unique_ptr<NodeViewMatch> match_ = nullptr;

  bool DoesOpTypePatternMatch(const OpTypePattern& pattern,
                              MutableNodeView* node_view, NodeViewMatch* match);

  // This function should be called after the pattern matcher has found
  // potential matched nodes (i.e. when DoesOpTypePatternMatch returns "true").
  // It performs a sanity check if the candidate nodes for removal in subgraph
  // fusion is indeed safe to remove.
  bool HasRemoveNodeExternalDependents() {
    for (const auto& node_idx : remove_node_indices_) {
      auto node_view = graph_view_->GetNode(node_idx);
      // Traverse all the Regular Fanouts. Fanouts are stored as vector of
      // vector, std::vector<std::vector<MutableFaninView>>. Note that
      // a MutableNodeView's fanouts are stored in a nested vector of
      // MutableFaninView type.
      auto fanouts_by_ports = node_view->GetRegularFanouts();
      for (const auto& fanouts : fanouts_by_ports) {
        for (const auto& fanout : fanouts) {
          if (!matched_node_indices_.count(fanout.node_index())) {
            return true;
          }
        }
      }
    }
    return false;
  }
};

}  // namespace utils
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_UTILS_PATTERN_HELPER_H_
