/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "tensorflow/core/graph/stream_subgraph.h"

namespace tensorflow {
namespace stream_subgraph {

namespace {

bool GetColocationConstraints(const Node* node,
                              std::vector<string>* constraints) {
  return TryGetNodeAttr(node->attrs(),
                        kColocationAttrName, constraints);
}

std::string GetDeviceNamePrefix(const std::string& device_name) {
  std::string prefix_token("device:GPU:");
  auto idx = device_name.find(prefix_token);
  if (idx == std::string::npos) {
    LOG(FATAL) << "Error device name, " << device_name;
  }
  std::string device_name_prefix =
      device_name.substr(0, idx + prefix_token.length());

  return device_name_prefix;
}

} // namesapce

void MarkStreamSubGraph(Graph* g, const MultiStreamOptions& opt) {
  int num_streams = opt.multi_stream_num();
  MultiStreamPartitionPolicy policy = opt.partition_policy();

  if (policy == MultiStreamPartitionPolicy::EMBEDDING_GRAPH_PARTITION) {
    MarkEmbeddingGraph(g, num_streams);
  }
}

void MarkEmbeddingGraph(Graph* g, int num_streams) {
  bool train_graph = false;

  // trained graph
  if (!g->IsTrainingGraph()) {
    return;
  }

  //for (Node* n : g->nodes()) {
  //  if (n->type_string() == "IsVariableInitialized" &&
  //      n->name() != "global_step/IsVariableInitialized") {
  //      return;
  //  }
  //}

  std::unordered_map<std::string, Node*> name_to_node;
  // User marked subgraph
  for (Node* n : g->nodes()) {
    name_to_node[n->name()] = n;

    if (n->assigned_device_name().find("device:GPU:") == std::string::npos ||
        n->def().attr().find("_stream_id") == n->def().attr().end()) {
      continue;
    }

    int stream_id = n->def().attr().at("_stream_id").i();
    std::string required_device =
        GetDeviceNamePrefix(n->assigned_device_name()) +
            std::to_string(stream_id);
    if (n->assigned_device_name() != required_device) {
      n->set_assigned_device_name(required_device);
    }
  }


  // Colocate nodes
  std::unordered_map<Node*, std::vector<Node*>> node_colocate_childs;
  std::unordered_set<Node*> colocate_nodes;
  for (Node* n : g->nodes()) {
    std::vector<string> constraints;
    bool has_constraints = GetColocationConstraints(n, &constraints);
    if (has_constraints) {
      for (const auto& constraint : constraints) {
        node_colocate_childs[name_to_node[constraint.substr(5)]].emplace_back(n);
      }
      colocate_nodes.insert(n);
    }
  }
  std::queue<Node*> q;
  for (auto pair : node_colocate_childs) {
    if (colocate_nodes.find(pair.first) == colocate_nodes.end()) {
      q.push(pair.first);
    }
  }
  while (!q.empty()) {
    Node* curr = q.front();
    q.pop();
    for (auto& n : node_colocate_childs[curr]) {
      if (curr->assigned_device_name() !=
          n->assigned_device_name()) {
        n->set_assigned_device_name(curr->assigned_device_name());
      }
      if (node_colocate_childs.find(n) != node_colocate_childs.end()) {
        q.push(n);
      }
    }
  }

	// Copy constant op
  for (Node* n : g->nodes()) {
    std::string prefix("/device:GPU:");
    if (n->assigned_device_name().find(prefix) ==
        std::string::npos) {
      continue;
    }

    //std::vector<const Edge*> edges_to_delete;
    std::vector<const Edge*> in_edges(n->in_edges().begin(),
                                      n->in_edges().end());
    for (const Edge* e : in_edges) {
      Node* input = e->src();
      std::string in_name = input->name();
      if (input->op_def().name() == "Const" &&
          input->assigned_device_name().find(prefix) != std::string::npos &&
          input->assigned_device_name() != n->assigned_device_name()) {
        std::string dev_prefix = GetDeviceNamePrefix(n->assigned_device_name());
        std::string dev_id_str =
            n->assigned_device_name().substr(dev_prefix.length());
        std::string copy_name(in_name + "/" + dev_id_str);

        // check if it's already copied
        if (name_to_node.find(copy_name) != name_to_node.end()) {
          g->AddEdge(name_to_node[copy_name], 0, n, e->dst_input());
        } else {
          // create a new Node
          Node* copied = g->CopyNode(input);
          copied->set_name(copy_name); 
          copied->set_assigned_device_name(n->assigned_device_name());
          g->AddEdge(copied, 0, n, e->dst_input());
          name_to_node[copy_name] = copied;
        }
        g->RemoveEdge(e);
      }
    }
  }
}

}  // namespace stream_subgraph
}  // namespace tensorflow
