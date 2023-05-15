/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <vector>

#include "tensorflow/core/graph/graph_util.h"

namespace tensorflow {
namespace graph_util {

namespace {
void MarkComputeGraph(const Graph* g, std::vector<bool>& is_var_relate) {
  // Get the starting node of the coloring algorithm
  std::queue<const Node*> q;
  for (Node* n : g->op_nodes()) {
    if (n->IsVariable() || n->IsKvVarHandle()
        || n->IsControlFlow()
        || n->type_string() == "VarHandleOp") {
      q.emplace(n);
    }
  }

  // mark compute graph node
  while (!q.empty()) {
    const Node* node = q.front();
    q.pop();
    is_var_relate[node->id()] = true;
    for (auto e : node->out_edges()) {
      if (!is_var_relate[e->dst()->id()]) {
        q.emplace(e->dst());
      }
    }
  }
}

void DealWithNode(Graph* g, Node* n,
      const std::vector<bool>& is_var_relate,
      std::queue<Node*>& queue,
      std::unordered_set<Node*>& has_visit_node,
      std::unordered_set<Node*>& boundary_node_set) {
  if (!n->IsConstant()) {
    for (auto edge : n->out_edges()) {
      Node* dst = edge->dst();
      if (is_var_relate[dst->id()]) {
        boundary_node_set.emplace(n);
      } else {
        queue.emplace(dst);
      }
    }
    return;
  }

  // classify the output edges of const op
  bool is_connect_to_marked_graph = false;
  std::unordered_set<const Edge*> data_edge_to_unmarked_graph;
  for (auto edge : n->out_edges()) {
    // skip control edge
    if (edge->IsControlEdge()) {
      continue;
    }

    Node* dst = edge->dst();
    if (is_var_relate[dst->id()]) {
      is_connect_to_marked_graph = true;
    } else {
      data_edge_to_unmarked_graph.emplace(edge);
      queue.emplace(dst);
    }
  }

  // const op connects to marked graph and unmarked graph at the same time,
  // duplicate a new const op node to avoid memcpy bewteen cpu and gpu.
  if (is_connect_to_marked_graph && !data_edge_to_unmarked_graph.empty()) {
    Node* new_node = g->CopyNode(n);
    std::string new_name(n->name() + "_duplicate");
    new_node->set_name(new_name);
    has_visit_node.emplace(new_node);

    for (auto edge : data_edge_to_unmarked_graph) {
      Node* dst_node = edge->dst();
      g->AddEdge(new_node, edge->src_output(), dst_node,
                    edge->dst_input());
      g->RemoveEdge(edge);
    }
  }
}

}  // namespace

void GetComputeGraphBoundaryNodes(
    Graph* g, std::unordered_set<Node*>& boundary_node_set) {
  // mark compute graph node
  std::vector<bool> is_var_relate(g->num_node_ids(), false);
  MarkComputeGraph(g, is_var_relate);

  // get boundary node
  std::unordered_set<Node*> has_visit_node;
  std::queue<Node*> queue;
  queue.emplace(g->source_node());
  while (!queue.empty()) {
    Node* n = queue.front();
    queue.pop();
    if (has_visit_node.find(n) != has_visit_node.end()) {
      continue;
    }

    has_visit_node.emplace(n);
    DealWithNode(g, n, is_var_relate, queue, has_visit_node,
                 boundary_node_set);
  }
}

}  // namespace graph_util
}  // namespace tensorflow
