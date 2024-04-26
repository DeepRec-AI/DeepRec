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
using DAG = std::vector<std::vector<int>>;
using Bigraph = std::vector<std::vector<bool>>;

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

DAG GraphToDAG(const Graph* g) {
  DAG dag;
  dag.resize(g->num_node_ids());
  for (auto node : g->nodes()) {
    for (auto edge : node->out_edges()) {
      int dst_id = edge->dst()->id();
      dag[node->id()].push_back(dst_id);
    }
  }

  return dag;
}

void DFS(int curr, const DAG& graph,
         std::vector<bool>& visited) {
  visited[curr] = true;
  const std::vector<int>& adjacent_nodes = graph[curr];
  for (auto n : adjacent_nodes) {
    if (!visited[n]) {
      DFS(n, graph, visited);
    }
  }
}

// TODO: Optimize the algorithm
std::vector<std::vector<bool>> GetReachableNodes(const DAG& dag) {
  std::vector<std::vector<bool>> reachable_nodes;
  int num_nodes = dag.size();
  for (int i = 0; i < num_nodes; i++) {
    std::vector<bool> reachable(num_nodes, false);
    DFS(i, dag, reachable);
    reachable[i] = false;
    reachable_nodes.push_back(std::move(reachable));
  }

  return reachable_nodes;
}

// Get minimum equivalent graph
DAG GetMEG(const DAG& dag) {
  const auto& reachable_nodes = GetReachableNodes(dag);
  int num_nodes = dag.size();
  DAG meg = dag;
  for (int i = 0; i < num_nodes; i++) {
    auto& meg_child_nodes = meg[i];
    auto& child_nodes = dag[i];
    for (auto child : child_nodes) {
      if (std::find(meg_child_nodes.begin(),
                    meg_child_nodes.end(), child) ==
          meg_child_nodes.end()) {
        continue;
      }

      for (auto another : child_nodes) {
        if (reachable_nodes[child][another]) {
          auto it = std::find(meg_child_nodes.begin(),
                              meg_child_nodes.end(), another);
          if (it != meg_child_nodes.end()) {
            meg_child_nodes.erase(it);
          }
        }
      }
    }
  }

  return meg;
}

Bigraph MEGToBigraph(const DAG& meg) {
  Bigraph bigraph;
  int num_nodes = meg.size();
  for (int i = 0; i < num_nodes; i++) {
    std::vector<bool> adjacency(num_nodes, false);
    for (auto child : meg[i]) {
      adjacency[child] = true;
    }
    bigraph.push_back(std::move(adjacency));
  }

  return bigraph;
}

Bigraph DAGToBigraph(const DAG& dag) {
  Bigraph bigraph;
  int num_nodes = dag.size();
  for (int i = 0; i < num_nodes; i++) {
    std::vector<bool> reachable(num_nodes, false);
    DFS(i, dag, reachable);
    reachable[i] = false;
    bigraph.push_back(std::move(reachable));
  }

  return bigraph;
}

DAG BuildStreamDAG(
    const DAG& dag,
    const std::vector<std::array<int, 2>>& stream_chains) {
  const auto& reachable_nodes = GetReachableNodes(dag);
  DAG stream_dag;
  for (int i = 0; i < stream_chains.size(); i++) {
    std::vector<int> ensuing_streams;
    auto chain_end = stream_chains[i][1];
    for (int j = 0; j < stream_chains.size(); j++) {
      auto chain_begin = stream_chains[j][0];
      if (reachable_nodes[chain_end][chain_begin]) {
        ensuing_streams.push_back(j);
      }
    }
    stream_dag.push_back(ensuing_streams);
  }

  return stream_dag;
}

bool FindMatching(int start, const Bigraph& graph,
                  std::vector<bool>& visited,
                  std::vector<int>& match_status) {
  int num = graph[0].size();
  for (int i = 0; i < num; i++) {
    if (graph[start][i] && !visited[i]) {
      visited[i] = true;
      int curr_match = match_status[i];
      if (match_status[i] == -1 ||
          FindMatching(curr_match, graph, visited, match_status)) {
        match_status[i] = start;
        return true;
      }
    }
  }

  return false;
}

std::vector<int> MaximumMatching(const Bigraph& graph) {
  int num = graph[0].size();
  std::vector<int> match_result(num, -1);
  int num_bigraph = graph.size();
  for (int i = 0; i < num_bigraph; i++) {
    std::vector<bool> visited(num, false);
    FindMatching(i, graph, visited, match_result);
  }

  return match_result;
}

std::tuple<std::vector<int>, std::vector<std::array<int, 2>>, int>
GetMapping(const std::vector<int>& matching) {
  int num_nodes = matching.size();
  std::vector<std::array<int, 2>> chains;
  for(int i = 0; i < num_nodes; i++) {
    auto it = std::find(matching.begin(), matching.end(), i);
    if (it == matching.end()) {
      chains.push_back({i, i});
    }
  }

  int group_num = 0;
  std::vector<int> mapping(num_nodes, -1);
  for (auto& chain : chains) {
    int group_id = group_num++;
    int curr = chain[1];
    while (true) {
      mapping[curr] = group_id;
      if (matching[curr] == -1) {
        chain[0] = curr;
        break;
      } else {
        curr = matching[curr];
      }
    }
  }

  return std::make_tuple(mapping, chains, group_num);
}

} // namesapce

void MarkStreamSubGraph(Graph* g, const MultiStreamOptions& opt) {
  // trained graph
  if (!g->IsTrainingGraph()) {
    return;
  }
  //for (Node* n : g->nodes()) {
  //  if (n->type_string() == "IsVariableInitialized" &&
  //      n->name() != "global_step/IsVariableInitialized") {
  //    return;
  //  }
  //}

  int num_streams = opt.multi_stream_num();
  MultiStreamPartitionPolicy policy = opt.partition_policy();
  if (policy == MultiStreamPartitionPolicy::EMBEDDING_GRAPH_PARTITION) {
    MarkEmbeddingGraph(g, num_streams);
  } else if (policy == MultiStreamPartitionPolicy::FULL_GRAPH_PARTITION) {
    MarkFullGraph(g, num_streams);
  } else {
    // Unrecognized policy
    return;
  }

  std::unordered_map<std::string, Node*> name_to_node;
  for (Node* n : g->nodes()) {
    name_to_node[n->name()] = n;
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

// Return stream id vector which indexed by node id
std::vector<int> GenerateNodeStreamId(const Graph* graph) {
  // Assign stream id nodes.
  const auto& dag = GraphToDAG(graph);
  const auto& meg = GetMEG(dag);
  const auto& bigraph = MEGToBigraph(meg);
  const auto& matching = MaximumMatching(bigraph);
  const auto& result = GetMapping(matching);
  std::vector<int> node_to_chain = std::get<0>(result);

  // Rematching stream, some streams can have the same id.
  const auto& stream_chains = std::get<1>(result);
  const auto& stream_dag = BuildStreamDAG(meg, stream_chains);
  const auto& stream_bigraph = DAGToBigraph(stream_dag);
  const auto& rematching = MaximumMatching(stream_bigraph);
  const auto& remapping = GetMapping(rematching);
  std::vector<int> chain_to_stream = std::get<0>(remapping);

  std::vector<int> stream_ids(node_to_chain.size(), -1);
  for (int node_id = 0; node_id < node_to_chain.size(); ++node_id) {
    stream_ids[node_id] = chain_to_stream[node_to_chain[node_id]];
  }

  return stream_ids;
}

void MarkFullGraph(Graph* g, int num_streams) {
  std::vector<int> node_stream_ids = GenerateNodeStreamId(g);

  std::unordered_map<std::string, Node*> name_to_node;
  for (Node* n : g->nodes()) {
    name_to_node[n->name()] = n;

    if (n->assigned_device_name().find("device:GPU:") ==
        std::string::npos) {
      continue;
    }

    int stream_id = node_stream_ids[n->id()] % num_streams;
    n->AddAttr("_stream_id", stream_id);

    std::string required_device =
        GetDeviceNamePrefix(n->assigned_device_name()) +
            std::to_string(stream_id);
    if (n->assigned_device_name() != required_device) {
      n->set_assigned_device_name(required_device);
    }
  }
}

void MarkEmbeddingGraph(Graph* g, int num_streams) {
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
}

}  // namespace stream_subgraph
}  // namespace tensorflow
