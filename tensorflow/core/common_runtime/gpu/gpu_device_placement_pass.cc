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

#if GOOGLE_CUDA

#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class DevicePlacementPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    if (options.session_options == nullptr) {
      return Status::OK();
    }

    bool is_enable_device_placement_optimization =
      options.session_options->config.graph_options().optimizer_options()
            .device_placement_optimization();
    if (is_enable_device_placement_optimization) {
      LOG(INFO) << "Run DevicePlacement Optimization";
    } else {
      return Status::OK();
    }

    Graph* graph = options.graph->get();
    if (graph == nullptr)
      return errors::Internal("a graph should be available");

    std::unique_ptr<Graph> device_graph(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, device_graph.get());

    std::unordered_set<Node*> boundary_node_set;
    GetDevicePlacementBoundaryNodes(device_graph.get(), boundary_node_set);
    if (boundary_node_set.empty()) {
      LOG(FATAL) << "DevicePlacementOptimization: Failed to get boundary_node, "
                    "disable DevicePlacementOptimization";
      return Status::OK();
    }

    // Get CPU device name, there should find only one CPU device
    std::vector<Device*> devices = options.device_set->devices();
    std::string cpu_device_name = "";
    GetCpuDeviceName(devices, cpu_device_name);
    if (cpu_device_name.empty()) {
      LOG(FATAL) << "DevicePlacementOptimization: Failed to get CPU Device, "
                    "disable DevicePlacementOptimization";
      return Status::OK();
    }

    // Put the nodes in front of the boundary nodes on the CPU
    PlaceNodesOnCPU(cpu_device_name, boundary_node_set, device_graph.get());

    options.graph->swap(device_graph);
    return Status::OK();
  }

 private:

  void MarkComputeGraph(const Graph* dest, std::vector<bool>& is_var_relate) {
    // Get the starting node of the coloring algorithm
    std::queue<const Node*> q;
    for (Node* n : dest->op_nodes()) {
      if (n->IsVariable() || n->IsKvVarHandle() || n->IsControlFlow() ||
	  n->type_string() == "VarHandleOp")
	q.emplace(n);
    }

    // mark compute graph node
    while (!q.empty()) {
      const Node* node = q.front();
      q.pop();
      is_var_relate[node->id()] = true;
      for (auto e : node->out_edges()) {
	if (!is_var_relate[e->dst()->id()])
	  q.emplace(e->dst());
      }
    }
  }

  void DealWithNode(Graph* dest, Node* n,
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
      if (edge->IsControlEdge())
	continue;

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
      Node* new_node = dest->CopyNode(n);
      std::string new_name(n->name() + "_duplicate");
      new_node->set_name(new_name);
      has_visit_node.emplace(new_node);

      for (auto edge : data_edge_to_unmarked_graph) {
	Node* dst_node = edge->dst();
        dest->AddEdge(new_node, edge->src_output(), dst_node,
                      edge->dst_input());
	dest->RemoveEdge(edge);
      }
    }
  }

  void GetDevicePlacementBoundaryNodes(
      Graph* dest, std::unordered_set<Node*> &boundary_node_set) {
    // mark compute graph node
    std::vector<bool> is_var_relate(dest->num_node_ids(), false);
    MarkComputeGraph(dest, is_var_relate);

    // get boundary node
    std::unordered_set<Node*> has_visit_node;
    std::queue<Node*> queue;
    queue.emplace(dest->source_node());
    while (!queue.empty()) {
      Node* n = queue.front();
      queue.pop();
      if (has_visit_node.find(n) != has_visit_node.end())
	continue;

      has_visit_node.emplace(n);
      DealWithNode(dest, n, is_var_relate, queue, has_visit_node,
                   boundary_node_set);
    }
  }

  void GetCpuDeviceName(const std::vector<Device*>& devices,
                        std::string& cpu_device_name) {
    for (auto iter = devices.begin(); iter != devices.end(); iter++) {
      if ((*iter)->device_type() == DEVICE_CPU) {
	cpu_device_name = (*iter)->name();
	break;
      }
    }
  }

  void PlaceNodesOnCPU(const std::string& cpu_device_name,
      const std::unordered_set<Node*>& boundary_node_set, Graph* device_graph) {

    auto set_stage_subgraph_node_device = [cpu_device_name](Node* n) {
      n->set_assigned_device_name(cpu_device_name);
    };

    std::vector<Node* > boundary_node_vec;
    for (const auto node : boundary_node_set) {
      boundary_node_vec.emplace_back(node);
    }
    ReverseDFSFrom(*device_graph, boundary_node_vec,
                   std::move(set_stage_subgraph_node_device), nullptr);
  }

};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 0,
                      DevicePlacementPass);

} // end of namespace tensorflow

#endif // endof GOOGLE_CUDA
