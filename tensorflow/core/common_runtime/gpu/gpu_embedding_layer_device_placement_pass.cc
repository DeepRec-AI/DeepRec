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

#include <string.h>
#include <vector>
#include <queue>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"

namespace tensorflow {

class EmbeddingLayerDevicePlacementPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    bool is_enable_embedding_layer_placement_optimization =
      options.session_options->config.graph_options().optimizer_options()
            .embedding_layer_device_placement_optimization();
    if (is_enable_embedding_layer_placement_optimization)
      LOG(INFO) << "Run EmbeddingLayerDevicePlacement Optimization";
    else
      return Status::OK();
    
    Graph* graph = options.graph->get();
    if (graph == nullptr)
      return errors::Internal("a graph should be available");

    std::unique_ptr<Graph> device_graph(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, device_graph.get());

    std::unordered_set<Node *> boundary_node_set;
    GetDevicePlacementBoundaryNodes(device_graph.get(), boundary_node_set);
    if (boundary_node_set.empty()) {
      LOG(FATAL) << "EmbeddingLayerDeivcePlace: Failed to get boundary_node, "
                    "disable EmbeddingLayerDevicePlacementOptimization";
      return Status::OK();
    }

    // Get CPU device name, there should find only one CPU device
    std::vector<Device *> devices = options.device_set->devices();
    std::string cpu_device_name = "";
    GetCpuDeviceName(devices, cpu_device_name);
    if (cpu_device_name.empty()) {
      LOG(FATAL) << "EmbeddingLayerPlacement: Failed to get CPU Device, "
                    "disable EmbeddingLayerDevicePlacementOptimization";
      return Status::OK();
    }

    // place embedding layer node on cpu
    PlaceEmbeddingLayerNodeOnCPU(cpu_device_name, boundary_node_set,
                                 device_graph.get());
    
    options.graph->swap(device_graph);
    return Status::OK();
  }

 private:
  void GetDevicePlacementBoundaryNodes(
      const Graph* dest, std::unordered_set<Node*>& boundary_node_set) {
    // Get the starting node of the coloring algorithm
    std::queue<const Node*> q;
    for (Node *n : dest->op_nodes()) {
      if (n->IsVariable() || n->IsKvVarHandle() || n->IsControlFlow() ||
	  n->type_string() == "VarHandleOp")
	q.push(n);
    }

    Node* source_node = dest->source_node();

    // mark compute graph node
    std::vector<bool> is_var_relate(dest->num_node_ids(), false);
    while (!q.empty()) {
      const Node *node = q.front();
      q.pop();
      is_var_relate[node->id()] = true;
      for (const Edge *e : node->out_edges()) {
	if (!is_var_relate[e->dst()->id()])
	  q.push(e->dst());
      }
    }

    // get boundary node
    std::queue<Node *> queue;
    std::unordered_set<Node *> has_visit_node;
    queue.push(source_node);
    while (!queue.empty()) {
      Node *n = queue.front();
      queue.pop();
      if (has_visit_node.find(n) != has_visit_node.end())
	continue;

      has_visit_node.insert(n);
      for (auto edge : n->out_edges()) {
	Node *dst = edge->dst();
	if (is_var_relate[dst->id()])
	  boundary_node_set.insert(n);
	else
	  queue.push(dst);
      }
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

  void PlaceEmbeddingLayerNodeOnCPU(
      const std::string& cpu_device_name,
      const std::unordered_set<Node*> boundary_node_set, Graph *device_graph) {
    
    auto set_stage_subgraph_node_device = [cpu_device_name](Node *n) {
      n->set_assigned_device_name(cpu_device_name);
    };

    std::vector<Node *> boundary_node_vec;
    for (const auto node : boundary_node_set) {
      boundary_node_vec.push_back(node);
    }
    ReverseDFSFrom(*device_graph, boundary_node_vec,
                   std::move(set_stage_subgraph_node_device), nullptr);    
  }
      
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 0,
                      EmbeddingLayerDevicePlacementPass);

} // end of namespace tensorflow

#endif // endof GOOGLE_CUDA
