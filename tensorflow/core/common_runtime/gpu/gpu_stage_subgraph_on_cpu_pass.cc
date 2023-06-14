/* Copyright 2023 The DeepRec Authors. All Rights Reserved.

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

#include <string>
#include <unordered_map>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class StageSubGraphOnCPUPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    if (options.session_options == nullptr) {
      return Status::OK();
    }

    bool is_enable_stage_subgraph_on_cpu =
        options.session_options->config.graph_options()
            .optimizer_options().stage_subgraph_on_cpu();
    if (is_enable_stage_subgraph_on_cpu) {
      LOG(INFO) << "Run StageSubGraphOnCPU Optimization";
    } else {
      return Status::OK();
    }

    Graph* graph = options.graph->get();
    if (graph == nullptr)
      return errors::Internal("a graph should be available.");

    std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, new_graph.get());

    // Place Stage SubGraph on CPU.
    PlaceStageSubGraphOnCPU(new_graph.get());

    options.graph->swap(new_graph);
    return Status::OK();
  }

 private:

  void PlaceStageSubGraphOnCPU(Graph* graph) {
    for (Node* n : graph->op_nodes()) {
      if (n->IsStage()) {
	std::vector<Node*> start_node;
	for (const Edge* e : n->in_edges())
	  start_node.emplace_back(e->src());

	auto set_stage_subgraph_node_device = [](Node *node) {
          std::string cpu_device_name;
          TF_CHECK_OK(DeviceNameUtils::DeviceNameToCpuDeviceName(
              node->assigned_device_name(), &cpu_device_name));
	  node->set_assigned_device_name(cpu_device_name);
	};
        ReverseDFSFrom(*graph, start_node,
                       std::move(set_stage_subgraph_node_device), nullptr);
      }
    }
  }
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 25,
                      StageSubGraphOnCPUPass);

} // End of namespace tensorflow

#endif // End of GOOGLE_CUDA
