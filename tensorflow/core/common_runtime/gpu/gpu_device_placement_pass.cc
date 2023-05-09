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
#include "tensorflow/core/graph/graph_util.h"
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
    graph_util::GetComputeGraphBoundaryNodes(device_graph.get(), boundary_node_set);
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
