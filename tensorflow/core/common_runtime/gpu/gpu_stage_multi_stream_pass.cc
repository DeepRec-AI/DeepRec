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

#include <unordered_set>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/algorithm.h"

namespace tensorflow {

class StageMultiStreamPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    if (options.session_options == nullptr) {
      return Status::OK();
    }
    
    bool is_enable_stage_multi_stream =
        options.session_options->config.graph_options()
            .optimizer_options().stage_multi_stream();
    if (is_enable_stage_multi_stream) {
      LOG(INFO) << "Run StageMultiStream Optimization";
    } else {
      return Status::OK();
    }

    Graph* graph = options.graph->get();
    if (graph == nullptr)
      return errors::Internal("a graph should be available.");

    std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, new_graph.get());

    // Get Stage Put node from graph.
    std::unordered_set<Node *> stage_put_node_set;
    GetStagePutNodes(new_graph.get(), stage_put_node_set);
    if (stage_put_node_set.empty()) {
      LOG(WARNING) << "Failed to get stage put node. "
	           << "StageMultiStream Optimization is disabled.";
      return Status::OK();
    }

    // Set 'stream_id' attribute for Stage subgraph,
    // the attribute value comes from the Stage Put node
    SetStreamIdForStageSubGraph(new_graph.get(), stage_put_node_set);

    options.graph->swap(new_graph);
    return Status::OK();
  }
  
 private:
  void GetStagePutNodes(const Graph* dest,
			std::unordered_set<Node *>& stage_put_node_set) {
    for (Node *n : dest->op_nodes()) {
      if (n->IsStage())
	stage_put_node_set.insert(n);
    }
  }

  void SetStreamIdForStageSubGraph(Graph* graph,
      std::unordered_set<Node*>& stage_put_node_set) {
    for (Node *node : stage_put_node_set) {
      // try to get stream id attribute from stage put node.
      const AttrValue *gpu_stream_idx_attr = node->attrs().Find("_stream_id");
      if (gpu_stream_idx_attr != nullptr) {
        int gpu_stream_idx = gpu_stream_idx_attr->i();
	auto set_stream_id_for_node = [gpu_stream_idx](Node *n) {
	  n->AddAttr("_stream_id", gpu_stream_idx);
	};
	ReverseDFSFrom(*graph, {node}, std::move(set_stream_id_for_node), nullptr);
      }        
    }
  }
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 25,
                      StageMultiStreamPass);

} // end of namespace tensorflow

#endif // end of GOOGLE_CUDA
