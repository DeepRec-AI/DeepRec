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
#include <vector>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"

namespace tensorflow {

class StagePackTransPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    bool is_enable = false;
    Status s;

    s = ReadBoolFromEnvVar("ENABLE_STAGE_PACK_TRANS", false, &is_enable);
    TF_CHECK_OK(s);
    if (!is_enable)
      return Status::OK();

    VLOG(1) << "Run StagePackTrans Optimization";

    Graph* graph = options.graph->get();
    if (graph == nullptr)
      return errors::Internal("a graph should be available.");
    std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, new_graph.get());

    std::unordered_set<Node*> stage_nodes;
    GetStageNodeOnGPU(new_graph, stage_nodes);

    for (Node* n : stage_nodes) {
      AddPackTransOp(new_graph, n);
    }

    options.graph->swap(new_graph);
    return Status::OK();
  }

 private:
  bool IsNodePlacedOnGPU(const Node* n) {
    if (n->assigned_device_name().find("GPU") != std::string::npos)
      return true;

    return false;
  }

  void GetStageNodeOnGPU(const std::unique_ptr<Graph>& graph,
                         std::unordered_set<Node*>& stage_nodes) {
    for (Node* n : graph->op_nodes()) {
      if (n->IsStage() && IsNodePlacedOnGPU(n)) {
	stage_nodes.emplace(n);
      }
    }
  }

  void GetEdgeNeedH2D(std::unique_ptr<Graph>& graph, Node* stage_node,
		      std::vector<const Edge*>& edges_vec,
		      std::vector<DataType>& dtype_vec,
		      std::vector<NodeDefBuilder::NodeOut>& src_list) {
    Status s;
    for (const Edge* e : stage_node->in_edges()) {
      if (e->IsControlEdge())
	continue;

      Node* src_node = e->src();
      // for src node placed on GPU, we need to check its output memtype.
      if (IsNodePlacedOnGPU(src_node)) {
	MemoryType src_output_memtype;
        s = MemoryTypeForOutput(DEVICE_GPU, graph.get(), src_node,
                                       e->src_output(), &src_output_memtype);
	TF_CHECK_OK(s);

	// skip the edge if src_output is already on device memory
	if (src_output_memtype == DEVICE_MEMORY)
	  continue;
      }

      // skip edge if src_output DataTypeSize is 0, because TensorPackTransH2DOp
      // can not handle it.
      DataType src_output_dtype = src_node->output_type(e->src_output());
      if (DataTypeSize(src_output_dtype) == 0)
	continue;

      edges_vec.emplace_back(e);
      dtype_vec.emplace_back(src_output_dtype);
      src_list.emplace_back(src_node->name(), e->src_output(),
                            src_output_dtype);
    }
  }

  void AddPackTransOp(std::unique_ptr<Graph>& graph, Node* stage_node) {
    // Find out the input edge which src output is placed on HOST_MEMORY
    std::vector<const Edge*> edges_vec;
    std::vector<DataType> dtype_vec;
    std::vector<NodeDefBuilder::NodeOut> src_list;
    GetEdgeNeedH2D(graph, stage_node, edges_vec, dtype_vec, src_list);
    if (edges_vec.size() == 0)
      return;

    NodeDef pack_h2d_ndef;
    std::string pack_h2d_name = stage_node->name() + "/TensorPackTransH2D";
    TF_CHECK_OK(NodeDefBuilder(pack_h2d_name,  "_TensorPackTransH2D")
		.Input(src_list)
		.Device(stage_node->assigned_device_name())
		.Attr("dtypes", DataTypeSlice(dtype_vec))
		.Finalize(&pack_h2d_ndef));
    Status s;
    Node* pack_h2d_node = graph->AddNode(pack_h2d_ndef, &s);
    TF_CHECK_OK(s);

    // Update Edge
    for (int i = 0; i < edges_vec.size(); i++) {
      const Edge* e = edges_vec[i];
      graph->AddEdge(e->src(), e->src_output(), pack_h2d_node, i);
      graph->UpdateEdge(pack_h2d_node, i, stage_node, e->dst_input());
    }
  }
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 26,
                      StagePackTransPass);

} // end of namespace tensorflow

#endif // end of GOOGLE_CUDA
