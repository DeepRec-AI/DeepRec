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

#define EIGEN_USE_THREADS

#include "tensorflow/core/grappler/optimizers/concat_cast_fusing.h"

#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/evaluation_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"

namespace tensorflow {
namespace grappler {
namespace {
struct ConcatWithCast {
  ConcatWithCast() = default;
  ConcatWithCast(int concat_id, int cast_id)
      : concat_id(concat_id), cast_id(cast_id) {}

  int concat_id = -1;
  int cast_id = -1;
};

DataType supported_types[] = {DataType::DT_FLOAT, DataType::DT_BFLOAT16};

bool FindConcatWithCast(const utils::MutableGraphView& graph_view,
    int node_index, ConcatWithCast* matched) {
  const auto* concat_node_view = graph_view.GetNode(node_index);

  if (concat_node_view->NumControllingFanins() > 0 ||
      concat_node_view->NumControlledFanouts() > 0) {
    return false;
  }

  const auto* node_def = concat_node_view->node();
  if (node_def == nullptr) return false;
  if (!IsConcat(*node_def)) return false;
  if (NodeIsOnGpu(node_def)) return false;

  if (concat_node_view->NumRegularFanouts() != 1) return false;
  const auto& concat_fanouts = concat_node_view->GetRegularFanout(0);
  // If concat's output is connected with more than one op, don't fuse
  if (concat_fanouts.size() > 1) return false;
  const auto* cast_node_view = concat_fanouts[0].node_view();
  const auto* cast_node_def = cast_node_view->node();
  if (!IsCast(*cast_node_def)) return false;

  auto& cast_attr = cast_node_def->attr();
  DataType srcT = cast_attr.at("SrcT").type();
  DataType dstT = cast_attr.at("DstT").type();
  bool src_exists = std::find(std::begin(supported_types),
      std::end(supported_types), srcT) != std::end(supported_types);
  bool dst_exists = std::find(std::begin(supported_types),
      std::end(supported_types), dstT) != std::end(supported_types);
  if (!src_exists || !dst_exists) {
    VLOG(2) << "ConcatCastFusion does not support following conversion: "
            << srcT << " -> " << dstT;
    return false;
  }
  if (srcT == dstT) {
    VLOG(2) << "ConcatCastFusion does not support conversion: "
            << srcT << " -> " << dstT
            << " when SrcT equals DstT";
    return false;
  }

  const ConcatWithCast pattern{node_index,
                               cast_node_view->node_index()};
  *matched = pattern;

  return true;
}
}
Status ConcatCastFusing::Optimize(Cluster* cluster, const GrapplerItem& item,
                                  GraphDef* optimized_graph) {
  Status status;
  *optimized_graph = item.graph;
  TF_RETURN_IF_ERROR(status);
  utils::MutableGraphView graph_view(optimized_graph, &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  const int num_nodes = item.graph.node_size();
  // invalidated_nodes - nodes that have been changed into a fused op
  // nodes_to_delete -  nodes that were fused into a fused op and are not needed anymore
  std::vector<bool> invalidated_nodes(num_nodes);
  std::vector<bool> nodes_to_delete(num_nodes);
  const GraphDef* graph = graph_view.graph();

  VLOG(3) << "Before concat cast graph rewrites: " << graph->DebugString();

  for (int i = 0; i < num_nodes; ++i) {
    if (invalidated_nodes[i] || nodes_to_delete[i]) {
      continue;
    }

    ConcatWithCast base;
    if (FindConcatWithCast(graph_view, i, &base)) {
      const auto* node_view = graph_view.GetNode(i);
      const auto& fused_node = graph->node(i);
      VLOG(2) << "Optimizing fused concat cast node " << SummarizeNodeDef(fused_node);

      // Adding fused concat+cast
      const NodeDef& concat = graph->node(base.concat_id);
      const NodeDef& cast = graph->node(base.cast_id);
      const std::size_t concat_num_inputs = node_view->NumRegularFanins();
      VLOG(2) << "Fuse " << concat.op() << " with Cast: "
              << " cast_name=" << cast.name();

      NodeDef fused_op;
      fused_op.set_name(cast.name());
      fused_op.set_op("FusedConcatCast");
      fused_op.set_device(concat.device());
      for (size_t j = 0; j < concat_num_inputs - 1; ++j) {
        fused_op.add_input(concat.input(j));  // inputs
      }
      fused_op.add_input(concat.input(concat_num_inputs - 1));  // axis

      auto* attr = fused_op.mutable_attr();
      auto& concat_attr = concat.attr();
      auto& cast_attr = cast.attr();
      (*attr)["N"] = concat_attr.at("N");
      (*attr)["Tidx"] = concat_attr.at("Tidx");

      (*attr)["SrcT"] = cast_attr.at("SrcT");
      (*attr)["DstT"] = cast_attr.at("DstT");
      (*attr)["Truncate"] = cast_attr.at("Truncate");

      utils::Mutation* mutation = graph_view.GetMutationBuilder();
      Status status;
      mutation->AddNode(std::move(fused_op), &status);
      TF_RETURN_IF_ERROR(status);
      TF_RETURN_IF_ERROR(mutation->Apply());
      invalidated_nodes[i+1] = true;
      nodes_to_delete[i] = true;
    }
  }

  // Remove not needed nodes
  utils::Mutation* mutation = graph_view.GetMutationBuilder();
  for (int i = 0; i < num_nodes; ++i) {
    if (nodes_to_delete[i]) {
      mutation->RemoveNode(graph_view.GetNode(i));
    }
  }
  TF_RETURN_IF_ERROR(mutation->Apply());
  *optimized_graph = *graph_view.graph();

  VLOG(3) << "After concat cast graph rewrites: " << optimized_graph->DebugString();
  return Status::OK();
}

void ConcatCastFusing::Feedback(Cluster* cluster, const GrapplerItem& item,
                               const GraphDef& optimize_output, double result) {
  // Nothing to do for ConcatCastFusing.
}

}  // namespace grappler
}  // namespace tensorflow
