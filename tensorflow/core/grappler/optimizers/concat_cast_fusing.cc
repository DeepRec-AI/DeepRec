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

#define EIGEN_USE_THREADS

#include "tensorflow/core/grappler/optimizers/concat_cast_fusing.h"

#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/evaluation_utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace grappler {
namespace {
struct Context {
  explicit Context(GrapplerItem* item, Status* status)
      : nodes_to_preserve(item->NodesToPreserve()),
        graph_view(&item->graph, status),
        graph_properties(*item),
        inferred_graph_properties(false) {}

  std::unordered_set<string> nodes_to_preserve;
  utils::MutableGraphView graph_view;
  GraphProperties graph_properties;
  bool inferred_graph_properties;
};

struct ConcatWithCast {
  ConcatWithCast() = default;
  ConcatWithCast(int concat_id, int cast_id)
      : concat_id(concat_id), cast_id(cast_id) {}

  int concat_id = -1;
  int cast_id = -1;
};

bool FindConcatWithCast(const Context& ctx, int node_index, ConcatWithCast* matched) {
    const auto* concat_node_view = ctx.graph_view.GetNode(node_index);

    if (concat_node_view->NumControllingFanins() > 0 || concat_node_view->NumControlledFanouts() > 0) return false;

    const auto* node_def = concat_node_view->node();
    if (node_def == nullptr) return false;
    if (!IsConcat(*node_def)) return false;

    //TODO: MKL Concat _MklConcatV2 can have 2 outputs
    if (concat_node_view->NumRegularFanouts() != 1) return false;
    const auto& concat_fanouts = concat_node_view->GetRegularFanout(0);
    // If concat's output is connected with more than one op, don't fuse
    if (concat_fanouts.size() > 1) return false;
    const auto* cast_node_view = concat_fanouts[0].node_view();
    const auto* cast_node_def = cast_node_view->node();
    if (!IsCast(*cast_node_def)) return false;

    const ConcatWithCast pattern{node_index,
                                 cast_node_view->node_index()};
    *matched = pattern;

    return true;
}
}
//TODO: work on constructors
ConcatCastFusing::ConcatCastFusing(RewriterConfig::Toggle opt_level,
                                 DeviceBase* cpu_device)
    : opt_level_(opt_level), cpu_device_(cpu_device) {
  resource_mgr_.reset(new ResourceMgr());
}

ConcatCastFusing::ConcatCastFusing(DeviceBase* cpu_device)
    : ConcatCastFusing(RewriterConfig::ON, cpu_device) {}

Status ConcatCastFusing::Optimize(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* optimized_graph) {
    if (cpu_device_ == nullptr) {
        owned_device_.reset(new DeviceSimple());
        cpu_device_ = owned_device_.get();
    }

    GrapplerItem mutable_item = item;
    Status status;
    TF_RETURN_IF_ERROR(status);
    //TODO: change context to just graph_view
    //utils::MutableGraphView graph_view = MutableGraphView(mutable_item.graph, &status);
    Context ctx(&mutable_item, &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
    const int num_nodes = item.graph.node_size();
    // invalidated_nodes - nodes that have been changed into a fused op
    // nodes_to_delete -  nodes that were fused into a fused op and are not needed anymore
    std::vector<bool> invalidated_nodes(num_nodes);
    std::vector<bool> nodes_to_delete(num_nodes);
    const GraphDef* graph = ctx.graph_view.graph();

    VLOG(3) << "Before concat cast graph rewrites: " << graph->DebugString();

    for (int i = 0; i < num_nodes; ++i) {
        if (invalidated_nodes[i] || nodes_to_delete[i]) {
            continue;
        }

        ConcatWithCast base;
        if (FindConcatWithCast(ctx, i, &base)) {
            const auto* node_view = ctx.graph_view.GetNode(i);
            const auto& fused_node = graph->node(i);
            VLOG(2) << "Optimizing fused concat cast node " << SummarizeNodeDef(fused_node);

            // TODO: add if for cases when src dtype == dst dtype
            // Adding fused concat+cast
            const NodeDef& concat = graph->node(base.concat_id);
            const NodeDef& cast = graph->node(base.cast_id);
            const std::size_t concat_num_inputs = node_view->NumRegularFanins();
            VLOG(2) << "Fuse " << concat.op() << " with Cast: "
                    << " cast_name=" << cast.name();

            NodeDef fused_op;
            fused_op.set_name(cast.name());
            fused_op.set_op("_FusedConcatCast");
            fused_op.set_device(concat.device());
            for (int j = 0; j < concat_num_inputs - 1; ++j)
                fused_op.add_input(concat.input(j));  // inputs
            fused_op.add_input(concat.input(concat_num_inputs - 1));  // axis

            auto* attr = fused_op.mutable_attr();
            auto& concat_attr = concat.attr();
            auto& cast_attr = cast.attr();
            //(*attr)["T"] = concat_attr.at("T");
            (*attr)["N"] = concat_attr.at("N");

            (*attr)["SrcT"] = cast_attr.at("SrcT");
            (*attr)["DstT"] = cast_attr.at("DstT");

            utils::Mutation* mutation = ctx.graph_view.GetMutationBuilder();
            Status status;
            mutation->AddNode(std::move(fused_op), &status);
            TF_RETURN_IF_ERROR(status);
            TF_RETURN_IF_ERROR(mutation->Apply());
            invalidated_nodes[i+1] = true;
            nodes_to_delete[i] = true;
        }
    }

    // Remove not needed nodes
    utils::Mutation* mutation = ctx.graph_view.GetMutationBuilder();
    for (int i = 0; i < num_nodes; ++i) {
        if (nodes_to_delete[i]) {
            mutation->RemoveNode(ctx.graph_view.GetNode(i));
        }
    }
    TF_RETURN_IF_ERROR(mutation->Apply());
    *optimized_graph = mutable_item.graph;

    VLOG(3) << "After concat cast graph rewrites: " << optimized_graph->DebugString();

    return Status::OK();
}

void ConcatCastFusing::Feedback(Cluster* cluster, const GrapplerItem& item,
                               const GraphDef& optimize_output, double result) {
  // Nothing to do for ConcatCastFusing.
}

}  // namespace grappler
}  // namespace tensorflow
