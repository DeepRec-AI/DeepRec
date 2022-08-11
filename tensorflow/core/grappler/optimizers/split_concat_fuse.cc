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

#include "tensorflow/core/grappler/optimizers/split_concat_fuse.h"

#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/evaluation_utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kFusedSplitConcat[] = "_FusedSplitConcat";

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

struct SplitWithConcat {
    SplitWithConcat() = default;
    SplitWithConcat(int split_id, int concat_id) 
                    : split_id(split_id), concat_id(concat_id){}
    
    int split_id = -1;
    int concat_id = -1;
};

bool FindSplitWithConcat(const Context& ctx, int node_index, SplitWithConcat* matched) {
    const auto* split_node_view = ctx.graph_view.GetNode(node_index); // split node
    if (split_node_view->NumControllingFanins() > 0 ||
        split_node_view->NumControlledFanouts() > 0) return false;
    
    const auto* node_def = split_node_view->node();
    if (node_def == nullptr) return false;
    if (!IsSplit(*node_def)) return false;
    if (split_node_view->NumRegularFanouts() < 2) return false;
    const auto& split_fanouts = split_node_view->GetRegularFanout(0);
    const auto* concat_node_view = split_fanouts[0].node_view(); // concat node
    const auto* concat_node_def = concat_node_view->node();
    if (!IsConcat(*concat_node_def)) return false;

    const SplitWithConcat pattern{node_index,
                                 concat_node_view->node_index()};
    *matched = pattern;

    return true;
}
}

SplitConcatFuse::SplitConcatFuse(RewriterConfig::Toggle opt_level,
                                 DeviceBase* cpu_device) 
                                 : opt_level_(opt_level), cpu_device_(cpu_device) {
    resource_mgr_.reset(new ResourceMgr());
}

SplitConcatFuse::SplitConcatFuse(DeviceBase* cpu_device)
                                 : SplitConcatFuse(RewriterConfig::ON, cpu_device) {}

Status AddSplitConcatFuseNode(Context* ctx,
                          int i,
                          const GraphDef* graph,
                          const SplitWithConcat& matched,
                          std::vector<bool>& invalidated_nodes,
                          std::vector<bool>& nodes_to_delete) {

    const auto* node_view = ctx->graph_view.GetNode(matched.split_id);
    const auto& fused_node = graph->node(matched.split_id);
    const auto* concat_view = ctx->graph_view.GetNode(matched.concat_id);

    VLOG(3) << "Optimizing fused Split Concat node " << SummarizeNodeDef(fused_node);

    const NodeDef& split = graph->node(matched.split_id);
    const NodeDef& concat = graph->node(matched.concat_id);
    const std::size_t split_num_inputs = node_view->NumRegularFanins();
    const int concat_num_inputs = concat_view->NumRegularFanins();
    const int split_num_fanouts = concat_view->NumRegularFanouts();

    VLOG(3) << "Fuse " << split.op() << " with Concat: "
            << " concat_name= " << concat.name();        

    NodeDef fused_op;
    fused_op.set_op(kFusedSplitConcat);
    fused_op.set_name(concat.name());
    fused_op.set_device(split.device());
    
    // Add inputs
    fused_op.add_input(split.input(0));  // 0: split_dim for split
    fused_op.add_input(split.input(1));  // 1: value
    fused_op.add_input(concat.input(concat_num_inputs - 1)); // 3: axis for concat

    auto* attrs = fused_op.mutable_attr();
    auto& split_attr = split.attr();
    auto& concat_attr = concat.attr();

    // Add attributes
    (*attrs)["num_split"] = split_attr.at("num_split"); // 0: num_split
    (*attrs)["T"] = split_attr.at("T");                 // 1: T
    (*attrs)["N"] = concat_attr.at("N");                // 2: N
    (*attrs)["Tidx"] = concat_attr.at("Tidx");          // 3: Tidx

    utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
    Status status;
    mutation->AddNode(std::move(fused_op), &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(mutation->Apply());
    invalidated_nodes[matched.concat_id] = true;
    nodes_to_delete[matched.split_id] = true;

    return Status::OK();
}

Status SplitConcatFuse::Optimize(Cluster* cluster, const GrapplerItem& item, GraphDef* optimized_graph) {
    if(cpu_device_ == nullptr){
        owned_device_.reset(new DeviceSimple());
        cpu_device_ = owned_device_.get();
    }

    GrapplerItem mutable_item = item;
    Status status;
    TF_RETURN_IF_ERROR(status);
    Context ctx(&mutable_item, &status);
    TF_RETURN_IF_ERROR(status);
    TF_RETURN_IF_ERROR(ctx.graph_view.SortTopologically(false, {}));
    const int num_nodes = item.graph.node_size();
    const GraphDef* graph = ctx.graph_view.graph();

    std::vector<bool> invalidated_nodes(num_nodes); // Nodes changed into fused op
    std::vector<bool> nodes_to_delete(num_nodes); // Fused nodes that are no longer needed

    VLOG(3) << "Before Split Concat graph rewrites: " << graph->DebugString();

    for(int i = 0; i < num_nodes; ++i){
        if (invalidated_nodes[i] || nodes_to_delete[i]) {
            continue;
        }

        SplitWithConcat fused_split_concat;
        if(FindSplitWithConcat(ctx, i, &fused_split_concat)) {
            const auto* node_view = ctx.graph_view.GetNode(i);
            const auto& fused_node = graph->node(i);
            string op_name = fused_node.op();
            TF_RETURN_IF_ERROR(AddSplitConcatFuseNode(&ctx,
                                                      i,
                                                      graph,
                                                      fused_split_concat,
                                                      invalidated_nodes,
                                                      nodes_to_delete));
        }
    }

    // Remove invalidated nodes
    utils::Mutation* mutation = ctx.graph_view.GetMutationBuilder();
    for (int i = 0; i < num_nodes; ++i){
        if(nodes_to_delete[i]) {
            mutation->RemoveNode(ctx.graph_view.GetNode(i));
        }
    }
    TF_RETURN_IF_ERROR(mutation->Apply());

    *optimized_graph = mutable_item.graph;
    VLOG(3) << "After Split Concat graph rewrites: " << optimized_graph->DebugString();

    return Status::OK();
}

void SplitConcatFuse::Feedback(Cluster* cluster, const GrapplerItem& item,
                               const GraphDef& optimized_graph, double result) {
    // Nothing to do for SplitConcatFuse
}

} // end namespace grappler
} // end namespace tensorflow
