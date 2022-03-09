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

bool FindConcatWithCast(const Context& ctx, int node_index) {
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

    for (int i = 0; i < num_nodes; ++i) {
        if (FindConcatWithCast(ctx, i)) {
            const auto* node_view = ctx.graph_view.GetNode(i);
            const auto* node_def = node_view->node();
            string op_name = node_def->op();
            std::cout << op_name << std::endl;
            std::cout << "\t Fanins: " << node_view->NumRegularFanins() << std::endl;
            std::cout << "\t Fanouts: " << node_view->NumRegularFanouts() << std::endl;
        }
    }

    *optimized_graph = mutable_item.graph;

    return Status::OK();
}

void ConcatCastFusing::Feedback(Cluster* cluster, const GrapplerItem& item,
                               const GraphDef& optimize_output, double result) {
  // Nothing to do for ConcatCastFusing.
}

}  // namespace grappler
}  // namespace tensorflow
