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

#include "tensorflow/core/common_runtime/graph_optimizer.h"

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/optimizer_cse.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace {
void VLogGraphDebugString(Graph* g) {
  GraphDef graph_def;
  g->ToGraphDef(&graph_def);
  VLOG(1) << "Grpah: " << graph_def.DebugString();
}

// Embedding ForwardBackward Joint Optimization, should before smart-stage
class EmbeddingForwardBackwardJointOptimizationPass : public GraphOptimizationPass {
 public:
  EmbeddingForwardBackwardJointOptimizationPass() : GraphOptimizationPass() {}

  Status Run(const GraphOptimizationPassOptions& options) override {
    bool embedding_fbj_opt = false;
    tensorflow::ReadBoolFromEnvVar("TF_EMBEDDING_FBJ_OPT",
                                   /*default_val=*/false, &embedding_fbj_opt);
    if (!embedding_fbj_opt) {
      VLOG(2) << "Graph Optimization Pass TF_EMBEDDING_FBJ_OPT is off.";
      return Status::OK();
    } else {
      VLOG(2) << "Graph Optimization Pass TF_EMBEDDING_FBJ_OPT is on.";
    }

    if (options.graph == nullptr) {
      // TODO(apassos) returning OK feels weird here as we can't do anything
      // without a graph, but some tests require this.
      return Status::OK();
    }
    Graph* g = options.graph->get();
    if (g == nullptr) {
      return errors::Internal(
          "Parallel concat removal should happen before partitioning and a "
          "graph should be available.");
    }
    for (Node* node : g->op_nodes()) {
      if (node->type_string() == "KvResourceGather" ||
          node->type_string() == "KvResourceGatherV1") {
        Node* gather_indices_node = nullptr;
        TF_CHECK_OK(node->input_node(1, &gather_indices_node));
        Edge* gather_edge = nullptr;
        Edge* apply_edge = nullptr;
        Node* apply_node = nullptr;
        Status s = FindEdgeAndNode(gather_indices_node, &gather_edge,
                                   &apply_edge, &apply_node);
        if (s.ok()) {
          Node* lookup_node = nullptr;
          Node* kv_variable_handle_node = nullptr;
          TF_CHECK_OK(CreateLookupNode(gather_edge, g, &kv_variable_handle_node, &lookup_node));
          g->AddEdge(lookup_node, 0,
                     apply_edge->dst(), apply_edge->dst_input());
          g->RemoveEdge(apply_edge);

          TF_CHECK_OK(ModifyApplyNode(apply_node, g));

          Node* gather_node = gather_edge->dst();
          Node* opt_gather_node;
          const Edge* default_edge;
          gather_node->input_edge(2, &default_edge);
          DataType dtype;
          TF_RETURN_IF_ERROR(GetNodeAttr(gather_node->attrs(), "dtype", &dtype));
          DataType tkeys;
          TF_RETURN_IF_ERROR(GetNodeAttr(gather_node->attrs(), "Tkeys", &tkeys));
          TF_RETURN_IF_ERROR(NodeBuilder(gather_node->name() + "/fb_opt2",
                                         "_OPT_KvResourceCollectEmbedding")
            .Input(kv_variable_handle_node, 0)
            .Input(gather_edge->src(), 0)
            .Input(lookup_node, 0)
            .Input(default_edge->src(), default_edge->src_output())
            .Attr("dtype", dtype)
            .Attr("Tkeys", tkeys)
            .Finalize(g, &opt_gather_node));
          opt_gather_node->set_assigned_device_name_index(gather_node->assigned_device_name_index());
          for (const Edge *e : gather_node->out_edges()) {
            g->AddEdge(opt_gather_node, e->src_output(),
                       e->dst(), e->dst_input());
          }
          g->RemoveNode(gather_node);

          VLogGraphDebugString(g);
        } else {
          LOG(ERROR) << "find edge and node failed:" << s.ToString();
        }
      }
    }
    return Status::OK();
  }

  Status GetApplyOpNode(Node* unique, Node** apply_node) {
    for (const Edge *edge : unique->out_edges()) {
      if (0 != edge->src_output()) {
        continue;
      }
      if (edge->dst()->IsKvSparseApply()) {
        *apply_node = edge->dst();
      } else if (edge->dst()->IsMetadata()) {
        // ignore Shape/Size/Rank Op
      } else {
        return errors::Unknown("Operation not match.");
      }
    }
    return Status::OK();
  }

  Status FindEdgeAndNode(Node* gather_indices_node,
                         Edge** gather_edge,
                         Edge** apply_edge,
                         Node** apply_node) {
    for (const Edge *e : gather_indices_node->out_edges()) {
      if (0 == e->src_output() && (e->dst()->type_string() == "KvResourceGather"
          || e->dst()->type_string() == "KvResourceGatherV1")) {
        if (e->dst()->input_type(1) == DT_INT64) {
          *gather_edge = const_cast<Edge*>(e);
        } else {
          LOG(ERROR) << "gather_edge is not DT_INT64";
        }
      } else if (0 == e->src_output() && e->dst()->IsKvSparseApply()) {
        *apply_node = e->dst();
        *apply_edge = const_cast<Edge*>(e);
      } else if (0 == e->src_output() && e->dst()->type_string() == "Reshape") {
        Node* reshape = e->dst();
        for (const Edge *e : reshape->out_edges()) {
          if (0 == e->src_output() && e->dst()->IsUnique()) {
            TF_RETURN_IF_ERROR(GetApplyOpNode(e->dst(), apply_node));
            *apply_edge = const_cast<Edge*>(e);
          }
        }
      } else if (0 == e->src_output() && e->dst()->type_string() == "Unique") {
        TF_RETURN_IF_ERROR(GetApplyOpNode(e->dst(), apply_node));
        *apply_edge = const_cast<Edge*>(e);
      }
    }
    if (*gather_edge && *apply_edge && *apply_node) {
      return Status::OK();
    } else {
      return errors::NotFound("not found edge and node");
    }
  }

  Status CreateLookupNode(Edge* gather_edge,
                          Graph* g,
                          Node** kv_variable_handle_node,
                          Node** lookup_node) {
    Node* gather_node = gather_edge->dst();
    TF_CHECK_OK(gather_node->input_node(0, kv_variable_handle_node));
    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(gather_node->attrs(), "dtype", &dtype));
    DataType tkeys;
    TF_RETURN_IF_ERROR(GetNodeAttr(gather_node->attrs(), "Tkeys", &tkeys));
    TF_RETURN_IF_ERROR(NodeBuilder(gather_node->name() + "/fb_opt1",
                                   "_OPT_KvResourceLookupID")
      .Input(*kv_variable_handle_node, 0)
      .Input(gather_edge->src(), 0)
      .Attr("dtype", dtype)
      .Attr("Tkeys", tkeys)
      .Finalize(g, lookup_node));
    (*lookup_node)->set_assigned_device_name_index(gather_node->assigned_device_name_index());

    VLOG(1) << "create lookup_node: " << (*lookup_node)->DebugString();
    return Status::OK();
  }

  Status ModifyApplyNode(Node* node, Graph* g) {
    Node* opt_node = nullptr;
    NodeBuilder node_builder = NodeBuilder(node->name() + "/fb_opt",
                               "_OPT_" + node->type_string());
    std::vector<const Edge*> nodes(node_builder.op_def().input_arg_size());
    for (const Edge *e : node->in_edges()) {
      if (e->IsControlEdge()) {
        node_builder.ControlInput(e->src());
      } else {
        nodes[e->dst_input()] = e;
      }
    }
    for (int i = 0; i < nodes.size(); ++i) {
      node_builder.Input(nodes[i]->src(), nodes[i]->src_output());
    }
    for (const auto& node_attr : node->attrs()) {
      node_builder.Attr(node_attr.first, node_attr.second);
    }
    TF_RETURN_IF_ERROR(node_builder.Finalize(g, &opt_node));
    opt_node->set_assigned_device_name_index(node->assigned_device_name_index());

    for (const Edge *e : node->out_edges()) {
      g->AddEdge(opt_node, e->src_output(),
                 e->dst(), e->dst_input());
    }
    g->RemoveNode(node);
    return Status::OK();
  }
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 23,
                      EmbeddingForwardBackwardJointOptimizationPass);

}  // namespace
}  // namespace tensorflow
