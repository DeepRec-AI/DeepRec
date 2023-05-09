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

#include <vector>
#include <string>
#include <queue>
#include <map>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

class SmartStagePass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    if (options.session_options == nullptr) {
      return Status::OK();
    }

    bool is_enable_smart_stage =
      options.session_options->config.graph_options()
          .optimizer_options().do_smart_stage();
    if (is_enable_smart_stage) {
      LOG(INFO) << "Run SmartStage Optimization";
    } else {
      return Status::OK();
    }

    Graph *graph = options.graph->get();
    if (graph == nullptr)
      return errors::Internal("a graph should be available.");
    std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, new_graph.get());

    // Get Target Node.
    std::vector<std::string> target_nodes;
    GetTargetNodesName(target_nodes);

    SmartStageGraph(new_graph, target_nodes);

    options.graph->swap(new_graph);
    return Status::OK();
  }

 private:
  void GetTargetNodesName(std::vector<std::string> & target_nodes) {
    std::string tn;
    ReadStringFromEnvVar("TARGET_NODES_NAME", "", &tn);
    for (std::string s : str_util::Split(tn, ';')) {
      target_nodes.push_back(s.substr(0, s.find_last_of(':')));
    }
  }

  void SmartStageGraph(std::unique_ptr<Graph>& g,
                       const std::vector<std::string>& target_nodes) {
    // Get Stage and UnStage node.
    std::map<std::string, Node*> stage_node_map;
    std::map<std::string, Node*> unstage_node_map;
    for (Node* n : g.get()->op_nodes()) {
      if (n->IsStage()) {
        std::string name = n->def().attr().at("shared_name").s();
        stage_node_map[name] = n;
      } else if (n->IsUnstage()) {
        std::string name = n->def().attr().at("shared_name").s();
        unstage_node_map[name] = n;
      }
    }

    for (auto it = stage_node_map.begin(); it != stage_node_map.end(); ++it) {
      if (unstage_node_map.find(it->first) != unstage_node_map.end()) {
        StageGraph(g.get(), it->second, unstage_node_map[it->first],
                   target_nodes);
      }
    }
  }

  void StageGraph(Graph* dest, Node* stage_node, Node* unstage_node,
        const std::vector<std::string>& target_nodes) {
    std::string s1 = stage_node->def().attr().at("shared_name").s();
    std::string s2 = unstage_node->def().attr().at("shared_name").s();
    CHECK(s1 == s2);

    std::vector<const Edge*> out_edges;
    for (const Edge* e : unstage_node->out_edges()) {
      if (!e->IsControlEdge()) {
        out_edges.push_back(e);
      }
    }

    std::unordered_set<Node *> source_node_set;
    for (const Edge* out_edge : out_edges) {
      const Edge* in_edge = NULL;
      int index = out_edge->src_output();
      Status s = stage_node->input_edge(index, &in_edge);
      TF_CHECK_OK(s);
      Node* dst = out_edge->dst();
      s = dest->UpdateEdge(in_edge->src(), in_edge->src_output(),
                           out_edge->dst(), out_edge->dst_input());
      TF_CHECK_OK(s);
      source_node_set.insert(dst);
    }

    std::vector<const Edge*> in_edges;
    for (auto* e : stage_node->in_edges()) {
      in_edges.push_back(e);
    }

    for (const Edge* e : in_edges) {
      dest->RemoveEdge(e);
    }

    std::vector<const Edge*> edge_vec;
    GetStagingEdges(*dest, source_node_set, target_nodes, edge_vec);

    ModifyGraph(dest, stage_node, unstage_node, edge_vec);
  }

  void GetStagingEdges(const Graph& dest,
      const std::unordered_set<Node *>& source_node_set,
      const std::vector<std::string>& target_nodes,
      std::vector<const Edge*>& edge_vec) {
    std::queue<const Node*> q;
    for (Node* n : dest.op_nodes()) {
      if (n->IsVariable() || n->IsKvVarHandle() || n->IsPlaceholder() ||
          n->IsControlFlow() || n->type_string() == "VarHandleOp" ||
          std::find(target_nodes.begin(), target_nodes.end(), n->name()) !=
              target_nodes.end()) {
        q.push(n);
      }
    }

    std::vector<bool> is_var_relate(dest.num_node_ids(), false);
    while (!q.empty()) {
      const Node* node = q.front();
      q.pop();
      is_var_relate[node->id()] = true;
      for (const Edge* e : node->out_edges()) {
        if (e->dst()->type_string() == "_OPT_KvResourceLookupID") {
          continue;
        } else if (!is_var_relate[e->dst()->id()]) {
          q.push(e->dst());
        }
      }
    }

    std::queue<Node *> queue;
    for (Node *n : source_node_set) {
      queue.push(n);
    }

    std::unordered_set<Node *> has_visit_node;
    while (!queue.empty()) {
      Node *n = queue.front();
      queue.pop();
      if (has_visit_node.find(n) != has_visit_node.end()) {
        continue;
      }

      has_visit_node.insert(n);
      for (auto edge : n->out_edges()) {
        Node *dst = edge->dst();
        if (is_var_relate[dst->id()]) {
          edge_vec.push_back(edge);
        } else {
          queue.push(dst);
        }
      }
    }
  }

  void ModifyGraph(Graph* dest, Node* stage_node, Node* unstage_node,
                   std::vector<const Edge*>& edge_vec) {
    std::vector<DataType> type_vec;
    int i = 0;
    std::map<std::string, int64> edge_map;
    std::vector<NodeDefBuilder::NodeOut> src_list;
    std::map<const Edge*, int64> edge_to_stage;
    std::map<const Edge*, int64> edge_to_unstage;
    for (const Edge* e : edge_vec) {
      if (e->IsControlEdge()) {
        // control flow is implemented by stage node and unstage node, remove control edge.
        dest->RemoveEdge(e);
        continue;
      }
      std::string name = e->src()->name() + std::to_string(e->src_output());
      if (edge_map.find(name) == edge_map.end()) {
        type_vec.push_back(e->src()->output_type(e->src_output()));
        src_list.emplace_back(e->src()->name(), e->src_output(), e->src()->output_type(e->src_output()));
        edge_to_stage[e] = i;
        edge_map[name] = i;
        ++i;
      }
      edge_to_unstage[e] = edge_map[name];
    }

    NodeDef node_def_stage;
    TF_CHECK_OK(NodeDefBuilder(stage_node->name(), "TensorBufferPut")
    .Device(stage_node->requested_device())
    .Input(src_list)
    .Attr("container", stage_node->def().attr().at("container"))
    .Attr("shared_capacity", stage_node->def().attr().at("shared_capacity"))
    .Attr("shared_name", stage_node->def().attr().at("shared_name"))
    .Attr("timeout_millis", stage_node->def().attr().at("timeout_millis"))
    .Finalize(&node_def_stage));
    if (stage_node->def().attr().contains("_stream_id")) {
      auto stream_id_attr = stage_node->def().attr().at("_stream_id");
      node_def_stage.mutable_attr()->insert({"_stream_id", stream_id_attr});
    }
    Status s;
    Node* stage_xxx = dest->AddNode(node_def_stage, &s);
    TF_CHECK_OK(s);
    dest->RemoveNode(stage_node);

    NodeDef node_def_unstage;
    TF_CHECK_OK(NodeDefBuilder(unstage_node->name(), "TensorBufferTake")
    .Device(unstage_node->requested_device())
    .Attr("container", unstage_node->def().attr().at("container"))
    .Attr("dtypes", DataTypeSlice(type_vec))
    .Attr("shared_capacity", unstage_node->def().attr().at("shared_capacity"))
    .Attr("shared_name", unstage_node->def().attr().at("shared_name"))
    .Attr("shared_threads", unstage_node->def().attr().at("shared_threads"))
    .Finalize(&node_def_unstage));
    Node* unstage_xxx = dest->AddNode(node_def_unstage, &s);
    TF_CHECK_OK(s);
    dest->RemoveNode(unstage_node);

    for (auto it = edge_to_stage.begin(); it != edge_to_stage.end(); ++it) {
      const Edge* e = it->first;
      dest->AddEdge(e->src(), e->src_output(), stage_xxx, it->second);
    }

    for (auto it = edge_to_unstage.begin(); it != edge_to_unstage.end(); ++it) {
      const Edge* e = it->first;
      Status s = dest->UpdateEdge(unstage_xxx, it->second, e->dst(), e->dst_input());
      TF_CHECK_OK(s);
    }
  }
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 24, SmartStagePass);

} // end of namespace tensorflow
