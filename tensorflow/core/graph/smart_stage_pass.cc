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

#include "tensorflow/cc/training/prefetch_runner.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

class SmartStagePass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    if (options.session_options == nullptr)
      return Status::OK();

    bool is_enable_smart_stage =
      options.session_options->config.graph_options()
          .optimizer_options().do_smart_stage();
    if (is_enable_smart_stage)
      LOG(INFO) << "Run SmartStage Optimization";
    else
      return Status::OK();

    auto smart_stage_options = options.session_options->config.graph_options()
                               .optimizer_options().smart_stage_options();

    Graph *graph = options.graph->get();
    if (graph == nullptr)
      return errors::Internal("a graph should be available.");
    std::unique_ptr<Graph> new_graph(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, new_graph.get());

    TF_RETURN_IF_ERROR(SmartStageGraph(new_graph, smart_stage_options));

    options.graph->swap(new_graph);
    return Status::OK();
  }

 private:
  Status SmartStageGraph(std::unique_ptr<Graph>& g,
                         const SmartStageOptions& options) {
    // Try to find Stage, UnStage node.
    Node* stage_node;
    Node* unstage_node;
    TF_RETURN_IF_ERROR(GetStageUnStageNode(g, stage_node, unstage_node));
    if (stage_node != nullptr && unstage_node != nullptr) {
      VLOG(1)
        << "SmartStage: Start searching from a user-specified position.";
      Status s =
          SmartStageFromStageUnStageNode(g, stage_node, unstage_node);
      return s;
    }

    // Try to find IteratorGetNext node.
    Node* get_next_node = nullptr;
    TF_RETURN_IF_ERROR(GetIteratorGetNextNode(g, get_next_node));
    if (get_next_node != nullptr) {
      VLOG(1) << "SmartStage: Start searching from IteratorGetNext node";
      Status s = SmartStageFromIteratorGetNextNode(g, get_next_node, options);
      return s;
    }

    LOG(WARNING) << "SmartStage: Failed to get starting position, SmartStage "
                    "is disabled. Or manually specify the starting position "
                    "using the 'tf.staged' interface";
    return Status::OK();
  }

  void GetTargetNodesName(std::unordered_set<std::string>& target_nodes) {
    std::string tn;
    ReadStringFromEnvVar("TARGET_NODES_NAME", "", &tn);
    for (std::string s : str_util::Split(tn, ';')) {
      target_nodes.insert(s.substr(0, s.find_last_of(':')));
    }
  }

  Status GetStageUnStageNode(const std::unique_ptr<Graph>& g,
                             Node*& stage_node, Node*& unstage_node) {
    stage_node = nullptr;
    unstage_node = nullptr;

    unsigned int stage_counter = 0;
    unsigned int unstage_counter = 0;
    for (Node* n : g->op_nodes()) {
      if (n->IsStage()) {
        stage_node = n;
        stage_counter++;
      } else if (n->IsUnstage()) {
        unstage_node = n;
        unstage_counter++;
      }
    }

    if (stage_counter != unstage_counter)
      return errors::Internal(
          "the number of Stage nodes and UnStage nodes does not match.");

    if (stage_counter > 1)
      return errors::Internal("there are multiple Stage nodes in the graph.");

    if (unstage_counter > 1)
      return errors::Internal("there are multiple UnStage nodes in the graph.");

    if (stage_counter == 1) {
      const std::string& s1 = stage_node->def().attr().at("shared_name").s();
      const std::string& s2 = unstage_node->def().attr().at("shared_name").s();
      if (s1 != s2)
        return errors::Internal(
            "the Stage node and the UnStage node in the graph do not match.");
    }

    return Status::OK();
  }

  Status GetIteratorGetNextNode(const std::unique_ptr<Graph>& g,
                                Node*& get_next_node) {
    get_next_node = nullptr;
    unsigned int counter = 0;
    for (Node* n : g->op_nodes()) {
      if (n->type_string() == "IteratorGetNext") {
        counter++;
        get_next_node = n;
      }
    }

    if (counter > 1)
      return errors::Internal(
          "there are multiple IteratorGetNext nodes in the graph.");

    return Status::OK();
  }

  Status SmartStageFromStageUnStageNode(std::unique_ptr<Graph>& g,
                                        Node* stage_node, Node* unstage_node) {
    // gather start_nodes and relink edge.
    std::vector<const Edge*> out_edges;
    out_edges.reserve(unstage_node->out_edges().size());
    for (const Edge* e : unstage_node->out_edges()) {
      // skip control edge.
      if (e->IsControlEdge())
        continue;
      out_edges.emplace_back(e);
    }
    std::unordered_set<const Node*> start_nodes;
    for (const Edge* out_edge : out_edges) {
      // gather start node.
      start_nodes.insert(out_edge->dst());
      // reconnect edges cut by Stage node and UnStage node.
      const Edge* in_edge = nullptr;
      int index = out_edge->src_output();
      TF_RETURN_IF_ERROR(stage_node->input_edge(index, &in_edge));
      TF_RETURN_IF_ERROR(g->UpdateEdge(in_edge->src(), in_edge->src_output(),
                                       out_edge->dst(), out_edge->dst_input()));
    }
    std::vector<const Edge*> in_edges;
    in_edges.reserve(stage_node->in_edges().size());
    for (const Edge* e : stage_node->in_edges())
      in_edges.emplace_back(e);
    for (const Edge* e : in_edges) {
      g->RemoveEdge(e);
    }

    std::unordered_set<const Node*> compute_graph_nodes;
    MarkComputeGraph(g, compute_graph_nodes);

    std::vector<const Edge*> stage_edges;
    GetStageEdges(g, start_nodes, compute_graph_nodes, stage_edges);
    Status s = AddStageNodeToGraph(g, stage_node, unstage_node, stage_edges,
                                   SmartStageOptions());
    return s;
  }

  Status SmartStageFromIteratorGetNextNode(std::unique_ptr<Graph>& g,
                                           const Node* get_next_node,
                                           const SmartStageOptions& options) {
    // gather start_nodes
    std::unordered_set<const Node*> start_nodes;
    start_nodes.insert(get_next_node);

    std::unordered_set<const Node*> compute_graph_nodes;
    MarkComputeGraph(g, compute_graph_nodes);

    std::vector<const Edge*> stage_edges;
    GetStageEdges(g, start_nodes, compute_graph_nodes, stage_edges);

    Status s = AddStageNodeToGraph(g, nullptr, nullptr, stage_edges, options);
    return s;
  }

  void MarkComputeGraph(const std::unique_ptr<Graph>& g,
                        std::unordered_set<const Node*>& compute_graph_nodes) {
    // get target nodes.
    std::unordered_set<std::string> target_nodes;
    GetTargetNodesName(target_nodes);

    // mark compute graph
    std::queue<const Node*> queue;
    for (const Node* n : g->op_nodes()) {
      if (n->IsVariable() || n->IsKvVarHandle() || n->IsPlaceholder() ||
          n->IsControlFlow() || n->type_string() == "VarHandleOp" ||
          target_nodes.count(n->name()) != 0) {
        queue.push(n);
      }
    }
    while (!queue.empty()) {
      const Node* node = queue.front();
      queue.pop();
      compute_graph_nodes.insert(node);
      for (const Edge* e : node->out_edges()) {
        if (e->dst()->type_string() == "_OPT_KvResourceLookupID") {
          continue;
        } else if (compute_graph_nodes.count(e->dst()) == 0) {
          queue.push(e->dst());
        }
      }
    }
  }

  void GetStageEdges(const std::unique_ptr<Graph>& g,
                     const std::unordered_set<const Node*>& start_nodes,
                     const std::unordered_set<const Node*>& compute_graph_nodes,
                     std::vector<const Edge*>& stage_edges) {
    std::queue<const Node*> queue;
    for (const Node* n : start_nodes) {
      queue.push(n);
    }

    std::unordered_set<const Node*> has_visit_node;
    while (!queue.empty()) {
      const Node* n = queue.front();
      queue.pop();
      if (has_visit_node.count(n) != 0)
        continue;

      has_visit_node.insert(n);
      for (const Edge* edge : n->out_edges()) {
        const Node* dst = edge->dst();
        if (compute_graph_nodes.count(dst) != 0)
          stage_edges.push_back(edge);
        else
          queue.push(dst);
      }
    }
  }

  Status GenerateStageNode(std::unique_ptr<Graph>& g, const Node* stage_node,
                           const std::vector<NodeDefBuilder::NodeOut>& src_list,
                           const SmartStageOptions& options,
                           Node*& new_stage_node,
                           std::string& stage_node_name) {
    NodeDef stage_node_def;
    if (stage_node) {
      stage_node_name = stage_node->name();
      auto builder =
          NodeDefBuilder(stage_node_name, "TensorBufferPut")
              .Device(stage_node->requested_device())
              .Input(src_list)
              .Attr("container", stage_node->def().attr().at("container"))
              .Attr("shared_capacity",
                    stage_node->def().attr().at("shared_capacity"))
              .Attr("shared_name", stage_node->def().attr().at("shared_name"))
              .Attr("timeout_millis",
                    stage_node->def().attr().at("timeout_millis"));

      if (stage_node->def().attr().contains("_stream_id"))
        builder.Attr("_stream_id", stage_node->def().attr().at("_stream_id"));

      TF_RETURN_IF_ERROR(builder.Finalize(&stage_node_def));
    } else {
      std::string name_prefix = "prefetch";
      if (!options.name().empty())
        name_prefix = options.name();
      stage_node_name = name_prefix + "/TensorBufferPut";

      auto builder = NodeDefBuilder(stage_node_name, "TensorBufferPut")
                         .Input(src_list)
                         .Attr("shared_capacity", options.capacity())
                         .Attr("shared_name", name_prefix)
                         .Attr("timeout_millis", options.timeout_millis());

      if (options.stage_subgraph_stream_id() > 0)
        builder.Attr("_stream_id", options.stage_subgraph_stream_id());

      TF_RETURN_IF_ERROR(builder.Finalize(&stage_node_def));
    }

    Status s;
    new_stage_node = g->AddNode(stage_node_def, &s);
    return s;
  }

  Status GenerateUnStageNode(std::unique_ptr<Graph>& g,
                             const Node* unstage_node,
                             const std::vector<DataType>& type_vec,
                             const SmartStageOptions& options,
                             Node*& new_unstage_node,
                             std::string& unstage_node_name) {
    NodeDef unstage_node_def;
    if (unstage_node) {
      unstage_node_name = unstage_node->name();
      TF_RETURN_IF_ERROR(
          NodeDefBuilder(unstage_node_name, "TensorBufferTake")
              .Device(unstage_node->requested_device())
              .Attr("container", unstage_node->def().attr().at("container"))
              .Attr("dtypes", DataTypeSlice(type_vec))
              .Attr("shared_capacity",
                    unstage_node->def().attr().at("shared_capacity"))
              .Attr("shared_name", unstage_node->def().attr().at("shared_name"))
              .Attr("shared_threads",
                    unstage_node->def().attr().at("shared_threads"))
              .Finalize(&unstage_node_def));
    } else {
      std::string name_prefix = "prefetch";
      if (!options.name().empty())
        name_prefix = options.name();
      unstage_node_name = name_prefix + "/TensorBufferTake";

      int num_clients = 1;
      if (options.num_clients() > 1)
        num_clients = options.num_clients();

      TF_RETURN_IF_ERROR(NodeDefBuilder(unstage_node_name, "TensorBufferTake")
                             .Attr("dtypes", DataTypeSlice(type_vec))
                             .Attr("shared_capacity", options.capacity())
                             .Attr("shared_name", name_prefix)
                             .Attr("shared_threads", num_clients)
                             .Finalize(&unstage_node_def));
    }

    Status s;
    new_unstage_node = g->AddNode(unstage_node_def, &s);
    return s;
  }

  Status GenerateStageControlNodes(std::unique_ptr<Graph>& g,
                                   const SmartStageOptions& options,
                                   std::string& cancel_node_name,
                                   std::string& resume_node_name,
                                   std::string& close_node_name) {
    std::string name_prefix = "prefetch";
    if (!options.name().empty())
      name_prefix = options.name();

    Status s;
    // Create cancel op.
    NodeDef cancel_node_def;
    cancel_node_name = name_prefix + "/TensorBufferCancel";
    TF_RETURN_IF_ERROR(NodeDefBuilder(cancel_node_name, "TensorBufferCancel")
                           .Attr("shared_name", name_prefix)
                           .Attr("shared_capacity", options.capacity())
                           .Finalize(&cancel_node_def));
    g->AddNode(cancel_node_def, &s);
    TF_RETURN_IF_ERROR(s);

    // Create resume op.
    NodeDef resume_node_def;
    resume_node_name = name_prefix + "/TensorBufferResume";
    TF_RETURN_IF_ERROR(NodeDefBuilder(resume_node_name, "TensorBufferCancel")
                           .Attr("is_cancelled", false)
                           .Attr("shared_name", name_prefix)
                           .Attr("shared_capacity", options.capacity())
                           .Finalize(&resume_node_def));
    g->AddNode(resume_node_def, &s);
    TF_RETURN_IF_ERROR(s);

    // Create close op.
    NodeDef close_node_def;
    close_node_name = name_prefix + "/TensorBufferClose";
    TF_RETURN_IF_ERROR(NodeDefBuilder(close_node_name, "TensorBufferClose")
                           .Attr("shared_name", name_prefix)
                           .Attr("shared_capacity", options.capacity())
                           .Finalize(&close_node_def));
    g->AddNode(close_node_def, &s);

    return s;
  }

  void CreatePrefetchRunner(const SmartStageOptions& options,
                            const std::string& fetch_op,
                            const std::string& cancel_op,
                            const std::string& resume_op,
                            const std::string& close_op) {
    auto runner_options = options.runner_options();
    for (size_t i = 0; i < options.num_threads(); i++)
      runner_options.add_fetch_ops(fetch_op);
    runner_options.set_cancel_op(cancel_op);
    runner_options.set_resume_op(resume_op);
    runner_options.set_close_op(close_op);

    std::string name_prefix = "prefetch";
    if (!options.name().empty())
      name_prefix = options.name();

    auto prefetch_runner_mgr = PrefetchRunnerMgr::singleton();
    prefetch_runner_mgr->RegisterPrefetchRunner(
        options.graph_key(), name_prefix + "_prefetch_runner", runner_options);
  }

  Status AddStageNodeToGraph(std::unique_ptr<Graph>& g,
                             Node* stage_node,
                             Node* unstage_node,
                             std::vector<const Edge*>& stage_edges,
                             const SmartStageOptions& options) {
    int index = 0;
    std::map<std::string, int64> edge_map;
    std::vector<DataType> type_vec;
    std::vector<NodeDefBuilder::NodeOut> src_list;
    std::map<const Edge*, int64> edge_to_stage;
    std::map<const Edge*, int64> edge_to_unstage;
    for (const Edge* e : stage_edges) {
      if (e->IsControlEdge()) {
        // control flow is implemented by stage node and unstage node, remove
        // control edge.
        g->RemoveEdge(e);
        continue;
      }
      std::string name = e->src()->name() + std::to_string(e->src_output());
      if (edge_map.count(name) == 0) {
        type_vec.push_back(e->src()->output_type(e->src_output()));
        src_list.emplace_back(e->src()->name(), e->src_output(),
                              e->src()->output_type(e->src_output()));
        edge_to_stage[e] = index;
        edge_map[name] = index;
        ++index;
      }
      edge_to_unstage[e] = edge_map[name];
    }

    Node* new_stage_node;
    std::string stage_node_name;
    TF_RETURN_IF_ERROR(GenerateStageNode(g, stage_node, src_list, options,
                                         new_stage_node, stage_node_name));

    Node* new_unstage_node;
    std::string unstage_node_name;
    TF_RETURN_IF_ERROR(GenerateUnStageNode(g, unstage_node, type_vec, options,
                                           new_unstage_node,
                                           unstage_node_name));

    for (auto it = edge_to_stage.begin(); it != edge_to_stage.end(); ++it) {
      const Edge* e = it->first;
      g->AddEdge(e->src(), e->src_output(), new_stage_node, it->second);
    }

    for (auto it = edge_to_unstage.begin(); it != edge_to_unstage.end(); ++it) {
      const Edge* e = it->first;
      TF_RETURN_IF_ERROR(
          g->UpdateEdge(new_unstage_node, it->second, e->dst(), e->dst_input()));
    }

    if (stage_node != nullptr && unstage_node != nullptr) {
      g->RemoveNode(stage_node);
      g->RemoveNode(unstage_node);
    } else {
      // need to create `tensor_buffer_cancel`, `tensor_buffer_resume`,
      // `tensor_buffer_close` and PrefetchRunner.
      std::string cancel_node_name;
      std::string resume_node_name;
      std::string close_node_name;
      TF_RETURN_IF_ERROR(GenerateStageControlNodes(
          g, options, cancel_node_name, resume_node_name, close_node_name));
      CreatePrefetchRunner(options, stage_node_name, cancel_node_name,
                           resume_node_name, close_node_name);
    }

    TF_RETURN_IF_ERROR(CheckGraphCircle(new_stage_node, new_unstage_node));

    return Status::OK();
  }

  Status CheckGraphCircle(Node* stage_node, Node* unstage_node) {
    std::unordered_set<const Node*> accessed;
    std::queue<const Node*> queue;
    queue.push(unstage_node);

    while (!queue.empty()) {
      const Node* node = queue.front();
      queue.pop();

      if (accessed.count(node) != 0)
        continue;

      accessed.insert(node);

      if (node == stage_node)
        return errors::Internal(
            "there is a cycle in the graph after smart stage.");

      for (const Edge* e : node->out_edges())
        queue.push(e->dst());
    }
    return Status::OK();
  }
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 24, SmartStagePass);

} // end of namespace tensorflow
