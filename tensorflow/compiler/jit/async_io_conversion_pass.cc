/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/async_io_conversion_pass.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace {

class AsyncIoConversionPassImpl {
 public:
  AsyncIoConversionPassImpl(Graph* graph, FunctionLibraryDefinition* flib_def,
                            bool enable_full_async_io)
      : graph_(graph),
        flib_def_(flib_def),
        enable_full_async_io_(enable_full_async_io) {}

  Status Run();

 private:
  // Inserts AsyncOutSends and AsyncOutRecvs for OutputTensor `o`. Accumulates
  // tensor names, created AsyncOutSends/Recvs into `tensor_names`, `sends`, and
  // `recvs`.
  Status InsertAsyncOutSendAndRecv(const OutputTensor& o,
                                   std::vector<string>* tensor_names,
                                   std::vector<Node*>* sends,
                                   std::vector<Node*>* recvs);

  // Inserts AsyncOutInit and AsyncOutDone for a cluster.
  Status InsertAsyncOutInitAndDone(const string& cluster_name,
                                   const std::vector<string>& tensor_names,
                                   const std::vector<Node*>& sends,
                                   const std::vector<Node*>& recvs);

  // Removes unnecessary input control edges to users of `o`.
  void RemoveUnnecessaryControlEdgesToUsersOf(const OutputTensor& o);

  // Collects per-cluster output edges into `cluster_out_edges_`.
  void CollectOutputEdgesForClusters();

 private:
  Graph* graph_;
  FunctionLibraryDefinition* flib_def_;
  bool enable_full_async_io_;
  // Maps cluster name to output edges (per cluster).
  absl::flat_hash_map<string, std::vector<const Edge*>> cluster_out_edges_;
};

// Builds XlaAsyncOutSend node.
xla::StatusOr<Node*> BuildXlaAsyncOutSendNode(Graph* g,
                                              const string& tensor_name,
                                              const string& device_name,
                                              const string& cluster_name,
                                              const OutputTensor& producer) {
  CHECK(producer.index != Graph::kControlSlot);
  NodeDef async_out_send_ndef;
  string node_name = absl::StrCat(tensor_name, "_XlaAsyncOutSend");
  node_name = absl::StrReplaceAll(node_name, {{":", "_"}});
  NodeDefBuilder async_out_send_builder(node_name, "_XlaAsyncOutSend");
  TF_RETURN_IF_ERROR(async_out_send_builder
                         .Attr("T", producer.node->output_type(producer.index))
                         .Input(producer.node->name(), producer.index,
                                producer.node->output_type(producer.index))
                         .Device(device_name)
                         .Finalize(&async_out_send_ndef));
  Status s;
  Node* async_out_send = g->AddNode(async_out_send_ndef, &s);
  TF_RETURN_IF_ERROR(s);
  async_out_send->set_assigned_device_name(device_name);
  async_out_send->AddAttr("device_name", device_name);
  async_out_send->AddAttr("tensor_name", tensor_name);
  async_out_send->AddAttr(kXlaClusterAttr, cluster_name);
  return async_out_send;
}

// Builds XlaAsyncOutRecv node.
xla::StatusOr<Node*> BuildXlaAsyncOutRecvNode(Graph* g,
                                              const string& tensor_name,
                                              const string& device_name,
                                              DataType dtype) {
  string node_name = absl::StrCat(tensor_name, "_XlaAsyncOutRecv");
  node_name = absl::StrReplaceAll(node_name, {{":", "_"}});
  NodeDef async_out_recv_ndef;
  string async_out_recv_op_name = "_XlaAsyncOutRecv";
  NodeDefBuilder async_out_recv_builder(node_name, async_out_recv_op_name);
  TF_RETURN_IF_ERROR(async_out_recv_builder.Attr("T", dtype)
                         .Device(device_name)
                         .Finalize(&async_out_recv_ndef));
  Status s;
  Node* async_out_recv = g->AddNode(async_out_recv_ndef, &s);
  TF_RETURN_IF_ERROR(s);
  async_out_recv->set_assigned_device_name(device_name);
  async_out_recv->AddAttr("device_name", device_name);
  async_out_recv->AddAttr("tensor_name", tensor_name);
  return async_out_recv;
}

// Builds XlaAsyncOutInit or XlaAsyncOutDone.
xla::StatusOr<Node*> BuildXlaAsyncOutInitOrDoneNode(
    Graph* g, const string& cluster_name,
    const std::vector<string>& tensor_names, const string& device_name,
    bool is_init) {
  NodeDef ndef;
  string type_string = is_init ? "_XlaAsyncOutInit" : "_XlaAsyncOutDone";
  string node_name = absl::StrCat(cluster_name, type_string);
  NodeDefBuilder ndef_builder(node_name, type_string);
  TF_RETURN_IF_ERROR(ndef_builder.Device(device_name).Finalize(&ndef));
  Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_RETURN_IF_ERROR(s);
  ret->set_assigned_device_name(device_name);
  ret->AddAttr("device_name", device_name);
  ret->AddAttr("tensor_names", tensor_names);
  return ret;
}

xla::StatusOr<Node*> BuildXlaAsyncOutInitNode(
    Graph* g, const string& cluster_name,
    const std::vector<string>& tensor_names, const string& device_name) {
  return BuildXlaAsyncOutInitOrDoneNode(g, cluster_name, tensor_names,
                                        device_name, /*is_init=*/true);
}

xla::StatusOr<Node*> BuildXlaAsyncOutDoneNode(
    Graph* g, const string& cluster_name,
    const std::vector<string>& tensor_names, const string& device_name) {
  return BuildXlaAsyncOutInitOrDoneNode(g, cluster_name, tensor_names,
                                        device_name, /*is_init=*/false);
}

std::string GetDeviceName(const Node& n) {
  return n.assigned_device_name().empty() ? n.requested_device()
                                          : n.assigned_device_name();
}

const absl::flat_hash_set<string>& GetAsyncOutDestOpList() {
  static absl::flat_hash_set<string> list = {
      "HorovodAllreduce",
  };
  return list;
}

// Returns whether `n1` and `n2` are in the same XLA cluster. If either of them
// is not in any cluster, return false. Otherwise, if they are both in clusters,
// check whether the cluster name is the same.
bool IsInTheSameCluster(const Node& n1, const Node& n2) {
  auto cluster_id1 = GetXlaClusterForNode(n1);
  if (!cluster_id1) {
    return false;
  }
  auto cluster_id2 = GetXlaClusterForNode(n2);
  if (!cluster_id2) {
    return false;
  }
  if (*cluster_id1 != *cluster_id2) {
    return false;
  }

  return true;
}

// Searches for a candidate node to trigger the XlaAsyncOutInit. To hide
// latency, we want this candidate to be far away enough, so that all the recv
// nodes can be executed before the `XlaRun` op starts. On the other hand, this
// triggering node needs to have the same control flow as the Init node.
Status SearchForAGoodTrigger(const Node& start, Node** trigger) {
  Node* candidate;
  TF_RETURN_IF_ERROR(start.input_node(0, &candidate));
  // Just a number that appears far enough.
  constexpr size_t distance_threshold = 64;
  for (size_t i = 0; i < distance_threshold;) {
    Node* pred;
    if (candidate->num_inputs() > 0) {
      // Follow through the first input data edge if any.
      TF_RETURN_IF_ERROR(candidate->input_node(0, &pred));
    } else {
      // Otherwise follow through a control edge.
      CHECK(candidate->in_edges().size() != 0)
          << " Candidate has no input edges, candidate = " << candidate->name();
      pred = (*candidate->in_edges().begin())->src();
    }

    if (pred->IsSource()) {
      // No farther can go.
      candidate = pred;
      break;
    } else if (pred->IsControlFlow()) {
      // Cannot cross control flow nodes.
      break;
    }

    // Continue the searching.
    if (!IsInTheSameCluster(*candidate, *pred)) {
      // Do not increase distance if `candidate` and `pred` are in the same
      // cluster. As nodes in the same cluster will become just one XlaRun op,
      // this more accurately calculates the real execution distance.
      ++i;
    }
    candidate = pred;
  }

  *trigger = candidate;
  return Status::OK();
}

Status AsyncIoConversionPassImpl::InsertAsyncOutInitAndDone(
    const string& cluster_name, const std::vector<string>& tensor_names,
    const std::vector<Node*>& sends, const std::vector<Node*>& recvs) {
  VLOG(2) << "Inserts AsyncOutInit and AsyncOutDone for cluster "
          << cluster_name;
  string device_name = GetDeviceName(*sends[0]);

  // Trigger ..> Init ..> Sends ..> Done
  //                  ..> Recvs ..>
  Node* trigger = nullptr;
  TF_RETURN_IF_ERROR(SearchForAGoodTrigger(*sends[0], &trigger));
  VLOG(2) << absl::StrCat("Adding AsyncOutInit and AsyncOutDone; ",
                          " Init will be triggered by node ", trigger->name());
  TF_ASSIGN_OR_RETURN(Node * async_out_init,
                      BuildXlaAsyncOutInitNode(graph_, cluster_name,
                                               tensor_names, device_name));
  TF_ASSIGN_OR_RETURN(Node * async_out_done,
                      BuildXlaAsyncOutDoneNode(graph_, cluster_name,
                                               tensor_names, device_name));
  graph_->AddEdge(trigger, /*x=*/Graph::kControlSlot, async_out_init,
                  /*y=*/Graph::kControlSlot);
  for (auto s : sends) {
    graph_->AddEdge(async_out_init, /*x=*/Graph::kControlSlot, s,
                    /*y=*/Graph::kControlSlot);
    graph_->AddEdge(s, /*x=*/Graph::kControlSlot, async_out_done,
                    /*y=*/Graph::kControlSlot);
  }
  for (auto r : recvs) {
    graph_->AddEdge(async_out_init, /*x=*/Graph::kControlSlot, r,
                    /*y=*/Graph::kControlSlot);
    graph_->AddEdge(r, /*x=*/Graph::kControlSlot, async_out_done,
                    /*y=*/Graph::kControlSlot);
  }

  return Status::OK();
}

Status AsyncIoConversionPassImpl::InsertAsyncOutSendAndRecv(
    const OutputTensor& o, std::vector<string>* tensor_names,
    std::vector<Node*>* sends, std::vector<Node*>* recvs) {
  VLOG(4) << "Insert AsyncOutSend and AsyncOutRecv for node " << o.node->name()
          << ", oidx = " << o.index;
  Node* n = o.node;
  int oidx = o.index;
  auto cluster_id = GetXlaClusterForNode(*n);
  CHECK(cluster_id);

  // Construct AsyncOutSend/Recv.
  string device_name = GetDeviceName(*n);
  string tensor_name = absl::StrCat(n->name(), ":", oidx);
  string cluster_name(*cluster_id);
  TF_ASSIGN_OR_RETURN(Node * async_out_send,
                      BuildXlaAsyncOutSendNode(graph_, tensor_name, device_name,
                                               cluster_name, {n, oidx}));
  TF_ASSIGN_OR_RETURN(Node * async_out_recv,
                      BuildXlaAsyncOutRecvNode(graph_, tensor_name, device_name,
                                               n->output_type(oidx)));
  tensor_names->push_back(tensor_name);
  sends->push_back(async_out_send);
  recvs->push_back(async_out_recv);

  // Add n -> AsyncOutSend.
  graph_->AddEdge(n, oidx, async_out_send, /*y*/ 0);

  // Update AsyncOutRecv -> users of node `n:oidx`.
  std::vector<const Edge*> edges_to_be_updated;
  for (const Edge* e : n->out_edges()) {
    if (e->src_output() != oidx || IsInTheSameCluster(*n, *e->dst())) {
      continue;
    }
    edges_to_be_updated.push_back(e);
  }

  for (const Edge* e : edges_to_be_updated) {
    if (e->IsControlEdge()) {
      graph_->AddEdge(async_out_recv, /*x=*/Graph::kControlSlot, e->dst(),
                      /*y=*/Graph::kControlSlot);
      graph_->RemoveEdge(e);
    } else {
      graph_->UpdateEdge(async_out_recv, /*new_src_index=*/0, e->dst(),
                         e->dst_input());
    }
  }

  return Status::OK();
}

bool IsConvertibleAndProfitable(const Edge& e, bool enable_full_async_io) {
  if (e.src()->output_type(e.src_output()) == DT_RESOURCE ||
      e.src()->IsConstant()) {
    // Do not convert resource and const outputs into async out to avoid
    // complexity due to interop between XLA and Tensorflow.
    return false;
  }

  if (enable_full_async_io ||
      GetAsyncOutDestOpList().contains(e.dst()->type_string())) {
    return true;
  }

  return false;
}

void AsyncIoConversionPassImpl::RemoveUnnecessaryControlEdgesToUsersOf(
    const OutputTensor& o) {
  VLOG(4) << "Remove unnecessary input control edges for users of node "
          << o.node->name();
  std::vector<const Edge*> to_remove;
  // Removes unnecessary input control edges to users of o.node.
  for (const Edge* out_edge : o.node->out_edges()) {
    if (out_edge->src_output() != o.index ||
        !GetAsyncOutDestOpList().contains(out_edge->dst()->type_string())) {
      continue;
    }

    for (const Edge* in_edge : out_edge->dst()->in_edges()) {
      // Remove the control `in_edge` from the same XLA cluster of o.node's.
      if (in_edge->IsControlEdge() &&
          IsInTheSameCluster(*in_edge->src(), *o.node)) {
        to_remove.push_back(in_edge);
      }
    }
  }

  for (const Edge* e : to_remove) {
    graph_->RemoveControlEdge(e);
  }
}

void AsyncIoConversionPassImpl::CollectOutputEdgesForClusters() {
  for (const Edge* e : graph_->edges()) {
    absl::optional<absl::string_view> src_cluster_id =
        GetXlaClusterForNode(*e->src());
    if (!src_cluster_id || IsInTheSameCluster(*e->src(), *e->dst())) {
      // Skip the edge if (a) the src node is not in a cluster at all, or (b)
      // the src node is in a cluster but the dst node is in the same cluster.
      continue;
    }

    cluster_out_edges_[string(*src_cluster_id)].push_back(e);
  }
}

Status AsyncIoConversionPassImpl::Run() {
  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("before_async_io_conversion_pass", *graph_, flib_def_);
  }

  // Collect output edges into `cluster_out_edges_`.
  CollectOutputEdgesForClusters();

  // For each cluster, insert AsyncOut operations.
  for (auto& p : cluster_out_edges_) {
    const string& cluster_name = p.first;
    const std::vector<const Edge*>& cluster_outputs = p.second;

    absl::flat_hash_set<OutputTensor, OutputTensor::Hash>
        cluster_outputs_to_convert;
    for (const Edge* e : cluster_outputs) {
      if (e->IsControlEdge()) {
        // Do not support control edges now.
        continue;
      }

      if (IsConvertibleAndProfitable(*e, enable_full_async_io_)) {
        cluster_outputs_to_convert.insert({e->src(), e->src_output()});
      }
    }

    if (cluster_outputs_to_convert.empty()) {
      // No output edges to be converted for the current cluster.
      continue;
    }

    std::vector<string> async_out_tensor_names;
    std::vector<Node*> async_out_sends;
    std::vector<Node*> async_out_recvs;
    for (const OutputTensor& o : cluster_outputs_to_convert) {
      // Remove unnecessary input control edges to users of `n:oidx`.
      //
      // TODO: generalize it by replacing the control edge also with
      // AsyncOutSend and AsyncOutRecv.
      RemoveUnnecessaryControlEdgesToUsersOf(o);

      // n:oidx -> AsyncOutSend
      //           AsyncOutRecv -> uses of `n:oidx`
      InsertAsyncOutSendAndRecv(o, &async_out_tensor_names, &async_out_sends,
                                &async_out_recvs);
    }

    // Insert AsyncOutInit and AsyncOutDone.
    InsertAsyncOutInitAndDone(cluster_name, async_out_tensor_names,
                              async_out_sends, async_out_recvs);
  }

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("async_io_conversion_pass", *graph_, flib_def_);
  }

  return Status::OK();
}

}  // namespace

Status AsyncIoConversionPass::Run(const GraphOptimizationPassOptions& options) {
  if (GetGlobalJitLevelForGraph(options) == OptimizerOptions::OFF) {
    return Status::OK();
  }

  int async_io_level = async_io_level_
                           ? *async_io_level_
                           : GetBuildXlaOpsPassFlags()->tf_xla_async_io_level;
  if (async_io_level < 0 || async_io_level > 2) {
    LOG(WARNING) << absl::StrCat("Unknown async_io_level = ", async_io_level,
                                 ".");
    return Status::OK();
  }
  if (async_io_level == 0) {
    VLOG(1) << "AsyncIoConversionPass is off.";
    return Status::OK();
  }

  VLOG(1) << "Run AsyncIoConversionPass.";

  Graph* graph = options.graph->get();
  FunctionLibraryDefinition* flib_def = options.flib_def;
  CHECK(flib_def != nullptr);

  bool enable_full_async_io = (async_io_level == 2);
  return AsyncIoConversionPassImpl{graph, flib_def, enable_full_async_io}.Run();
}

}  // namespace tensorflow
