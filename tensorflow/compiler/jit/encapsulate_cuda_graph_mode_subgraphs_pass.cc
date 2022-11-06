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

#include "tensorflow/compiler/jit/encapsulate_cuda_graph_mode_subgraphs_pass.h"

#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/graphcycles/graphcycles.h"
#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"
#include "tensorflow/compiler/jit/mark_for_cuda_graph_mode_pass.h"
#include "tensorflow/compiler/jit/shape_inference_helpers.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

const char* const kCgmodeCompiledAttr = "_CgmodeCompiledKernel";
const char* const kCgmodeNumConstantArgsAttr = "_CgmodeNumConstantArgs";
const char* const kCgmodeNumResourceArgsAttr = "_CgmodeNumResourceArgs";
const char* const kCgmodeHostTransferSequencerAttr =
    "_cgmode_host_transfer_sequencer";

namespace {

bool AreAllParentsGuaranteedConst(
    const Node& n,
    const absl::flat_hash_set<const Node*>& runtime_const_nodes) {
  if (n.type_string() == "GuaranteeConst") {
    // If the current node is itself a cast-to-const, no need
    // to look at the incoming edges.
    return true;
  }

  bool all_parents_const = true;
  bool atleast_one_non_control_edge = false;
  for (const Edge* in : n.in_edges()) {
    atleast_one_non_control_edge =
        atleast_one_non_control_edge || !in->IsControlEdge();
    if (!in->IsControlEdge() && runtime_const_nodes.count(in->src()) == 0) {
      all_parents_const = false;
      break;
    }
  }
  return all_parents_const && atleast_one_non_control_edge;
}

void MarkGuaranteedConstants(
    const Graph& graph,
    const std::vector<std::pair<const Node*, Node*>>& src_arg_pairs) {
  absl::flat_hash_set<const Node*> guaranteed_const_nodes;
  std::vector<const Node*> srcs;
  srcs.reserve(src_arg_pairs.size());
  for (const auto& src_arg : src_arg_pairs) {
    srcs.push_back(src_arg.first);
  }
  ReverseDFSFrom(
      graph, srcs, /*enter=*/nullptr,
      /*leave=*/[&guaranteed_const_nodes](const Node* n) {
        // TODO(vinuraja): Doesn't work in the presence of loops.
        if (AreAllParentsGuaranteedConst(*n, guaranteed_const_nodes)) {
          guaranteed_const_nodes.insert(n);
        }
      });

  for (auto& src_arg : src_arg_pairs) {
    if (guaranteed_const_nodes.count(src_arg.first) != 0) {
      VLOG(1) << "Guaranteed const found: " << src_arg.first->DebugString();
      src_arg.second->AddAttr("_is_guaranteed_constant", true);
    }
  }
}

struct OutputInputTensorPairHasher {
  uint64 operator()(std::pair<OutputTensor, InputTensor> const& s) const {
    return Hash64Combine(OutputTensor::Hash()(s.first),
                         InputTensor::Hash()(s.second));
  }
};

// TODO(phawkins) add a canonical copy of these operator names and refactor
// everything to use it.
static const char* const kArgOp = "_Arg";
static const char* const kRetValOp = "_Retval";

class Encapsulator {
 public:
  Encapsulator(string group_attribute, Graph const* graph_in)
      : group_attribute_(std::move(group_attribute)), graph_in_(graph_in) {}

  // Find subgraphs marked with 'group_attribute', and build a new
  // subgraph, one for each value of 'group_attribute'.
  Status SplitIntoSubgraphs();

  // Build a FunctionDef for each subgraph, and add it 'library'. The values of
  // the 'group_attribute' annotations become the function names.
  // If 'reuse_existing_functions' is set, use an existing function with the
  // same name, if any.
  // If 'rewrite_subgraph_fn' is set, it is applied to each subgraph before
  // function conversion.
  Status ValidateAndBuild(const ValidateCgmodeSubgraphFn& rewrite_subgraph_fn,
                          FunctionLibraryDefinition* library);

  // Write a copy of the input graph to 'graph_out', where the subgraphs are
  // replaced with calls to the new functions.
  Status BuildOutputGraph(Graph* graph_out);

 private:
  // A subgraph of the input, all marked with a common 'group_attribute'
  // value.
  //
  // In the following simple example, A, B, ..., E are nodes in the original
  // graph. The group attributes g are each shown as either 0 or empty.
  //
  //  A  -->  B  -->  C  -->  D  -->  E
  //  g:      g:0     g:0     g:0     g:
  //
  // The example is rewritten to two graphs; one on the host and one to be
  // compiled. The host graph is as follows.
  //
  //  A  -->  Call  -->  E
  //
  // The compiled cluster is as follows.
  //
  //  Arg  --> B  --> C  --> D --> Retval
  class Subgraph {
   public:
    // Creates a graph to build the subgraph in, if it doesn't already exist,
    // using the same op registry and versions as graph_in.
    Node* MakeNodeImage(const Graph* graph_in, Node* node);

    // Returns the graph the subgraph is being built in.
    Graph* GetGraph() const;

    // Builds a FunctionDef, and adds it to 'library'. The value of the
    // 'group_attribute' annotations becomes the function name.  If
    // 'reuse_existing_functions' is set, use an existing function with the same
    // name, if any.  If 'rewrite_subgraph_fn' is set, it is applied to the
    // subgraph before function conversion.
    Status ValidateAndBuild(const string& name_in,
                            const ValidateCgmodeSubgraphFn& rewrite_subgraph_fn,
                            FunctionLibraryDefinition* library);

    // Adds the function call node to graph_out.
    Status AddFunctionCallNode(
        const std::unordered_map<const Node*, Node*>& node_images,
        Graph* graph_out);

    // Returns the Node that the inputs and outputs of the function should be
    // wired up to.
    Node* GetCallNode() const;

    // Returns the index of the arg that the dst of edge should connect to.
    int GetArgIndexForEdge(const Edge* edge) const;

    // Returns the index of the result that the src of edge should connect to.
    int GetResultIndexForEdge(const Edge* edge) const;

    // Creates an _Arg node for the src node of edge, and add its index to
    // args_by_src_, if none exists yet. Also adds its index to args_by_dst_,
    // and adds the edge within the subgraph from the _Arg node to the image of
    // the dst node.
    Status RecordArg(const Edge* edge,
                     const std::unordered_map<const Node*, Node*>& node_images,
                     std::vector<std::pair<const Node*, Node*>>* src_arg_pairs);

    // Records the src of the given edge as a control result of the graph.
    // Used during graph to function conversion to tie control results to
    // the function signature.
    Status RecordControlResult(
        const Edge* edge,
        const std::unordered_map<const Node*, Node*>& node_images);

    // Creates a _Retval node for the src node of edge, and add it to results_,
    // if none exists yet. If a new _Retval node is created, also adds the edge
    // within the subgraph from the src to the _Retval node.
    Status RecordResult(
        const Edge* edge,
        const std::unordered_map<const Node*, Node*>& node_images);

    // Creates the sequencer node if it doesn't exist, adding it to graph_out.
    Status MakeSequencingNode(const string& subgraph_name, Graph* graph_out);

    // If there is a sequencer node, adds a control edge from the sequencer to
    // the call node.
    void ConnectSequencerToCallNode(Graph* graph_out);

    Status ReplaceFunctionDef(FunctionLibraryDefinition* library);

   private:
    // The subgraph extracted from the input graph, suitable for being turned
    // into a cuda_graph_exec. Inputs are fed by _Arg nodes, and outputs are
    // returned by _Retval nodes.
    std::unique_ptr<Graph> graph_;

    // Which device are these nodes on? Used to assign a device to the call
    // node.
    string device_;

    // NodeDef for the function call node.
    NodeDef call_node_def_;

    // Name that is used for the call node. This may not be
    // call_node_def_.name() if the client supplies a rewrite lambda.
    string function_def_name_;

    // Placeholder node simulating the host compute key in the output graph.
    // Not owned.
    Node* host_compute_key_placeholder_ = nullptr;

    // Function call node in the output graph. Not owned.
    Node* call_node_;

    cudaGraph_t cuda_graph_;
    cudaGraphExec_t cuda_graph_exec_;

    // Maps from source (producer node/slot) and destination
    // (consumer node/slot) tensors in the input graph to _Arg numbers in
    // the subgraph. The source map is one-to-one, whereas the dest map may be
    // many-to-one.
    std::unordered_map<OutputTensor, int, OutputTensor::Hash> args_by_src_;
    std::unordered_map<InputTensor, int, InputTensor::Hash> args_by_dst_;

    // The arguments to the subgraph, in order.
    std::vector<Node*> args_;

    // Map from source tensor in the input graph to result #.
    std::unordered_map<OutputTensor, int, OutputTensor::Hash> results_;

    // Set of node names that are the source of a control output of the
    // subgraph. We store strings here so that we can tolerate nodes being
    // removed from the graph.
    absl::flat_hash_set<string> control_output_nodes_;

    // NoOp node in the output graph that is sequenced after the call node.
    Node* sequencer_ = nullptr;
  };

  // Returns the key attribute associated with a node in attr. Sets either
  // result to the empty string if the respective attribute is not found.
  Status GetGroupNameAttr(Node const* node, string* attr) const;

  // Copies edges local to a subgraph. Adds _Arg and _Retval nodes to
  // subgraphs for data edges that cross subgraph boundaries.
  Status CopySubgraphEdges(
      const std::unordered_map<const Node*, Node*>& node_images,
      std::vector<std::pair<const Node*, Node*>>* src_arg_pairs);

  // Copies all marked nodes to a subgraph. Does nothing for unmarked nodes.
  Status CopySubgraphNodes(std::unordered_map<const Node*, Node*>* node_images);

  // Copies all nodes that aren't in a compiled subgraph to the output graph.
  Status CopyNodesToOutputGraph(
      Graph* graph_out, std::unordered_map<const Node*, Node*>* node_images);

  // Adds function call nodes for each compiled subgraph.
  Status AddCudaGraphModeCallNodes(
      const std::unordered_map<const Node*, Node*>& node_images,
      Graph* graph_out);

  // Finds the image of an edge source in the output graph. If the edge crosses
  // a subgraph boundary it is the output of a call node, otherwise it is a node
  // in the output graph.
  Status FindOutputImageOfEdgeSrc(
      const string& src_group_id, const string& dst_group_id,
      const std::unordered_map<const Node*, Node*>& node_images,
      const Node* original_src_node, Node** src_image);

  // Finds an edge source slot in the output graph. If the edge crosses a
  // subgraph boundary it is a slot on the output of a call node, otherwise it
  // is a slot on a node in the output graph.
  int FindOutputSlotOfEdgeSrc(const string& src_group_id,
                              const string& dst_group_id, const Edge* edge);

  // Finds the image of an edge destination in the output graph. If the edge
  // crosses a subgraph boundary it is the input of a call node, otherwise it is
  // a node in the output graph.
  Status FindOutputImageOfEdgeDst(
      const string& src_group_id, const string& dst_group_id,
      const std::unordered_map<const Node*, Node*>& node_images,
      const Node* original_dst_node, Node** dst_image);

  // Finds an edge destination slot in the output graph. If the edge crosses a
  // subgraph boundary it is a slot on the input of a call node, otherwise it is
  // a slot on a node in the output graph.
  int FindOutputSlotOfEdgeDst(const string& src_group_id,
                              const string& dst_group_id, const Edge* edge);

  // Copies a single edge to the output graph. The edge is either entirely
  // within the output graph, or crosses into or out of a compiled subgraph.
  Status CopyEdgeToOutputGraph(
      const Edge* edge, const string& src_group_id, const string& dst_group_id,
      const std::unordered_map<const Node*, Node*>& node_images,
      Graph* graph_out,
      std::unordered_set<std::pair<OutputTensor, InputTensor>,
                         OutputInputTensorPairHasher>* edges_added);

  // Adds all edges to the output graph.
  Status AddEdgesToOutputGraph(
      const std::unordered_map<const Node*, Node*>& node_images,
      Graph* graph_out);

  // Makes a copy of graph containing only nodes that are ancestors of at least
  // one node in send_from_host_nodes and store it in pruned_graph. On exit
  // nodes_images contains a mapping from nodes in graph to nodes in
  // pruned_graph. All functions in the copied graph are inlined.
  Status MakePrunedGraphCopyAndInline(
      const Graph& graph, const std::vector<Node*>& sink_nodes,
      std::unique_ptr<Graph>* pruned_graph,
      std::unordered_map<const Node*, Node*>* node_images,
      FunctionLibraryDefinition* library);

  const string group_attribute_;
  const Graph* graph_in_;

  std::unordered_map<string, Subgraph> subgraphs_;

  TF_DISALLOW_COPY_AND_ASSIGN(Encapsulator);
};

namespace {

// Return in 'sorted' a topological sort of clusters according to the
// dependencies encoded in ancestors. clusters is the list of all clusters
// including clusters that are not present in the ancestors map. has_successors
// is the set of clusters that are ancestors of some other cluster.
void TopologicalClusterSort(
    const std::unordered_set<string>& clusters,
    const std::unordered_set<string>& has_successors,
    const std::unordered_map<string, std::unordered_set<string>>& ancestors,
    std::vector<string>* sorted) {
  // The nodes are placed in 'sorted' in topological order.
  sorted->clear();
  // We don't use the standard DFS because we are not operating on Node*
  // objects.
  struct Work {
    string cluster;
    bool leave;
  };
  std::set<string> visited;
  std::vector<Work> stack;
  // Seed the processing list with clusters that have no successors.
  for (const auto& cluster : clusters) {
    if (has_successors.find(cluster) == has_successors.end()) {
      stack.push_back({cluster, false});
    }
  }
  while (!stack.empty()) {
    const Work item = stack.back();
    stack.pop_back();
    if (item.leave) {
      sorted->push_back(item.cluster);
      continue;
    }

    if (visited.find(item.cluster) != visited.end()) continue;
    visited.insert(item.cluster);

    stack.push_back({item.cluster, true});
    const auto& iter = ancestors.find(item.cluster);
    if (iter != ancestors.end()) {
      for (const auto& ancestor : iter->second) {
        stack.push_back({ancestor, false});
      }
    }
  }
  CHECK(sorted->size() == clusters.size());
}

}  // namespace

Node* Encapsulator::Subgraph::GetCallNode() const { return call_node_; }

int Encapsulator::Subgraph::GetArgIndexForEdge(const Edge* edge) const {
  return args_by_dst_.at(InputTensor(edge->dst(), edge->dst_input()));
}

int Encapsulator::Subgraph::GetResultIndexForEdge(const Edge* edge) const {
  return results_.at(OutputTensor(edge->src(), edge->src_output()));
}

Node* Encapsulator::Subgraph::MakeNodeImage(const Graph* graph_in, Node* node) {
  if (!graph_) {
    graph_.reset(new Graph(graph_in->op_registry()));
    graph_->set_versions(graph_in->versions());
  }

  // TODO(b/116981129): Enhance how the device for the encapsulated subgraph is
  // determined. In case of hard placement, ensure all the encapsulated nodes
  // have the same requested device, which in turn will be the requested device
  // for the entire encapsulated subgraph. In case of soft placement, use a
  // deterministic approach to fill in the requested device. Handle co-location
  // constraints similarly if they exist.
  if (device_.empty()) {
    device_ = node->assigned_device_name().empty()
                  ? node->requested_device()
                  : node->assigned_device_name();
  }

  return graph_->CopyNode(node);
}

Graph* Encapsulator::Subgraph::GetGraph() const { return graph_.get(); }

Status Encapsulator::Subgraph::RecordArg(
    const Edge* edge, const std::unordered_map<const Node*, Node*>& node_images,
    std::vector<std::pair<const Node*, Node*>>* src_arg_pairs) {
  Node* src_node = edge->src();
  int src_slot = edge->src_output();
  std::unordered_map<OutputTensor, int, OutputTensor::Hash>::iterator iter;
  bool inserted;
  std::tie(iter, inserted) = args_by_src_.emplace(
      OutputTensor(src_node, src_slot), args_by_src_.size());
  int arg_index = iter->second;
  if (inserted) {
    NodeDef arg_def;
    NodeDefBuilder builder(
        absl::StrCat(src_node->name(), "_", src_slot, "_arg"), kArgOp,
        NodeDebugInfo(src_node->def()));
    DataType dtype = edge->dst()->input_type(edge->dst_input());
    builder.Attr("T", dtype);
    builder.Attr("index", arg_index);
    Status s = builder.Finalize(&arg_def);
    if (!s.ok()) return s;

    Node* arg = graph_->AddNode(arg_def, &s);
    if (!s.ok()) return s;

    src_arg_pairs->push_back({src_node, arg});
    args_.push_back(arg);
  }
  Node* dst_node = edge->dst();
  Node* dst_image = node_images.at(dst_node);
  int dst_slot = edge->dst_input();
  args_by_dst_[InputTensor(dst_node, dst_slot)] = arg_index;
  graph_->AddEdge(args_[arg_index], 0, dst_image, dst_slot);
  return Status::OK();
}

Status Encapsulator::Subgraph::RecordControlResult(
    const Edge* edge,
    const std::unordered_map<const Node*, Node*>& node_images) {
  Node* src_node = edge->src();
  Node* src_image = node_images.at(src_node);
  control_output_nodes_.insert(src_image->name());
  return Status::OK();
}

Status Encapsulator::Subgraph::RecordResult(
    const Edge* edge,
    const std::unordered_map<const Node*, Node*>& node_images) {
  Node* src_node = edge->src();
  Node* src_image = node_images.at(src_node);
  int src_slot = edge->src_output();
  std::unordered_map<OutputTensor, int, OutputTensor::Hash>::iterator iter;
  bool inserted;
  std::tie(iter, inserted) =
      results_.emplace(OutputTensor(src_node, src_slot), results_.size());
  int ret_index = iter->second;
  if (inserted) {
    NodeDef ret_def;
    NodeDefBuilder builder(
        absl::StrCat(src_node->name(), "_", src_slot, "_retval"), kRetValOp,
        NodeDebugInfo(src_node->def()));
    DataType dtype = src_node->output_type(src_slot);
    builder.Attr("T", dtype);
    builder.Attr("index", ret_index);
    builder.Input(src_image->name(), src_slot, dtype);
    Status s = builder.Finalize(&ret_def);
    if (!s.ok()) return s;
    Node* ret = graph_->AddNode(ret_def, &s);
    if (!s.ok()) return s;

    graph_->AddEdge(src_image, src_slot, ret, 0);
  }
  return Status::OK();
}

Status Encapsulator::Subgraph::MakeSequencingNode(const string& subgraph_name,
                                                  Graph* graph_out) {
  if (sequencer_ == nullptr) {
    NodeDef seq_def;
    // TODO(shikharagarwal): What source node should we use for errors?
    NodeDefBuilder builder(absl::StrCat(subgraph_name, "_sequencer"), "NoOp");
    builder.Attr(kCgmodeHostTransferSequencerAttr, subgraph_name);
    builder.Device(device_);
    Status s = builder.Finalize(&seq_def);
    if (!s.ok()) return s;

    sequencer_ = graph_out->AddNode(seq_def, &s);
    if (!s.ok()) return s;
  }
  return Status::OK();
}

void Encapsulator::Subgraph::ConnectSequencerToCallNode(Graph* graph_out) {
  if (sequencer_ != nullptr) {
    VLOG(2) << "ConnectSequencerToCallNode";
    graph_out->AddControlEdge(sequencer_, call_node_,
                              /* allow_duplicates= */ true);
  }
}

Status Encapsulator::Subgraph::ValidateAndBuild(
    const string& name_in, const ValidateCgmodeSubgraphFn& rewrite_subgraph_fn,
    FunctionLibraryDefinition* library) {
  // name_in is copied here because name may be modified below if
  // rewrite_subgraph_fn is true.
  string name = name_in;
  call_node_def_.set_op(name);
  call_node_def_.set_name(name);
  call_node_def_.set_device(device_);
  AddNodeAttr(kCgmodeCompiledAttr, true, &call_node_def_);

  function_def_name_ = name;

  FunctionDef fdef;
  auto lookup = [this](const Node* node) -> absl::optional<string> {
    if (control_output_nodes_.contains(node->name())) {
      return absl::make_optional(node->name());
    }
    return absl::nullopt;
  };
  // Verify that the graph has well-formed control flow structure.
  std::vector<ControlFlowInfo> dummy;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph_.get(), &dummy));
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*graph_, name, lookup, &fdef));

  const FunctionDef* original_fdef = library->Find(name);
  if (original_fdef == nullptr) {
    TF_RETURN_IF_ERROR(library->AddFunctionDef(fdef));
  } else if (!FunctionDefsEqual(*original_fdef, fdef)) {
    TF_RETURN_IF_ERROR(library->ReplaceFunction(name, fdef));
  }

  if (rewrite_subgraph_fn) {
    TF_RETURN_IF_ERROR(rewrite_subgraph_fn(&graph_, &call_node_def_));
  }
  return Status::OK();
}

Status Encapsulator::Subgraph::ReplaceFunctionDef(
    FunctionLibraryDefinition* library) {
  const string& name = function_def_name_;

  FunctionDef fdef;
  TF_RETURN_IF_ERROR(GraphToFunctionDef(*graph_, name, &fdef));

  if (VLOG_IS_ON(1)) {
    VLOG(2) << "Replace function def " << name;
    DumpGraphToFile(absl::StrCat("replace_encapsulate_fdef_graph_", name),
                    *graph_, library);
    DumpFunctionDefToFile(absl::StrCat("replace_encapsulate_fdef_", name),
                          fdef);
  }

  TF_RETURN_IF_ERROR(library->ReplaceFunction(name, fdef));
  return Status::OK();
}

Status Encapsulator::Subgraph::AddFunctionCallNode(
    const std::unordered_map<const Node*, Node*>& node_images,
    Graph* graph_out) {
  Status s;
  call_node_ = graph_out->AddNode(call_node_def_, &s);
  if (!s.ok()) return s;

  // Copy the assigned device and the key_annotation over.
  call_node_->set_assigned_device_name(device_);

  return Status::OK();
}

Status Encapsulator::GetGroupNameAttr(Node const* node, string* attr) const {
  AttrSlice attrs = node->attrs();
  attr->clear();
  bool found_group_attribute = false;
  for (const auto& node_attr : attrs) {
    if (node_attr.first == group_attribute_) {
      TF_RETURN_IF_ERROR(AttrValueHasType(node_attr.second, "string"));
      *attr = node_attr.second.s();
      found_group_attribute = true;
      break;
    }
  }
  return Status::OK();
}

bool IsInSubgraph(const string& group_id) { return !group_id.empty(); }

Status Encapsulator::CopySubgraphNodes(
    std::unordered_map<const Node*, Node*>* node_images) {
  for (Node* node : graph_in_->op_nodes()) {
    string group_id;
    TF_RETURN_IF_ERROR(GetGroupNameAttr(node, &group_id));
    if (!IsInSubgraph(group_id)) continue;

    Subgraph& subgraph = subgraphs_[group_id];
    Node* image = subgraph.MakeNodeImage(graph_in_, node);
    image->ClearAttr(group_attribute_);
    (*node_images)[node] = image;
  }
  return Status::OK();
}

Status Encapsulator::CopySubgraphEdges(
    const std::unordered_map<const Node*, Node*>& node_images,
    std::vector<std::pair<const Node*, Node*>>* src_arg_pairs) {
  for (const Edge* edge : graph_in_->edges()) {
    string src_group_id;
    TF_RETURN_IF_ERROR(GetGroupNameAttr(edge->src(), &src_group_id));
    string dst_group_id;
    TF_RETURN_IF_ERROR(GetGroupNameAttr(edge->dst(), &dst_group_id));
    Node* src_image = gtl::FindWithDefault(node_images, edge->src(), nullptr);
    Node* dst_image = gtl::FindWithDefault(node_images, edge->dst(), nullptr);

    // Copy edges that are local to a subgraph.
    if (IsInSubgraph(src_group_id) && IsInSubgraph(dst_group_id) &&
        src_group_id == dst_group_id) {
      Graph* g = subgraphs_[src_group_id].GetGraph();
      if (edge->IsControlEdge()) {
        g->AddControlEdge(src_image, dst_image,
                          /* allow_duplicates= */ true);
      } else {
        g->AddEdge(src_image, edge->src_output(), dst_image, edge->dst_input());
      }
      continue;
    }

    // Record 'src' as an output of its subgraph, if applicable.
    if (IsInSubgraph(src_group_id)) {
      if (!edge->IsControlEdge()) {
        DataType dtype = edge->src()->output_type(edge->src_output());
        if (IsRefType(dtype)) {
          return errors::InvalidArgument(
              "Ref Tensors (e.g., Variables) are not supported as results: "
              "tensor ",
              edge->src()->name(), ":", edge->src_output());
        }
      }

      Subgraph& src_subgraph = subgraphs_[src_group_id];
      if (edge->IsControlEdge()) {
        TF_RETURN_IF_ERROR(src_subgraph.RecordControlResult(edge, node_images));
      } else {
        TF_RETURN_IF_ERROR(src_subgraph.RecordResult(edge, node_images));
      }
    }

    // Record 'dst' as an input of its subgraph, if applicable.
    if (IsInSubgraph(dst_group_id)) {
      // Look at the type of the destination not the source, since Ref output
      // Tensors can be automatically cast to non-Ref Tensors at the
      // destination.
      if (!edge->IsControlEdge()) {
        DataType dtype = edge->dst()->input_type(edge->dst_input());
        if (IsRefType(dtype)) {
          return errors::InvalidArgument(
              "Ref Tensors (e.g., Variables) are not supported as args: "
              "tensor ",
              edge->src()->name(), ":", edge->src_output());
        }
      }

      Subgraph& dst_subgraph = subgraphs_[dst_group_id];
      // Ignore control edges entering the subgraph. We will lift them onto
      // the enclosing call operators in BuildOutputGraph().
      if (!edge->IsControlEdge()) {
        TF_RETURN_IF_ERROR(
            dst_subgraph.RecordArg(edge, node_images, src_arg_pairs));
      }
    }
  }
  return Status::OK();
}

Status Encapsulator::SplitIntoSubgraphs() {
  Status s;

  // Map from input graph nodes to subgraph nodes.
  std::unordered_map<const Node*, Node*> node_images;

  // Each entry of src_arg_pairs is a pair whose first element is a node in the
  // original graph that has an output edge in the subgraph, and whose second
  // element is the arg node in the subgraph that it sends to. The vector will
  // be filled in below in AddArgs.
  std::vector<std::pair<const Node*, Node*>> src_arg_pairs;

  TF_RETURN_IF_ERROR(CopySubgraphNodes(&node_images));
  TF_RETURN_IF_ERROR(CopySubgraphEdges(node_images, &src_arg_pairs));

  // MarkGuaranteedConstants(*graph_in_, src_arg_pairs);

  for (auto& entry : subgraphs_) {
    Subgraph& subgraph = entry.second;
    FixupSourceAndSinkEdges(subgraph.GetGraph());
  }

  return s;
}

Status Encapsulator::ValidateAndBuild(
    const ValidateCgmodeSubgraphFn& rewrite_subgraph_fn,
    FunctionLibraryDefinition* library) {
  for (auto& subgraph_entry : subgraphs_) {
    string name = subgraph_entry.first;
    Subgraph& subgraph = subgraph_entry.second;
    TF_RETURN_IF_ERROR(
        subgraph.ValidateAndBuild(name, rewrite_subgraph_fn, library));
  }
  return Status::OK();
}

Status Encapsulator::CopyNodesToOutputGraph(
    Graph* graph_out, std::unordered_map<const Node*, Node*>* node_images) {
  for (Node* node : graph_in_->op_nodes()) {
    string group_id;
    TF_RETURN_IF_ERROR(GetGroupNameAttr(node, &group_id));

    // Don't copy nodes that are going to be encapsulated.
    if (IsInSubgraph(group_id)) continue;

    Node* image = graph_out->CopyNode(node);
    (*node_images)[node] = image;
  }
  (*node_images)[graph_in_->source_node()] = graph_out->source_node();
  (*node_images)[graph_in_->sink_node()] = graph_out->sink_node();
  return Status::OK();
}

Status Encapsulator::AddCudaGraphModeCallNodes(
    const std::unordered_map<const Node*, Node*>& node_images,
    Graph* graph_out) {
  for (auto& subgraph_entry : subgraphs_) {
    TF_RETURN_IF_ERROR(
        subgraph_entry.second.AddFunctionCallNode(node_images, graph_out));
  }
  return Status::OK();
}

Status Encapsulator::FindOutputImageOfEdgeSrc(
    const string& src_group_id, const string& dst_group_id,
    const std::unordered_map<const Node*, Node*>& node_images,
    const Node* original_src_node, Node** src_image) {
  if (IsInSubgraph(src_group_id)) {
    // The edge is from a subgraph to a regular node in the output graph so
    // use the subgraph's call node output.
    *src_image = subgraphs_.at(src_group_id).GetCallNode();
  } else {
    // The source of the edge is in the output graph so use the node image in
    // the output graph.
    *src_image = node_images.at(original_src_node);
  }
  return Status::OK();
}

int Encapsulator::FindOutputSlotOfEdgeSrc(const string& src_group_id,
                                          const string& dst_group_id,
                                          const Edge* edge) {
  if (IsInSubgraph(src_group_id)) {
    const Subgraph& src_subgraph = subgraphs_.at(src_group_id);
    // 'src' is in a subgraph and 'dst' is a regular node in the output
    // graph. Use the corresponding call output instead.
    return src_subgraph.GetResultIndexForEdge(edge);
  } else {
    // The source of the edge is in the output graph so use the regular edge
    // slot.
    return edge->src_output();
  }
}

Status Encapsulator::FindOutputImageOfEdgeDst(
    const string& src_group_id, const string& dst_group_id,
    const std::unordered_map<const Node*, Node*>& node_images,
    const Node* original_dst_node, Node** dst_image) {
  if (IsInSubgraph(dst_group_id)) {
    // The edge is to a subgraph from a regular node in the output graph so
    // use the subgraph's call node input.
    *dst_image = subgraphs_.at(dst_group_id).GetCallNode();
  } else {
    // The destination of the edge is in the output graph so use the node image
    // in the output graph.
    *dst_image = node_images.at(original_dst_node);
  }
  return Status::OK();
}

int Encapsulator::FindOutputSlotOfEdgeDst(const string& src_group_id,
                                          const string& dst_group_id,
                                          const Edge* edge) {
  if (IsInSubgraph(dst_group_id)) {
    const Subgraph& dst_subgraph = subgraphs_.at(dst_group_id);
    // 'dst' is in a subgraph and 'src' is a regular node in the output
    // graph. Use the corresponding call input instead.
    return dst_subgraph.GetArgIndexForEdge(edge);
  } else {
    // The destination of the edge is in the output graph so use the regular
    // edge slot.
    return edge->dst_input();
  }
}

Status Encapsulator::CopyEdgeToOutputGraph(
    const Edge* edge, const string& src_group_id, const string& dst_group_id,
    const std::unordered_map<const Node*, Node*>& node_images, Graph* graph_out,
    std::unordered_set<std::pair<OutputTensor, InputTensor>,
                       OutputInputTensorPairHasher>* edges_added) {
  Node* src_image;
  TF_RETURN_IF_ERROR(FindOutputImageOfEdgeSrc(
      src_group_id, dst_group_id, node_images, edge->src(), &src_image));
  Node* dst_image;
  TF_RETURN_IF_ERROR(FindOutputImageOfEdgeDst(
      src_group_id, dst_group_id, node_images, edge->dst(), &dst_image));

  // If this is a control edge then copy it and return. Lift control edges onto
  // the enclosing call operator.
  if (edge->IsControlEdge()) {
    // Add the control edge, if we have not already added it, using the images
    // determined above (potentially call operators or RecvAtHost/SendFromHost).
    if (edges_added
            ->emplace(OutputTensor(src_image, -1), InputTensor(dst_image, -1))
            .second) {
      graph_out->AddControlEdge(src_image, dst_image,
                                /* allow_duplicates= */ true);
    }

    return Status::OK();
  }

  int src_output = FindOutputSlotOfEdgeSrc(src_group_id, dst_group_id, edge);

  int dst_input = FindOutputSlotOfEdgeDst(src_group_id, dst_group_id, edge);

  // Add the edge, if we have not already added it.
  if (edges_added
          ->emplace(OutputTensor(src_image, src_output),
                    InputTensor(dst_image, dst_input))
          .second) {
    graph_out->AddEdge(src_image, src_output, dst_image, dst_input);
  }
  return Status::OK();
}

Status Encapsulator::AddEdgesToOutputGraph(
    const std::unordered_map<const Node*, Node*>& node_images,
    Graph* graph_out) {
  // Set of edges already added to the output graph, represented as (src, dst)
  // pairs. We use the set to deduplicate edges; multiple edges in the input
  // graph may map to one edge in the output graph.
  std::unordered_set<std::pair<OutputTensor, InputTensor>,
                     OutputInputTensorPairHasher>
      edges_added;

  for (const Edge* edge : graph_in_->edges()) {
    string src_group_id;
    TF_RETURN_IF_ERROR(GetGroupNameAttr(edge->src(), &src_group_id));
    string dst_group_id;
    TF_RETURN_IF_ERROR(GetGroupNameAttr(edge->dst(), &dst_group_id));

    // Ignore edges that are strictly contained within one subgraph, unless
    // we are constructing parallel check graphs.
    if (IsInSubgraph(src_group_id) && IsInSubgraph(dst_group_id) &&
        src_group_id == dst_group_id) {
      continue;
    }

    // We have an edge that crosses a cluster boundary or is entirely within the
    // unclustered graph.
    TF_RETURN_IF_ERROR(CopyEdgeToOutputGraph(edge, src_group_id, dst_group_id,
                                             node_images, graph_out,
                                             &edges_added));
  }

  for (auto& subgraph_entry : subgraphs_) {
    Subgraph& subgraph = subgraph_entry.second;
    subgraph.ConnectSequencerToCallNode(graph_out);
  }

  return Status::OK();
}

namespace {

// Adds a dummy Const node to graph_out. The "constant" has the type of
// data_type and the shape indicated in 'shape'. The dummy node is not a valid
// Const node because it does not have any value defined, but this doesn't
// matter because it will only be used subsequently for shape inference. (It
// would be possible to add a switch statement over data_type to create a value
// for the constant, but that would entail maintaining the logic as new types
// are added, and is not necessary.) If the node being replaced was within a
// control flow frame, adds appropriate Enter nodes so that the use of the Const
// is well-formed.
Node* AddDummyShapedNode(const Node* src_node, int src_port,
                         const std::vector<ControlFlowInfo>& control_flow_info,
                         const TensorShapeProto& shape, Graph* graph_out) {
  DataType data_type = src_node->output_type(src_port);
  TensorProto dummy_proto;
  dummy_proto.set_dtype(data_type);
  *dummy_proto.mutable_tensor_shape() = shape;
  // Don't set any value field in the proto, since it is only going to be used
  // for shape inference.

  GraphDefBuilder::Options options(graph_out, /*status=*/nullptr);
  NodeBuilder node_builder(options.GetNameForOp("KnownShape"), "Const",
                           options.op_registry());
  node_builder.Attr("dtype", data_type).Attr("value", dummy_proto);
  Node* node = options.FinalizeBuilder(&node_builder);
  // Add any Enter nodes required to bring the constant to the correct control
  // flow frame.
  while (!control_flow_info[src_node->id()].frame_name.empty()) {
    NodeDebugInfo debug_info(*src_node);
    NodeBuilder enter_builder(options.GetNameForOp("Enter"), "Enter",
                              options.op_registry(), &debug_info);
    enter_builder.Attr("frame_name",
                       control_flow_info[src_node->id()].frame_name);
    enter_builder.Attr("is_constant", true);
    enter_builder.Input(node, 0);
    Node* enter_node = options.FinalizeBuilder(&enter_builder);
    // Adopt the new Enter node as the value in the current frame.
    node = enter_node;
    // Recurse to the parent frame to see if more Enter nodes need to be added.
    src_node = control_flow_info[src_node->id()].parent_frame;
  }
  return node;
}

}  // namespace

Status Encapsulator::MakePrunedGraphCopyAndInline(
    const Graph& graph, const std::vector<Node*>& sink_nodes,
    std::unique_ptr<Graph>* pruned_graph,
    std::unordered_map<const Node*, Node*>* node_images,
    FunctionLibraryDefinition* library) {
  // First copy all ancestor nodes of sink_nodes into a new graph.
  pruned_graph->reset(new Graph(library));
  (*pruned_graph)->set_versions(graph.versions());
  ReverseDFSFrom(graph, sink_nodes,
                 /*enter=*/nullptr,
                 /*leave=*/[&](Node* n) {
                   if (!n->IsSource()) {
                     Node* copied = (*pruned_graph)->CopyNode(n);
                     node_images->emplace(n, copied);
                   }
                 });

  // Add all the edges between copied nodes.
  for (auto entry : *node_images) {
    const Node* orig = entry.first;
    Node* image = entry.second;
    for (const Edge* out_edge : orig->out_edges()) {
      auto iter = node_images->find(out_edge->dst());
      if (iter != node_images->end()) {
        // The source and destination are both in the copied graph.
        (*pruned_graph)
            ->AddEdge(image, out_edge->src_output(), iter->second,
                      out_edge->dst_input());
      }
    }
  }

  // Find all the function call nodes, and inline them.
  std::vector<Node*> function_nodes;
  for (auto node : (*pruned_graph)->nodes()) {
    const OpRegistrationData* op_reg_data;
    TF_RETURN_IF_ERROR(library->LookUp(node->type_string(), &op_reg_data));
    if (op_reg_data->is_function_op) {
      function_nodes.push_back(node);
    }
  }
  for (auto node : function_nodes) {
    VLOG(2) << "Inlining function " << node->name();
    const FunctionDef* fdef = library->Find(node->type_string());
    if (fdef == nullptr) {
      return errors::Internal("Failed to find function ", node->type_string(),
                              " in function library.");
    }
    std::unique_ptr<FunctionBody> fbody;
    TF_RETURN_IF_ERROR(
        FunctionDefToBodyHelper(*fdef, node->attrs(), library, &fbody));

    InlineFunctionBodyOptions inline_opts;
    TF_RETURN_IF_ERROR(InlineFunctionBody(*library, pruned_graph->get(), node,
                                          fbody.get(), inline_opts));
  }

  return Status::OK();
}

Status Encapsulator::BuildOutputGraph(Graph* graph_out) {
  // Map from nodes in the input graph to nodes in the output graph.
  std::unordered_map<const Node*, Node*> node_images;

  TF_RETURN_IF_ERROR(CopyNodesToOutputGraph(graph_out, &node_images));
  TF_RETURN_IF_ERROR(AddCudaGraphModeCallNodes(node_images, graph_out));
  TF_RETURN_IF_ERROR(AddEdgesToOutputGraph(node_images, graph_out));
  VLOG(1) << "graph in: " << DebugString(graph_in_);
  for (auto n : graph_in_->nodes()) {
    VLOG(1) << n->DebugString();
  }
  VLOG(1) << "graph out: " << DebugString(graph_out);
  for (auto n : graph_out->nodes()) {
    VLOG(1) << n->DebugString();
  }

  return Status::OK();
}

}  // anonymous namespace

Status EncapsulateCGModeSubgraphsInFunctions(
    string group_attribute, const Graph& graph_in,
    const ValidateCgmodeSubgraphFn& rewrite_subgraph_fn,
    std::unique_ptr<Graph>* graph_out, FunctionLibraryDefinition* library) {
  Encapsulator encapsulator(std::move(group_attribute), &graph_in);
  TF_RETURN_IF_ERROR(encapsulator.SplitIntoSubgraphs());

  TF_RETURN_IF_ERROR(
      encapsulator.ValidateAndBuild(rewrite_subgraph_fn, library));

  std::unique_ptr<Graph> out(new Graph(library));
  out->set_versions(graph_in.versions());
  TF_RETURN_IF_ERROR(encapsulator.BuildOutputGraph(out.get()));

  *graph_out = std::move(out);
  return Status::OK();
}

// Finds the types of the _Arg nodes, indexed by position.
static Status GetArgTypes(const Graph& graph, DataTypeVector* types) {
  for (Node* n : graph.op_nodes()) {
    if (n->type_string() == kArgOp) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      if (index < 0 || index >= types->size()) {
        return errors::InvalidArgument("Invalid argument number");
      }
      (*types)[index] = n->output_type(0);
    }
  }
  return Status::OK();
}

// Renumber the indices of _Arg nodes in a graph, according to
// 'permutation' that maps old indices to new indices.
static Status RenumberArguments(Graph* graph,
                                const std::vector<int>& permutation) {
  for (Node* n : graph->op_nodes()) {
    if (n->type_string() == kArgOp) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      if (index < 0 || index >= permutation.size()) {
        return errors::InvalidArgument("Invalid argument number");
      }
      n->AddAttr("index", permutation[index]);
    }
  }
  return Status::OK();
}

Status EncapsulateCGModeSubgraphsPass::Run(
    const GraphOptimizationPassOptions& options) {
  if (!options.session_options->config.gpu_options().cuda_graph_enable_jit()) {
    return Status::OK();
  }
  VLOG(1) << "EncapsulateCGModeSubgraphsPass::Run";
  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("encapsulate_cuda_graph_mode_subgraphs_before",
                    **options.graph, options.flib_def);
  }

  std::unique_ptr<Graph> graph_out;

  // TODO create a cuda graph run session
  FunctionLibraryDefinition* const library = options.flib_def;
  // validate subgraphs for cuda graph execution
  SessionOptions session_options;
  auto* device_count = session_options.config.mutable_device_count();
  device_count->insert({"GPU", 1});
  std::vector<std::unique_ptr<Device>> devices;

  DeviceFactory* gpu_factory = DeviceFactory::GetFactory("GPU");
  if (!gpu_factory) {
    return errors::NotFound(
        "GPU Factory not registered. Can't run EncapsulateCGModeSubgraphsPass");
  }
  TF_RETURN_IF_ERROR(gpu_factory->CreateDevices(
      session_options, "/job:localhost/replica:0/task:0", &devices));
  if (devices.empty()) {
    return errors::NotFound(
        "Failed to create a GPU device for EncapsulateCGModeSubgraphsPass");
  }

  std::unique_ptr<DeviceMgr> device_mgr =
      absl::make_unique<DeviceMgr>(std::move(devices));
  OptimizerOptions opts;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(device_mgr.get(),
                                        options.session_options->env,
                                        TF_GRAPH_DEF_VERSION, library, opts));
  FunctionLibraryRuntime* flr =
      pflr->GetFLR("/job:localhost/replica:0/task:0/device:GPU:0");

  if (flr == nullptr) {
    return errors::Internal(
        "Failed to create and retrieve function library runtime to run "
        "cuda graph mode validation");
  }
  auto rewrite_subgraph = [flr](std::unique_ptr<Graph>* subgraph,
                                NodeDef* node) {
    // TODO: instantiate subgraphs via flr and validate cuda graph constraints.
    return Status::OK();
  };

  TF_RETURN_WITH_CONTEXT_IF_ERROR(EncapsulateCGModeSubgraphsInFunctions(
                                      kCGModeClusterAttr, **options.graph,
                                      rewrite_subgraph, &graph_out, library),
                                  "EncapsulateCGModeSubgraphsPass failed");

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("encapsulate_cuda_graph_mode_subgraphs_after", *graph_out,
                    options.flib_def);
  }

  *options.graph = std::move(graph_out);
  return Status::OK();
}

bool IsCgmodeCompiledKernel(const Node& node) {
  bool is_compiled_by_cgmode = false;
  bool has_cgmode_compiled_attr =
      TryGetNodeAttr(node.attrs(), kCgmodeCompiledAttr,
                     &is_compiled_by_cgmode) &&
      is_compiled_by_cgmode;
  return has_cgmode_compiled_attr ? is_compiled_by_cgmode : false;
}

}  // namespace tensorflow
