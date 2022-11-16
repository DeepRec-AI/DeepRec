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

#include "tensorflow/compiler/jit/build_cuda_graph_mode_ops_pass.h"

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/logging_ops.h"
#include "tensorflow/compiler/jit/cuda_graph_mode_cluster_util.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/device_util.h"
#include "tensorflow/compiler/jit/encapsulate_cuda_graph_mode_subgraphs_pass.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/cc/ops/xla_jit_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace {
struct DebuggingOpts {
  // If true, insert Print nodes to print every output from an cluster.
  bool print_outputs;

  // If true, insert CheckNumerics nodes for every floating point typed input to
  // an cluster.
  bool check_input_numerics;

  // If true, insert CheckNumerics nodes for every floating point typed output
  // from an cluster.
  bool check_output_numerics;
};

void MoveOutgoingEdges(Graph* g, Node* old_node, Node* new_node) {
  std::vector<const Edge*> out_edges(old_node->out_edges().begin(),
                                     old_node->out_edges().end());
  for (const Edge* edge : out_edges) {
    // TODO(sanjoy): This does not update NodeDef inputs.  To be able to update
    // NodeDef inputs we first need to fix encapsulate_subgraphs_pass to fix up
    // the NodeDef inputs to the function call nodes.
    g->AddEdge(new_node, edge->src_output(), edge->dst(), edge->dst_input());
    g->RemoveEdge(edge);
  }
}

// Returns a data value that is dead iff `control` is dead.
Output ControlToData(const Scope& scope, Node* control) {
  Output data = ops::Const(scope.WithOpName("ctrl_as_data"),
                           Tensor(DT_BOOL, TensorShape({0})));
  scope.graph()->AddControlEdge(control, data.node());
  return Output(data.node());
}

// Returns an operation that can be control-depended on that is dead iff `data`
// is dead.
Operation DataToControl(const Scope& scope, Output data) {
  return Operation(
      ops::Identity(scope.WithOpName("data_as_ctrl"), data).node());
}

// Replaces each outgoing edge from `old_node` with a merge node that merges in
// the corresponding output from `new_node`.
void MergeOutgoingDataEdges(const Scope& s, Node* old_node, Node* new_node,
                            absl::string_view cluster_name,
                            const DebuggingOpts& debugging_opts) {
  if (!s.status().ok()) {
    return;
  }

  std::vector<Output> merged_outputs(old_node->num_outputs(), Output(nullptr));

  std::vector<const Edge*> data_edges;
  absl::c_copy_if(old_node->out_edges(), std::back_inserter(data_edges),
                  [](const Edge* e) { return !e->IsControlEdge(); });

  for (const Edge* e : data_edges) {
    int oidx = e->src_output();
    Output merged_output = merged_outputs[oidx];
    if (merged_output.node() == nullptr) {
      Output new_output(new_node, oidx);
      if (debugging_opts.print_outputs) {
        string cpu_device = "/job:localhost/replica:0/task:0/device:CPU:0";
        ops::Print print_op(s.WithOpName("print_", oidx)
                                .WithDevice(cpu_device)
                                .WithAssignedDevice(cpu_device),
                            new_output, {new_output},
                            ops::Print::Attrs{}
                                .Message(absl::StrCat("output ", oidx, " from ",
                                                      old_node->name(), " is "))
                                .FirstN(1000)
                                .Summarize(-1));
        new_output = print_op;
      }

      if (debugging_opts.check_output_numerics &&
          DataTypeIsFloating(new_output.type())) {
        ops::CheckNumerics check_numerics_op(
            s.WithOpName("check_output_", oidx)
                .WithDevice(new_node->requested_device())
                .WithAssignedDevice(new_node->assigned_device_name()),
            new_output,
            absl::StrCat("CheckNumerics failed for output ", oidx, "(",
                         new_output.name(), ") from cluster ", cluster_name));
        new_output = check_numerics_op;
      }

      ops::_XlaMerge xla_merge_op(s.WithOpName("merge_oidx_", oidx),
                                  Output(old_node, oidx), new_output);
      merged_output = merged_outputs[oidx] = xla_merge_op.output;
    }

    Node* dst = e->dst();
    int dst_idx = e->dst_input();

    s.graph()->RemoveEdge(e);
    s.graph()->AddEdge(merged_output.node(), merged_output.index(), dst,
                       dst_idx);
  }
}

// Replaces each control successor of `old_node` to execute whenever either
// `old_node` or `new_node` is executed.
void MergeOutgoingControlEdges(const Scope& s, Node* old_node, Node* new_node) {
  if (!s.status().ok()) {
    return;
  }

  std::vector<const Edge*> ctrl_edges;
  absl::c_copy_if(old_node->out_edges(), std::back_inserter(ctrl_edges),
                  [](const Edge* e) { return e->IsControlEdge(); });

  if (ctrl_edges.empty()) {
    return;
  }

  if (ctrl_edges.size() == 1 && ctrl_edges.front()->dst()->IsSink()) {
    // Avoid creating a Merge node if we can just add an edge to _SINK
    // instead.
    s.graph()->AddControlEdge(new_node, s.graph()->sink_node());
    return;
  }

  // We can't merge control edges directly so we instead first "convert" them to
  // normal values that can be merged, merge the values and then "convert" the
  // merged value back into control.
  //
  // NB! We need to copy out the outgoing control edges before constructing
  // old_ctrl_as_data otherwise the control edge from old_node to the constant
  // in ControlToData will be present in ctrl_edges.

  Output old_ctrl_as_data = ControlToData(s, old_node);
  Output new_ctrl_as_data = ControlToData(s, new_node);

  ops::Merge ctrl_merge_as_data(s.WithOpName("ctrl_merge"),
                                {old_ctrl_as_data, new_ctrl_as_data});
  Operation ctrl_merge = DataToControl(s, ctrl_merge_as_data.output);

  for (const Edge* e : ctrl_edges) {
    s.graph()->AddControlEdge(ctrl_merge.node(), e->dst());
    s.graph()->RemoveControlEdge(e);
  }
}

struct CgmodeClusterInfo {
  std::vector<Output> non_constant_inputs;
  NameAttrList function;
};

Output IncomingEdgeAsOutput(const Edge* e) {
  return Output(e->src(), e->src_output());
}

Status GetCgmodeClusterInfo(Node* n, CgmodeClusterInfo* result) {
  int num_non_constant_inputs = n->num_inputs();

  std::vector<const Edge*> input_edges_vector;
  TF_RETURN_IF_ERROR(n->input_edges(&input_edges_vector));
  absl::Span<const Edge*> input_edges(input_edges_vector);

  absl::c_transform(input_edges.subspan(0, num_non_constant_inputs),
                    std::back_inserter(result->non_constant_inputs),
                    IncomingEdgeAsOutput);

  result->function.set_name(n->type_string());
  *result->function.mutable_attr() = n->def().attr();
  return Status::OK();
}

Status CopyIncomingControlEdges(Graph* g, Node* from, Node* to) {
  for (const Edge* e : from->in_edges()) {
    if (e->IsControlEdge()) {
      g->AddControlEdge(e->src(), to);
    }
  }

  return Status::OK();
}

void RemoveAllIncomingControlEdges(Graph* g, Node* n) {
  std::vector<const Edge*> incoming_ctrl_edges;
  absl::c_copy_if(n->in_edges(), std::back_inserter(incoming_ctrl_edges),
                  [](const Edge* e) { return e->IsControlEdge(); });
  for (const Edge* e : incoming_ctrl_edges) {
    g->RemoveControlEdge(e);
  }
}

// Returns true (into `result`) if a node placed on `device` must be compiled.
Status DeviceRequiresCompilation(const jit::DeviceInfoCache& device_info_cache,
                                 jit::DeviceId device, bool* result) {
  const XlaOpRegistry::DeviceRegistration* registration =
      device_info_cache.GetCompilationDevice(device);
  *result = registration->autoclustering_policy ==
            XlaOpRegistry::AutoclusteringPolicy::kAlways;
  return Status::OK();
}

// Replaces `n` with a `PartitionedCall` op that calls the same function.
Status ReplaceFunctionCallWithPartitionedCall(
    const GraphOptimizationPassOptions& options,
    const FunctionLibraryDefinition& flib_def, Node* n, Graph* g,
    const NameAttrList& func, const Scope& root) {
  string config_string = options.session_options->config.SerializeAsString();

  int input_count = absl::c_count_if(
      n->in_edges(), [](const Edge* e) { return !e->IsControlEdge(); });

  std::vector<Output> args(input_count);
  for (const Edge* e : n->in_edges()) {
    if (!e->IsControlEdge()) {
      args[e->dst_input()] = Output(e->src(), e->src_output());
    }
  }

  ops::PartitionedCall call(
      root.WithOpName("partitioned_call"), args, n->output_types(), func,
      ops::PartitionedCall::Attrs{}.ConfigProto(config_string));

  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) {
      g->AddControlEdge(e->src(), call.operation.node());
    }
  }

  std::vector<const Edge*> edges_to_delete;

  for (const Edge* e : n->out_edges()) {
    edges_to_delete.push_back(e);
    if (e->IsControlEdge()) {
      g->AddControlEdge(call.operation.node(), e->dst());
    } else {
      g->AddEdge(call.operation.node(), e->src_output(), e->dst(),
                 e->dst_input());
    }
  }

  for (const Edge* e : edges_to_delete) {
    g->RemoveEdge(e);
  }

  g->RemoveNode(n);
  return Status::OK();
}

xla::StatusOr<jit::DeviceId> InferDeviceForCluster(
    jit::DeviceInfoCache* device_info_cache, Node* n,
    const string& function_name, const FunctionLibraryDefinition& flib_def) {
  const FunctionDef* func_def = flib_def.Find(function_name);
  TF_RET_CHECK(func_def) << "Could not find " << function_name;

  jit::DeviceSet device_set;

  for (const NodeDef& ndef : func_def->node_def()) {
    VLOG(3) << ndef.DebugString();
    if (!ndef.device().empty()) {
      TF_ASSIGN_OR_RETURN(jit::DeviceId device_id,
                          device_info_cache->GetIdFor(ndef.device()));
      device_set.Insert(device_id);
    }
  }

  if (!n->assigned_device_name().empty()) {
    // TODO(sanjoy): We need this because EncapsulateSubgraphsPass drops device
    // assignment when constant folding.  We should fix EncapsulateSubgraphsPass
    // instead.
    TF_ASSIGN_OR_RETURN(jit::DeviceId device_id,
                        device_info_cache->GetIdFor(n->assigned_device_name()));
    device_set.Insert(device_id);
  }

  TF_ASSIGN_OR_RETURN(jit::DeviceId result,
                      PickDeviceForXla(*device_info_cache, device_set,
                                       /*allow_mixing_unknown_and_cpu=*/true));
  VLOG(2) << "For " << function_name << " PickDeviceForXla("
          << device_info_cache->DebugString(device_set) << ") -> "
          << device_info_cache->GetNameFor(result);
  return result;
}

std::vector<Output> GetCgmodeRunArgs(const Scope& s,
                                     const CgmodeClusterInfo& cluster_info) {
  std::vector<Output> cgmode_run_args;
  cgmode_run_args.reserve(cluster_info.non_constant_inputs.size());
  int input_idx = 0;
  for (const Output& o : cluster_info.non_constant_inputs) {
    cgmode_run_args.push_back(o);
    input_idx++;
  }
  return cgmode_run_args;
}

Status ReplaceNodeWithCgmodeCompileAndCgmodeRun(
    jit::DeviceInfoCache* device_info_cache,
    const GraphOptimizationPassOptions& options,
    const FunctionLibraryDefinition& flib_def, Graph* g, Node* n) {
  CgmodeClusterInfo cluster_info;
  TF_RETURN_IF_ERROR(GetCgmodeClusterInfo(n, &cluster_info));

  TF_ASSIGN_OR_RETURN(
      jit::DeviceId device,
      InferDeviceForCluster(device_info_cache, n, cluster_info.function.name(),
                            flib_def));

  string device_name_str = string(device_info_cache->GetNameFor(device));

  Status status;
  Scope root = NewInternalScope(g, &status, /*refiner=*/nullptr)
                   .NewSubScope(n->name())
                   .WithDevice(n->requested_device())
                   .WithAssignedDevice(device_name_str);

  ops::_CgmodeCompile cgmode_compile(root.WithOpName("cgmode_compile"),
                                     /*args=*/cluster_info.non_constant_inputs,
                                     cluster_info.function);
  TF_RETURN_IF_ERROR(CopyIncomingControlEdges(
      g, /*from=*/n, /*to=*/cgmode_compile.key.node()));

  std::vector<Output> cgmode_run_args = GetCgmodeRunArgs(root, cluster_info);

  // "Strict" compilation:  every _CgmodeCompile invocation must compile the
  // cluster.
  ops::_CgmodeRun cgmode_run(root.WithOpName("cgmode_run"), cgmode_run_args,
                             cgmode_compile.key, n->output_types());

  MoveOutgoingEdges(g, /*old_node=*/n,
                    /*new_node=*/cgmode_run.operation.node());
  g->RemoveNode(n);

  return Status::OK();
}
}  // namespace

Status BuildCgmodeOpsPass::Run(const GraphOptimizationPassOptions& options) {
  if (!options.session_options->config.gpu_options().cuda_graph_enable_jit()) {
    return Status::OK();
  }
  Graph* graph = options.graph->get();
  // Copy out the nodes we want to rewrite to avoid modifying the graph while we
  // iterate on graph->op_nodes().
  std::vector<Node*> cgmode_compiled_kernels;
  absl::c_copy_if(graph->op_nodes(),
                  std::back_inserter(cgmode_compiled_kernels),
                  [](const Node* n) {
                    if (n->IsSend() || n->IsRecv() || n->IsControlFlow()) {
                      return false;
                    }

                    // Only compile nodes that are marked for compilation by the
                    // compilation-marking pass (via 'attr_name').
                    return IsCgmodeCompiledKernel(*n);
                  });

  jit::DeviceInfoCache device_info_cache;

  for (Node* n : cgmode_compiled_kernels) {
    TF_RETURN_IF_ERROR(ReplaceNodeWithCgmodeCompileAndCgmodeRun(
        &device_info_cache, options, *options.flib_def, graph, n));
  }

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("build_cgmode_ops", *graph, options.flib_def);
  }
  // debug
  VLOG(1) << "dump graph after build cuda graph " << DebugString(graph);
  GraphCycles cycles_graph;
  TF_ASSIGN_OR_RETURN(bool cycle_detection_graph_ok,
                      CreateCycleDetectionGraph(graph, &cycles_graph));
  if (!cycle_detection_graph_ok) {
    return errors::Internal("Could not form cycle detection graph");
  }
  std::vector<Node*> compile_nodes;
  std::vector<Node*> gpu_nodes;
  for (Node* n : graph->op_nodes()) {
    if (n->type_string() == "_CgmodeCompile") {
      compile_nodes.emplace_back(n);
      continue;
    }
    const string& device_name_str = !n->assigned_device_name().empty()
                                        ? n->assigned_device_name()
                                        : n->requested_device();
    DeviceNameUtils::ParsedName full_device_name;
    DeviceNameUtils::ParseFullName(device_name_str, &full_device_name);
    if (full_device_name.type == DEVICE_GPU) {
      gpu_nodes.emplace_back(n);
    }
  }
  std::vector<Node*> sorted_compiled_nodes;
  for (Node* n : compile_nodes) {
    bool added = false;
    for (int i = 0; i < sorted_compiled_nodes.size(); i++) {
      if (cycles_graph.IsReachable(n->id(), sorted_compiled_nodes[i]->id())) {
        sorted_compiled_nodes.insert(sorted_compiled_nodes.begin() + i, n);
        added = true;
        break;
      }
    }
    if (!added) {
      sorted_compiled_nodes.emplace_back(n);
    }
  }

  for (int i = 0; i < sorted_compiled_nodes.size(); i++) {
    if (i != sorted_compiled_nodes.size() - 1) {
      if (!cycles_graph.IsReachable(sorted_compiled_nodes[i]->id(),
                                    sorted_compiled_nodes[i + 1]->id())) {
        graph->AddControlEdge(sorted_compiled_nodes[i],
                              sorted_compiled_nodes[i + 1]);
      }
    }
    VLOG(4) << "sorted compile nodes: " << sorted_compiled_nodes[i]->name();
  }

  for (Node* n : gpu_nodes) {
    bool added = false;
    for (int i = 0; i < sorted_compiled_nodes.size(); i++) {
      if (cycles_graph.IsReachable(n->id(), sorted_compiled_nodes[i]->id())) {
        if (i != 0) {
          graph->AddControlEdge(sorted_compiled_nodes[i - 1], n);
        }
        added = true;
        break;
      }
    }
    if (!added && sorted_compiled_nodes.size() > 0) {
      graph->AddControlEdge(sorted_compiled_nodes.back(), n);
    }
  }
  VLOG(4) << "dump graph after re-org cuda graph " << DebugString(graph);
  return Status::OK();
}
}  // namespace tensorflow
