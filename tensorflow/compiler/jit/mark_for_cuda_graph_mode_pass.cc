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

#include "tensorflow/compiler/jit/mark_for_cuda_graph_mode_pass.h"

#include <atomic>
#include <deque>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/jit/compilability_check_util.h"
#include "tensorflow/compiler/jit/cuda_graph_mode_cluster_util.h"
#include "tensorflow/compiler/jit/deadness_analysis.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/device_util.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/graphcycles/graphcycles.h"
#include "tensorflow/compiler/jit/resource_operation_safety_analysis.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/resource_operation_table.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/union_find.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

absl::flat_hash_map<string, std::vector<string>>* GetCgmodeWhitelistTable() {
  // Table format: category name: {list of TF operations in that category}
  static absl::flat_hash_map<string, std::vector<string>>* result =
      new absl::flat_hash_map<string, std::vector<string>>{
          // Unary
          {"PW",
           {"ComplexAbs", "Angle", "Conj", "Abs", "Acos", "Acosh", "Asin",
            "Atan", "Atanh", "Ceil", "Cos", "Cosh", "Sin", "Exp", "Expm1",
            "Floor", "IsFinite", "IsInf", "IsNan", "Inv", "Reciprocal", "Log",
            "Log1p", "Invert", "LogicalNot", "Neg", "Rint", "Round", "Rsqrt",
            "Sigmoid", "Sign", "Sinh", "Softplus", "Softsign", "Sqrt", "Square",
            "Tan", "Tanh", "Real", "Imag", "Erf", "Erfc", "Lgamma", "Digamma",
            // Binary
            "Add", "AddV2", "Sub", "Mul", "Div", "Atan2", "Complex", "DivNoNan",
            "MulNoNan", "FloorDiv", "Xlogy", "Xdivy", "FloorMod", "BitwiseAnd",
            "BitwiseOr", "BitwiseXor", "LeftShift", "RightShift", "LogicalAnd",
            "LogicalOr", "Mod", "Maximum", "Minimum", "RealDiv",
            "ReciprocalGrad", "RsqrtGrad", "SqrtGrad", "TruncateDiv",
            "TruncateMod", "Equal", "NotEqual", "Greater", "GreaterEqual",
            "Less", "LessEqual", "SigmoidGrad", "SoftplusGrad", "SoftsignGrad",
            "TanhGrad", "Pow", "SquaredDifference", "ApproximateEqual",
            // Others
            "AddN", "Bitcast", "Cast", "ClipByValue", "Const", "Empty",
            "Identity", "IdentityN", "Relu", "Relu6", "ReluGrad", "Relu6Grad",
            "LeakyReluGrad", "Elu", "EluGrad", "Selu", "SeluGrad", "Select",
            "SelectV2", "Transpose", "ConjugateTranspose",
            "_UnaryOpsComposition",
            // The following 4 operations are converted to identity
            "PlaceholderWithDefault", "PreventGradient", "StopGradient",
            "Snapshot"}},
          // clang-format off
    {"RED",
     {"All", "Any", "Min", "Max", "Mean", "Prod", "Sum"}},
          // clang-format on
          {"PWRED",
           {"ArgMax", "ArgMin", "DiagPart", "Softmax",
            "SparseSoftmaxCrossEntropyWithLogits", "LogSoftmax"}},
          {"REDUCEWINDOW",
           {"ArgMax", "ArgMin", "DiagPart", "Softmax",
            "SparseSoftmaxCrossEntropyWithLogits", "LogSoftmax"}},
          {"REDUCEWINDOWPW", {"BiasAddGrad", "LRN", "LRNGrad"}},
          {"BN",
           {"FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3",
            "_FusedBatchNormEx", "FusedBatchNormGrad", "FusedBatchNormGradV2",
            "FusedBatchNormGradV3"}},
          {"SORT", {"TopKV2"}},  // XLA version much faster then TF version.
          {"MISC",
           // clang-format off
     {"BroadcastTo", "ExpandDims", "Fill", "NoOp",
      "Range", "Rank", "Reshape", "Shape", "ShapeN", "Size", "Squeeze",
      "Transpose", "ZerosLike", "OnesLike", "BiasAdd" /*PW + Broadcast*/,
      "BroadcastArgs", "BroadcastGradientArgs", "OneHot", "Concat", "ConcatV2",
      "ConcatOffset", "Const", "MirrorPad", "Pack", "Pad", "PadV2", "Reverse",
      "ReverseV2", "ReverseSequence", "Slice", "Split", "SplitV",
      "StridedSlice", "StridedSliceGrad", "ResourceStridedSliceAssign",
      "Tile", "Transpose", "InvertPermutation", "Unpack"}},
          {"VERIFIED",
           // clang-format off
     {"Mul", "Sub", "Prod", "ZerosLike", "Select", "Less", "ExpandDims",
      "MatMul"}}};
  // clang-format on
  return result;
}
namespace {
using DeadnessPredicate = DeadnessAnalysis::DeadnessPredicate;
using jit::DeviceId;
using jit::DeviceSet;
using xla::StatusOr;

// The clusters we create here are eventually lowered into an
// _CgmodeCompile/_CgmodeRun pair with a TF executor "fallback" that uses the
// function call.
const char* kCGModeAlreadyClustered = "_CGModeAlreadyClustered";

class MarkForCudaGraphModePassImpl {
 public:
  struct DebugOptions {
    // If true, do not respect the results of deadness analysis.
    bool ignore_deadness_checks;

    // If true, do not do safety checks to preserve TensorFlow's resource
    // variable concurrency semantics.
    bool ignore_resource_variable_checks;

    // If true, do not respect the _CGModeCompile=false attribute.
    bool ignore_cgmode_compile_attr;

    int max_cluster_size;
    int min_cluster_size;

    // Compiler fuel for the auto-clustering algorithm.
    std::atomic<int64>* fuel;

    bool dump_graphs;
  };

  MarkForCudaGraphModePassImpl(
      DebugOptions debug_options, Graph* graph,
      FunctionLibraryDefinition* flib_def, Env* env,
      OptimizerOptions::GlobalJitLevel global_jit_level)
      : debug_options_(debug_options),
        graph_(graph),
        flib_def_(flib_def),
        env_(env),
        global_jit_level_(global_jit_level) {}

  Status Run();

 private:
  // Represents a "cluster" or a connected subgraph of a TensorFlow graph.
  class Cluster {
   public:
    // Constructs a trivial cluster representing a single TF node.
    Cluster(int tf_graph_node_id, int effective_cluster_size,
            bool has_functional_control_flow, DeviceSet devices,
            absl::optional<DeviceId> resource_op_device,
            absl::optional<int> resource_var_operation_node_id,
            absl::optional<DeadnessPredicate> deadness_predicate,
            bool is_cuda_graph_mode_attr_true,
            absl::optional<string> cuda_graph_mode_scope)
        : cycles_graph_node_id_(tf_graph_node_id),
          effective_cluster_size_(effective_cluster_size),
          has_functional_control_flow_(has_functional_control_flow),
          devices_(std::move(devices)),
          resource_op_device_(resource_op_device),
          deadness_predicate_(deadness_predicate),
          is_cuda_graph_mode_attr_true_(is_cuda_graph_mode_attr_true),
          cuda_graph_mode_scope_(std::move(cuda_graph_mode_scope)) {
      if (resource_var_operation_node_id.has_value()) {
        resource_var_operation_node_ids_.push_back(
            *resource_var_operation_node_id);
      }
    }

    // Merges `other` into this cluster, and clears `other`.  This method is
    // closely tied with the implementation of `MarkForCudaGraphModePassImpl`.
    void Merge(Cluster* other);

    // If this is a trivial cluster containing only one node then return the ID
    // of that node.  May not be called otherwise.
    int GetIdOfOnlyNode() const {
      DCHECK_EQ(cluster_size(), 1);
      return cycles_graph_node_id();
    }

    // The number of TF nodes in this cluster.
    int cluster_size() const { return cluster_size_; }

    // The ID of the cluster as represented in `cycles_graph_`.
    int cycles_graph_node_id() const { return cycles_graph_node_id_; }

    // The size of the cluster excluding constant and identity nodes.
    int effective_cluster_size() const { return effective_cluster_size_; }

    // True if the cluster has functional control flow like `If` and `While`.
    bool has_functional_control_flow() const {
      return has_functional_control_flow_;
    }

    // The set of devices nodes in the cluster are placed on.
    const DeviceSet& devices() const { return devices_; }

    // If the cluster has a resource operation then the device the resource
    // operation is placed on.  A cluster may have resource ops placed only on a
    // single device.
    const absl::optional<DeviceId>& resource_op_device() const {
      return resource_op_device_;
    }

    // If not nullopt the a predicate that is true iff the cluster is alive.
    // Otherwise the user has (unsafely) disabled deadness analysis.  If this is
    // unset on a single Cluster instance then it is unset on all Cluster
    // instances.
    const absl::optional<DeadnessPredicate>& deadness_predicate() const {
      return deadness_predicate_;
    }

    // If true then the cluster has a CgmodeCompile=true attribute on one of its
    // nodes.
    bool is_cuda_graph_mode_attr_true() const {
      return is_cuda_graph_mode_attr_true_;
    }

    // If not nullopt then the all nodes in the cluster either do not have the
    // CgmodeScope attribute set or have it set to the value returned.
    const absl::optional<string>& cuda_graph_mode_scope() const {
      return cuda_graph_mode_scope_;
    }

    // Returns the TF graph node IDs for the resource variable operations in
    // this cluster.
    absl::Span<const int> resource_var_operation_node_ids() const {
      return resource_var_operation_node_ids_;
    }

    string DebugString(const Graph& graph) const {
      Node* node = graph.FindNodeId(cycles_graph_node_id());
      if (!node) {
        // This should never happen but we try to be resilient because this is a
        // debugging aid.
        return absl::StrCat("NULL NODE IN #", cycles_graph_node_id());
      }

      return absl::StrCat("<", node->name(), " + ", cluster_size(), " others #",
                          cycles_graph_node_id(), ">");
    }

   private:
    int cluster_size_ = 1;
    int cycles_graph_node_id_;
    int effective_cluster_size_;
    bool has_functional_control_flow_;
    DeviceSet devices_;
    absl::optional<DeviceId> resource_op_device_;
    absl::optional<DeadnessPredicate> deadness_predicate_;
    bool is_cuda_graph_mode_attr_true_;
    absl::optional<string> cuda_graph_mode_scope_;
    std::vector<int> resource_var_operation_node_ids_;

    TF_DISALLOW_COPY_AND_ASSIGN(Cluster);
  };

  // If `cluster` has only a single node then returns that, otherwise returns
  // nullptr.
  Node* GetOnlyNodeIn(const Cluster& cluster);

  // Returns true if `cluster` is a trivial cluster containing a "sink like"
  // node -- a NoOp node that only the Sink node control depends on.
  bool IsSinkLike(const Cluster& cluster);

  // Returns true if `cluster` looks like an "i++" operation on an integer
  // scalar resource variable.
  bool IsScalarIntegerResourceOperation(const Cluster& cluster);

  bool IsGpuOp(const Node* node) {
    const string& device_name_str = !node->assigned_device_name().empty()
                                        ? node->assigned_device_name()
                                        : node->requested_device();
    DeviceNameUtils::ParsedName full_device_name;
    DeviceNameUtils::ParseFullName(device_name_str, &full_device_name);
    return (full_device_name.type == DEVICE_GPU);
  }
  // ---------------------------------------------------------------------------
  // The pass proceeds in four steps, out of which `RunEdgeContractionLoop` and
  // `CreateClusters` do most of the heavy lifting.

  // Initializes some internal data structures.
  //
  // If this returns false then Initialize exited early (either because there is
  // nothing to do or we saw a graph that we can't handle) and not all the
  // fields in this MarkForCudaGraphModePassImpl instance are set up.
  StatusOr<bool> Initialize();

  // Runs through the entire cluster graph in post-order and calls `fn(from,
  // to)` on each edge.  `fn(from, to)` is expected to return true if it was
  // able to contract `from`->`to`.
  //
  // Returns true if `fn` returned true for any edge.
  template <typename FnTy>
  StatusOr<bool> ForEachEdgeInPostOrder(FnTy fn);

  // Contracts as many edges as possible to create GraphMode clusters.  After
  // this finishes the clustering decisions made are implicitly stored in
  // `clusters_`.
  Status RunEdgeContractionLoop();

  // Manifests the clustering decisions into the TF graph by tagging nodes with
  // an `_CgmodeCluster` attribute.  Also some basic filter logic, like
  // tf_cgmode_min_cluster_size, are applied here.
  Status CreateClusters();

  Status DumpDebugInfo();

  // Tries to contract the edge from cluster `from` to cluster `to`.  Returns
  // true if successful.
  StatusOr<bool> TryToContractEdge(Cluster* from, Cluster* to);

  // Populates `clusters_`.
  Status BuildInitialClusterSet();

  StatusOr<bool> ClusteringWillIntroduceInterDeviceDependency(
      const Cluster& from, const Cluster& to);

  // Returns true if the devices in `cluster_a` and `cluster_b` are compatible
  // and therefore not a hindrance for combining the two clusters into a larger
  // cluster.
  StatusOr<bool> AreDevicesCompatible(const Cluster& cluster_a,
                                      const Cluster& cluster_b);

  void DumpPostClusteringGraphs();

  Cluster* MakeNewCluster(int cycles_graph_node_id, int effective_cluster_size,
                          bool has_functional_control_flow,
                          const DeviceSet& device_set,
                          absl::optional<DeviceId> resource_op_device,
                          absl::optional<int> resource_var_operation_node_id,
                          absl::optional<DeadnessPredicate> deadness_predicate,
                          bool is_cuda_graph_mode_attr_true,
                          absl::optional<string> cuda_graph_mode_scope) {
    cluster_storage_.push_back(absl::make_unique<Cluster>(
        cycles_graph_node_id, effective_cluster_size,
        has_functional_control_flow, device_set, resource_op_device,
        resource_var_operation_node_id, deadness_predicate,
        is_cuda_graph_mode_attr_true, cuda_graph_mode_scope));
    return cluster_storage_.back().get();
  }

  absl::optional<string> GetCGModeScope(Node* n);

  // Returns the cluster for node `n`.  If two nodes, N1 and N2, are placed in
  // the same cluster by the clustering algorithm then this function will return
  // the same Cluster instance for N1 and N2.
  //
  // Returns nullptr if `n` is not a compilation candidate.
  Cluster* GetClusterForNode(Node* n) {
    return cluster_for_node_[n->id()].Get();
  }

  // Returns the cluster for a node in `cycles_graph_`.  This uses the same
  // underlying map because of how we set things up, but we can do an additional
  // CHECK in this accessor.
  //
  // Returns nullptr if `node_id` is not a compilation candidate.
  Cluster* GetClusterForCyclesGraphNode(int node_id) {
    // We have to check `graph_->FindNodeId(node) == nullptr` because we add all
    // nodes in [0, graph_->num_node_ids()) to the cycle detection graph but the
    // TF graph may be missing some node ids.
    if (node_id >= graph_->num_node_ids() ||
        graph_->FindNodeId(node_id) == nullptr) {
      return nullptr;
    }
    Cluster* cluster = cluster_for_node_[node_id].Get();
    if (cluster) {
      DCHECK_EQ(cluster->cycles_graph_node_id(), node_id);
    }
    return cluster;
  }

  bool LogNotContractableAndReturnFalse(Cluster* from, Cluster* to,
                                        absl::string_view reason);

  // Finds a path in `cycles_graph_` from `from` to `to` that is not a direct
  // edge from `from` to `to`.
  //
  // Tries to find a path that contains at least one unclusterable node.
  std::vector<int> FindAlternatePathForDebugging(int from, int to);

  // Returns a string representing `cycles_graph_node_id`.  If the node is
  // unclusterable (either it is a phatom "frame" node or is not a compilation
  // candidate) then set `*found_unclustered` to true.
  string DebugStringForCyclesGraphNode(int node_id, bool* found_unclustered);

  // We could not contract the edge from `from` to `to`.  Return a string
  // describing an alternate path from `from` to `to` (besides the direct edge
  // from `from` to `to`) which would have created a cycle had we contracted the
  // edge.
  //
  // Tries (if possible) to find a path that contains at least one unclusterable
  // node as it is surprising to the user if we print "A->B could not be
  // contracted because of the path [P,Q,R]" where P, Q and R are all clusters
  // since in that case a natural question is why we could not form a {A, P, Q,
  // R, B} cluster.
  string DescribePotentialCycle(int from, int to);

  // Merge the clusters `cluster_from` and `cluster_to`.  After this step the
  // larger combined cluster is represented by `cluster_from`'s ID in
  // `cycles_graph_`.
  bool MergeClusters(Cluster* cluster_from, Cluster* cluster_to) {
    int from = cluster_from->cycles_graph_node_id();
    int to = cluster_to->cycles_graph_node_id();

    if (!cycles_graph_.ContractEdge(from, to)) {
      VLOG(3) << "Could not contract " << cluster_from->DebugString(*graph_)
              << " -> " << cluster_to->DebugString(*graph_)
              << " because contracting the edge would create a cycle via "
              << DescribePotentialCycle(from, to) << ".";
      return false;
    }

    // Merge the clusters.
    cluster_from->Merge(cluster_to);

    // Merge the UnionFind<Cluster*>.
    cluster_for_node_[from].Merge(&cluster_for_node_[to]);

    return true;
  }

  string EdgeContractionFailureMsg(Cluster* from, Cluster* to,
                                   absl::string_view reason) {
    return absl::StrCat("Could not contract ", from->DebugString(*graph_),
                        " -> ", to->DebugString(*graph_), " because ", reason,
                        ".");
  }

  DebugOptions debug_options_;
  Graph* graph_;
  FunctionLibraryDefinition* flib_def_;
  Env* env_;
  OptimizerOptions::GlobalJitLevel global_jit_level_;
  absl::flat_hash_map<const Cluster*, bool> should_compile_cluster_cache_;
  jit::DeviceInfoCache device_info_cache_;

  bool initialized_ = false;
  bool edges_contracted_ = false;
  bool clusters_created_ = false;

  std::vector<std::unique_ptr<Cluster>> cluster_storage_;
  std::vector<UnionFind<Cluster*>> cluster_for_node_;
  GraphCycles cycles_graph_;
  OrderedNodeSet compilation_candidates_;
  std::unique_ptr<DeadnessAnalysis> deadness_analysis_;
  int64 iteration_count_ = 0;
  absl::flat_hash_set<std::pair<int, int>> unsafe_resource_deps_;
};

std::vector<int> MarkForCudaGraphModePassImpl::FindAlternatePathForDebugging(
    int from, int to) {
  std::vector<int> rpo = cycles_graph_.AllNodesInPostOrder();
  absl::c_reverse(rpo);

  // best_pred_for_node[n] contains a predecessor of `n` that has an
  // unclusterable node in some path from `from` to itself.
  // best_pred_for_node[n] is unpopulated for nodes that are not reachable from
  // `from`.  We build this table up inductively by traversing the cycles graph
  // in RPO.
  absl::flat_hash_map<int, int> best_pred_for_node;
  best_pred_for_node[from] = -1;

  int rpo_index = 0, current_rpo_node;
  do {
    current_rpo_node = rpo[rpo_index++];
    absl::optional<int> some_pred, preferred_pred;
    for (int pred : cycles_graph_.Predecessors(current_rpo_node)) {
      if (!best_pred_for_node.contains(pred)) {
        continue;
      }

      // Ignore the from->to edge since we're trying to find an alternate path.
      if (current_rpo_node == to && pred == from) {
        continue;
      }

      some_pred = pred;
      if (GetClusterForCyclesGraphNode(pred) == nullptr) {
        preferred_pred = pred;
      }
    }

    if (some_pred || preferred_pred) {
      best_pred_for_node[current_rpo_node] =
          preferred_pred.has_value() ? *preferred_pred : *some_pred;
    }
  } while (current_rpo_node != to);

  auto get_best_pred = [&](int n) {
    auto it = best_pred_for_node.find(n);
    CHECK(it != best_pred_for_node.end());
    return it->second;
  };

  std::vector<int> path;
  int current_path_node = get_best_pred(to);
  while (current_path_node != from) {
    path.push_back(current_path_node);
    current_path_node = get_best_pred(current_path_node);
  }

  absl::c_reverse(path);
  return path;
}

string MarkForCudaGraphModePassImpl::DebugStringForCyclesGraphNode(
    int cycles_graph_node_id, bool* found_unclustered) {
  Cluster* cluster = GetClusterForCyclesGraphNode(cycles_graph_node_id);
  if (cluster) {
    return cluster->DebugString(*graph_);
  }

  *found_unclustered = true;
  if (cycles_graph_node_id >= graph_->num_node_ids()) {
    return absl::StrCat("<oob #", cycles_graph_node_id, ">");
  }

  Node* node = graph_->FindNodeId(cycles_graph_node_id);
  if (!node) {
    return absl::StrCat("<bad #", cycles_graph_node_id, ">");
  }

  return node->name();
}

string MarkForCudaGraphModePassImpl::DescribePotentialCycle(int from, int to) {
  std::vector<string> path_str;
  bool found_unclustered = false;
  absl::c_transform(FindAlternatePathForDebugging(from, to),
                    std::back_inserter(path_str), [&](int node_id) {
                      return DebugStringForCyclesGraphNode(node_id,
                                                           &found_unclustered);
                    });
  return absl::StrCat(!found_unclustered ? "(all clusters) " : "", "[",
                      absl::StrJoin(path_str, ","), "]");
}

void MarkForCudaGraphModePassImpl::Cluster::Merge(Cluster* other) {
  // We keep our own cycles_graph_node_id_ to mirror what GraphCycles does.

  // Clearing out data structures in `other` is just a memory saving
  // optimization and not needed for correctness.

  cluster_size_ += other->cluster_size_;
  effective_cluster_size_ += other->effective_cluster_size_;
  has_functional_control_flow_ |= other->has_functional_control_flow_;

  devices_.UnionWith(other->devices_);

  DCHECK(!(resource_op_device_.has_value() &&
           other->resource_op_device_.has_value()) ||
         *resource_op_device_ == *other->resource_op_device_)
      << "AreDevicesCompatible should have returned false otherwise!";

  if (!resource_op_device_.has_value()) {
    resource_op_device_ = other->resource_op_device_;
  }

  is_cuda_graph_mode_attr_true_ |= other->is_cuda_graph_mode_attr_true_;

  if (!cuda_graph_mode_scope_.has_value()) {
    cuda_graph_mode_scope_ = std::move(other->cuda_graph_mode_scope_);
  }

  resource_var_operation_node_ids_.reserve(
      resource_var_operation_node_ids_.size() +
      other->resource_var_operation_node_ids_.size());
  absl::c_copy(other->resource_var_operation_node_ids_,
               std::back_inserter(resource_var_operation_node_ids_));
  other->resource_var_operation_node_ids_.clear();
}

Status IgnoreResourceOpForSafetyAnalysis(
    jit::DeviceInfoCache* device_info_cache, const Node& n, bool* ignore) {
  // If a resource operation is assigned to XLA_CPU or XLA_GPU explicitly then
  // ignore it during resource operation safety analysis.  We need this hack
  // because of two reasons:
  //
  //  1. Operations assigned to XLA_CPU and XLA_GPU have to always be compiled.
  //  2. We don't support live-out values of type DT_RESOURCE and live-in values
  //     of type DT_RESOURCE that are not resource variables.
  //
  // Together these imply we cannot let resource variable safety analysis
  // constrain e.g. a TensorArrayV3->TensorArrayAssignV3 edge to be in different
  // clusters: both of them will have to be clustered because of (1) and we
  // won't be able to keep the edge between the two as neither the input to the
  // second XLA cluster nor the output from the first XLA cluster are supported
  // because of (2).
  //
  // TODO(b/113100872): This can be fixed if the TensorFlow representation for
  // TensorArray and Stack on the XLA_{C|G}PU devices were the same in XLA; then
  // (2) would no longer hold.

  if (n.assigned_device_name().empty()) {
    *ignore = false;
    return Status::OK();
  }

  TF_ASSIGN_OR_RETURN(
      const XlaOpRegistry::DeviceRegistration* registration,
      device_info_cache->GetCompilationDevice(n.assigned_device_name()));

  if (!registration) {
    *ignore = true;
  } else {
    *ignore = registration->cluster_resource_variable_ops_unsafely;
  }
  return Status::OK();
}

StatusOr<bool> MarkForCudaGraphModePassImpl::Initialize() {
  TF_RET_CHECK(!initialized_ && !edges_contracted_ && !clusters_created_);
  initialized_ = true;

  std::vector<Node*> sorted_nodes;
  for (Node* node : graph_->op_nodes()) {
    sorted_nodes.push_back(node);
  }
  std::sort(sorted_nodes.begin(), sorted_nodes.end(), NodeComparatorID());
  for (Node* node : sorted_nodes) {
    compilation_candidates_.insert(node);
  }
  VLOG(2) << "compilation_candidates_.size() = "
          << compilation_candidates_.size();

  TF_ASSIGN_OR_RETURN(bool cycle_detection_graph_ok,
                      CreateCycleDetectionGraph(graph_, &cycles_graph_));
  if (!cycle_detection_graph_ok) {
    VLOG(2) << "Could not form cycle detection graph";
    return false;
  }

  if (!debug_options_.ignore_deadness_checks) {
    TF_RETURN_IF_ERROR(DeadnessAnalysis::Run(*graph_, &deadness_analysis_));
  }

  // Each compilation candidate belongs to a cluster. The cluster's
  // representative names the node in the 'cycles' graph that represents the
  // cluster.
  TF_RETURN_IF_ERROR(BuildInitialClusterSet());
  return true;
}

template <typename FnTy>
StatusOr<bool> MarkForCudaGraphModePassImpl::ForEachEdgeInPostOrder(FnTy fn) {
  bool changed = false;
  for (int32 node : cycles_graph_.AllNodesInPostOrder()) {
    Cluster* cluster_from = GetClusterForCyclesGraphNode(node);
    if (!cluster_from) {
      continue;
    }

    // Make a copy of the set of successors because we may modify the graph in
    // TryToContractEdge.
    std::vector<int32> successors_copy =
        cycles_graph_.SuccessorsCopy(cluster_from->cycles_graph_node_id());

    for (int to : successors_copy) {
      iteration_count_++;

      Cluster* cluster_to = GetClusterForCyclesGraphNode(to);
      if (!cluster_to) {
        continue;
      }

      TF_ASSIGN_OR_RETURN(bool contracted_edge, fn(cluster_from, cluster_to));
      changed |= contracted_edge;
    }
  }

  return changed;
}

Node* MarkForCudaGraphModePassImpl::GetOnlyNodeIn(const Cluster& cluster) {
  return cluster.cluster_size() == 1
             ? graph_->FindNodeId(cluster.GetIdOfOnlyNode())
             : nullptr;
}

bool MarkForCudaGraphModePassImpl::IsSinkLike(const Cluster& cluster) {
  if (Node* n = GetOnlyNodeIn(cluster)) {
    return n->type_string() == "NoOp" && n->out_edges().size() == 1 &&
           (*n->out_edges().begin())->dst()->IsSink();
  }

  return false;
}

bool MarkForCudaGraphModePassImpl::IsScalarIntegerResourceOperation(
    const Cluster& cluster) {
  Node* n = GetOnlyNodeIn(cluster);
  if (!n) {
    return false;
  }

  if (n->type_string() != "AssignAddVariableOp" &&
      n->type_string() != "AssignSubVariableOp") {
    return false;
  }

  DataType dtype;
  if (!TryGetNodeAttr(n->def(), "dtype", &dtype) || !DataTypeIsInteger(dtype)) {
    return false;
  }

  Node* const_input = nullptr;
  for (const Edge* e : n->in_edges()) {
    if (!e->IsControlEdge() && e->src()->IsConstant()) {
      const_input = e->src();
      break;
    }
  }

  if (!const_input) {
    return false;
  }

  const TensorProto* proto = nullptr;
  if (!TryGetNodeAttr(const_input->def(), "value", &proto)) {
    return false;
  }

  return TensorShapeUtils::IsScalar(proto->tensor_shape());
}

Status MarkForCudaGraphModePassImpl::RunEdgeContractionLoop() {
  TF_RET_CHECK(initialized_ && !edges_contracted_ && !clusters_created_);
  edges_contracted_ = true;

  // TODO(hpucha): Handle the case where kCGModeClusterAttr is already set (for
  // example, from the Grappler fusion pass).

  // In general there are multiple maximal clusterings, but they are not all
  // equally performant.  Some clustering decision are likely to improve
  // performance much more than others, and we cannot order contractions on this
  // cost function, nor can we look at global information while deciding on
  // individual edges to contract.  Instead, we will make decisions on these
  // important edges then make decisions on all other edges, causing the highest
  // chance of all most important edges to be contracted.
  //
  // An example of where this might occur is with a digraph:
  // {A -> B, B -> C, A -> X, X -> C} where B is a Size operation and X is
  // not-compilable. In this case, the valid clusterings are {A,B} or {B,C}. B
  // should be clustered with A because it will prevent a potentially large
  // tensor from A being computed and copied.
  //
  // To choose better maximal clusterings we make multiple iterations over the
  // graph in post-order, where each such iteration is called a "phase".

  // Phase 0: contract metadata operations with their producer.

  VLOG(4) << "Running phase 0";
  TF_RETURN_IF_ERROR(
      ForEachEdgeInPostOrder([&](Cluster* from, Cluster* to) -> StatusOr<bool> {
        // Shape consuming operations are desirable to cluster with their
        // operands because they return a small set of scalar values after
        // consuming a large amount of data.  For example, given a graph X -> Y
        // -> Size -> Z, where the possible clustering is [{X, Y, Size}, {Z}] or
        // [{X, Y}, {Size, Z}], the better clustering is Size with Y because the
        // output of size will be a small tensor while Y is a potentially large
        // tensor that must be computed and possible transposed/copied before
        // the second cluster executes.
        Node* n = GetOnlyNodeIn(*to);
        bool is_shape_consumer_op = n && IsShapeConsumerOp(*n);
        if (!is_shape_consumer_op) {
          return false;
        }

        return TryToContractEdge(from, to);
      })
          .status());

  // Phase 1: apply a heuristic to ensure that we don't mess up clustering due
  // to "group_deps".  After this phase most edges should have been contracted.

  VLOG(4) << "Running phase 1";
  TF_RETURN_IF_ERROR(
      ForEachEdgeInPostOrder([&](Cluster* from, Cluster* to) -> StatusOr<bool> {
        // We split out this phase to get good clustering in the presence of a
        // specific pattern seen in some graphs:
        //
        // digraph {
        //   ApplyWeightUpdates_0 -> "iteration++"
        //   ApplyWeightUpdates_1 -> "iteration++"
        //   ApplyWeightUpdates_2 -> "iteration++"
        //   ApplyWeightUpdates_0 -> Computation_A
        //   ApplyWeightUpdates_1 -> Computation_B
        //   ApplyWeightUpdates_2 -> Computation_C
        //   Computation_A -> NoOp
        //   Computation_B -> NoOp
        //   Computation_C -> NoOp
        //   "iteration++" -> NoOp
        // }
        //
        // In the graph above we can't cluster iteration++ with any of the
        // gradient update operations since that will break the TF resource
        // variable memory model.  Given that constraint the ideal clustering
        // would be to put all the gradient updates and all of the Computation_*
        // nodes in one cluster, and leave iteration++ and NoOp unclustered.
        //
        // A naive post-order traversal would not create this good clustering,
        // however.  Instead it will first create a cluster that puts
        // Computation_* nodes, the NoOp and iteration++ node in a single
        // cluster, after which it will fail to put any of the
        // ApplyWeightUpdates_* nodes into this cluster. To avoid this fate we
        // instead run a pass that avoids contracting edges _into_ NoOps like
        // the above, and avoid clustering edges _from_ "iteration++" like the
        // above.  Then we run a second pass that contracts the edges we could
        // not contract the first time around.

        if (IsSinkLike(*to)) {
          return false;
        }

        if (IsScalarIntegerResourceOperation(*from)) {
          return false;
        }

        return TryToContractEdge(from, to);
      })
          .status());

  // Phase 2: contract any remaining edges.  After this phase we should have a
  // maximal clustering:
  //
  // A. We visit a cluster only after maximally clustering all its children.
  // B. By the time we're done with a node all of its children that could have
  //    been absorbed into the node have been absorbed.
  // C. We have an invariant that making a cluster larger does not make edges
  //    leaving it more contractable. That is, if we have
  //    digraph { X->Y; Y->Z; } then collapsing X->Y does not make it possible
  //    to contract Y->Z if Y->Z was not contractible originally.
  VLOG(4) << "Running phase 2";
  TF_RETURN_IF_ERROR(ForEachEdgeInPostOrder([&](Cluster* from, Cluster* to) {
                       return TryToContractEdge(from, to);
                     })
                         .status());

  // Check that the conclusion made above (that iterating over the graph once in
  // post order gives a maximal clustering) holds.  Once the linear time
  // post-order scheme has been battle tested we can move this to happen only in
  // debug builds.
  VLOG(2) << "Checking idempotence";
  TF_ASSIGN_OR_RETURN(bool changed,
                      ForEachEdgeInPostOrder([&](Cluster* from, Cluster* to) {
                        return TryToContractEdge(from, to);
                      }));
  TF_RET_CHECK(!changed);

  return Status::OK();
}

std::atomic<int64> cluster_sequence_num;

int64 GetNextClusterSequenceNumber() { return cluster_sequence_num++; }

Status MarkForCudaGraphModePassImpl::CreateClusters() {
  TF_RET_CHECK(initialized_ && edges_contracted_ && !clusters_created_);
  clusters_created_ = true;

  // Names for each cluster.
  std::unordered_map<int, string> cluster_names;

  if (debug_options_.dump_graphs) {
    DumpGraphToFile("before_mark_for_compilation", *graph_, flib_def_);
  }

  // Mark clusters for compilation that:
  // * are explicitly marked for compilation (_CgmodeCompile=true), or
  // * have more than debug_options_.cgmode_min_cluster_size elements
  // (applicable
  //   only if compilation is enabled, otherwise there will be no such
  //   candidates).
  for (Node* n : compilation_candidates_) {
    Cluster* cluster = GetClusterForNode(n);
    if (!cluster) {
      continue;
    }

    // We assume that functional If and While nodes have at least
    // min_cluster_size non-trivial nodes in them.  It would be more principled
    // to (recursively) verify this fact, but that's probably not worth the
    // trouble.

    if (cluster->effective_cluster_size() >= debug_options_.min_cluster_size ||
        cluster->has_functional_control_flow() ||
        cluster->is_cuda_graph_mode_attr_true()) {
      string& name = cluster_names[cluster->cycles_graph_node_id()];

      if (name.empty()) {
        name = absl::StrCat("cluster_", GetNextClusterSequenceNumber());
      }

      // TODO: add kCGModeClusterAttr for encapsulation
      n->AddAttr(kCGModeClusterAttr, name);
      n->AddAttr(kCGModeAlreadyClustered, true);
      VLOG(1) << "Assigning node " << n->name() << " to cluster " << name;
    }
  }

  return Status::OK();
}

Status MarkForCudaGraphModePassImpl::DumpDebugInfo() {
  TF_RET_CHECK(initialized_ && edges_contracted_ && clusters_created_);

  if (debug_options_.dump_graphs) {
    DumpPostClusteringGraphs();
  }

  return Status::OK();
}

StatusOr<bool>
MarkForCudaGraphModePassImpl::ClusteringWillIntroduceInterDeviceDependency(
    const Cluster& cluster_from, const Cluster& cluster_to) {
  // If any of the consumer's producers are on a different device, do not
  // cluster these nodes. This prevents other work on this device from being
  // delayed by work on other devices. We consider predecessors of the entire
  // cluster rather than just the inputs to the node to prevent the cluster
  // still being combined in cases where the 'to' cluster has multiple
  // dependencies on the 'from' cluster and another dependency leads to a
  // merging of the clusters.
  //
  // Example:
  // Cluster0:GPU0 -> Cluster1:GPU0
  //               -> Cluster2:GPU1
  // Even if, Cluster0 and Cluster1 could be combined, it would harm parallelism
  // of the model by delaying execution of Cluster2 until all of Cluster1 had
  // finished, rather than them being independent.
  for (const auto& in_id :
       cycles_graph_.Predecessors(cluster_to.cycles_graph_node_id())) {
    const Cluster* cluster_in = GetClusterForCyclesGraphNode(in_id);
    if (cluster_in) {
      TF_ASSIGN_OR_RETURN(bool devices_compatible,
                          AreDevicesCompatible(cluster_to, *cluster_in));
      if (!devices_compatible) {
        return true;
      }
      TF_ASSIGN_OR_RETURN(devices_compatible,
                          AreDevicesCompatible(cluster_from, *cluster_in));
      if (!devices_compatible) {
        return true;
      }
    }
  }

  // Do the operation described above, also in reverse. Parallelism can also be
  // ruined by a producer that is used by the same device and other devices.
  // Prevent clustering with its consumers to allow the other devices to be
  // unblocked as soon as possible.
  //
  // Example:
  // Cluster0:GPU0 -> Cluster2:GPU0
  // Cluster1:GPU1 /
  // Even if, Cluster0 and Cluster2 could be combined, it would harm parallelism
  // of the model by delaying execution of Cluster0 until all of Cluster1 had
  // finished, rather than them being independent.
  for (const auto& out_id :
       cycles_graph_.Successors(cluster_from.cycles_graph_node_id())) {
    const Cluster* cluster_out = GetClusterForCyclesGraphNode(out_id);
    if (cluster_out) {
      TF_ASSIGN_OR_RETURN(bool devices_compatible,
                          AreDevicesCompatible(cluster_from, *cluster_out));
      if (!devices_compatible) {
        return true;
      }
      TF_ASSIGN_OR_RETURN(devices_compatible,
                          AreDevicesCompatible(cluster_to, *cluster_out));
      if (!devices_compatible) {
        return true;
      }
    }
  }

  return false;
}

absl::optional<string> MarkForCudaGraphModePassImpl::GetCGModeScope(
    Node* node) {
  // Look for either _CGModeScope or _CGModeInternalScope on both nodes to guide
  // clustering.  If both nodes have a scope and the scopes do not match, do
  // not cluster along this edge.  If even one of the nodes lacks a scope
  // attribute, then it is treated as a "bridge" and a cluster may be created
  // along it.
  //
  // The difference between _CGModeScope and _CGModeInternalScope is that
  // _CGModeScope is provided by users through jit_scope APIs, while
  // _CGModeInternalScope is automatically generated by the ClusterScopingPass
  // when auto_jit is on.  As such, we respect _CGModeScope only when auto_jit
  // is off, while respecting _CGModeInternalScope only when auto_jit is on.
  //
  // We may want to restrict the _CGModeScope behavior to require all nodes
  // marked with _CGModeCompile=true to also have a _CGModeScope property set
  // (and raise an error otherwise); but for now we don't do this.

  if (global_jit_level_ != OptimizerOptions::OFF) {
    // If global_jit_level_ is ON, respect only _CgmodeInternalScope.
    const string& scope =
        GetNodeAttrString(node->attrs(), kCGModeInternalScopeAttr);
    if (!scope.empty()) {
      return scope;
    }
  } else {
    // If global_jit_level_ is OFF, respect only _CgmodeScope.
    const string& scope = GetNodeAttrString(node->attrs(), kCGModeScopeAttr);
    if (!scope.empty()) {
      return scope;
    }
  }

  return absl::nullopt;
}

Status MarkForCudaGraphModePassImpl::BuildInitialClusterSet() {
  auto ignore_resource_ops = [&](const Node& n, bool* ignore) {
    return IgnoreResourceOpForSafetyAnalysis(&device_info_cache_, n, ignore);
  };

  std::vector<std::pair<int, int>> unsafe_resource_deps_vect;
  TF_RETURN_IF_ERROR(ComputeIncompatibleResourceOperationPairs(
      *graph_, flib_def_, ignore_resource_ops, &unsafe_resource_deps_vect));
  absl::c_copy(
      unsafe_resource_deps_vect,
      std::inserter(unsafe_resource_deps_, unsafe_resource_deps_.begin()));

  cluster_for_node_.resize(graph_->num_node_ids());
  for (Node* node : graph_->nodes()) {
    // adopt a blacklist here to prevent erroneous compilation.
    if (node->IsSource() || node->IsSink() || node->IsArg() ||
        node->IsRetval() || node->IsRunGraph() || node->IsSend() ||
        node->IsHostSend() || node->IsVariable() || node->IsRecv() ||
        node->IsHostRecv() || node->IsFuseRecv() || node->IsHostFuseRecv() ||
        node->IsStage() || node->IsUnstage() || node->IsSwitch() ||
        node->IsNextIteration() || node->IsMerge() || node->IsEnter() ||
        node->IsExit() || node->IsLoopCond() || node->IsControlTrigger() ||
        node->IsGetSessionHandle() || node->IsGetSessionTensor() ||
        node->IsDeleteSessionTensor() || node->IsControlFlow() ||
        node->IsScopedAllocator() || node->IsCollective() ||
        node->IsMetadata() || node->IsFakeParam() ||
        node->IsPartitionedCall() || node->IsKvVarHandle() ||
        node->IsApplyFtrlOps() || node->IsSparseApplyFtrlOps() ||
        node->IsPlaceholder() || node->type_string() == "NoOp" ||
        node->type_string() == "Reshape" ||
        node->type_string() == "VarHandleOp" ||
        node->type_string() == "AssignVariableOp" ||
        node->type_string() == "RestoreV2" || node->type_string() == "All" ||
        node->type_string() == "Tile" || node->type_string() == "Sum" ||
        node->type_string() == "Cast" || node->type_string() == "Slice" ||
        node->type_string() == "StridedSlice" ||
        node->type_string() == "Where" || node->type_string() == "Unique" ||
        node->type_string() == "OneHot" ||
        node->type_string() == "SparseReshape" ||
        node->type_string() == "SparseToDense" ||
        node->type_string() == "GatherV2" || node->type_string() == "Pack" ||
        node->type_string() == "ConcatV2" ||
        node->type_string() == "SparseFillEmptyRows" ||
        node->type_string() == "SparseSegmentSum") {
      cluster_for_node_[node->id()].Get() = nullptr;
      continue;
    }

    int effective_cluster_size =
        (node->IsIdentity() || node->IsConstant()) ? 0 : 1;

    bool has_functional_control_flow = node->IsWhileNode() || node->IsIfNode();

    absl::optional<DeadnessPredicate> deadness_predicate;
    if (deadness_analysis_) {
      TF_ASSIGN_OR_RETURN(
          deadness_predicate,
          deadness_analysis_->GetPredicateFor(node, Graph::kControlSlot));
    }

    // only process gpu ops
    if (!IsGpuOp(node)) {
      cluster_for_node_[node->id()].Get() = nullptr;
      continue;
    }
    // exclude all ops whose inputs/outputs are on CPU
    // because rendezvous cannot be processed within a function body now
    bool has_invalid_args = false;
    for (const Node* in : node->in_nodes()) {
      if (!IsGpuOp(in) || in->IsVariable() || in->IsKvVarHandle()) {
        has_invalid_args = true;
        break;
      }
    }
    if (has_invalid_args) {
      cluster_for_node_[node->id()].Get() = nullptr;
      continue;
    }

    bool has_invalid_rets = false;
    for (const Node* out : node->out_nodes()) {
      if (!IsGpuOp(out) || out->IsVariable() || out->IsKvVarHandle()) {
        has_invalid_rets = true;
        break;
      }
    }
    if (has_invalid_rets) {
      cluster_for_node_[node->id()].Get() = nullptr;
      continue;
    }

    absl::flat_hash_map<string, std::vector<string>>* whitelist_table =
        tensorflow::GetCgmodeWhitelistTable();

    MarkForCudaGraphModePassFlags* flags = GetMarkForCudaGraphModePassFlags();
    absl::flat_hash_set<string> whitelist;
    for (auto s : absl::StrSplit(flags->tf_cgmode_ops_to_cluster, ",")) {
      if (whitelist_table->contains(s)) {
        auto v = whitelist_table->at(s);
        whitelist.insert(v.begin(), v.end());
      } else if (!s.empty()) {
        whitelist.insert(string(s));
      }
    }
    if (VLOG_IS_ON(2) && !whitelist.empty()) {
      std::vector<string> vwhitelist(whitelist.begin(), whitelist.end());
      absl::c_sort(vwhitelist);
      VLOG(2) << "CUDA Graph clustering will only consider the following TF "
                 "operations: "
              << absl::StrJoin(vwhitelist, " ");
    }
    if (whitelist.size() > 0 && !whitelist.contains(node->def().op())) {
      VLOG(1) << "Rejecting TF operation " << node->def().op()
              << " as it is not listed in --tf_cgmode_ops_to_cluster.";
      cluster_for_node_[node->id()].Get() = nullptr;
      continue;
    }

    // TODO: make device cache working on grah mode
    const string& device_name_str = !node->assigned_device_name().empty()
                                        ? node->assigned_device_name()
                                        : node->requested_device();
    TF_ASSIGN_OR_RETURN(DeviceId device,
                        device_info_cache_.GetIdFor(device_name_str));

    bool is_resource_op = HasResourceInputOrOutput(*node);
    absl::optional<DeviceId> resource_op_device;
    if (is_resource_op) {
      // Temporarily disable resource_op in graph mode
      cluster_for_node_[node->id()].Get() = nullptr;
      continue;
    }

    // record the resource variable operation
    absl::optional<int> resource_var_operation_node_id;
    if (is_resource_op || MayCallFunction(*node, flib_def_)) {
      resource_var_operation_node_id = node->id();
    }

    bool is_cuda_graph_mode_attr_true = false;

    bool cuda_graph_mode_attr;
    if (TryGetNodeAttr(node->attrs(), kCGModeCompileAttr,
                       &cuda_graph_mode_attr)) {
      is_cuda_graph_mode_attr_true |= cuda_graph_mode_attr;
    }

    if (flib_def_->GetAttr(*node, kCGModeCompileAttr, &cuda_graph_mode_attr)
            .ok()) {
      is_cuda_graph_mode_attr_true |= cuda_graph_mode_attr;
    }

    DeviceSet devices;
    devices.Insert(device);

    Cluster* new_cluster = MakeNewCluster(
        /*cycles_graph_node_id=*/node->id(),
        /*effective_cluster_size=*/effective_cluster_size,
        /*has_functional_control_flow=*/has_functional_control_flow, devices,
        resource_op_device, resource_var_operation_node_id, deadness_predicate,
        /*is_cuda_graph_mode_attr_true=*/is_cuda_graph_mode_attr_true,
        GetCGModeScope(node));

    cluster_for_node_[node->id()].Get() = new_cluster;
  }

  return Status::OK();
}

StatusOr<bool> IsIdentityDrivingConstsInLoop(Node* node) {
  if (!node->IsIdentity()) {
    return false;
  }

  // Check if the Identity is driven by a Switch on its true path.
  auto it = absl::c_find_if(node->in_edges(), [](const Edge* e) {
    return e->src()->IsSwitch() && e->src_output() == 1;
  });
  if (it == node->in_edges().end()) {
    return false;
  }
  const Node* switch_node = (*it)->src();

  // Check if the Switch is driven by LoopCond.
  const Node* maybe_loop_cond;
  TF_RETURN_IF_ERROR(switch_node->input_node(1, &maybe_loop_cond));
  if (!maybe_loop_cond->IsLoopCond()) {
    return false;
  }

  // Check if the Identity is driving any const nodes through a control edge.
  bool driving_any_consts =
      absl::c_any_of(node->out_edges(), [](const Edge* e) {
        return e->dst()->IsConstant() && e->IsControlEdge();
      });
  if (!driving_any_consts) {
    return false;
  }

  return true;
}

bool MarkForCudaGraphModePassImpl::LogNotContractableAndReturnFalse(
    Cluster* from, Cluster* to, absl::string_view reason) {
  VLOG(3) << EdgeContractionFailureMsg(from, to, reason);
  return false;
}

StatusOr<bool> MarkForCudaGraphModePassImpl::TryToContractEdge(Cluster* from,
                                                               Cluster* to) {
  DCHECK(from->deadness_predicate().has_value() ==
         to->deadness_predicate().has_value());
  if (from->deadness_predicate() != to->deadness_predicate()) {
    VLOG(3) << EdgeContractionFailureMsg(
        from, to,
        absl::StrCat(
            "the two nodes have mismatching deadness: ",
            deadness_analysis_->DebugString(*from->deadness_predicate()),
            " and ",
            deadness_analysis_->DebugString(*to->deadness_predicate())));
    return false;
  }

  TF_ASSIGN_OR_RETURN(bool devices_compatible,
                      AreDevicesCompatible(*from, *to));
  if (!devices_compatible) {
    return LogNotContractableAndReturnFalse(
        from, to, "the two nodes have incompatible devices");
  }

  if (from->cuda_graph_mode_scope().has_value() &&
      to->cuda_graph_mode_scope().has_value() &&
      *from->cuda_graph_mode_scope() != *to->cuda_graph_mode_scope()) {
    return LogNotContractableAndReturnFalse(
        from, to, "the two nodes have mismatching Cgmode scopes");
  }

  // Don't exceed the maximum cluster size.
  if (from->cluster_size() + to->cluster_size() >
      debug_options_.max_cluster_size) {
    return LogNotContractableAndReturnFalse(
        from, to, "the new cluster will be larger than the max cluster size");
  }

  TF_ASSIGN_OR_RETURN(bool will_introduce_cross_device_dependency,
                      ClusteringWillIntroduceInterDeviceDependency(*from, *to));

  if (will_introduce_cross_device_dependency) {
    return LogNotContractableAndReturnFalse(
        from, to, "the new cluster will introduce a cross device dependency");
  }

  // Check if contracting this edge will break the resource variable concurrency
  // semantics.  In theory this is quadratic in the number of nodes, but seems
  // to not be a problem in practice so far.
  if (!debug_options_.ignore_resource_variable_checks) {
    for (int resource_var_from : from->resource_var_operation_node_ids()) {
      for (int resource_var_to : to->resource_var_operation_node_ids()) {
        // If unsafe_resource_deps_ contains {A, B} then
        //
        //  a. A and B are resource operations.
        //  b. A and B cannot be placed in the same cluster.
        //  c. There is no path from B to A in the cycles graph (but there may
        //     be a path from A to B).
        //
        // So check the legality of the edge contraction by checking if any of
        // the n^2 pairs of resource variable operations are forbidden.
        if (unsafe_resource_deps_.contains(
                {resource_var_from, resource_var_to})) {
          return LogNotContractableAndReturnFalse(
              from, to,
              "the new cluster would break resource variable semantics");
        }
      }
    }
  }

  return MergeClusters(from, to);
}

Status MarkForCudaGraphModePassImpl::Run() {
  TF_ASSIGN_OR_RETURN(bool initialized, Initialize());
  if (!initialized) {
    // Initialization exited early which means this instance of
    // MarkForCudaGraphModePassImpl is not set up to run the subsequent phases.
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(RunEdgeContractionLoop());
  TF_RETURN_IF_ERROR(CreateClusters());
  TF_RETURN_IF_ERROR(DumpDebugInfo());

  return Status::OK();
}

void MarkForCudaGraphModePassImpl::DumpPostClusteringGraphs() {
  DumpGraphToFile("mark_for_cuda_graph_mode", *graph_, flib_def_);

  // We also dump out an annoated version of the TF graph where the nodes
  // names are prefixed with the cluster names.  This can help visualizing the
  // clustering decisions on TensorBoard.
  Graph new_graph(graph_->op_registry());
  CopyGraph(*graph_, &new_graph);

  for (Node* n : new_graph.nodes()) {
    if (absl::optional<absl::string_view> cluster_name =
            GetCGModeClusterForNode(*n)) {
      n->set_name(absl::StrCat(*cluster_name, "/", n->name()));
    } else if (n->type_string() == "VarHandleOp") {
      n->set_name(absl::StrCat("varhandle/", n->name()));
    } else {
      // There is room for improvement here.  In particular, it may help to
      // split these unclustered nodes into classes where every node in a
      // specific class has edges to and from the same set of clusters.
      n->set_name(absl::StrCat("unclustered/", n->name()));
    }
  }

  DumpGraphToFile("mark_for_cuda_graph_mode_annotated", new_graph, flib_def_);
}

string RatioToString(int numerator, int denominator) {
  return absl::StrFormat("%d / %d (%.2f%%)", numerator, denominator,
                         (100.0 * numerator) / denominator);
}

StatusOr<bool> MarkForCudaGraphModePassImpl::AreDevicesCompatible(
    const Cluster& cluster_a, const Cluster& cluster_b) {
  DeviceSet devices = cluster_a.devices();
  devices.UnionWith(cluster_b.devices());

  TF_ASSIGN_OR_RETURN(
      absl::optional<jit::DeviceId> maybe_chosen_device,
      MaybePickDeviceForXla(device_info_cache_, devices,
                            /*allow_mixing_unknown_and_cpu=*/false));
  if (!maybe_chosen_device.has_value()) {
    return false;
  }

  jit::DeviceId chosen_device = *maybe_chosen_device;

  // If we are able to pick a device `chosen_device` for the larger cluster, the
  // resource operations in `cluster_a` and `cluster_b` must be placed on the
  // same device as `chosen_device`.  This is because the _CgmodeCompile and
  // _CgmodeRun kernels are going to run on and therefore try to access the
  // resource variables from `chosen_device`, which will be an error if the
  // resource variables are placed on some other device.
  auto resource_op_device_ok =
      [&](absl::optional<DeviceId> resource_op_device) {
        return !resource_op_device.has_value() ||
               *resource_op_device == chosen_device;
      };

  return resource_op_device_ok(cluster_a.resource_op_device()) &&
         resource_op_device_ok(cluster_b.resource_op_device());
}

Status MarkForCompilation(
    const GraphOptimizationPassOptions& options,
    const MarkForCudaGraphModePassImpl::DebugOptions& debug_options) {
  Graph* graph = options.graph->get();
  FunctionLibraryDefinition* flib_def = options.flib_def;

  // Deadness analysis expects a graph with source and sink edges properly
  // connected but sometimes the incoming graph does not follow this invariant.
  // So fix up the source and sink edges before calling into deadness analysis.
  FixupSourceAndSinkEdges(graph);

  // See explanation on `kCGModeAlreadyClustered`.
  for (Node* n : graph->nodes()) {
    if (n->attrs().Find(kCGModeAlreadyClustered)) {
      return Status::OK();
    }
  }

  return MarkForCudaGraphModePassImpl{debug_options, graph, flib_def,
                                      options.session_options != nullptr
                                          ? options.session_options->env
                                          : Env::Default(),
                                      GetGlobalJitLevelForGraph(options)}
      .Run();
}

std::atomic<int64>* GetPointerToFuel(int64 initial_value) {
  static std::atomic<int64>* fuel = [&]() {
    std::atomic<int64>* fuel = new std::atomic<int64>;
    *fuel = initial_value;
    return fuel;
  }();

  return fuel;
}
}  // anonymous namespace

Status MarkForCudaGraphModePass::Run(
    const GraphOptimizationPassOptions& options) {
  if (!options.session_options->config.gpu_options().cuda_graph_enable_jit()) {
    return Status::OK();
  }
  MarkForCudaGraphModePassFlags* flags = GetMarkForCudaGraphModePassFlags();

  MarkForCudaGraphModePassImpl::DebugOptions debug_options;
  debug_options.ignore_deadness_checks =
      flags->tf_cgmode_disable_deadness_safety_checks_for_debugging;
  debug_options.ignore_resource_variable_checks =
      flags->tf_cgmode_disable_resource_variable_safety_checks_for_debugging;
  debug_options.ignore_cgmode_compile_attr = false;
  debug_options.max_cluster_size = flags->tf_cgmode_max_cluster_size;
  debug_options.min_cluster_size = flags->tf_cgmode_min_cluster_size;
  debug_options.fuel = GetPointerToFuel(flags->tf_cgmode_clustering_fuel);
  debug_options.dump_graphs = flags->tf_cgmode_clustering_debug;

  return MarkForCompilation(options, debug_options);
}

Status MarkForCudaGraphModePass::RunForTest(
    const GraphOptimizationPassOptions& options,
    bool disable_deadness_analysis) {
  MarkForCudaGraphModePassFlags* flags = GetMarkForCudaGraphModePassFlags();

  MarkForCudaGraphModePassImpl::DebugOptions debug_options;
  debug_options.ignore_deadness_checks = disable_deadness_analysis;
  debug_options.ignore_resource_variable_checks =
      flags->tf_cgmode_disable_resource_variable_safety_checks_for_debugging;
  debug_options.ignore_cgmode_compile_attr = true;
  debug_options.max_cluster_size = flags->tf_cgmode_max_cluster_size;
  debug_options.min_cluster_size = flags->tf_cgmode_min_cluster_size;
  debug_options.fuel = GetPointerToFuel(flags->tf_cgmode_clustering_fuel);
  debug_options.dump_graphs = flags->tf_cgmode_clustering_debug;
  return MarkForCompilation(options, debug_options);
}

namespace testing {
void ResetClusterSequenceNumber() { cluster_sequence_num = 0; }

}  // namespace testing
}  // namespace tensorflow
