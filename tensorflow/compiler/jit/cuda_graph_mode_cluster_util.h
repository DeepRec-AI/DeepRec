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

// Contains utilities for clustering compilable graph nodes for Graph Mode.

#ifndef TENSORFLOW_COMPILER_JIT_CUDA_GRAPH_MODE_CLUSTER_UTIL_H_
#define TENSORFLOW_COMPILER_JIT_CUDA_GRAPH_MODE_CLUSTER_UTIL_H_

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/graphcycles/graphcycles.h"
#include "tensorflow/compiler/jit/xla_activity.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

// The attribute that marks nodes to be grouped into functions by the
// encapsulate subgraphs pass.

// The attribute that marks nodes in a cluster to be placed outside the graph mode
// compilation by the encapsulate subgraphs pass.
const char* const kCGModeClusterAttr = "_CGModeCluster";

using OrderedNodeSet = std::set<Node*, NodeComparatorID>;

// Returns the cluster in which `node` is placed if it is in an cluster,
// otherwise returns nullopt.
absl::optional<absl::string_view> GetCGModeClusterForNode(const Node& node);

// Removes `node_def` its cluster (by clearing its _CgmodeCluster attribute).
void RemoveFromCGModeCluster(NodeDef* node_def);

// Removes `node` its cluster (by clearing its _CgmodeCluster attribute).
void RemoveFromCGModeCluster(Node* node);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_CUDA_GRAPH_MODE_CLUSTER_UTIL_H_
