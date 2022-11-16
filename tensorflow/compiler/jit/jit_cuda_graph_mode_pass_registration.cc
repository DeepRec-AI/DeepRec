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
#include "tensorflow/compiler/jit/clone_constants_for_better_clustering.h"
#include "tensorflow/compiler/jit/encapsulate_cuda_graph_mode_subgraphs_pass.h"
#include "tensorflow/compiler/jit/mark_for_cuda_graph_mode_pass.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

// POST_REWRITE_FOR_EXEC passes that support auto-clustering to enable CUDA Graph Mode:

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 5,
                     CloneConstantsForBetterClusteringPass);

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 10,
                     MarkForCudaGraphModePass);

// The EncapsulateCGModeSubgraphsPass pass must run after the MarkForCudaGraphModePass. We
// also need to run it after the graph been rewritten to have _Send nodes added
// for fetches. Before the _Send nodes are added, fetch nodes are identified by
// name, and encapsulation might remove that node from the graph.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 50,
                     EncapsulateCGModeSubgraphsPass);

// Must run after EncapsulateCGModeSubgraphsPass.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 60,
                     BuildCgmodeOpsPass);

}  // namespace tensorflow
