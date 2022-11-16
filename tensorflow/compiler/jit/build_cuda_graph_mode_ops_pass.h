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

#ifndef TENSORFLOW_COMPILER_JIT_BUILD_CUDA_GRAPH_MODE_OPS_PASS_H_
#define TENSORFLOW_COMPILER_JIT_BUILD_CUDA_GRAPH_MODE_OPS_PASS_H_

#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Adds _CgmodeCompile and _CgmodeRun operations to the TF graph that compiles
// and executes (using CUDA Graph) TF function calls marked with
// "_CgmodeCompiledKernel".
class BuildCgmodeOpsPass : public GraphOptimizationPass {
 public:
  explicit BuildCgmodeOpsPass() {}

  Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_BUILD_CUDA_GRAPH_MODE_OPS_PASS_H_
