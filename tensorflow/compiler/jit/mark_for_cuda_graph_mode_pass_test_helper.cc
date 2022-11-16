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

#include "tensorflow/compiler/jit/mark_for_cuda_graph_mode_pass_test_helper.h"

#include "tensorflow/compiler/jit/cluster_scoping_pass.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
/*static*/ Status MarkForCudaGraphModePassTestHelper::MarkForCompilation(
    std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def,
    MarkForCudaGraphModePassTestHelper::Options options) {
  // Assign all unassigned nodes to the CPU device.
  static const char* kGpuDevice = "/job:localhost/replica:0/task:0/gpu:0";
  for (Node* n : (*graph)->nodes()) {
    if (n->assigned_device_name().empty()) {
      n->set_assigned_device_name(kGpuDevice);
    }
  }

  SessionOptions session_options;
  session_options.config.mutable_gpu_options()->set_cuda_graph_enable_jit(true);
  GraphOptimizationPassOptions opt_options;
  opt_options.graph = graph;
  opt_options.session_options = &session_options;
  opt_options.flib_def = flib_def;

  MarkForCudaGraphModePass mark_for_cuda_graph_mode_pass;
  return mark_for_cuda_graph_mode_pass.RunForTest(
      opt_options,
      /*disable_deadness_analysis=*/options.disable_deadness_analysis);
}

/*static*/ Status MarkForCudaGraphModePassTestHelper::MarkForCompilation(
    std::unique_ptr<Graph>* graph,
    MarkForCudaGraphModePassTestHelper::Options options) {
  FunctionDefLibrary flib;
  FunctionLibraryDefinition flib_def((*graph)->op_registry(), flib);
  return MarkForCompilation(graph, &flib_def, options);
}
}  // namespace tensorflow
