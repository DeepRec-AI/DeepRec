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
#if GOOGLE_CUDA

#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class ValidateCudaGraphModePass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    if (options.session_options->config.gpu_options().cuda_graph_enable_jit()) {
      return Status::OK();
    }
    bool has_invalid_graph = false;
    if (options.graph == nullptr) {
      return Status::OK();
    }
    Graph* graph = options.graph->get();
    if (graph == nullptr) {
      return errors::Internal("a graph should be available.");
    }
    for (Node* node : graph->op_nodes()) {
      // validate device
      DeviceNameUtils::ParsedName assigned_device;
      if (!DeviceNameUtils::ParseFullName(node->assigned_device_name(),
                                          &assigned_device)) {
        continue;
      }
      if (assigned_device.type != DEVICE_GPU) {
        if (!node->IsPlaceholder() && !node->IsConstant()) {
          LOG(WARNING) << "Op node: " << node->name()
                       << " has been assigned a non-gpu device: "
                       << node->assigned_device_name();
          has_invalid_graph = true;
          break;
        }
      }
      // validate dynamic shape
      if (HasNodeAttr(node->def(), "shape")) {
        auto shape_proto = node->def().attr().at("shape").shape();
        int total_dims = shape_proto.dim().size();
        if (total_dims > 1 && shape_proto.dim(total_dims - 1).size() < 0) {
          LOG(WARNING) << "Op node: " << node->name()
                       << " has unkown and non-first dimension detected";
          has_invalid_graph = true;
          break;
        }
      }
    }

    SessionOptions* sess_opts =
        const_cast<SessionOptions*>(options.session_options);
    sess_opts->config.mutable_cuda_graph_mode_options()->set_has_invalid_graph(
        has_invalid_graph);
    return Status::OK();
  }
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PLACEMENT, 0,
                      ValidateCudaGraphModePass);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
