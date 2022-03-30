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

#ifndef SERVING_PROCESSOR_FRAMEWORK_UTIL_UTILS_H_
#define SERVING_PROCESSOR_FRAMEWORK_UTIL_UTILS_H_

#include <unordered_map>
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace tensorflow {
namespace processor {

// Return true when node has dynamic shape at any input or output, false else.
std::unordered_map<std::string, bool> GetNodesHasDynamicShapeMap(const GraphDef& gdef);

// Return true when node has control input edge, false else.
std::unordered_map<std::string, bool> GetNodesHasControlFlowInputs(const GraphDef& gdef);

// Whether any output has a dynamic shape
bool HasDynamicShapeOutput(NodeDef* node);

Status AddDefaultAttributes(const GraphDef& input_graph_def,
                            GraphDef* output_graph_def);

} // namespace processor
} // namespace tensorflow

#endif // SERVING_PROCESSOR_FRAMEWORK_UTIL_UTILS_H_

