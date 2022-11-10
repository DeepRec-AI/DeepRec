/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPH_STREAM_SUBGRAPH_H_
#define TENSORFLOW_CORE_GRAPH_STREAM_SUBGRAPH_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace stream_subgraph {

void MarkStreamSubGraph(Graph* g, const MultiStreamOptions& opt);

// Assign embedding graphs stream.
void MarkEmbeddingGraph(Graph* g, int num_streams);

}  // namespace stream_subgraph
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_STREAM_SUBGRAPH_H_
