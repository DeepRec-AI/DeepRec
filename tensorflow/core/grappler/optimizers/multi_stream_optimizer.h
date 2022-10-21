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
#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_MULTI_STREAM_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_MULTI_STREAM_OPTIMIZER_H_

#include <atomic>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
class Graph;

namespace grappler {

class MultiStreamOptimizer : public GraphOptimizer {
 public:
  MultiStreamOptimizer(const MultiStreamOptions& opt);

  string name() const override { return "multi_stream_optimizer"; }

  bool UsesFunctionLibrary() const override { return false; }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override {
    // Nothing to do for MultiStreamOptimizer
  }

 private:
  Status SplitEmbeddingGraph(
      const GrapplerItem& item, GraphDef* optimized_graph);

  Status MarkEmbeddingGraphNodes(
      const std::vector<NodeDef*> start_nodes,
      std::unordered_map<std::string, std::vector<NodeDef*>> output_edges,
      GraphDef* optimized_graph);

 private:
  MultiStreamOptions opt_;
};

}  // namespace grappler
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_MULTI_STREAM_OPTIMIZER_H_
