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

#ifndef ODL_PROCESSOR_CORE_GRAPH_OPTIMIZER_H_
#define ODL_PROCESSOR_CORE_GRAPH_OPTIMIZER_H_

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace odl_processor {

// Tensorflow GraphDef will be splited into
// two subgraph, one for tf and another for IREE.
struct ClusteredGraphInfo {
  GraphDef tf_subgraph;
  GraphDef iree_subgraph;
  SignatureDef tf_signature;
};

// clustering strategr base class
class CluteringStrategy {
 public:
  CluteringStrategy() {}
  virtual ~CluteringStrategy() {}
  virtual void Run(const MetaGraphDef&, ClusteredGraphInfo*) = 0;
  // @gdef: tf graph_def
  // @inputs: root ops of the graph def
  // @outputs: target ops of the graph def
  virtual void Run(const GraphDef& gdef,
                   std::vector<std::string>& inputs,
                   std::vector<std::string>& outputs,
                   ClusteredGraphInfo*) = 0;
};

class StaticShapeCluteringStrategy
    : public CluteringStrategy {
 public:
  StaticShapeCluteringStrategy() {}
  void Run(const MetaGraphDef&, ClusteredGraphInfo*);
  void Run(const GraphDef& gdef,
           std::vector<std::string>& inputs,
           std::vector<std::string>& outputs,
           ClusteredGraphInfo*);
};

ClusteredGraphInfo ClusteringGraphDef(
    const MetaGraphDef& mgdef,
    CluteringStrategy* cluster_strategy = nullptr);

// @saved_model_str: the saved model pb string
MetaGraphDef GetMetaGraphDefFromSavedModel(
    const std::string& saved_model_str);
 
} // namespace odl_processor
} // namespace tensorflow

#endif  // ODL_PROCESSOR_CORE_GRAPH_OPTIMIZER_H_

