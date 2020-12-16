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

#ifndef ODL_PROCESSOR_FRAMEWORK_GRAPH_OPTIMIZER_H_
#define ODL_PROCESSOR_FRAMEWORK_GRAPH_OPTIMIZER_H_

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

namespace tensorflow {
namespace processor {

/// IREE

// Tensorflow GraphDef will be splited into
// two subgraph, one for tf and another for IREE.
// TODO: FIXME split it into two structures
struct ClusteredGraphInfo {
  GraphDef tf_subgraph;
  GraphDef iree_subgraph;
  SignatureDef tf_signature;
  SavedModel tf_saved_model;
  SavedModel iree_saved_model;
};

// clustering strategr base class
class CluteringStrategy {
 public:
  CluteringStrategy() {}
  virtual ~CluteringStrategy() {}
  virtual void Run(const std::string& tag,
                   const SavedModel&,
                   ClusteredGraphInfo*) {}
  virtual void Run(const MetaGraphDef&, ClusteredGraphInfo*) {}
  // @gdef: tf graph_def
  // @inputs: root ops of the graph def
  // @outputs: target ops of the graph def
  virtual void Run(const GraphDef& gdef,
                   std::vector<std::string>& inputs,
                   std::vector<std::string>& outputs,
                   ClusteredGraphInfo*) {}
};

class StaticShapeCluteringStrategy
    : public CluteringStrategy {
 public:
  StaticShapeCluteringStrategy() {}
  void Run(const std::string& tag,
           const SavedModel&,
           ClusteredGraphInfo*);
  void Run(const MetaGraphDef&, ClusteredGraphInfo*);
  void Run(const GraphDef& gdef,
           std::vector<std::string>& inputs,
           std::vector<std::string>& outputs,
           ClusteredGraphInfo*);

  // Function will return static ops set
  void GetStaticGraphOps(
      const GraphDef& gdef,
      std::vector<std::string>& inputs,
      std::vector<std::string>& outputs,
      std::unordered_set<std::string>* static_ops_name);
  // Get dynamic and static signature def
  void GetDynamicAndStaticSignatureDef(
      const MetaGraphDef&,
      std::map<string, SignatureDef>* dynamic_sdef,
      std::map<string, SignatureDef>* static_sdef);
  // Get dynamic and static meta graph
  void GetDynamicAndStaticMetaGraphDef(
      const MetaGraphDef& mgdef,
      MetaGraphDef* dynamic_mgdef,
      MetaGraphDef* static_mgdef);
};

ClusteredGraphInfo ClusteringGraphDef(
    const std::string& tag,
    const SavedModel& saved_model,
    CluteringStrategy* cluster_strategy = nullptr);

ClusteredGraphInfo ClusteringGraphDef(
    const std::string& tag,
    const std::string& saved_model_str,
    CluteringStrategy* cluster_strategy = nullptr);

ClusteredGraphInfo ClusteringGraphDef(
    const MetaGraphDef& mgdef,
    CluteringStrategy* cluster_strategy = nullptr);

// @saved_model_str: the saved model pb string
MetaGraphDef GetMetaGraphDefFromSavedModel(
    const std::string& saved_model_str);

// @saved_model_str: the saved model text
MetaGraphDef GetMetaGraphDefFromSavedModelText(
    const std::string& str);

/// Tensorflow

const std::string& GetInitDefKey();
const std::string& GetModelVersionNodeName();
const std::string& GetStoragePointerNodeName();
const std::string& GetInitNodeName();

struct GraphOptimizerOption {
  // Convert EV ops to HashTable ops
  // to support local graph execution.
  bool native_tf_mode = false;
};

class GraphOptimizer {
 public:
  explicit GraphOptimizer(
      const std::string& signature_name,
      MetaGraphDef* mgdef,
      GraphOptimizerOption& option);
  virtual ~GraphOptimizer() {}
  virtual Status Optimize() = 0;

 protected:
  Graph graph_; // graph of meta_graph_def_.graph_def()
  std::string signature_name_;
  MetaGraphDef* meta_graph_def_ = nullptr; // not owned
  GraphOptimizerOption option_;
};

class SavedModelOptimizer : public GraphOptimizer {
 public:
  explicit SavedModelOptimizer(
      const std::string& signature_name,
      MetaGraphDef* mgdef,
      GraphOptimizerOption& option);
  ~SavedModelOptimizer();
  Status Optimize() override;

 private:
  Status GetFeature2IdAttr(
      const std::string& name,
      AttrValue* attr_value);

 private:
  Status RunNativeTFGraphPass();
  Status RunODLGraphPass();

  // Convert EV related ops to HashTable ops
  Status ConvertToHashTableOps();

  // TODO: Only support EV now
  // Add Lookup and Insert ops,
  // then remove KvResourceGather and KvResourceImportV2 ops.
  Status ConvertKVOps();

  // Rewrite default value op when not found the variable key.
  Status RewriteDefaultValueOp();

  // Remove unused signature def
  Status FreezeSignatureDef();

  // Add a init-op to initialize variable, redis for example
  Status AddVariableInitSubGraph();

  // Add full and delta ckpt update subgraph
  Status AddFullAndDeltaUpdateSubGraph();

  // Add placeholder nodes
  // include storage pointer
  Status AddStoragePlaceholderNode();

  // Each feature name will be map to a uint64 num,
  // the num will be as the prefix of a query key.
  Status GenerateIdsForFeatures();

  Node* storage_pointer_node_ = nullptr;// storage placeholder node
  std::unordered_map<std::string, int> feature_names_to_ids;
};

} // namespace processor
} // namespace tensorflow

#endif // ODL_PROCESSOR_FRAMEWORK_GRAPH_OPTIMIZER_H_

