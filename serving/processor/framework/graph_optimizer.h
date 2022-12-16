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

#ifndef SERVING_PROCESSOR_FRAMEWORK_GRAPH_OPTIMIZER_H_
#define SERVING_PROCESSOR_FRAMEWORK_GRAPH_OPTIMIZER_H_

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
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
const std::string& GetIncrCkptNodeName();
const std::string& GetKvRestoreAllNameSuffix();
const std::string& GetKvIncrRestoreAllNameSuffix();
const std::string& GetDenseRestoreAllNameSuffix();
 
struct GraphOptimizerOption {
  // Convert EV ops to HashTable ops
  // to support local graph execution.
  bool native_tf_mode = false;

  // The embeddings which will be partitioned can be found
  // in GraphOptimizerOption.shard_embedding_names.
  // GraphOptimizerOption.partition_id and
  // GraphOptimizerOptionshard_instance_count are for
  // partiton policy.
  // GraphOptimizerOptionshard_instance_count means the slices
  // the embedding will be partitioned, GraphOptimizerOption.partition_id
  // means which slice current instance will load.
  //
  // shard embedding or not
  bool shard_embedding = false;
  // which embeddings will be sharded
  std::vector<std::string> shard_embedding_names;
  // current instance partition id
  int partition_id = -1;
  int shard_instance_count = 0;

  // multi tiered embedding
  embedding::StorageType st = embedding::StorageType::DEFAULT;
  std::string path;
  std::vector<int64> size;
};

struct SrcInfo {
  Node* src_node;
  int src_slot;
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

  // Add op to restore delta model which contain
  // EV ops.
  Status AddIncrRestoreOps();
  Status ConvertKvImportToKvInsert(
      Node* import_op, Node** insert_op);
  Status CreateIncrRestoreOp(
      Node* import_op, Node** restore_op);
  Status GetIncrRestoreOpInputs(
    const Node* restore_op,
    std::vector<SrcInfo>& input_nodes);

  // Convert EV related ops to HashTable ops
  Status ConvertToHashTableOps();

  Status ConvertToHashTableOp(
      Node* node, std::vector<SrcInfo>& input_info);
  Status ConvertToHashLookupOp(
      Node* node, std::vector<SrcInfo>& input_info);
  Status ConvertToHashImportOp(
      Node* node, std::vector<SrcInfo>& input_info);

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

  // Create dense and sparse restore ops
  Status CreateDenseAndSparseRestoreOp();

  // Add placeholder nodes
  // include storage pointer
  Status AddStoragePlaceholderNode();

  // Add node for feed version
  Status AddVersionNode();

  // Import incr ckpt flag
  Status AddIncrCkptFlag();

  // Each feature name will be map to a uint64 num,
  // the num will be as the prefix of a query key.
  Status GenerateIdsForFeatures();

  // Partition embeddings according partition policy.
  Status RewriteEmbeddingLookupGraph(
    std::unordered_map<std::string, std::vector<Node*>>& var_parts_map,
    std::unordered_map<std::string, std::vector<Node*>>& origin_import_nodes);
 
  Status FindVariableParts(
      std::unordered_map<std::string, std::vector<Node*> >& var_parts);

  Status FindKvResourceImportNode(
      std::unordered_map<std::string, std::vector<Node*>>& var_parts,
      std::unordered_map<std::string, std::vector<Node*>>& import_nodes);

  Node* FindRestoreShardNode();

  Node* UpdateRestoreShardNodeInputs(
      std::unordered_map<std::string, std::vector<Node*>>& origin_import_nodes,
      std::vector<Node*>& new_kv_import_nodes);

  Status RewriteEmbeddingVariableAttr(embedding::StorageType st, const std::string& path,
                                      const std::vector<int64>& size);

  Node* storage_pointer_node_ = nullptr;// storage placeholder node
  Node* version_node_ = nullptr; // version placeholder node
  Node* incr_ckpt_node_ = nullptr; // indicate if import incr ckpt
  std::unordered_map<std::string, int> feature_names_to_ids;
};

} // namespace processor
} // namespace tensorflow

#endif // SERVING_PROCESSOR_FRAMEWORK_GRAPH_OPTIMIZER_H_

