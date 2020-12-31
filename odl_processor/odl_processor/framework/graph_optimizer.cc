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

#include <queue>
#include <unordered_set>

#include "odl_processor/framework/graph_optimizer.h"
#include "odl_processor/framework/util/utils.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"

namespace tensorflow {
namespace processor {

/// IREE

namespace {

std::unordered_set<std::string> GetBlackOpsSet() {
  std::unordered_set<std::string> s = {
    "Unique",
    "SparseSegmentMean",
    "SparseSegmentMeanGrad",
    "SparseSegmentSum",
    "SparseSegmentSumGrad",
    "SparseSegmentSqrt",
    "SparseSegmentSqrtNGrad",
    "UnsortedSegmentSum",
    "UnsortedSegmentSumGrad",
    "UnsortedSegmentMax",
    "UnsortedSegmentMaxGrad",
    "HashTableV2",
    "LookupTableFindV2",
    "ParseExample",
    "Bucketize"
    // Add other ops
  };

  return s;
}

} // namespace


MetaGraphDef GetMetaGraphDefFromSavedModel(
    const std::string& saved_model_str) {
  SavedModel saved_model;
  if (!saved_model.ParseFromString(saved_model_str)) {
    LOG(FATAL) << "Can not parse saved model from pb string.";
  }

  // TODO: How about many meta graphs?
  MetaGraphDef meta_gdef = saved_model.meta_graphs()[0];

  return meta_gdef;
}

MetaGraphDef GetMetaGraphDefFromSavedModelText(
    const std::string& str) {
  SavedModel saved_model;
  if (!tensorflow::protobuf::TextFormat::ParseFromString(str, &saved_model)) {
    LOG(FATAL) << "Can not parse saved model from text.";
  }

  // TODO: How about many meta graphs?
  MetaGraphDef meta_gdef = saved_model.meta_graphs()[0];

  return meta_gdef;
} 

// Function will return static ops set
void StaticShapeCluteringStrategy::GetStaticGraphOps(
    const GraphDef& gdef,
    std::vector<std::string>& inputs,
    std::vector<std::string>& outputs,
    std::unordered_set<std::string>* static_ops_name) {
  std::unordered_map<std::string, bool> dynamic_ops_map =
      GetNodesHasDynamicShapeMap(gdef);

  std::unordered_map<std::string, bool> has_control_flow_input =
      GetNodesHasControlFlowInputs(gdef);

  std::unordered_map<std::string, const NodeDef*> nodes;
  for (const NodeDef& node : gdef.node()) {
    nodes[node.name()] = &node;
  }

  std::unordered_set<const NodeDef*> static_nodes;
  std::unordered_set<const NodeDef*> visited;
  std::queue<const NodeDef*> q;
  for (auto output : outputs) {
    q.push(nodes[output]);
    visited.insert(nodes[output]);
  }

  std::unordered_set<std::string> black_ops = GetBlackOpsSet();
  while (!q.empty()) {
    const NodeDef* curr_node = q.front();
    // 1) no control edge
    // 2) no dynamic shape
    // 3) no blacklist ops
    if (!has_control_flow_input[curr_node->name()] &&
        !dynamic_ops_map[curr_node->name()] &&
        black_ops.find(curr_node->op()) == black_ops.end()) {
      // Add op into static ops set
      //(*static_ops)[curr_node->name()] = curr_node;
      static_ops_name->insert(curr_node->name());

      for (auto in_name : curr_node->input()) {
        size_t offset = in_name.find(":");
        in_name = in_name.substr(0, offset);
        if (visited.find(nodes[in_name]) == visited.end()) {
          q.push(nodes[in_name]);
          visited.insert(nodes[in_name]);
        }
      }
    }

    q.pop();
  }
}

void StaticShapeCluteringStrategy::Run(
    const GraphDef& gdef,
    std::vector<std::string>& inputs,
    std::vector<std::string>& outputs,
    ClusteredGraphInfo* clustered_graph_info) {
  std::unordered_map<std::string, bool> dynamic_ops_map =
      GetNodesHasDynamicShapeMap(gdef);

  std::unordered_map<std::string, bool> has_control_flow_input =
      GetNodesHasControlFlowInputs(gdef);

  std::unordered_map<std::string, const NodeDef*> nodes;
  for (const NodeDef& node : gdef.node()) {
    nodes[node.name()] = &node;
  }

  GraphDef& dynamic_graph_def = clustered_graph_info->tf_subgraph;
  GraphDef& static_graph_def = clustered_graph_info->iree_subgraph;

  std::unordered_set<const NodeDef*> static_nodes;
  std::unordered_set<const NodeDef*> visited;
  std::queue<const NodeDef*> q;
  for (auto output : outputs) {
    q.push(nodes[output]);
    visited.insert(nodes[output]);
  }

  std::unordered_set<std::string> black_ops = GetBlackOpsSet();
  while (!q.empty()) {
    const NodeDef* curr_node = q.front();
    // 1) no control edge
    // 2) no dynamic shape
    // 3) no blacklist ops
    if (!has_control_flow_input[curr_node->name()] &&
        !dynamic_ops_map[curr_node->name()] &&
        black_ops.find(curr_node->op()) == black_ops.end()) {
      // Add op into static_graph_def
      NodeDef* new_node = static_graph_def.add_node();
      new_node->CopyFrom(*curr_node);
      static_nodes.insert(curr_node);

      for (auto in_name : curr_node->input()) {
        size_t offset = in_name.find(":");
        in_name = in_name.substr(0, offset);
        if (visited.find(nodes[in_name]) == visited.end()) {
          q.push(nodes[in_name]);
          visited.insert(nodes[in_name]);
        }
      }
    }

    q.pop();
  }

  // TODO: Add version and library ops

  // TODO: Add placeholder here

  for (const NodeDef& node : gdef.node()) {
    if (static_nodes.find(&node) == static_nodes.end()) {
      NodeDef* new_node = dynamic_graph_def.add_node();
      new_node->CopyFrom(node);
    }
  }

  // TODO: Add tf_signature
}

void StaticShapeCluteringStrategy::Run(
    const MetaGraphDef& mgdef,
    ClusteredGraphInfo* clustered_graph_info) {
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;

  // TODO: FIXME consider several signature_defs here
  // Now only consider one signature_def
  for (auto sdef : mgdef.signature_def()) {
    for (auto input : sdef.second.inputs()) {
      // convert "input_example_tensor:0" to "input_example_tensor"
      std::string name = input.second.name();
      size_t offset = name.find(":");
      inputs.push_back(name.substr(0, offset));
    }

    for (auto output : sdef.second.outputs()) {
      // convert "Reshape_2:0" to "Reshape_2"
      std::string name = output.second.name();
      size_t offset = name.find(":");
      outputs.push_back(name.substr(0, offset));
    }

    // TODO: FIXME
    break;
  }

  Run(mgdef.graph_def(), inputs, outputs,
      clustered_graph_info);
}

void StaticShapeCluteringStrategy::GetDynamicAndStaticSignatureDef(
    const MetaGraphDef& mgdef,
    std::map<string, SignatureDef>* dynamic_sdef,
    std::map<string, SignatureDef>* static_sdef) {
  Graph graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.allow_internal_ops = true;
 
  Status s = ConvertGraphDefToGraph(opts, mgdef.graph_def(), &graph);
  if (!s.ok()) {
    LOG(FATAL) << "can not convert graphdef to graph, " << s.error_message();
  }
  std::unordered_map<std::string, Node*> nodes_map;
  for (Node* n : graph.nodes()) {
    nodes_map[n->name()] = n;
  }

  // maybe have many signature_defs in the meta graphdef
  for (auto sdef : mgdef.signature_def()) {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    // filter the dupilication ops, like
    // key: "in_A:0" and key: "in_A:1",
    std::unordered_set<std::string> existed;
    for (auto input : sdef.second.inputs()) {
      // convert "input_example_tensor:0" to "input_example_tensor"
      std::string name = input.second.name();
      size_t offset = name.find(":");
      name = name.substr(0, offset);
      if (existed.find(name) != existed.end()) continue;
      existed.insert(name);
      inputs.push_back(name);
    }

    existed.clear();
    for (auto output : sdef.second.outputs()) {
      // convert "Reshape_2:0" to "Reshape_2"
      std::string name = output.second.name();
      size_t offset = name.find(":");
      name = name.substr(0, offset);
      if (existed.find(name) != existed.end()) continue;
      existed.insert(name);
      outputs.push_back(name);
    }

    // get those ops which will be insert into static graphdef
    std::unordered_set<std::string> static_ops_name;
    GetStaticGraphOps(mgdef.graph_def(), inputs, outputs, &static_ops_name);

    std::string signature_def_key(sdef.first);
    // should create a static graphdef
    if (static_ops_name.size() > 0) {
      (*dynamic_sdef)[signature_def_key] = sdef.second;
      (*static_sdef)[signature_def_key] = sdef.second;
      (*dynamic_sdef)[signature_def_key].clear_inputs();
      (*dynamic_sdef)[signature_def_key].clear_outputs();
      (*static_sdef)[signature_def_key].clear_inputs();
      (*static_sdef)[signature_def_key].clear_outputs();

      // 1) set signature_def output of static/dynamic graphdef
      for (auto output : sdef.second.outputs()) {
        std::string name = output.second.name();
        size_t offset = name.find(":");
        name = name.substr(0, offset);
        if (static_ops_name.find(name) != static_ops_name.end()) {
          (*(*static_sdef)[signature_def_key].mutable_outputs())[output.first] =
              output.second;
        } else {
          (*(*dynamic_sdef)[signature_def_key].mutable_outputs())[output.first] =
              output.second;
        }
      }

      // 2) set signature_def input of static/dynamic graphdef
      for (auto input : sdef.second.inputs()) {
        std::string name = input.second.name();
        size_t offset = name.find(":");
        name = name.substr(0, offset);
        if (static_ops_name.find(name) != static_ops_name.end()) {
          (*(*static_sdef)[signature_def_key].mutable_inputs())[input.first] =
              input.second;
        } else {
          (*(*dynamic_sdef)[signature_def_key].mutable_inputs())[input.first] =
              input.second;
        }
      }
 
      std::unordered_map<std::string, const NodeDef*> nodes;
      for (const NodeDef& node : mgdef.graph_def().node()) {
        nodes[node.name()] = &node;
      }

      // 3) create signature_def of those ops connect dynamic and static graph
      for (auto& op_name : static_ops_name) {
        for (std::string in_name : nodes[op_name]->input()) {
          // NOTE(jiankeng.pt): NO control edge here, so no need to consider '^AAA' case
          auto offset = in_name.find(":");
          in_name = in_name.substr(0, offset);
          // the node named 'in_name' in dynamic graph,
          // so need add the node into the input signature of static graph,
          // and add it into the output signature of dynamic graph.
          if (static_ops_name.find(in_name) == static_ops_name.end()) {
            std::vector<std::vector<int>> shapes;
            NodeDef* tmp_in_def = const_cast<NodeDef*>(nodes[in_name]);
            AttrValue attr_value = (*tmp_in_def->mutable_attr())["_output_shapes"];
            int output_tensor_count = -1;
            Status s = AttrValueHasType(attr_value, "list(shape)");
            if (!s.ok()) {
              s = AttrValueHasType(attr_value, "shape");
              if (!s.ok()) {
                LOG(FATAL) << "Can not found the _output_shapes attr, "
                           << "we dont know the output tensor count.";
              }
              std::vector<int> op_shape;
              op_shape.push_back(attr_value.shape().dim().size());
              shapes.push_back(op_shape);
              output_tensor_count = 1;
            } else {
              output_tensor_count = 0;
              for (const auto& curr_shape : attr_value.list().shape()) {
                ++output_tensor_count;
                std::vector<int> op_shape;
                for (auto d : curr_shape.dim()) {
                  op_shape.push_back(d.size());
                }
                shapes.push_back(op_shape);
              }
            }

            // add signature
            std::string input_base_name = "static_sig_inputs_" + in_name + "_";
            std::string output_base_name = "dynamic_sig_outputs_" + in_name + "_";
            for (auto i = 0; i < output_tensor_count; ++i) {
              std::string target_name = in_name + ":" + std::to_string(i);
              TensorInfo tinfo;
              tinfo.set_name(target_name);
              tinfo.set_dtype(nodes_map[in_name]->output_type(i));
              for (size_t j = 0; j < shapes[i].size(); ++j) {
                tinfo.mutable_tensor_shape()->add_dim()->set_size(shapes[i][j]);
              }

              // input signature for static graph
              std::string in_key_name = input_base_name + std::to_string(i);
              (*(*static_sdef)[signature_def_key].mutable_inputs())[in_key_name] = tinfo;

              // output signature for dynamic graph
              std::string out_key_name = output_base_name + std::to_string(i);
              (*(*dynamic_sdef)[signature_def_key].mutable_outputs())[out_key_name] = tinfo;
            }
          }
        }
      }
    } else {
      (*dynamic_sdef)[signature_def_key] = sdef.second;
      (*static_sdef)[signature_def_key] = SignatureDef();
    }
  }
}

void StaticShapeCluteringStrategy::GetDynamicAndStaticMetaGraphDef(
    const MetaGraphDef& mgdef,
    MetaGraphDef* dynamic_mgdef,
    MetaGraphDef* static_mgdef) {
  std::map<string, SignatureDef> dynamic_sdef, static_sdef;
  GetDynamicAndStaticSignatureDef(mgdef, &dynamic_sdef, &static_sdef);

  *dynamic_mgdef = mgdef;
  dynamic_mgdef->clear_signature_def();
  auto dyn_sig_def = dynamic_mgdef->mutable_signature_def();
  for (auto sdef : dynamic_sdef) {
    (*dyn_sig_def)[sdef.first] = sdef.second;
  }

  *static_mgdef = mgdef;
  static_mgdef->clear_signature_def();
  auto sta_sig_def = static_mgdef->mutable_signature_def();
  for (auto sdef : static_sdef) {
    (*sta_sig_def)[sdef.first] = sdef.second;
  }
}

void StaticShapeCluteringStrategy::Run(
    const std::string& tag,
    const SavedModel& saved_model,
    ClusteredGraphInfo* clustered_graph_info) {
  clustered_graph_info->tf_saved_model.set_saved_model_schema_version(
      saved_model.saved_model_schema_version());
  clustered_graph_info->iree_saved_model.set_saved_model_schema_version(
      saved_model.saved_model_schema_version());

  // maybe have many meta_graphs here, select the
  // meta graphdef according the tag.
  for (const MetaGraphDef& mgdef : saved_model.meta_graphs()) {
    bool is_target = false;
    for (auto t : mgdef.meta_info_def().tags()) {
      if (t == tag) {
        is_target = true;
        break;
      }
    }
    if (!is_target) continue;

    MetaGraphDef* dynamic_mgdef =
      clustered_graph_info->tf_saved_model.add_meta_graphs();
    MetaGraphDef* static_mgdef =
      clustered_graph_info->iree_saved_model.add_meta_graphs();

    GetDynamicAndStaticMetaGraphDef(mgdef, dynamic_mgdef, static_mgdef);

    break;
  }
  // return:
  // clustered_graph_info->tf_saved_model
  // clustered_graph_info->iree_saved_model
}

ClusteredGraphInfo ClusteringGraphDef(
    const MetaGraphDef& mgdef,
    CluteringStrategy* cluster_strategy) {
  static StaticShapeCluteringStrategy static_strategy;
  if (cluster_strategy == nullptr) {
    cluster_strategy = &static_strategy;
  }

  ClusteredGraphInfo info;
  cluster_strategy->Run(mgdef, &info);

  return info;
}

ClusteredGraphInfo ClusteringGraphDef(
    const std::string& tag,
    const SavedModel& saved_model,
    CluteringStrategy* cluster_strategy) {
  static StaticShapeCluteringStrategy static_strategy;
  if (cluster_strategy == nullptr) {
    cluster_strategy = &static_strategy;
  }

  ClusteredGraphInfo info;
  cluster_strategy->Run(tag, saved_model, &info);

  return info;
}

ClusteredGraphInfo ClusteringGraphDef(
    const std::string& tag,
    const std::string& saved_model_str,
    CluteringStrategy* cluster_strategy) {
  SavedModel saved_model;
  if (!tensorflow::protobuf::TextFormat::ParseFromString(
      saved_model_str, &saved_model)) {
    LOG(FATAL) << "Can not parse saved model from text.";
  }

  return ClusteringGraphDef(tag, saved_model, cluster_strategy);
}

/// Tensorflow

// TODO: FIXME these names may be existed in the graph
// should a method to get the unique name.

const std::string& GetInitDefKey() {
  static std::string init_def_key("GlobalODL/InitKv");
  return init_def_key;
}

const std::string& GetModelVersionNodeName() {
  static std::string name("GlobalODL/ModelVersion");
  return name;
}

const std::string& GetStoragePointerNodeName() {
  static std::string name("GlobalODL/StoragePointer");
  return name;
}

const std::string& GetInitNodeName() {
  static std::string name("GlobalODL/KvInit");
  return name;
}

GraphOptimizer::GraphOptimizer(
    const std::string& signature_name,
    MetaGraphDef* mgdef,
    GraphOptimizerOption& option)
  : graph_(OpRegistry::Global()),
    signature_name_(signature_name),
    meta_graph_def_(mgdef),
    option_(option) {

  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.allow_internal_ops = true;
  Status s = ConvertGraphDefToGraph(
      opts, meta_graph_def_->graph_def(), &graph_);
  if (!s.ok()) {
    LOG(FATAL) << "Can not convert graphdef to graph, "
               << s.error_message();
  }
}


SavedModelOptimizer::SavedModelOptimizer(
    const std::string& signature_name,
    MetaGraphDef* mgdef,
    GraphOptimizerOption& option)
  : GraphOptimizer(signature_name, mgdef, option) {
}

SavedModelOptimizer::~SavedModelOptimizer() {

}

Status SavedModelOptimizer::Optimize() {
  if (option_.native_tf_mode) {
    return RunNativeTFGraphPass();
  }

  return RunODLGraphPass();
}

Status SavedModelOptimizer::RunNativeTFGraphPass() {
  Status s = ConvertToHashTableOps();
  if (!s.ok()) {
    return s;
  }
  // Add other passes here

  // replace the graph def in saved_model_bundle
  graph_.ToGraphDef(meta_graph_def_->mutable_graph_def());

  return Status::OK();
}

Status SavedModelOptimizer::RunODLGraphPass() {
  // Generate ids for every feature
  Status s = GenerateIdsForFeatures();
  if (!s.ok()) return s;

  // Add node for feed version
  s = AddVersionNode();

  // Add a placeholder for storage pointer
  s = AddStoragePlaceholderNode();
  if (!s.ok()) return s;

  // Add KvInit Op
  s = AddVariableInitSubGraph();
  if (!s.ok()) return s;

  s = FreezeSignatureDef();
  if (!s.ok()) return s;

  // For example:
  // convert KvResourceGather to KvLookup
  // convert KvResourceImportV2 to KvImport
  s = ConvertKVOps();
  if (!s.ok()) return s;

  s = RewriteDefaultValueOp();
  if (!s.ok()) return s;

  // replace the graph def in saved_model_bundle
  graph_.ToGraphDef(meta_graph_def_->mutable_graph_def());

  // Add other passes here
  return Status::OK();
}

namespace {

Status ReplaceNode(
    const std::string& op_name,
    Node* node, Graph* graph,
    std::vector<SrcInfo>& input_info,
    std::unordered_map<std::string, const AttrValue*>& attr_info) {

  // Create KvLookup/KvImport op here and remove the node
  NodeDef new_node_def;
  new_node_def.set_name(node->name());
  new_node_def.set_op(op_name);

  // Set attrs
  for (auto attr : attr_info) {
    (*new_node_def.mutable_attr())[attr.first] = *(attr.second);
  }

  Status status;
  Node *new_node = graph->AddNode(new_node_def, &status);
  if (!status.ok()) return status;

  // Add input egdes, from slot 0 -> n
  for (size_t i = 0; i < input_info.size(); ++i) {
    int idx = i;
    if (input_info[i].src_slot == Graph::kControlSlot) {
      idx = Graph::kControlSlot;
    }
    graph->AddEdge(input_info[i].src_node,
                   input_info[i].src_slot,
                   new_node, idx);
  }

  // Add output edges
  for (const Edge* edge : node->out_edges()) {
    graph->AddEdge(new_node, edge->src_output(),
                   edge->dst(), edge->dst_input());
  }

  // remove current node
  graph->RemoveNode(node);

  return Status::OK();
}

Status ReplaceNode(
    const std::string& op_name,
    Node* node, Graph* graph,
    std::vector<SrcInfo>& input_info,
    std::unordered_map<std::string, std::string>& attr_info_map) {
  std::unordered_map<std::string, const AttrValue*> attr_info;
  for (auto m : attr_info_map) {
    const tensorflow::AttrValue* attr = node->attrs().Find(m.first);
    if (attr == nullptr) {
      return tensorflow::errors::Internal(
          "Miss attr: ", m.first, " in the node, ",
          node->DebugString());
    }
    attr_info[m.second] = attr;
  }

  return ReplaceNode(op_name, node, graph,
                     input_info, attr_info);
}

// for KvVarHandleOp and MutableHashTableOfTensorsV2
Status GetShapeValue(Node* node, int* dim) {
  AttrValue* dim_len_value =
      const_cast<AttrValue*>(node->attrs().Find("shape"));
  if (!dim_len_value) {
    dim_len_value = const_cast<AttrValue*>(node->attrs().Find("value_shape"));
  }

  Status s = AttrValueHasType(*dim_len_value, "shape");
  if (!s.ok()) {
    return tensorflow::errors::Internal(
        "Miss shape attr in the KvVarHandleOp, ",
        node->DebugString());
  }

  if (dim_len_value->shape().dim().size() == -1) {
    return tensorflow::errors::Internal(
        "Dim value of the shape in the KvVarHandleOp is unknown, ",
        node->DebugString());
  }

  *dim = dim_len_value->shape().dim(0).size();

  return Status::OK();
}

bool IsKvOps(const Node* node) {
  return node->op_def().name() == "KvResourceGather" ||
         node->op_def().name() == "KvVarHandleOp" ||
         node->op_def().name() == "KvResourceImportV2";
}

Status GetInputNodesInfo(std::vector<SrcInfo>* input_info,
                         const Node* node) {
  int edge_count = 0;
  if (IsKvOps(node)) {
    (*input_info).resize(node->num_inputs());
    for (const Edge* edge : node->in_edges()) {
      if (edge->IsControlEdge()) {
        (*input_info).push_back({edge->src(), edge->src_output()});
      } else {
        ++edge_count;
        (*input_info)[edge->dst_input()] = {edge->src(), edge->src_output()};
      }
    }

    if (edge_count != node->num_inputs()) {
      return tensorflow::errors::Internal(
          "Edge count not match, node = ", node->DebugString(),
          ", need ", std::to_string(node->num_inputs()),
          ", got ", std::to_string(edge_count));
    }
  }

  return Status::OK();
}

Status Create1DStringConstOp(const std::string& name,
                             tensorflow::DataType type,
                             const std::string& value,
                             Graph* graph, Node** new_node) {
  NodeDef const_def;
  const_def.set_name(name);
  const_def.set_op("Const");
  auto* attr = const_def.mutable_attr();
  (*attr)["dtype"].set_type(type);
  Tensor const_tensor(type, TensorShape({1}));
  const_tensor.vec<std::string>()(0) = value;
  const_tensor.AsProtoTensorContent((*attr)["value"].mutable_tensor());

  Status s_add_node;
  *new_node = graph->AddNode(const_def, &s_add_node);
  return s_add_node;
}

Status CreateRestoreOp(const std::string& name,
                       std::vector<SrcInfo>& input_info,
                       std::vector<DataType>& types,
                       Graph* graph, Node** new_node) {
  NodeDef restore_def;
  restore_def.set_name(name);
  restore_def.set_op("RestoreV2");
  DataTypeVector dtypes;
  for (auto t : types) {
    dtypes.push_back(t);
  }
  AttrValue attr_value;
  SetAttrValue(dtypes, &attr_value);
  (*restore_def.mutable_attr())["dtypes"] = attr_value;
  Status status;
  *new_node = graph->AddNode(restore_def, &status);
  if (!status.ok()) return status;

  // Add input egdes
  for (size_t i = 0; i < input_info.size(); ++i) {
    graph->AddEdge(input_info[i].src_node,
                   input_info[i].src_slot,
                   *new_node, i);
  }

  return Status::OK();
}

// create 1-D const op, filled by zeros
Status CreateDefaultValueNode(Graph* graph, int dim_size,
                              Node** new_node, DataType type,
                              const std::string& name) {
  NodeDef const_def;
  const_def.set_name(name);
  const_def.set_op("Const");
  auto* attr = const_def.mutable_attr();
  (*attr)["dtype"].set_type(type);
  Tensor const_tensor(type, TensorShape({dim_size}));
  for (int i = 0; i < dim_size; ++i) {
    switch (type) {
      case DT_INT32: {
        const_tensor.vec<int>()(i) = 0;
        break;
      }
      case DT_INT64: {
        const_tensor.vec<int64>()(i) = 0;
        break;
      }
      case DT_FLOAT: {
        const_tensor.vec<float>()(i) = 0.0;
        break;
      }
      case DT_DOUBLE: {
        const_tensor.vec<double>()(i) = 0.0;
        break;
      }
      case DT_STRING: {
        const_tensor.vec<std::string>()(i) = "";
        break;
      }
      default: {
        LOG(FATAL) << "CreateDefaultValueNode not support type : "
                   << type;
      }
    }
  }
  const_tensor.AsProtoTensorContent((*attr)["value"].mutable_tensor());

  Status s_add_node;
  *new_node = graph->AddNode(const_def, &s_add_node);
  return s_add_node;
}

Status GetNodeAttr(const Node* node,
                   const std::string& attr_name,
                   AttrValue** attr) {
  *attr = const_cast<AttrValue*>(node->attrs().Find(attr_name));
  if (*attr == nullptr) {
     return tensorflow::errors::Internal(
         "Miss attr: ", attr_name ," in the node, ",
         node->DebugString());
  }

  return Status::OK();
}

 Status AddPlaceholder(Graph* g, const std::string& name,
                       DataType type, Node** target_node) {
  // Add a placeholder for 'name' which set by user.
  NodeDef def;
  def.set_name(name);
  def.set_op("Placeholder");
  (*def.mutable_attr())["dtype"].set_type(type);
  Status status;
  *target_node = g->AddNode(def, &status);
  return status;
}

} // namespace

Status SavedModelOptimizer::GetFeature2IdAttr(
    const std::string& name,
    AttrValue* attr_value) {
  if (feature_names_to_ids.find(name) ==
      feature_names_to_ids.end()) {
    return tensorflow::errors::Internal(
        "Not found a id of the featue. ", name);
  }

  SetAttrValue(feature_names_to_ids[name],
               attr_value);
  return Status::OK();
}

Status SavedModelOptimizer::ConvertKVOps() {

  // Find sparse lookup/Import ops and replace them
  // with KvLookup and KvImport
  for (Node* node : graph_.nodes()) {
    // Get input edges
    std::vector<SrcInfo> input_info;
    Status s_input_info = GetInputNodesInfo(&input_info, node);
    TF_RETURN_IF_ERROR(s_input_info);
    int edge_count = node->num_inputs();

    std::unordered_map<std::string, const AttrValue*> attr_info;

    if (node->op_def().name() == "KvResourceGather") {
      if (!storage_pointer_node_ || !version_node_) {
        return tensorflow::errors::Internal(
            "Not found a storage pointer or version node in the graph.");
      }

      std::vector<SrcInfo> gather_input_info;
      // indices
      gather_input_info.push_back(SrcInfo{input_info[1].src_node,
                                          input_info[1].src_slot});
      // default_value
      gather_input_info.push_back(SrcInfo{input_info[2].src_node,
                                          input_info[2].src_slot});
      // storage pointer
      gather_input_info.push_back(SrcInfo{storage_pointer_node_, 0});

      // model version
      gather_input_info.push_back(SrcInfo{version_node_, 0});

      // control edges
      for (size_t i = edge_count; i < input_info.size(); ++i) {
        gather_input_info.push_back(input_info[i]);
      }

      AttrValue feature_name_value;
      SetAttrValue(input_info[0].src_node->name(), &feature_name_value);

      AttrValue feature_name_to_id_value;
      Status s_feature_to_id = GetFeature2IdAttr(
          input_info[0].src_node->name(),
          &feature_name_to_id_value);
      if (!s_feature_to_id.ok()) return s_feature_to_id;

      // get resource shape attr
      int dim_len_value = 0;
      Status s_get_dim = GetShapeValue(input_info[0].src_node, &dim_len_value);
      if (!s_get_dim.ok()) return s_get_dim;

      AttrValue dim_len_value_int;
      SetAttrValue(dim_len_value, &dim_len_value_int);

      AttrValue* dtype_value =
        const_cast<AttrValue*>(node->attrs().Find("dtype"));
      AttrValue* tkeys_value =
        const_cast<AttrValue*>(node->attrs().Find("Tkeys"));
      if (!dtype_value || !tkeys_value) {
        return tensorflow::errors::Internal(
            "Miss dtype or Tkeys attr, ",
            node->DebugString());
      }
      attr_info["feature_name"] = &feature_name_value;
      attr_info["feature_name_to_id"] = &feature_name_to_id_value;
      attr_info["dim_len"] = &dim_len_value_int;
      attr_info["dtype"] = dtype_value;
      attr_info["Tkeys"] = tkeys_value;

      Status s_replace = ReplaceNode(
          "KvLookup", node, &graph_, gather_input_info, attr_info);
      if (!s_replace.ok()) {
        return tensorflow::errors::Internal(
            "Replace Kv ops with lookup or import ops failed, ",
            s_replace.error_message());
      }

    } else if (node->op_def().name() == "KvResourceImportV2") {
      if (!storage_pointer_node_ || !version_node_) {
        return tensorflow::errors::Internal(
            "Not found a storage pointer or version node in the graph.");
      }

      std::vector<SrcInfo> import_input_info;
      // prefix
      import_input_info.push_back(SrcInfo{input_info[0].src_node,
                                          input_info[0].src_slot});
      // tensor_names
      import_input_info.push_back(SrcInfo{input_info[3].src_node,
                                          input_info[3].src_slot});
      // storage pointer
      import_input_info.push_back(SrcInfo{storage_pointer_node_, 0});

      // model version
      import_input_info.push_back(SrcInfo{version_node_, 0});

      // control edges
      for (size_t i = edge_count; i < input_info.size(); ++i) {
        import_input_info.push_back(input_info[i]);
      }

      AttrValue feature_name_value;
      SetAttrValue(input_info[1].src_node->name(), &feature_name_value);
      AttrValue feature_name_to_id_value;
      Status s_feature_to_id = GetFeature2IdAttr(
          input_info[1].src_node->name(),
          &feature_name_to_id_value);
      if (!s_feature_to_id.ok()) return s_feature_to_id;

      // get resource shape attr
      int dim_len_value = 0;
      Status s_get_dim = GetShapeValue(input_info[1].src_node, &dim_len_value);
      if (!s_get_dim.ok()) return s_get_dim;

      AttrValue dim_len_value_int;
      SetAttrValue(dim_len_value, &dim_len_value_int);
      attr_info["dim_len"] = &dim_len_value_int;

      AttrValue* dtype_value =
        const_cast<AttrValue*>(node->attrs().Find("dtype"));
      AttrValue* tkeys_value =
        const_cast<AttrValue*>(node->attrs().Find("Tkeys"));
      if (!dtype_value || !tkeys_value) {
        return tensorflow::errors::Internal(
            "Miss dtype or Tkeys attr, ",
            node->DebugString());
      }

      attr_info["feature_name"] = &feature_name_value;
      attr_info["feature_name_to_id"] = &feature_name_to_id_value;
      attr_info["dtype"] = dtype_value;
      attr_info["Tkeys"] = tkeys_value;

      Status s_replace = ReplaceNode(
          "KvImport", node, &graph_, import_input_info, attr_info);
      if (!s_replace.ok()) {
        return tensorflow::errors::Internal(
            "Replace Kv ops with lookup or import ops failed, ",
            s_replace.error_message());
      }
    }
  }

  return Status::OK();
}

Status SavedModelOptimizer::FreezeSignatureDef() {
  std::map<string, SignatureDef> new_signature_def;
  bool found = false;
  for (auto sdef : meta_graph_def_->signature_def()) {
    if (sdef.first == signature_name_) {
      new_signature_def[signature_name_] = sdef.second;
      found = true;
      break;
    }
  }

  if (!found) {
    return tensorflow::errors::Internal(
        "Not found the signature_def with user specified signature name.",
        signature_name_);
  }

  meta_graph_def_->clear_signature_def();
  auto sig_def = meta_graph_def_->mutable_signature_def();
  for (auto sdef : new_signature_def) {
    (*sig_def)[sdef.first] = sdef.second;
  }

  // Add Init kv def
  Node* init_op = nullptr;
  for (Node* node : graph_.nodes()) {
    if (node->name() == GetInitNodeName()) {
      init_op = node;
      break;
    }
  }

  if (!init_op) {
    return tensorflow::errors::Internal(
        "Not found the init kv op in the graph.",
        GetInitNodeName());
  }

  (*sig_def)[GetInitDefKey()] = SignatureDef();
  TensorInfo tinfo;
  tinfo.set_name(GetInitNodeName()+":0");
  (*(*sig_def)[GetInitDefKey()].mutable_outputs())["init_op"] = tinfo;

  return Status::OK();
}

Status SavedModelOptimizer::RewriteDefaultValueOp() {
  return Status::OK();
}

Status SavedModelOptimizer::AddVersionNode() {
  // Add a placeholder for version which set by user.
  return AddPlaceholder(&graph_, GetModelVersionNodeName(),
                        DT_UINT64, &version_node_);
}

Status SavedModelOptimizer::AddStoragePlaceholderNode() {
  // Add a placeholder for storage pointer which set by user.
  return AddPlaceholder(&graph_, GetStoragePointerNodeName(),
                        DT_UINT64, &storage_pointer_node_);
}

Status SavedModelOptimizer::AddVariableInitSubGraph() {
  // Add a init_op to initialize storage like redis.
  NodeDef init_op_def;
  init_op_def.set_name(GetInitNodeName());
  init_op_def.set_op("KvInit");

  std::vector<std::string> feature_names;
  for (Node* node : graph_.nodes()) {
    // Add resource name as feature_names attr
    if (node->op_def().name() == "KvResourceImportV2") {
      feature_names.push_back(node->name());
    }
  }
  AddNodeAttr("feature_names", feature_names, &init_op_def);

  Status status;
  Node* init_op_node = graph_.AddNode(init_op_def, &status);
  return status;
}

Status SavedModelOptimizer::GenerateIdsForFeatures() {
  int id = 0;
  for (Node* node : graph_.nodes()) {
    if (node->op_def().name() == "KvVarHandleOp") {
      feature_names_to_ids[node->name()] = id++;
    }
  }

  return Status::OK();
}

Status SavedModelOptimizer::AddFullAndDeltaUpdateSubGraph() {
  return Status::OK();
}

Status SavedModelOptimizer::ConvertToHashTableOp(
    Node* node, std::vector<SrcInfo>& input_info) {
  std::unordered_map<std::string, std::string> attr_info_map;

  // KvVarHandleOp -> MutableHashTableV2
  attr_info_map["container"] = "container";
  attr_info_map["shared_name"] = "shared_name";
  attr_info_map["dtype"] = "value_dtype";
  attr_info_map["Tkeys"] = "key_dtype";
  attr_info_map["shape"] = "value_shape";

  return ReplaceNode(
      "MutableHashTableOfTensorsV2", node, &graph_, input_info,
      attr_info_map);
}

Status SavedModelOptimizer::ConvertToHashLookupOp(
    Node* node, std::vector<SrcInfo>& input_info) {
  std::unordered_map<std::string, std::string> attr_info_map;

  // KvResourceGather -> LookupTableFindV2
  attr_info_map["dtype"] = "Tout";
  attr_info_map["Tkeys"] = "Tin";

  AttrValue* dtype_attr = nullptr;
  Status s_attr = GetNodeAttr(node, "dtype", &dtype_attr);
  TF_RETURN_IF_ERROR(s_attr);

  // create a defaualt value node
  Node* default_value_node = nullptr;
  int dim = 0;
  Status s_shape_value = GetShapeValue(input_info[0].src_node, &dim);
  TF_RETURN_IF_ERROR(s_shape_value);
  // shape = [dim]
  Status s_default_value = CreateDefaultValueNode(
      &graph_, dim, &default_value_node, dtype_attr->type(),
      input_info[2].src_node->name() + "/new_default_value");
  TF_RETURN_IF_ERROR(s_default_value);

  // use newly created default value node(filled by zeros)
  input_info[2].src_node = default_value_node;
  input_info[2].src_slot = 0;
  return ReplaceNode(
      "LookupTableFindV2", node, &graph_, input_info,
      attr_info_map);
}

Status SavedModelOptimizer::ConvertToHashImportOp(
    Node* node, std::vector<SrcInfo>& input_info) {
  std::unordered_map<std::string, std::string> attr_info_map;
  int edge_count = node->num_inputs();

  // KvResourceImportV2 -> LookupTableImportV2
  AttrValue* key_attr = nullptr;
  Status s_attr = GetNodeAttr(node, "Tkeys", &key_attr);
  TF_RETURN_IF_ERROR(s_attr);

  AttrValue* dtype_attr = nullptr;
  s_attr = GetNodeAttr(node, "dtype", &dtype_attr);
  TF_RETURN_IF_ERROR(s_attr);

  // create restore_op to restore values and keys
  // create const "tensor_names-keys" and "tensor_names-values" op
  Node* empty_shape_and_slices_node = nullptr;
  Status s_add_node = Create1DStringConstOp(
      node->name() + "/const/shape_and_slices", DT_STRING,
      "", &graph_, &empty_shape_and_slices_node);
  TF_RETURN_IF_ERROR(s_add_node);

  AttrValue* prefix_attr = nullptr;
  s_attr = GetNodeAttr(input_info[3].src_node, "value", &prefix_attr);
  TF_RETURN_IF_ERROR(s_attr);
  Tensor prefix_tensor;
  bool success = prefix_tensor.FromProto(prefix_attr->tensor());
  if (!success) {
    return errors::Internal("Can not parse prefix_attr->tensor()"
                            "to prefix_tensor");
  }

  std::string prefix = prefix_tensor.scalar<string>()();

  Node* tensor_names_key_node = nullptr;
  s_add_node = Create1DStringConstOp(
      node->name() + "/const/keys", DT_STRING,
      prefix + "-keys",
      &graph_, &tensor_names_key_node);
  TF_RETURN_IF_ERROR(s_add_node);

  Node* tensor_names_value_node = nullptr;
  s_add_node = Create1DStringConstOp(
      node->name() + "/const/values", DT_STRING,
      prefix + "-values",
      &graph_, &tensor_names_value_node);
  TF_RETURN_IF_ERROR(s_add_node);

  // 1) create keys restore_op
  std::vector<SrcInfo> restore_input_info;
  restore_input_info.push_back(input_info[0]);
  restore_input_info.push_back(SrcInfo{tensor_names_key_node, 0});
  restore_input_info.push_back(SrcInfo{empty_shape_and_slices_node, 0});
  std::vector<DataType> key_types;
  key_types.push_back(key_attr->type());

  Node* restore_key_node = nullptr;
  s_add_node = CreateRestoreOp(node->name() + "/keys/restore",
                               restore_input_info, key_types,
                               &graph_, &restore_key_node);
  TF_RETURN_IF_ERROR(s_add_node);

  // 2) create values restore_op
  std::vector<DataType> value_types;
  value_types.push_back(dtype_attr->type());
  restore_input_info[1] = SrcInfo{tensor_names_value_node, 0};
  Node* restore_value_node = nullptr;
  s_add_node = CreateRestoreOp(node->name() + "/values/restore",
                               restore_input_info, value_types,
                               &graph_, &restore_value_node);
  TF_RETURN_IF_ERROR(s_add_node);
 
  std::vector<SrcInfo> import_input_info;
  import_input_info.push_back(input_info[1]);
  import_input_info.push_back(SrcInfo{restore_key_node, 0});
  import_input_info.push_back(SrcInfo{restore_value_node, 0});
  // control edges
  for (size_t i = edge_count; i < input_info.size(); ++i) {
    import_input_info.push_back(input_info[i]);
  }

  attr_info_map["Tkeys"] = "Tin";
  attr_info_map["dtype"] = "Tout";

  return ReplaceNode(
      "LookupTableImportV2", node, &graph_, import_input_info,
      attr_info_map);
}

Status SavedModelOptimizer::ConvertToHashTableOps() {

  for (Node* node : graph_.nodes()) {
    // Get input edges
    std::vector<SrcInfo> input_info;
    Status s_input_info = GetInputNodesInfo(&input_info, node);
    TF_RETURN_IF_ERROR(s_input_info);

    Status s_replace;
    if (node->op_def().name() == "KvVarHandleOp") {
      // KvVarHandleOp -> MutableHashTableOfTensorsV2
      s_replace = ConvertToHashTableOp(node, input_info);
    } else if (node->op_def().name() == "KvResourceGather") {
      // KvResourceGather -> LookupTableFindV2
      s_replace = ConvertToHashLookupOp(node, input_info);
    } else if (node->op_def().name() == "KvResourceImportV2") {
      // KvResourceImportV2 -> LookupTableImportV2
      s_replace = ConvertToHashImportOp(node, input_info);
    }

    if (!s_replace.ok()) {
      return tensorflow::errors::Internal(
          "Replace Kv ops with hash-table, hash-lookup"
          "or hash-import ops failed, ",
           s_replace.error_message());
    }
  }

  return Status::OK();
}

} // namespace processor
} // namespace tensorflow

