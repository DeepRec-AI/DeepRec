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

#include "serving/processor/framework/graph_optimizer.h"
#include "serving/processor/framework/util/utils.h"
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

static const int gather_input_resource_slot = 0;
static const int gather_input_indice_slot = 1;
static const int gather_input_default_val_slot = 2;
static const int import_input_prefix_slot = 0;
static const int import_input_resource_slot = 1;
static const int import_input_tname_slot = 4;

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

const std::string& GetIncrCkptNodeName() {
  static std::string name("GlobalODL/ImportIncrCkpt");
  return name;
}

const std::string& GetInitNodeName() {
  static std::string name("GlobalODL/KvInit");
  return name;
}

const std::string& GetKvRestoreAllNameSuffix() {
  static std::string suffix("/Kv_all");
  return suffix;
}

const std::string& GetKvIncrRestoreAllNameSuffix() {
  static std::string suffix("/Kv_incr_all");
  return suffix;
}

const std::string& GetDenseRestoreAllNameSuffix() {
  static std::string suffix("/Dense_all");
  return suffix;
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

  // Add default attributes
  GraphDef converted_graph_def;
  Status s = processor::AddDefaultAttributes(
    meta_graph_def_->graph_def(), &converted_graph_def);
  if (!s.ok()) {
    LOG(WARNING) << "Add default attrs to graphdef failed. "
                 << s.error_message();
  }
  s = ConvertGraphDefToGraph(
      opts, converted_graph_def/*meta_graph_def_->graph_def()*/, &graph_);
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
  if (option_.shard_embedding) {
    // Find all variables ops which should be rewrite,
    // include partitioned or non-partitoned variables
    std::unordered_map<std::string, std::vector<Node*>> var_parts;
    TF_RETURN_IF_ERROR(FindVariableParts(var_parts));

    std::unordered_map<std::string, std::vector<Node*>> import_nodes;
    TF_RETURN_IF_ERROR(FindKvResourceImportNode(var_parts, import_nodes));

    // Rewrite subgraph
    TF_RETURN_IF_ERROR(RewriteEmbeddingLookupGraph(var_parts, import_nodes));
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

  s = AddIncrCkptFlag();
  if (!s.ok()) return s;

  // Add KvInit Op
  s = AddVariableInitSubGraph();
  if (!s.ok()) return s;

  // Create dense and sparse restore ops
  s = CreateDenseAndSparseRestoreOp();
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

Status CreateRestoreAllNode(const std::string& name,
                            const std::string& op_name,
                            std::vector<SrcInfo>& inputs,
                            Graph* graph) {
  NodeDef def;
  def.set_name(name);
  def.set_op(op_name);
  Status status;
  Node *new_node = graph->AddNode(def, &status);
  if (!status.ok()) return status;

  // Add input egdes
  int dst_in_slot = Graph::kControlSlot;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].src_slot != Graph::kControlSlot) {
      LOG(FATAL) << "RestoreAll op only allow control eages. "
                 << "Current input node: " << inputs[i].src_node->DebugString()
                 << ", slot: " << inputs[i].src_slot;
    }

    graph->AddEdge(inputs[i].src_node, inputs[i].src_slot,
                   new_node, dst_in_slot);
  }

  return Status::OK();
}

Status FindGatherNode(Node* n, const std::unordered_set<std::string>& stop_nodes,
    Node** gather_node) {
  *gather_node = nullptr;
  std::unordered_set<Node*> pushed;
  std::queue<Node*> q;
  q.push(n);
  pushed.insert(n);
  while (!q.empty()) {
    Node* curr = q.front();
    q.pop();
    if (curr->op_def().name() == "KvResourceGather") {
      if (*gather_node) {
        LOG(FATAL) << "Find many KvResourceGather op from variable: " << n->DebugString()
                   << ", gather1: " << (*gather_node)->DebugString()
                   << ", gather2: " << curr->DebugString();
      }
      *gather_node = curr;
      continue;
    }

    if (stop_nodes.find(curr->op_def().name()) == stop_nodes.end()) {
      for (const Edge* edge : curr->out_edges()) {
        if (pushed.find(edge->dst()) == pushed.end()) {
          pushed.insert(edge->dst());
          q.push(edge->dst());
        }
      }
    }
  }

  return Status::OK();
}

bool IsDynamicStitchOp(Node* n) {
  return n->op_def().name() == "DynamicStitch" ||
         n->op_def().name() == "DynamicStitchFast" ||
         n->op_def().name() == "ParallelDynamicStitch";
}

Node* FindDynamicStitchNode(Node* gather_node) {
  // Supported pattern:
  //
  //        KvResourceGather
  //              |
  //           Identity
  //              |
  //             ...
  //              |
  //           Identity
  //              |
  //     ParallelDynamicStitch
  //
  if (gather_node->num_outputs() != 1) {
    LOG(FATAL) << "Gather node has more than one edge, "
               << gather_node->DebugString();
  }

  for (const Edge* edge : gather_node->out_edges()) {
    Node* dest = edge->dst();
    while (dest->op_def().name() == "Identity") {
      Node* tmp = nullptr;
      for (const Edge* e : dest->out_edges()) {
        tmp = e->dst();
        break;
      }
      dest = tmp;
    }

    if (IsDynamicStitchOp(dest)) {
      return dest;
    }
  }

  return nullptr;
}

Node* CheckDynamicStitchNode(std::vector<Node*> gather_nodes) {
  Node* ds_node = nullptr;
  for (size_t i = 0; i < gather_nodes.size(); ++i) {
    Node* tmp = FindDynamicStitchNode(gather_nodes[i]);
    if (!ds_node) ds_node = tmp;
    if (!tmp) {
      LOG(FATAL) << "Gather node do not output to a DynamicStitch node, "
                 << gather_nodes[i]->DebugString();
    }
    if (tmp != ds_node) {
      LOG(FATAL) << "Gather Node output to different DynamicStitch node"
                 << ", gather node1: " << gather_nodes[0]->DebugString()
                 << ", DynamicStitch node1: " << ds_node->DebugString()
                 << ", gather node2: " << gather_nodes[i]->DebugString()
                 << ", DynamicStitch node2: " << tmp->DebugString(); 
    }
  }

  return ds_node;
}

Node* CheckReshapeNode(Node* dynamic_stitch) {
  Node* reshape_node = nullptr;
  for (const Edge* edge : dynamic_stitch->out_edges()) {
    if (!edge->IsControlEdge() &&
        edge->dst()->op_def().name() == "Reshape") {
      if (reshape_node) {
        LOG(FATAL) << "Found many reshape node edges from DynamicStitch node: "
                   << dynamic_stitch->DebugString();
      }
      reshape_node = edge->dst();
    }
  }

  if (!reshape_node) {
    LOG(FATAL) << "Can not found reshape node edges from DynamicStitch node: "
               << dynamic_stitch->DebugString(); 
  }

  return reshape_node;
}

Node* InternalFindUniqueNode(Node* n) {
  std::unordered_set<Node*> pushed;
  std::queue<Node*> q;
  q.push(n);
  pushed.insert(n);
  while (!q.empty()) {
    Node* curr = q.front();
    q.pop();
    if (curr->op_def().name() == "Unique" ||
        curr->op_def().name() == "UniqueV2") {
      return curr;
    }

    for (const Edge* edge : curr->in_edges()) {
      if (edge->IsControlEdge()) continue;
      if (pushed.find(edge->src()) == pushed.end()) {
        pushed.insert(edge->src());
        q.push(edge->src());
      }
    }
  }

  return nullptr;
}

Node* CheckUniqueNode(std::vector<Node*> gather_nodes) {
  Node* unique_node = nullptr;
 
  for (size_t i = 0; i < gather_nodes.size(); ++i) {
    Node* tmp_node = InternalFindUniqueNode(gather_nodes[i]);
    if (!tmp_node) {
      LOG(FATAL) << "Can not found unique node from current gather node: "
                 << gather_nodes[i]->DebugString();
    }
    if (!unique_node) unique_node = tmp_node;

    if (tmp_node != unique_node) {
      LOG(FATAL) << "Found different unique nodes from two gather node "
                 << ", gather node1: " << gather_nodes[0]->DebugString()
                 << ", unique node1: " << unique_node->DebugString()
                 << ", gather node2: " << gather_nodes[i]->DebugString()
                 << ", unique node2: " << tmp_node->DebugString();
    }
  }

  return unique_node;
}

Status ModifySharedNameAttr(Node* new_var_op, int curr_partition_id) {
  std::string shared_name;
  TF_RETURN_IF_ERROR(GetNodeAttr(new_var_op->attrs(), "shared_name", &shared_name));
  if (shared_name == "") {
    LOG(FATAL) << "KvVarHandleOp can not contain the shared_name attr: "
               << new_var_op->DebugString();
  }

  auto offset = shared_name.find("/part_");
  if (offset == std::string::npos) {
    shared_name = shared_name + "/part_" + std::to_string(curr_partition_id);
  } else {
    // Inference graph is not contain slot variable node, so we do not need to consider
    // xx/yy/zz/part_0/aa pattern here
    shared_name = shared_name.substr(0, offset) +
        "/part_" + std::to_string(curr_partition_id);
  }

  new_var_op->ClearAttr("shared_name");
  new_var_op->AddAttr("shared_name", shared_name);

  return Status::OK();
}

Node* CheckInputDefaultValNode(Node* node, int slot, int* src_slot) {
  for (const Edge* edge : node->in_edges()) {
    if (!edge->IsControlEdge() && edge->dst_input() == slot) {
      *src_slot = edge->src_output();
      return edge->src();
    }
  }

  LOG(FATAL) << "Can not found DefaultValNode in in_edges: "
             << node->DebugString() << ", slot: " << slot;

  return nullptr;
}

Status CreateScalarStringConstOp(const std::string& name,
                                 const std::string& value,
                                 Graph* graph, Node** new_node) {
  NodeDef const_def;
  const_def.set_name(name);
  const_def.set_op("Const");
  auto* attr = const_def.mutable_attr();
  (*attr)["dtype"].set_type(DT_STRING);
  Tensor const_tensor(DT_STRING, TensorShape({}));
  const_tensor.scalar<std::string>()() = value;
  const_tensor.AsProtoTensorContent((*attr)["value"].mutable_tensor());

  Status s_add_node;
  *new_node = graph->AddNode(const_def, &s_add_node);
  return s_add_node;
}

Node* CreateKvResourceImportNode(Graph* graph, Node* origin_import_node,
    Node* new_var_op, int partition_id, int shard_instance_count) {
  // create new tensor_name node
  std::string shared_name;
  if (!GetNodeAttr(new_var_op->attrs(), "shared_name", &shared_name).ok()) {
    LOG(FATAL) << "Get nodet attr failed, attr: shared_name, node: "
               << new_var_op->DebugString();
  }
  Node* new_tensor_name_node = nullptr;
  Status status;
  status = CreateScalarStringConstOp(
      new_var_op->name()+"/tensor_names",
      shared_name, graph, &new_tensor_name_node);
  if (!status.ok()) {
    LOG(FATAL) << "Create tensor name const node failed, "
               << status.error_message();
  }

  NodeDef new_import_opdef = origin_import_node->def();
  new_import_opdef.set_name(new_var_op->name()+"/KvResourceImportV2");
  new_import_opdef.set_input(1, new_var_op->name());
  new_import_opdef.set_input(3, new_tensor_name_node->name());
  Node* new_import_op = graph->AddNode(new_import_opdef, &status);
  if (!status.ok()) {
    LOG(FATAL) << "Create new import node failed, "
               << status.error_message()
               << ", origin_import_node: "
               << origin_import_node->DebugString();
  }
  new_import_op->ClearAttr("partition_id");
  new_import_op->ClearAttr("partition_num");
  new_import_op->AddAttr("partition_id", partition_id);
  new_import_op->AddAttr("partition_num", shard_instance_count);

  // origin import node will be deleted later
  //graph->RemoveNode(origin_import_node);

  return new_import_op;
}

Status CreateNewEmbeddingLookupGraph(
    Graph* graph, Node* origin_var_node, const std::string& new_var_name,
    Node* gather_node, Node* reshape_node, Node* unique_node,
    Node* default_val_node, int default_val_node_slot, int curr_partition_id,
    int shard_instance_count, Node* origin_import_node,
    Node** new_kv_import_op) {
  Status status;
  // Add new KvVarHandleOp node
  NodeDef new_var_opdef = origin_var_node->def();
  new_var_opdef.set_name(
      new_var_name+"/odl_var_part/part_"+std::to_string(curr_partition_id));
  Node* new_var_op = graph->AddNode(new_var_opdef, &status);
  TF_RETURN_IF_ERROR(status);
  // reset the shared_name attr
  TF_RETURN_IF_ERROR(ModifySharedNameAttr(new_var_op, curr_partition_id));

  // Add new KvResourceGather node
  NodeDef new_gather_def = gather_node->def();
  new_gather_def.set_name(new_var_op->name()+"/KvResourceGather");
  new_gather_def.clear_input();
  Node* new_gather_op = graph->AddNode(new_gather_def, &status);
  TF_RETURN_IF_ERROR(status);
  // No need add control edges here.
  graph->AddEdge(new_var_op, 0, new_gather_op, 0);
  graph->AddEdge(unique_node, 0, new_gather_op, 1);
  graph->AddEdge(default_val_node, default_val_node_slot, new_gather_op, 2);

  // Add New Identity Node
  DataType gather_dtype;
  if (!GetNodeAttr(new_gather_op->attrs(), "dtype", &gather_dtype).ok()) {
    LOG(FATAL) << "Get nodet attr failed, attr: dtype, node: "
               << new_gather_op->DebugString();
  }
  NodeDef identity_op_def;
  identity_op_def.set_name(new_var_op->name()+"/Identity");
  identity_op_def.set_op("Identity");
   (*identity_op_def.mutable_attr())["T"].set_type(gather_dtype);
  identity_op_def.add_input(new_var_op->name()+"/KvResourceGather");
  Node* identity_op_node = graph->AddNode(identity_op_def, &status);
  if (!status.ok()) {
    LOG(FATAL) << "Create Identity node failed when CreateNewEmbeddingLookupGraph.";
  }

  for (const Edge* edge : reshape_node->out_edges()) {
    int slot = 0;
    if (edge->IsControlEdge()) {
      slot = Graph::kControlSlot;
    }

    graph->AddEdge(identity_op_node, slot, edge->dst(),
                   edge->dst_input());
  }

  // Create new KvResourceImportV2 node
  *new_kv_import_op =
      CreateKvResourceImportNode(graph, origin_import_node,
                                 new_var_op, curr_partition_id,
                                 shard_instance_count);
 
  return Status::OK();
}

void DeleteOriginEmbeddingLookupGraphNodes(
    const std::vector<Node*>& var_parts, Graph* graph) {
  for (Node* n : var_parts) {
    std::unordered_set<Node*> pushed;
    std::queue<Node*> q;
    q.push(n);
    pushed.insert(n);
    while (!q.empty()) {
      Node* curr = q.front();
      q.pop();

      for (const Edge* edge : curr->out_edges()) {
        if (pushed.find(edge->dst()) == pushed.end()) {
          pushed.insert(edge->dst());
          q.push(edge->dst());
        }
      }

      graph->RemoveNode(curr);
    }
  }
}

} // namespace

Status SavedModelOptimizer::RewriteEmbeddingLookupGraph(
    std::unordered_map<std::string, std::vector<Node*>>& var_parts_map,
    std::unordered_map<std::string, std::vector<Node*>>& origin_import_nodes) {
  int curr_partition_id = option_.partition_id;
  int shard_instance_count = option_.shard_instance_count;
  std::vector<Node*> new_kv_import_nodes;
 std::unordered_map<std::string, bool> delete_origin_graph_nodes;
  for (auto vp : var_parts_map) {
    if (vp.second.size() == 1 && vp.first == vp.second[0]->name()) {
      delete_origin_graph_nodes[vp.first] = false;
      // non-partitoned variable
      // rename node, from "xx/yy/zz" to "xx/yy/zz/part_curr_partition_id",
      // this will let KvResourceImportOp load variable slice according
      // the partition policy.
      Status status;
      NodeDef new_var_opdef = vp.second[0]->def();
      new_var_opdef.set_name(vp.second[0]->name() + "/odl_var_part/part_" + std::to_string(curr_partition_id));
      Node* new_var_op = graph_.AddNode(new_var_opdef, &status);
      TF_RETURN_IF_ERROR(status);

      // reset the shared_name attr
      TF_RETURN_IF_ERROR(ModifySharedNameAttr(new_var_op, curr_partition_id));

      for (const Edge* edge : vp.second[0]->out_edges()) {
        graph_.AddEdge(new_var_op, edge->src_output(),
                       edge->dst(), edge->dst_input());
      }

      // remove old var node
      graph_.RemoveNode(vp.second[0]);

      // Create new KvResourceImportV2 node
      Node* new_kv_import_op =
          CreateKvResourceImportNode(&graph_, origin_import_nodes[vp.first][0],
                                     new_var_op, curr_partition_id, shard_instance_count);
      new_kv_import_nodes.push_back(new_kv_import_op);
    } else {
      delete_origin_graph_nodes[vp.first] = true;
      // partitioned variable, like 'xx/yy/zz/part_0', 'xx/yy/zz/part_1' ...
      // delete these part nodes, and create "xx/yy/zz/part_curr_partition_id" node,
      // then, rewrite the embedding lookup subgraph.
      std::unordered_set<std::string> stop_nodes
          { "SparseSegmentSqrtN", "SparseSegmentSum",
            "SparseSegmentMean", "DynamicStitch",
            "ParallelDynamicStitch" };
      std::vector<Node*> gather_nodes;
      gather_nodes.resize(vp.second.size());
      for (size_t i = 0; i < vp.second.size(); ++i) {
        gather_nodes[i] = nullptr;
        TF_RETURN_IF_ERROR(FindGatherNode(vp.second[i], stop_nodes, &gather_nodes[i]));
      }

      //                           Unique
      //                             |
      //                            ...
      //                             | 
      //                       DynamicPartition
      //                             |
      //          ----------------------------------------
      //          |                  |                   |
      //    KvResourceGather  KvResourceGather_1  KvResourceGather_2
      //          |                  |                   |
      //          ----------------------------------------
      //                             |
      //                       DynamicStitch
      //                             |
      //                          Reshape
      //                             |
      //                      SparseSegmentMean (or clip ops)
      //
      Node* dynamic_stitch_node = CheckDynamicStitchNode(gather_nodes);
      Node* reshape_node = CheckReshapeNode(dynamic_stitch_node);

      Node* unique_node = CheckUniqueNode(gather_nodes);

      // create a default value node, not use origin default value.
      // Node* origin_default_val_node = CheckInputDefaultValNode(gather_nodes[0], 2, &src_slot);
      Node* default_val_node = nullptr;
      int src_slot = 0; 
      int default_value_dim = 0;
      TF_RETURN_IF_ERROR(GetShapeValue(vp.second[0], &default_value_dim));
      DataType default_val_type;
      TF_RETURN_IF_ERROR(GetNodeAttr(gather_nodes[0]->attrs(),
                                     "dtype", &default_val_type));
      // shape = [dim]
      TF_RETURN_IF_ERROR(CreateDefaultValueNode(
          &graph_, default_value_dim, &default_val_node, default_val_type,
          vp.first+"/odl_var_part/part_"+std::to_string(curr_partition_id) +"/default_value"));

      //  Convert subgraph above to below:
      //
      //                 Unique
      //                   |
      //      KvResourceGather_partition_id
      //                   |
      //                Identity
      //                   |
      //             SparseSegmentMean (or clip ops)
      //
      Node* new_kv_import_op = nullptr;
      TF_RETURN_IF_ERROR(
          CreateNewEmbeddingLookupGraph(&graph_, vp.second[0], vp.first, gather_nodes[0],
                                        reshape_node, unique_node, default_val_node,
                                        src_slot, curr_partition_id, shard_instance_count,
                                        origin_import_nodes[vp.first][0], &new_kv_import_op));
      new_kv_import_nodes.push_back(new_kv_import_op);

      // delete Reshape and DynamicStitch node
      graph_.RemoveNode(reshape_node);
      graph_.RemoveNode(dynamic_stitch_node);
    }
  }

  // create new restore_op
  Node* restore_shard =
      UpdateRestoreShardNodeInputs(origin_import_nodes, new_kv_import_nodes);

  // delete nodes from variable node(KvVarHandleOp) to Reshape node
  for (auto vp : var_parts_map) {
    if (delete_origin_graph_nodes[vp.first]) {
      DeleteOriginEmbeddingLookupGraphNodes(vp.second, &graph_);
    }
  }

  // Add IncrRestore node
  std::vector<SrcInfo> input_nodes;
  for (const Edge* edge : restore_shard->in_edges()) {
    input_nodes.push_back({edge->src(), edge->src_output()});
  }
  return CreateRestoreAllNode(
      meta_graph_def_->saver_def().restore_op_name() +
          GetKvIncrRestoreAllNameSuffix(),
      "NoOp", input_nodes, &graph_);
}

Status SavedModelOptimizer::FindKvResourceImportNode(
    std::unordered_map<std::string, std::vector<Node*>>& var_parts,
    std::unordered_map<std::string, std::vector<Node*>>& import_nodes) {
  for (auto vp : var_parts) {
    for (Node* n : vp.second) {
      for (const Edge* edge : n->out_edges()) {
        if (edge->dst()->op_def().name() == "KvResourceImportV2") {
          import_nodes[vp.first].push_back(edge->dst());
        }
      }
    }
    // NOTE: Why import_nodes num is twice to var?
    // Please see the details in KvResourceImport defination.
    if (import_nodes.find(vp.first) == import_nodes.end() ||
        import_nodes[vp.first].size() != 2 * var_parts[vp.first].size()) {
      LOG(FATAL) << "KvVarHandle node count should be twice to KvResourceImportV2 node count "
                 << ", " << vp.first << ", " << import_nodes[vp.first].size() << " VS " << var_parts[vp.first].size();
    }
  }

  return Status::OK();
}

Status SavedModelOptimizer::FindVariableParts(
    std::unordered_map<std::string, std::vector<Node*>>& var_parts) {
  // TODO: only support embedding variable currently
  for (Node* node : graph_.nodes()) {
    if (node->op_def().name() == "KvVarHandleOp") {
      for (auto sname : option_.shard_embedding_names) {
        if (node->name() == sname ||
            node->name().find(sname+"/part_") != std::string::npos) {
          var_parts[sname].push_back(node);
        }
      }
    }
  }

  for (auto sname : option_.shard_embedding_names) {
    if (var_parts.find(sname) == var_parts.end() ||
        var_parts[sname].size() == 0) {
      return tensorflow::errors::Internal(
          "Can not found variable info in graph: ", sname);
    }
  }

  return Status::OK();
}

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
      gather_input_info.push_back(
          SrcInfo{input_info[gather_input_indice_slot].src_node,
                  input_info[gather_input_indice_slot].src_slot});
      // default_value
      gather_input_info.push_back(
          SrcInfo{input_info[gather_input_default_val_slot].src_node,
                  input_info[gather_input_default_val_slot].src_slot});
      // storage pointer
      gather_input_info.push_back(SrcInfo{storage_pointer_node_, 0});

      // model version
      gather_input_info.push_back(SrcInfo{version_node_, 0});

      // control edges
      for (size_t i = edge_count; i < input_info.size(); ++i) {
        gather_input_info.push_back(input_info[i]);
      }

      AttrValue feature_name_value;
      SetAttrValue(
          input_info[gather_input_resource_slot].src_node->name(), &feature_name_value);

      AttrValue feature_name_to_id_value;
      Status s_feature_to_id = GetFeature2IdAttr(
          input_info[gather_input_resource_slot].src_node->name(),
          &feature_name_to_id_value);
      if (!s_feature_to_id.ok()) return s_feature_to_id;

      // get resource shape attr
      int dim_len_value = 0;
      Status s_get_dim = GetShapeValue(
          input_info[gather_input_resource_slot].src_node, &dim_len_value);
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
      import_input_info.push_back(
          SrcInfo{input_info[import_input_prefix_slot].src_node,
                  input_info[import_input_prefix_slot].src_slot});
      // tensor_names
      import_input_info.push_back(
          SrcInfo{input_info[import_input_tname_slot].src_node,
                  input_info[import_input_tname_slot].src_slot});
      // storage pointer
      import_input_info.push_back(SrcInfo{storage_pointer_node_, 0});

      // model version
      import_input_info.push_back(SrcInfo{version_node_, 0});

      // incr ckpt flag
      if (!incr_ckpt_node_) {
        return tensorflow::errors::Internal(
            "Not found a incr_ckpt node in the graph.");
      }
      import_input_info.push_back(SrcInfo{incr_ckpt_node_, 0});

      // control edges
      for (size_t i = edge_count; i < input_info.size(); ++i) {
        import_input_info.push_back(input_info[i]);
      }

      AttrValue feature_name_value;
      SetAttrValue(
          input_info[import_input_resource_slot].src_node->name(), &feature_name_value);
      AttrValue feature_name_to_id_value;
      Status s_feature_to_id = GetFeature2IdAttr(
          input_info[import_input_resource_slot].src_node->name(),
          &feature_name_to_id_value);
      if (!s_feature_to_id.ok()) return s_feature_to_id;

      // get resource shape attr
      int dim_len_value = 0;
      Status s_get_dim = GetShapeValue(
          input_info[import_input_resource_slot].src_node, &dim_len_value);
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

Status SavedModelOptimizer::AddIncrCkptFlag() {
  // Add a placeholder for incr_ckpt flag
  return AddPlaceholder(&graph_, GetIncrCkptNodeName(),
                        DT_BOOL, &incr_ckpt_node_);
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

namespace {
void GetResourceNodeFromRestoreOpInputs(
    const Node* restore_op,
    std::vector<SrcInfo>& kv_nodes,
    std::vector<SrcInfo>& dense_nodes) {
  for (const Edge* edge : restore_op->in_edges()) {
    const Node* src = edge->src();
    if (src->type_string() != "NoOp") {
      LOG(FATAL) << "Restore op " << restore_op->name()
                 << " has input node which type is not NoOp."
                 << src->DebugString();
    }

    for (const Edge* inner_edge : src->in_edges()) {
      Node* inner_src = inner_edge->src();
      if (inner_src->op_def().name() == "KvResourceImportV2") {
        kv_nodes.push_back({inner_src, inner_edge->src_output()});
      } else {
        dense_nodes.push_back({inner_src, inner_edge->src_output()});
      }
    }
  }
}
}

Status SavedModelOptimizer::CreateDenseAndSparseRestoreOp() {
  const std::string restore_op_name =
      meta_graph_def_->saver_def().restore_op_name();
  Node* restore_op = nullptr;
  for (Node* node : graph_.nodes()) {
    if (node->name() == restore_op_name) {
      restore_op = node;
      break;
    }
  }

  if (!restore_op) {
    return errors::Internal(
        "Can not find restore op: " + restore_op_name);
  }

  std::vector<SrcInfo> kv_nodes;
  std::vector<SrcInfo> dense_nodes;
  if (restore_op->in_edges().size() == 1) {
    GetResourceNodeFromRestoreOpInputs(restore_op, kv_nodes, dense_nodes);
  } else {
    for (const Edge* edge : restore_op->in_edges()) {
      const Node* sub_restore_op = edge->src();
      if (sub_restore_op->in_edges().size() != 1) {
        LOG(FATAL) << "Sub restore all op only allow 1 input: "
                   << sub_restore_op->DebugString();
      }
      GetResourceNodeFromRestoreOpInputs(sub_restore_op, kv_nodes, dense_nodes);
    }
  }

  Status s = CreateRestoreAllNode(
      restore_op_name + GetKvRestoreAllNameSuffix(),
      "NoOp", kv_nodes, &graph_);
  if (!s.ok()) return s;

  return CreateRestoreAllNode(
      restore_op_name + GetDenseRestoreAllNameSuffix(),
     "NoOp", dense_nodes, &graph_);
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

namespace {

#define SET_DEFAULT_CONST_ATTR(pattr, type)         \
do {                                                \
  Tensor val(type, TensorShape({}));                \
  if (type == DT_BOOL) {                            \
    val.scalar<bool>()() = false;                   \
  } else if (type == DT_DOUBLE) {                   \
    val.scalar<double>()() = 0.0;                   \
  } else if (type == DT_FLOAT) {                    \
    val.scalar<float>()() = 0.0;                    \
  } else if (type == DT_INT64) {                    \
    val.scalar<int64>()() = 0;                      \
  } else if (type == DT_INT32) {                    \
    val.scalar<int>()() = 0;                        \
  } else if (type == DT_STRING) {                   \
    val.scalar<string>()() = "";                    \
  } else {                                          \
    LOG(FATAL) << "Not supported FakeConstOp type"; \
  }                                                 \
  val.AsProtoTensorContent((*pattr)["value"].mutable_tensor()); \
} while(0)

Status CreateFakeConstOp(const std::string& name,
                         DataType type, Graph* graph,
                         Node** fake_const,
                         Tensor* content = nullptr) {
  NodeDef def;
  def.set_name(name);
  def.set_op("Const");
  auto* attr = def.mutable_attr();
  (*attr)["dtype"].set_type(type);
  if (content != nullptr) {
    content->AsProtoTensorContent((*attr)["value"].mutable_tensor());
  } else {
    SET_DEFAULT_CONST_ATTR(attr, type);
  }
  Status status;
  *fake_const = graph->AddNode(def, &status);
  return status;
}

Status CreateStringJoinOp(const std::string& name,
                          std::vector<SrcInfo> input_info,
                          Graph* graph, Node** join_node) {
  NodeDef def;
  def.set_name(name);
  def.set_op("StringJoin");
  AttrValue attr_N;
  attr_N.set_i(input_info.size());
  def.mutable_attr()->insert({"N", attr_N});
  Status status;
  *join_node = graph->AddNode(def, &status);
  TF_RETURN_IF_ERROR(status);

  for (size_t i = 0; i < input_info.size(); ++i) {
    graph->AddEdge(input_info[i].src_node, input_info[i].src_slot,
                   *join_node, i);
  }

  return status;
}

Status CreateConcatOp(const std::string& name,
                      DataType type,
                      std::vector<SrcInfo> input_info,
                      Graph* graph, Node** concat_op) {
  NodeDef def;
  def.set_name(name);
  def.set_op("Concat");
  AttrValue attr_T;
  attr_T.set_type(type);
  def.mutable_attr()->insert({"T", attr_T});
  AttrValue attr_N;
  attr_N.set_i(input_info.size() - 1);
  def.mutable_attr()->insert({"N", attr_N});
  Status status;
  *concat_op = graph->AddNode(def, &status);
  TF_RETURN_IF_ERROR(status);

  for (size_t i = 0; i < input_info.size(); ++i) {
    graph->AddEdge(input_info[i].src_node, input_info[i].src_slot,
                   *concat_op, i);
  }

  return status;
}

}

Status SavedModelOptimizer::CreateIncrRestoreOp(
    Node* import_op, Node** restore_op) {
  NodeDef incr_restore_def;
  incr_restore_def.set_name(import_op->name() + "_IncrRestore");
  incr_restore_def.set_op("IncrRestore");
  DataTypeVector dtypes;
  // Tkeys, dtype, int64 (for version)
  AttrValue* key_attr = nullptr;
  Status s_attr = GetNodeAttr(import_op, "Tkeys", &key_attr);
  TF_RETURN_IF_ERROR(s_attr);
  AttrValue* dtype_attr = nullptr;
  s_attr = GetNodeAttr(import_op, "dtype", &dtype_attr);
  TF_RETURN_IF_ERROR(s_attr);
  dtypes.push_back(key_attr->type());
  dtypes.push_back(dtype_attr->type());
  dtypes.push_back(tensorflow::DataType::DT_INT64);

  AttrValue attr_value;
  SetAttrValue(dtypes, &attr_value);
  (*incr_restore_def.mutable_attr())["dtypes"] = attr_value;
  Status status;
  *restore_op = graph_.AddNode(incr_restore_def, &status);
  TF_RETURN_IF_ERROR(status);

  std::vector<SrcInfo> src_info;
  Status s_src_info = GetInputNodesInfo(&src_info, import_op);
  TF_RETURN_IF_ERROR(s_src_info);

  // Add input egdes
  std::vector<SrcInfo> input_info;
  // input: prefix
  input_info.push_back({src_info[import_input_prefix_slot].src_node,
                        src_info[import_input_prefix_slot].src_slot});

  // input: tensor_names
  // concat key/value/version strings to one tensor
  Node* kvv_concat_dim_op = nullptr;
  Tensor kvv_concat_dim_val(DT_INT32, TensorShape({}));
  kvv_concat_dim_val.scalar<int>()() = 0;
  TF_RETURN_IF_ERROR(
      CreateFakeConstOp(import_op->name() + "_IncrRestore/kvv_concat_dim",
                        DataType::DT_INT32, &graph_, &kvv_concat_dim_op,
                        &kvv_concat_dim_val));
  std::vector<SrcInfo> kvv_concat_inputs;
  kvv_concat_inputs.push_back({kvv_concat_dim_op, 0});
 
  std::string tensor_kind_names[3] = {"keys", "values", "versions"};
  for (int i = 0; i < 3; ++i) {
    Tensor suffix(DT_STRING, TensorShape({1}));
    suffix.flat<string>()(0) = "-" + tensor_kind_names[i];

    Node* suffix_node = nullptr;
    TF_RETURN_IF_ERROR(
        CreateFakeConstOp(import_op->name() + "_IncrRestore/" +
                              tensor_kind_names[i] + "_suffix",
                          DataType::DT_STRING, &graph_, &suffix_node, &suffix));

    std::vector<SrcInfo> join_inputs;
    // tensor name
    join_inputs.push_back({src_info[import_input_tname_slot].src_node,
                           src_info[import_input_tname_slot].src_slot});
    join_inputs.push_back({suffix_node, 0});
    Node* join_node = nullptr;
    TF_RETURN_IF_ERROR(
        CreateStringJoinOp(import_op->name() + "_IncrRestore/" +
                               tensor_kind_names[i] + "_join",
                           join_inputs, &graph_, &join_node));

    kvv_concat_inputs.push_back({join_node, 0});
  }

  Node* kvv_concat_op = nullptr;
  TF_RETURN_IF_ERROR(
      CreateConcatOp(import_op->name() + "_IncrRestore/kvv_concat",
                     DataType::DT_STRING, kvv_concat_inputs, &graph_,
                     &kvv_concat_op));

  // tensor_names input, shape=[3]
  input_info.push_back({kvv_concat_op, 0});

  // fake input: shape_and_slices
  Node* shape_slices_op = nullptr;
  TF_RETURN_IF_ERROR(
      CreateFakeConstOp(import_op->name() + "_IncrRestore/shape_and_slices",
                        DataType::DT_STRING, &graph_, &shape_slices_op));
  // input: is_sparse = true
  Node* is_sparse_op = nullptr;
  Tensor sparse_val(DT_BOOL, TensorShape({}));
  sparse_val.scalar<bool>()() = true;
  TF_RETURN_IF_ERROR(
      CreateFakeConstOp(import_op->name() + "_IncrRestore/is_sparse",
                        DataType::DT_BOOL, &graph_, &is_sparse_op, &sparse_val));

  input_info.push_back({shape_slices_op, 0});
  input_info.push_back({is_sparse_op, 0});
 
  // input: in_tensors
  Node* key_tensor_op = nullptr;
  TF_RETURN_IF_ERROR(
      CreateFakeConstOp(import_op->name() + "_IncrRestore/key_tensor",
                        dtypes[0], &graph_, &key_tensor_op));
  Node* val_tensor_op = nullptr;
  TF_RETURN_IF_ERROR(
      CreateFakeConstOp(import_op->name() + "_IncrRestore/val_tensor",
                        dtypes[1], &graph_, &val_tensor_op));
  Node* version_tensor_op = nullptr;
  TF_RETURN_IF_ERROR(
      CreateFakeConstOp(import_op->name() + "_IncrRestore/version_tensor",
                        dtypes[2], &graph_, &version_tensor_op));

  input_info.push_back({key_tensor_op, 0});
  input_info.push_back({val_tensor_op, 0});
  input_info.push_back({version_tensor_op, 0});

  for (size_t i = 0; i < input_info.size(); ++i) {
    graph_.AddEdge(input_info[i].src_node,
                   input_info[i].src_slot,
                   *restore_op, i);
  }

  return Status::OK();
}

Status SavedModelOptimizer::ConvertKvImportToKvInsert(
    Node* import_op, Node** insert_op) {
  Node* incr_restore_op = nullptr;
  // ev
  Status s = CreateIncrRestoreOp(import_op, &incr_restore_op);
  TF_RETURN_IF_ERROR(s);

  NodeDef kv_insert_def;
  kv_insert_def.set_name(import_op->name() + "_KvResourceInsert");
  kv_insert_def.set_op("KvResourceInsert");

  AttrValue* dtype_value =
      const_cast<AttrValue*>(import_op->attrs().Find("dtype"));
  AttrValue* tkeys_value =
      const_cast<AttrValue*>(import_op->attrs().Find("Tkeys"));
  if (!dtype_value || !tkeys_value) {
    LOG(FATAL) << "Miss dtype or Tkeys attr, "
               << import_op->DebugString();
  }
  (*kv_insert_def.mutable_attr())["dtype"] = *dtype_value;
  (*kv_insert_def.mutable_attr())["Tkeys"] = *tkeys_value;

  *insert_op = graph_.AddNode(kv_insert_def, &s);
  TF_RETURN_IF_ERROR(s);

  std::vector<SrcInfo> input_info;
  Status s_input_info = GetInputNodesInfo(&input_info, import_op);
  TF_RETURN_IF_ERROR(s_input_info);

  // input_edge: resource_handle
  graph_.AddEdge(input_info[1].src_node, input_info[1].src_slot,
                 *insert_op, 0);
  // input_edge: keys
  graph_.AddEdge(incr_restore_op, 0, *insert_op, 1);
  // input_edge: values
  graph_.AddEdge(incr_restore_op, 1, *insert_op, 2);
  // input_edge: versions
  graph_.AddEdge(incr_restore_op, 2, *insert_op, 3);

  return Status::OK();
}

Node* SavedModelOptimizer::FindRestoreShardNode() {
  const std::string restore_op_name =
      meta_graph_def_->saver_def().restore_op_name();
  Node* restore_op = nullptr;
  for (Node* node : graph_.nodes()) {
    if (node->name() == restore_op_name) {
      restore_op = node;
      break;
    }
  }

  if (!restore_op) {
    LOG(FATAL) << "Can not find restore op: "
               << restore_op_name;
  }

  if (restore_op->in_edges().size() != 1) {
    LOG(FATAL) << "Restore all op only allow 1 input: "
               << restore_op->DebugString();
  }

  Node* restore_shard = nullptr;
  for (const Edge* edge : restore_op->in_edges()) {
    restore_shard = edge->src();
    if (restore_shard->type_string() != "NoOp") {
      LOG(FATAL) << "Restore op " << restore_op_name
                 << " has input node which type is not NoOp."
                 << restore_shard->DebugString();
    }
  }

  return restore_shard; 
}

Status SavedModelOptimizer::GetIncrRestoreOpInputs(
    const Node* restore_op,
    std::vector<SrcInfo>& input_nodes) {
  for (const Edge* edge : restore_op->in_edges()) {
    const Node* src = edge->src();
    if (src->type_string() != "NoOp") {
      LOG(FATAL) << "Restore op " << restore_op->name()
                 << " has input node which type is not NoOp."
                 << src->DebugString();
    }    

    for (const Edge* inner_edge : src->in_edges()) {
      Node* inner_src = inner_edge->src();
      // KvResourceImportV2 -> KvResourceInsert
      if (inner_src->op_def().name() == "KvResourceImportV2") {
        if (inner_edge->src_output() != -1) {
          LOG(FATAL) << "Invalid tensorflow graph, restore_shard op is: "
                     << src->DebugString() << ", KvResourceImportV2 op is: "
                     << inner_src->DebugString();
        }    

        Node* insert_op = nullptr;
        Status s = ConvertKvImportToKvInsert(inner_src, &insert_op);
        TF_RETURN_IF_ERROR(s);

        input_nodes.push_back({insert_op, inner_edge->src_output()});
      } else {
        input_nodes.push_back({inner_src, inner_edge->src_output()});
      }    
    }    
  }

  return Status::OK();
}

Status SavedModelOptimizer::AddIncrRestoreOps() {
  const std::string restore_op_name =
      meta_graph_def_->saver_def().restore_op_name();
  Node* restore_op = nullptr;
  for (Node* node : graph_.nodes()) {
    if (node->name() == restore_op_name) {
      restore_op = node;
      break;
    }
  }

  if (!restore_op) {
    return errors::Internal(
        "Can not find restore op: " + restore_op_name);
  }

  std::vector<SrcInfo> input_nodes;
  Status s;
  if (restore_op->in_edges().size() == 1) {
    s = GetIncrRestoreOpInputs(restore_op, input_nodes);
  } else {
    for (const Edge* edge : restore_op->in_edges()) {
      const Node* sub_restore_op = edge->src();
      if (sub_restore_op->in_edges().size() != 1) {
        LOG(FATAL) << "Sub restore all op only allow 1 input: "
                   << sub_restore_op->DebugString();
      }
      s = GetIncrRestoreOpInputs(sub_restore_op, input_nodes);
    }
  }

  TF_RETURN_IF_ERROR(s);

  return CreateRestoreAllNode(
      meta_graph_def_->saver_def().restore_op_name() +
          GetKvIncrRestoreAllNameSuffix(),
      "NoOp", input_nodes, &graph_);
}

Node* SavedModelOptimizer::UpdateRestoreShardNodeInputs(
    std::unordered_map<std::string, std::vector<Node*>>& origin_import_nodes,
    std::vector<Node*>& new_kv_import_nodes) {
  std::unordered_set<Node*> m;
  for (auto node_map : origin_import_nodes) {
    for (Node* n : node_map.second) { m.insert(n); }
  }

  Node* shard_node = FindRestoreShardNode();

  // remove odl import node, origin edges
  std::vector<Node*> delete_candidate;
  for (const Edge* edge : shard_node->in_edges()) {
    if (m.find(edge->src()) != m.end()) {
      delete_candidate.push_back(edge->src());
    }
  }
  for (Node* n : delete_candidate) {
    graph_.RemoveNode(n);
  }

  // add new edges
  for (Node* n : new_kv_import_nodes) {
    graph_.AddEdge(n, Graph::kControlSlot,
                   shard_node, Graph::kControlSlot);
  }

  return shard_node;
}

} // namespace processor
} // namespace tensorflow

