/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_BASE_H_
#define TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_BASE_H_

#include "tensorflow/core/graph/template_base.h"

namespace tensorflow {

class TemplateSelectBase: public TemplateBase {
 public:
  TemplateSelectBase() {
  }

  const string name() {
    return "select_base";
  }

  bool add_subgraph(std::map<std::string, MatchedNode>& nodes,
      std::string name_prefix, Graph* g,
      std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs) override {
    DataType datatype = get_data_type(nodes[first_key_].node);
    if (datatype != DT_FLOAT && datatype != DT_INT32 && datatype != DT_INT64) {
      LOG(INFO) << "Drop fusion template[" << name() << "] match op[" << nodes[first_key_].node->DebugString() << "]";
      return false;
    } else {
      LOG(INFO) << "Fusion template[" << name() << "] match op[" << nodes[first_key_].node->name() <<
          "][new_name:" << name_prefix << "_" << name() << "]";
    }

    Node* node_const_zero = add_zero_like_node(nodes, name_prefix, g, inputs, outputs);
    if (!node_const_zero) {
      LOG(WARNING) << "Add zero_like node failed";
      return false;
    }

    Node* node_select_add = add_select_node(nodes, name_prefix, g, inputs, outputs);
    if (!node_select_add) {
      LOG(WARNING) << "Add select node failed";
      return false;
    }

    return rebuild_graph(g, inputs, outputs, node_const_zero, node_select_add);
  }

  bool CheckDynamicInputs(
      const Node* node, const TempNode* temp_node, int dy_mode, 
      std::vector<const Edge*>& fused_op_inputs, 
      std::map<const std::string, TempNode>& temp_node_map,
      std::map<std::string, MatchedNode>& matched_node_map) override {
    return false;
  } 

  bool CheckDynamicOutputs(
      const Node* node, const TempNode* temp_node, int dy_mode, 
      std::vector<std::vector<const Edge*>>& fused_op_outputs, 
      std::map<const std::string, TempNode>& temp_node_map,
      std::map<std::string, MatchedNode>& matched_node_map) override {
    return false;
  }

  DataType get_data_type(const Node* node) {
    DataType datatype;
    if (GetNodeAttr(node->def(), "T", &datatype) != Status::OK()) {
      return DT_INVALID;
    }
    return datatype;
  }

 protected:
  virtual Node* add_zero_like_node(
      std::map<std::string, MatchedNode>& nodes,
      std::string name_prefix, Graph* g,
      std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs) {
    // construct zero const
    NodeDef const_zero;
    const_zero.set_op("Const");
    const_zero.set_name(name_prefix + "_const_zero_" + name());

    DataType datatype = get_data_type(nodes[first_key_].node);
    AttrValue attr_type;
    attr_type.set_type(datatype);
    const_zero.mutable_attr()->insert({"dtype", attr_type});

    Tensor tensor_zero(datatype, {});
    if (datatype == DT_FLOAT) {
      tensor_zero.scalar<float>()() = 0;
    } else if (datatype == DT_INT32) {
      tensor_zero.scalar<int32>()() = 0;
    } else if (datatype == DT_INT64) {
      tensor_zero.scalar<int64>()() = 0;
    }

    AttrValue value_zero;
    tensor_zero.AsProtoTensorContent(value_zero.mutable_tensor());
    const_zero.mutable_attr()->insert({"value", value_zero});

    // Add node
    Status status;
    Node* node_const_zero = g->AddNode(const_zero, &status);
    if (status != Status::OK() || !node_const_zero) {
      LOG(WARNING) << "Add const_zero node failed: " << status.error_message();
      return NULL;
    }

    return node_const_zero;
  }

  virtual Node* add_select_node(
      std::map<std::string, MatchedNode>& nodes,
      std::string name_prefix, Graph* g,
      std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs) {
    // construct select op
    NodeDef def_select;
    def_select.set_op("Select");
    def_select.set_name(name_prefix + "_" + name());

    if (inputs.size() >= 2) {
      add_input(def_select, inputs[0]);
      add_input(def_select, inputs[1]);
      def_select.add_input(name_prefix + "_const_zero");
    } else {
      LOG(WARNING) << "Input size[" << inputs.size() << "] is less then 2";
      return NULL;
    }

    auto matmul_attr = nodes["select_op"].node->def().attr();
    def_select.set_device(nodes["select_op"].node->def().device());
    def_select.mutable_attr()->insert({"T", matmul_attr.at("T")});

    // Add node
    Status status;
    Node* node_select_add = g->AddNode(def_select, &status);
    if (status != Status::OK() || !node_select_add) {
      LOG(WARNING) << "Add node failed: " << status.error_message();
      return NULL;
    }

    return node_select_add;
  }

  virtual bool rebuild_graph(Graph* g, std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs,
      Node* node_const_zero, Node* node_select_add) {
    if (inputs.size() < 2 || outputs.size() < 2) {
      LOG(WARNING) << "Input size[" << inputs.size() << "] is less then 2 or output size["
          << outputs.size() << "] is less then 2";
      return false;
    }

    add_iedge(g, node_select_add, 0, inputs[0]);
    add_iedge(g, node_select_add, 1, inputs[1]);
    g->AddEdge(node_const_zero, 0, node_select_add, 2);

    add_oedges(g, node_select_add, 0, outputs[1]);
    return true;
  }

};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_BASE_H_
