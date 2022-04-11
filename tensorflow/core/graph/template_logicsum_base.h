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

#ifndef TENSORFLOW_CORE_GRAPH_TEMPLATE_LOGICSUM_BASE_H_
#define TENSORFLOW_CORE_GRAPH_TEMPLATE_LOGICSUM_BASE_H_

#include "tensorflow/core/graph/template_base.h"

namespace tensorflow {

class TemplateLogicSumBase: public TemplateBase {
 public:
  TemplateLogicSumBase() {
    const TempNode n0 = {
      .key = "greater_0",
      .op = "Greater",
      .inputs = {"0","1"},
      .outputs = {{"logic_or_0","logic_and_0"}}
    };
    temp_nodes_.emplace_back(n0);

    const TempNode n1 = {
      .key = "greater_1",
      .op = "Greater",
      .inputs = {"2","3"},
      .outputs = {{"logic_or_0","logic_and_0"}}
    };
    temp_nodes_.emplace_back(n1);

    const TempNode n2 = {
      .key = "logic_or_0",
      .op = "LogicalOr",
      .inputs = {"greater_0","greater_1"},
      .outputs = {{"logic_xor_0"}}
    };
    temp_nodes_.emplace_back(n2);

    const TempNode n3 = {
      .key = "logic_and_0",
      .op = "LogicalAnd",
      .inputs = {"greater_0","greater_1"},
      .outputs = {{"logic_not_0"}}
    };
    temp_nodes_.emplace_back(n3);

    const TempNode n4 = {
      .key = "logic_not_0",
      .op = "LogicalNot",
      .inputs = {"logic_and_0"},
      .outputs = {{"logic_xor_0"}}
    };
    temp_nodes_.emplace_back(n4);

    const TempNode n5 = {
      .key = "logic_xor_0",
      .op = "LogicalAnd",
      .inputs = {"logic_or_0","logic_not_0"},
      .outputs = {{"cast_0"}}
    };
    temp_nodes_.emplace_back(n5);

    const TempNode n6 = {
      .key = "cast_0",
      .op = "Cast",
      .inputs = {"logic_xor_0"},
      .outputs = {{"sum_0"}}
    };
    temp_nodes_.emplace_back(n6);

    const TempNode n7 = {
      .key = "sum_0",
      .op = "Sum",
      .inputs = {"cast_0","4"},
      .outputs = {{"neg_0"}}
    };
    temp_nodes_.emplace_back(n7);

    const TempNode n8 = {
      .key = "neg_0",
      .op = "Neg",
      .inputs = {"sum_0"},
      .outputs = {{"0"}}
    };
    temp_nodes_.emplace_back(n8);

    first_key_   = "greater_0";
    num_inputs_  = 5;
    num_outputs_ = 1;

  }

  const string name() {
    return "logicsum_base";
  }

  bool add_subgraph(std::map<std::string, MatchedNode>& nodes,
    std::string name_prefix, Graph* g,
    std::vector<const Edge*>& inputs,
    std::vector<std::vector<const Edge*>>& outputs) override {
    if (!CheckInputs(inputs)) {
      LOG(WARNING) << "Input check failed";
      return false;
    }
    LOG(INFO) << "Fusion template[" << name() << "] match op[" << nodes[first_key_].node->name() <<
          "][new_name:" << name_prefix << "_" << name() << "]";

    Node* node_fused_logicsum = add_fused_logicsum_node(nodes, name_prefix, g, inputs, outputs);
    if (!node_fused_logicsum) {
      LOG(WARNING) << "Add node_fused_logicsum node failed";
      return false;
    }

    return rebuild_graph(g, inputs, outputs, node_fused_logicsum);
  }

  bool CheckConstZeroNode(const NodeDef& node_def) {
    Tensor val;
    Status s = GetNodeAttr(node_def, "value", &val);
    if (val.dtype() == DT_FLOAT) {
      auto v = val.flat<float>();
      for (int i = 0; i < v.size(); i++) {
        if (fabs(v(i)) > 1e-6) {
          return false;
        }
      }
      return true;
    }

    return false;
  }

  bool CheckInputs(std::vector<const Edge*>& inputs) {
    if (inputs.size() > 4) {
      return CheckConstZeroNode(inputs[1]->src()->def()) && CheckConstZeroNode(inputs[3]->src()->def());
    } else {
      return false;
    }
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

 protected:
  virtual Node* add_fused_logicsum_node(
      std::map<std::string, MatchedNode>& nodes,
      std::string name_prefix, Graph* g,
      std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs) {
    // construct fused_logicsum node
    NodeDef fused_logicsum_node;
    add_input(fused_logicsum_node, inputs[0]);
    add_input(fused_logicsum_node, inputs[2]);
    fused_logicsum_node.set_op("LogicalSum");
    fused_logicsum_node.set_name(name_prefix + name());
    fused_logicsum_node.set_device(nodes["greater_0"].node->def().device());
    AttrValue dtype_attr;
    dtype_attr.set_type(DT_FLOAT);
    fused_logicsum_node.mutable_attr()->insert({"T", dtype_attr});
    // Add node
    Status status;
    Node* node_fused_logicsum_node = g->AddNode(fused_logicsum_node, &status);
    if (status != Status::OK() || !node_fused_logicsum_node) {
      LOG(WARNING) << "Add fused_logicsum node failed: " << status.error_message();
      return NULL;
    }

    return node_fused_logicsum_node;
  }

  virtual bool rebuild_graph(Graph* g, std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs,
      Node* node_fused_logicsum_node) {
    if (inputs.size() < 5 || outputs.size() > 1) {
      LOG(WARNING) << "Input size[" << inputs.size() << "] is less then 5 or output size["
          << outputs.size() << "] is more then 1";
      return false;
    }

    add_iedge(g, node_fused_logicsum_node, 0, inputs[0]);
    add_iedge(g, node_fused_logicsum_node, 1, inputs[2]);
    add_oedges(g, node_fused_logicsum_node, 0, outputs[0]);
    return true;
  }

};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPH_TEMPLATE_LOGICSUM_BASE_H_
