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

#ifndef TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_THEN_SCALAR_IN_GRAD_H_
#define TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_THEN_SCALAR_IN_GRAD_H_

#include "tensorflow/core/graph/template_select_base.h"

namespace tensorflow {

class TemplateSelectThenScalarInGrad: public TemplateSelectBase {
 public:
  TemplateSelectThenScalarInGrad() {
    const TempNode zeros_like = {
      .key = "zeros_like_op",
      .op = "ZerosLike",
      .inputs = {"2"},
      .outputs = {{"select_op", "0"}}
    };
    temp_nodes_.push_back(zeros_like);

    const TempNode select = {
      .key = "select_op",
      .op = "Select",
      .inputs = {"0", "zeros_like_op", "1"},
      .outputs = {{"1"}},
      .deps_inputs = {},
      .deps_outputs = {"2"}
    };
    temp_nodes_.push_back(select);

    first_key_ = "select_op";
    num_inputs_ = 3;
    num_outputs_ = 3;
  }

  const string name() {
    return "select_then_scalar_in_grad";
  }

 protected:
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
      def_select.add_input(name_prefix + "_const_zero_" + name());
      add_input(def_select, inputs[1]);
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
    if (status != Status::OK()) {
      LOG(WARNING) << "Add node failed: " << status.error_message();
      return NULL;
    }

    return node_select_add;
  }

  virtual bool rebuild_graph(Graph* g, std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs,
      Node* node_const_zero, Node* node_select_add) {
    if (inputs.size() < 2 || outputs.size() < 3) {
      LOG(WARNING) << "Input size[" << inputs.size() << "] is less then 2 or output size["
          << outputs.size() << "] is less then 3";
      return false;
    }

    add_iedge(g, node_select_add, 0, inputs[0]);
    g->AddEdge(node_const_zero, 0, node_select_add, 1);
    add_iedge(g, node_select_add, 2, inputs[1]);

    add_oedges(g, node_select_add, 0, outputs[1]);
    add_oedges(g, node_select_add, -1, outputs[2]);
    return true;
  }

};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_THEN_SCALAR_IN_GRAD_H_
