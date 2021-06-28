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

#ifndef TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_ELSE_SCALAR_IN_GRAD_H_
#define TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_ELSE_SCALAR_IN_GRAD_H_

#include "tensorflow/core/graph/template_select_base.h"

namespace tensorflow {

class TemplateSelectElseScalarInGrad: public TemplateSelectBase {
 public:
  TemplateSelectElseScalarInGrad() {
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
      .inputs = {"0", "1", "zeros_like_op"},
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
    return "select_else_scalar_in_grad";
  }

 protected:
  virtual bool rebuild_graph(Graph* g, std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs,
      Node* node_const_zero, Node* node_select_add) {
    if (inputs.size() < 2 || outputs.size() < 3) {
      LOG(WARNING) << "Input size[" << inputs.size() << "] is less then 2 or output size["
          << outputs.size() << "] is less then 3";
      return false;
    }

    add_iedge(g, node_select_add, 0, inputs[0]);
    add_iedge(g, node_select_add, 1, inputs[1]);
    g->AddEdge(node_const_zero, 0, node_select_add, 2);

    add_oedges(g, node_select_add, 0, outputs[1]);
    add_oedges(g, node_select_add, -1, outputs[2]);
    return true;
  }

};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_ELSE_SCALAR_IN_GRAD_H_
