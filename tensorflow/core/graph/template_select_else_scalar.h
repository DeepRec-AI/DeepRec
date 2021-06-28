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

#ifndef TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_ELSE_SCALAR_H_
#define TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_ELSE_SCALAR_H_

#include "tensorflow/core/graph/template_select_base.h"

namespace tensorflow {

class TemplateSelectElseScalar: public TemplateSelectBase {
 public:
  TemplateSelectElseScalar() {
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
      .outputs = {{"1"}}
    };
    temp_nodes_.push_back(select);

    first_key_ = "select_op";
    num_inputs_ = 3;
    num_outputs_ = 2;
  }

  const string name() {
    return "select_else_scalar";
  }

};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_ELSE_SCALAR_H_
