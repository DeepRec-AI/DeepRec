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

#ifndef TENSORFLOW_CORE_GRAPH_TEMPLATE_SPARSE_INNER_FLATTEN_H_
#define TENSORFLOW_CORE_GRAPH_TEMPLATE_SPARSE_INNER_FLATTEN_H_

#include "tensorflow/core/graph/template_base.h"
#include "tensorflow/core/framework/node_def_util.h"

namespace tensorflow {

namespace {
template <class T>
inline void SetNodeAttr(const string& key, const T& value, NodeDef* node) {
  AttrValue attr_value;
  SetAttrValue(value, &attr_value);
  auto* attr_map = node->mutable_attr();
  (*attr_map)[key] = attr_value;
}

template <>
inline void SetNodeAttr(const string& key, const Tensor& tensor,
    NodeDef* node) {
  TensorProto tensor_proto;
  tensor.AsProtoTensorContent(&tensor_proto);
  SetNodeAttr(key, tensor_proto, node);
}
}

class TemplateSparseInnerFlatten : public TemplateBase {
 public:
  TemplateSparseInnerFlatten() {
    const TempNode n0 = {
      .key = "const_stack",
      .op = "Const",
      .inputs = {},
      .outputs = {{"strided_slice_0"}}
    };
    temp_nodes_.emplace_back(n0);

    const TempNode n1 = {
      .key = "const_stack_1",
      .op = "Const",
      .inputs = {},
      .outputs = {{"strided_slice_0"}}
    };
    temp_nodes_.emplace_back(n1);

    const TempNode n2 = {
      .key = "const_stack_2",
      .op = "Const",
      .inputs = {},
      .outputs = {{"strided_slice_0"}}
    };
    temp_nodes_.emplace_back(n2);

    const TempNode n3 = {
      .key = "strided_slice_0",
      .op = "StridedSlice",
      .inputs = {"0", "const_stack", "const_stack_1", "const_stack_2"},
      .outputs = {{"prod"}}
    };
    temp_nodes_.emplace_back(n3);
    
    const TempNode n4 = {
      .key = "const_stack_3",
      .op = "Const",
      .inputs = {},
      .outputs = {{"strided_slice_1"}}
    };
    temp_nodes_.emplace_back(n4);

    const TempNode n5 = {
      .key = "const_stack_4",
      .op = "Const",
      .inputs = {},
      .outputs = {{"strided_slice_1"}}
    };
    temp_nodes_.emplace_back(n5);

    const TempNode n6 = {
      .key = "const_stack_5",
      .op = "Const",
      .inputs = {},
      .outputs = {{"strided_slice_1"}}
    };
    temp_nodes_.emplace_back(n6);

    const TempNode n7 = {
      .key = "strided_slice_1",
      .op = "StridedSlice",
      .inputs = {"0", "const_stack_3", "const_stack_4",
        "const_stack_5"},
      .outputs = {{"concat"}}
    };
    temp_nodes_.emplace_back(n7);

    const TempNode n8 = {
      .key = "const_prod",
      .op = "Const",
      .inputs = {},
      .outputs = {{"prod"}}
    };
    temp_nodes_.emplace_back(n8);

    const TempNode n9 = {
      .key = "prod",
      .op = "Prod",
      .inputs = {"strided_slice_0", "const_prod"},
      .outputs = {{"pack"}}
    };
    temp_nodes_.emplace_back(n9);

    const TempNode n10 = {
      .key = "pack",
      .op = "Pack",
      .inputs = {"prod"},
      .outputs = {{"concat"}}
    };
    temp_nodes_.emplace_back(n10);

    const TempNode n11 = {
      .key = "const_axis",
      .op = "Const",
      .inputs = {},
      .outputs = {{"concat"}}
    };
    temp_nodes_.emplace_back(n11);

    const TempNode n12 = {
      .key = "concat",
      .op = "ConcatV2",
      .inputs = {"strided_slice_1", "pack", "const_axis"},
      .outputs = {{"sparse_reshape"}}
    };
    temp_nodes_.emplace_back(n12);

    const TempNode n13 = {
      .key = "sparse_reshape",
      .op = "SparseReshape",
      .inputs = {"1", "0", "concat"},
      .outputs = {{"0"}, {"1"}}
    };
    temp_nodes_.emplace_back(n13);

    first_key_ = "const_stack";
    num_inputs_ = 2;
    num_outputs_ = 2;
  }

  Node* NewRank(const std::string& name_prefix,
      const std::map<std::string, MatchedNode>& nodes, Graph* g) {
    auto stack = nodes.find("const_stack");
    Tensor tstack;
    GetNodeAttr(stack->second.node->attrs(), "value", &tstack);
    auto stack_val = tstack.flat<int32>()(0);
    
    Tensor rank(DT_INT64, {});
    rank.scalar<int64>()() = stack_val + 1;
    
    NodeDef const_node;
    const_node.set_op("Const");
    const_node.set_name(name_prefix + "new_rank");
    SetNodeAttr("value", rank, &const_node);
    SetNodeAttr("dtype", DT_INT64, &const_node);

    Status status;
    auto ret = g->AddNode(const_node, &status);
    if (!status.ok()) {
      VLOG(1) << status.error_message();
    }
    return ret;
  }

  bool add_subgraph(std::map<std::string, MatchedNode>& nodes,
      std::string name_prefix, Graph* g,
      std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs) override {
    NodeDef fused_def;
    fused_def.set_op("SparseInnerFlatten");
    fused_def.set_name(name_prefix + "_sparse_inner_flatten");
    auto origin_device = inputs[0]->dst()->def().device();
    fused_def.set_device(origin_device);

    add_input(fused_def, inputs[0]);
    add_input(fused_def, inputs[1]);
    fused_def.add_input("new_rank:0");
    
    Status status;
    Node* fused_node = g->AddNode(fused_def, &status);
    if (status != Status::OK()) {
      VLOG(1) << status.error_message();
      return false;
    }

    Node* new_rank = NewRank(name_prefix, nodes, g);
    if (new_rank == nullptr) {
      return false;
    }

    add_iedge(g, fused_node, 0, inputs[1]);
    add_iedge(g, fused_node, 1, inputs[0]);
    g->AddEdge(new_rank, 0, fused_node, 2); 

    add_oedges(g, fused_node, 0, outputs[0]);
    add_oedges(g, fused_node, 1, outputs[1]);
    return true;
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
};

// Sub Graph After constant folding and common expression elimation
// TODO: use this sub graph pattern
/*class TemplateSparseInnerFlattenV2 : public TemplateBase {
 public:
  TemplateSparseInnerFlattenV2() {
    const TempNode n0 = {
      .key = "const_stack",
      .op = "Const",
      .inputs = {},
      .outputs = {{"strided_slice_0", "strided_slice_1", "prod"}}
    };
    temp_nodes_.emplace_back(n0);

    const TempNode n1 = {
      .key = "const_stack_1",
      .op = "Const",
      .inputs = {},
      .outputs = {{"strided_slice_0", "strided_slice_0", "strided_slice_1", "strided_slice_1"}}
    };
    temp_nodes_.emplace_back(n1);

    const TempNode n3 = {
      .key = "strided_slice_0",
      .op = "StridedSlice",
      .inputs = {"0", "const_stack_1", "const_stack", "const_stack_1"},
      .outputs = {{"prod"}}
    };
    temp_nodes_.emplace_back(n3);
    
    const TempNode n7 = {
      .key = "strided_slice_1",
      .op = "StridedSlice",
      .inputs = {"0", "const_stack", "const_stack_1",
        "const_stack_1"},
      .outputs = {{"concat"}}
    };
    temp_nodes_.emplace_back(n7);

    const TempNode n9 = {
      .key = "prod",
      .op = "Prod",
      .inputs = {"strided_slice_0", "const_stack"},
      .outputs = {{"pack"}}
    };
    temp_nodes_.emplace_back(n9);

    const TempNode n10 = {
      .key = "pack",
      .op = "Pack",
      .inputs = {"prod"},
      .outputs = {{"concat"}}
    };
    temp_nodes_.emplace_back(n10);

    const TempNode n11 = {
      .key = "const_axis",
      .op = "Const",
      .inputs = {},
      .outputs = {{"concat"}}
    };
    temp_nodes_.emplace_back(n11);

    const TempNode n12 = {
      .key = "concat",
      .op = "ConcatV2",
      .inputs = {"strided_slice_1", "pack", "const_axis"},
      .outputs = {{"sparse_reshape"}}
    };
    temp_nodes_.emplace_back(n12);

    const TempNode n13 = {
      .key = "sparse_reshape",
      .op = "SparseReshape",
      .inputs = {"1", "0", "concat"},
      .outputs = {{"0"}, {"1", "2", "3", "4"}}
    };
    temp_nodes_.emplace_back(n13);

    first_key_ = "const_stack";
    num_inputs_ = 2;
    num_outputs_ = 2;
  }

  bool add_subgraph(std::map<std::string, MatchedNode>& nodes,
      std::string name_prefix, Graph* g,
      std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs) override {
    NodeDef fused_def;
    fused_def.set_op("SparseInnerFlatten2D");
    fused_def.set_name(name_prefix + "_sparse_inner_flatten");
    add_input(fused_def, inputs[0]);
    add_input(fused_def, inputs[1]);

    Status status;
    Node* fused_node = g->AddNode(fused_def, &status);
    if (status != Status::OK()) {
      VLOG(1) << status.error_message();
      return false;
    }

    add_iedge(g, fused_node, 0, inputs[1]);
    add_iedge(g, fused_node, 1, inputs[0]);

    add_oedges(g, fused_node, 0, outputs[0]);
    add_oedges(g, fused_node, 1, outputs[1]);
    return true;
  }
};*/

}  // namespace tensorflow
#endif // TENSORFLOW_CORE_GRAPH_TEMPLATE_SPARSE_INNER_FLATTEN_H_
