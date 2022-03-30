// @file template_select_pruning.h
// @author jiancai.ljc(jiancai.ljc@alibaba-inc.com)
// @date 2021/03/26 16:36:31
// @brief 

#ifndef TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_PRUNING_BASE_H_
#define TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_PRUNING_BASE_H_

#include "tensorflow/core/graph/template_base.h"

namespace tensorflow {
class TemplateSelectPruningBase : public TemplateBase {
 public:
  TemplateSelectPruningBase() {
  }

  void declare_graph() {
    const TempNode zeros_like_for_select = {
      .key = "zeros_like_for_select",
      .op = "ZerosLike",
      .inputs = {"0"},
      .outputs = {{"select_0", "select_1"}}
    };
    temp_nodes_.push_back(zeros_like_for_select);

    const TempNode select_0 = {
      .key = "select_0",
      .op = "Select",
      .inputs = {"1", "2", "zeros_like_for_select"},
      .outputs = {{"control_dependency_0"}},
      .deps_inputs = {},
      .deps_outputs = {"group_dependency"}
    };
    temp_nodes_.push_back(select_0);

    const TempNode select_1 = {
      .key = "select_1",
      .op = "Select",
      .inputs = {"1", "zeros_like_for_select", "2"},
      .outputs = {{"control_dependency_1"}},
      .deps_inputs = {},
      .deps_outputs = {"group_dependency"}
    };
    temp_nodes_.push_back(select_1);

    const TempNode group_dependency = {
      .key = "group_dependency",
      .op = "NoOp",
      .inputs = {},
      .outputs = {{}},
      .deps_inputs = {"select_0", "select_1"},
      .deps_outputs = {"control_dependency_0", "control_dependency_1"}
    };
    temp_nodes_.push_back(group_dependency);
  }

  const string name() {
    return "select_pruning_base";
  }

  virtual const std::vector<std::string> node_to_remove() {
    std::vector<std::string> to_del_node;
    return to_del_node;
  }

  bool add_subgraph(std::map<std::string, MatchedNode>& nodes,
                    std::string name_prefix, Graph* g,
                    std::vector<const Edge*>& inputs,
                    std::vector<std::vector<const Edge*>>& outputs) override {
    LOG(INFO) << "Found match op by " << name() << " " << nodes[first_key_].node->name();

    const std::vector<std::string> to_del_node = node_to_remove();
    for (auto name : to_del_node) {
      const Node* select_node = nodes[name].node;
      for (Node* node : g->nodes()) {
        if (node->name() == select_node->name()) {
          LOG(INFO) << "remove node: " << node->name();
          g->RemoveNode(node);
          break;
        }
      }
    }

    return true;
  }

  bool CheckDynamicInputs(
      const Node* node, const TempNode* temp_node, int dy_mode, 
      std::vector<const Edge*>& fused_op_inputs, 
      std::map<const std::string, TempNode>& temp_node_map,
      std::map<std::string, MatchedNode>& matched_node_map) override {
    return true;
  }

  bool CheckDynamicOutputs(
      const Node* node, const TempNode* temp_node, int dy_mode, 
      std::vector<std::vector<const Edge*>>& fused_op_outputs, 
      std::map<const std::string, TempNode>& temp_node_map,
      std::map<std::string, MatchedNode>& matched_node_map) override {
    return false;
  }
};

}  // namespace tensorflow

#endif // TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_PRUNING_BASE_H_
