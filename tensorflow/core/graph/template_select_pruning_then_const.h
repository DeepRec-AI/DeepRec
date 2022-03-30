// @file template_select_pruning.h
// @author jiancai.ljc(jiancai.ljc@alibaba-inc.com)
// @date 2021/03/26 16:36:31
// @brief 

#ifndef TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_PRUNING_THEN_CONST_H_
#define TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_PRUNING_THEN_CONST_H_

#include "tensorflow/core/graph/template_select_pruning_base.h"

namespace tensorflow {
class TemplateSelectPruningThenConst : public TemplateSelectPruningBase {
 public:
  TemplateSelectPruningThenConst() {
    declare_graph();

    const TempNode control_dependency_0 = {
      .key = "control_dependency_0",
      .op = "Identity",
      .inputs = {"select_0"},
      .outputs = {{}},
      .deps_inputs = {"group_dependency"},
      .deps_outputs = {"1"}
    };
    temp_nodes_.push_back(control_dependency_0);

    const TempNode control_dependency_1 = {
      .key = "control_dependency_1",
      .op = "Identity",
      .inputs = {"select_1"},
      .outputs = {{"0"}},
      .deps_inputs = {"group_dependency"},
      .deps_outputs = {}
    };
    temp_nodes_.push_back(control_dependency_1);

    first_key_ = "zeros_like_for_select";
    num_inputs_ = 3;
    num_outputs_ = 2;
  }

  const string name() {
    return "select_pruning_then_const";
  }

  const std::vector<std::string> node_to_remove() {
    std::vector<std::string> to_del_node;
    to_del_node.push_back("select_0");
    to_del_node.push_back("control_dependency_0");
    return to_del_node;
  }

};

}  // namespace tensorflow

#endif // TENSORFLOW_CORE_GRAPH_TEMPLATE_SELECT_PRUNING_THEN_CONST_H_
