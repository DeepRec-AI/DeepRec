#include "tensorflow/core/grappler/optimizers/dice_fusion.h"

#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/evaluation_utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"

namespace tensorflow {
namespace grappler {
namespace dicefusion {

struct DicePattern {
  int sub_1_id = -1;
  int sub_2_id = -1;
  int sigmoid_id = -1;
  int mul_1_id = 1;
  int mul_2_id = -1;
  int mul_3_id = -1;
  int mul_4_id = -1;
  int add_id = -1;
  int sub_1_input_1_id = -1;
  int sub_1_input_2_id = -1;
  int sub_2_input_id = -1;
  int mul_1_input_id = -1;
  int mul_3_input_id = -1;
};

bool IsSigmoid(const NodeDef& node) { return node.op() == "Sigmoid"; }

int GetOutputNodeIndex(const utils::MutableGraphView& graph_view,
                       int node_index, int output_index) {
  const auto* node_view = graph_view.GetNode(node_index);
  const auto& fanouts = node_view->GetRegularFanout(0);
  int size = fanouts.size();

  if (output_index < 0 || output_index >= size) {
    return -1;
  }
  return fanouts[output_index].node_index();
}

void ListOutputNodeIndex(const utils::MutableGraphView& graph_view,
                         int node_index, std::vector<int>& output) {
  const auto* node_view = graph_view.GetNode(node_index);
  const auto& fanouts = node_view->GetRegularFanout(0);
  int size = fanouts.size();

  output.clear();
  for (int i = 0; i < size; ++i) {
    output.emplace_back(fanouts[i].node_index());
  }

  return;
}

bool IsNodeViewMatch(const utils::MutableGraphView& graph_view, int node_index,
                     int num_outputs, bool (*func)(const NodeDef& node)) {
  if (node_index < 0) return false;
  const auto* node_view = graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  if (node_def == nullptr) return false;
  if (!func(*node_def)) return false;

  if (num_outputs >= 0) {
    if (node_view->NumRegularFanouts() != num_outputs) return false;
    const auto& fanouts = node_view->GetRegularFanout(0);
    if (fanouts.size() != num_outputs) return false;
  }

  return true;
}

bool IsNodeViewMatchSub_1(const utils::MutableGraphView& graph_view,
                          int node_index, DicePattern* pattern) {
  if (!IsNodeViewMatch(graph_view, node_index, 1, IsSub)) return false;
  const auto* node_view = graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  const int num_inputs = node_view->NumRegularFanins();
  if (num_inputs != 2) return false;
  pattern->sub_1_input_1_id = node_view->GetRegularFanin(0).node_index();
  pattern->sub_1_input_2_id = node_view->GetRegularFanin(1).node_index();

  pattern->mul_1_id = GetOutputNodeIndex(graph_view, node_index, 0);
  return true;
}

bool IsNodeViewMatchMul_1(const utils::MutableGraphView& graph_view,
                          int node_index, DicePattern* pattern) {
  if (!IsNodeViewMatch(graph_view, node_index, 1, IsMul)) return false;
  const auto* node_view = graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  const int num_inputs = node_view->NumRegularFanins();
  if (num_inputs != 2) return false;

  for (int j = 0; j < num_inputs; j++) {
    auto& fanin = node_view->GetRegularFanin(j);
    int index = fanin.node_index();
    if (index == pattern->sub_1_id)
      continue;
    else if (pattern->mul_1_input_id == -1)
      pattern->mul_1_input_id = index;
    else
      return false;
  }

  if (pattern->mul_1_input_id == -1) return false;
  pattern->sigmoid_id = GetOutputNodeIndex(graph_view, node_index, 0);
  return true;
}

bool IsNodeViewMatchSigmoid(const utils::MutableGraphView& graph_view,
                            int node_index, DicePattern* pattern) {
  if (!IsNodeViewMatch(graph_view, node_index, 2, IsSigmoid)) return false;
  const auto* node_view = graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  std::vector<int> output;
  ListOutputNodeIndex(graph_view, node_index, output);
  if (output.size() != 2) return false;
  for (int j = 0; j < 2; j++) {
    if (IsNodeViewMatch(graph_view, output.at(j), 1, IsMul) &&
        pattern->mul_4_id == -1) {
      pattern->mul_4_id = output.at(j);
    } else if (IsNodeViewMatch(graph_view, output.at(j), 1, IsSub) &&
               pattern->sub_2_id == -1) {
      pattern->sub_2_id = output.at(j);
    } else {
      return false;
    }
  }

  return true;
}

bool IsNodeViewMatchSub_2(const utils::MutableGraphView& graph_view,
                          int node_index, DicePattern* pattern) {
  if (!IsNodeViewMatch(graph_view, node_index, 1, IsSub)) return false;
  const auto* node_view = graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  const int num_inputs = node_view->NumRegularFanins();
  if (num_inputs != 2) return false;

  for (int j = 0; j < num_inputs; j++) {
    auto& fanin = node_view->GetRegularFanin(j);
    int index = fanin.node_index();
    if (index == pattern->sigmoid_id)
      continue;
    else if (pattern->sub_2_input_id == -1)
      pattern->sub_2_input_id = index;
    else
      return false;
  }

  if (pattern->sub_2_input_id == -1) return false;
  pattern->mul_2_id = GetOutputNodeIndex(graph_view, node_index, 0);
  return true;
}

bool IsNodeViewMatchMul_2(const utils::MutableGraphView& graph_view,
                          int node_index, DicePattern* pattern) {
  if (!IsNodeViewMatch(graph_view, node_index, 1, IsMul)) return false;
  const auto* node_view = graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  const int num_inputs = node_view->NumRegularFanins();
  if (num_inputs != 2) return false;

  for (int j = 0; j < num_inputs; j++) {
    auto& fanin = node_view->GetRegularFanin(j);
    int index = fanin.node_index();
    if (index != pattern->sub_2_id && index != pattern->sub_1_input_1_id &&
        index != pattern->sub_1_input_2_id)
      return false;
  }

  pattern->mul_3_id = GetOutputNodeIndex(graph_view, node_index, 0);
  return true;
}

bool IsNodeViewMatchMul_3(const utils::MutableGraphView& graph_view,
                          int node_index, DicePattern* pattern) {
  if (!IsNodeViewMatch(graph_view, node_index, 1, IsMul)) return false;
  const auto* node_view = graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  const int num_inputs = node_view->NumRegularFanins();
  if (num_inputs != 2) return false;

  for (int j = 0; j < num_inputs; j++) {
    auto& fanin = node_view->GetRegularFanin(j);
    int index = fanin.node_index();
    if (index == pattern->mul_2_id)
      continue;
    else if (pattern->mul_3_input_id == -1)
      pattern->mul_3_input_id = index;
    else
      return false;
  }

  if (pattern->mul_3_input_id == -1) return false;
  pattern->add_id = GetOutputNodeIndex(graph_view, node_index, 0);
  return true;
}

bool IsNodeViewMatchMul_4(const utils::MutableGraphView& graph_view,
                          int node_index, DicePattern* pattern) {
  if (!IsNodeViewMatch(graph_view, node_index, 1, IsMul)) return false;
  const auto* node_view = graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  const int num_inputs = node_view->NumRegularFanins();
  if (num_inputs != 2) return false;

  for (int j = 0; j < num_inputs; j++) {
    auto& fanin = node_view->GetRegularFanin(j);
    int index = fanin.node_index();
    if (index != pattern->sub_1_input_1_id &&
        index != pattern->sub_1_input_2_id && index != pattern->sigmoid_id)
      return false;
  }

  if (pattern->add_id != GetOutputNodeIndex(graph_view, node_index, 0))
    return false;

  return true;
}

bool FindDicePattern(const utils::MutableGraphView& graph_view, int node_index,
                     DicePattern* matched) {
  DicePattern pattern;
  std::vector<int> outs;
  auto* node_view = graph_view.GetNode(node_index);
  if (node_view->NumControllingFanins() > 0 ||
      node_view->NumControlledFanouts() > 0)
    return false;

  pattern.sub_1_id = node_index;
  // Match sub_1, get input index and mul_1 index
  if (!IsNodeViewMatchSub_1(graph_view, node_index, &pattern)) return false;
  // Match mul, get constant input index and sigmoid index
  if (!IsNodeViewMatchMul_1(graph_view, pattern.mul_1_id, &pattern))
    return false;
  // Match sigmoid, get sub_2 & mul_4 index
  if (!IsNodeViewMatchSigmoid(graph_view, pattern.sigmoid_id, &pattern))
    return false;
  // Match sub_2, get constant input index
  if (!IsNodeViewMatchSub_2(graph_view, pattern.sub_2_id, &pattern))
    return false;
  // Match mul_2, get mul_3 index and check input index
  if (!IsNodeViewMatchMul_2(graph_view, pattern.mul_2_id, &pattern))
    return false;
  // Match mul_3, get add index and input index
  if (!IsNodeViewMatchMul_3(graph_view, pattern.mul_3_id, &pattern))
    return false;
  // Match mul_4, check add and input index
  if (!IsNodeViewMatchMul_4(graph_view, pattern.mul_4_id, &pattern))
    return false;

  *matched = pattern;
  return true;
}

string get_node_by_tensor(string tensor_name) {
  auto position = tensor_name.find(":");
  if (position != string::npos)
    tensor_name.erase(tensor_name.begin() + position, tensor_name.end());
  if (tensor_name[0] == '^') tensor_name.erase(tensor_name.begin());

  return tensor_name;
}
}  // namespace dicefusion

Status DiceFusion::Optimize(Cluster* cluster, const GrapplerItem& item,
                            GraphDef* output) {
#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)
  Status status;
  *output = item.graph;
  utils::MutableGraphView graph_view(output, &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(graph_view.SortTopologically(/*ignore_cycles=*/false, {}));
  const int num_nodes = item.graph.node_size();
  // invalidated_nodes - nodes that have been changed into a fused op
  // nodes_to_delete -  nodes that were fused into a fused op and are not needed
  // anymore
  std::vector<bool> invalidated_nodes(num_nodes);
  std::vector<bool> nodes_to_delete(num_nodes);
  const GraphDef* graph = graph_view.graph();

  VLOG(3) << "Before Dice fusion rewrites: " << graph->DebugString();

  for (int i = 0; i < num_nodes; ++i) {
    if (invalidated_nodes[i] || nodes_to_delete[i]) continue;

    dicefusion::DicePattern base;
    if (dicefusion::FindDicePattern(graph_view, i, &base)) {
      const auto& fused_node = graph->node(i);
      VLOG(2) << "Optimizing fused dice node " << SummarizeNodeDef(fused_node);

      std::vector<int> dead_nodes = {
          base.sub_1_id, base.sub_2_id, base.sigmoid_id, base.mul_1_id,
          base.mul_2_id, base.mul_3_id, base.mul_4_id,   base.sub_2_input_id};
      for (auto& id : dead_nodes) nodes_to_delete[id] = true;
      invalidated_nodes[base.add_id] = true;

      // Adding fused dice op
      const NodeDef& add = graph->node(base.add_id);

      NodeDef dice_op;
      dice_op.set_name(add.name());
      dice_op.set_op("Dice");
      dice_op.set_device(add.device());

      // Get input
      std::vector<int> nodes_with_inputs = {base.mul_4_id, base.sub_1_id,
                                            base.mul_1_id, base.mul_3_id};
      std::set<std::string> names;
      for (int i = 0; i < dead_nodes.size(); ++i) {
        const auto* node_view = graph_view.GetNode(dead_nodes.at(i));
        const auto* node_def = node_view->node();
        names.insert(node_def->name());
      }
      std::set<std::string> added_input;
      for (int i = 0; i < nodes_with_inputs.size(); ++i) {
        const int id = nodes_with_inputs.at(i);
        const auto* node_view = graph_view.GetNode(id);
        const auto* node_def = node_view->node();
        const std::size_t num_inputs = node_view->NumRegularFanins();
        for (size_t j = 0; j < num_inputs; ++j) {
          std::string node_name =
              dicefusion::get_node_by_tensor(node_def->input(j));
          if (names.find(node_name) != names.end()) continue;
          if (added_input.find(node_name) != added_input.end()) continue;
          added_input.insert(node_name);
          dice_op.add_input(node_def->input(j));
        }
      }

      auto* attr = dice_op.mutable_attr();
      auto& add_attr = add.attr();

      (*attr)["T"] = add_attr.at("T");

      utils::Mutation* mutation = graph_view.GetMutationBuilder();
      Status status;
      mutation->AddNode(std::move(dice_op), &status);
      TF_RETURN_IF_ERROR(status);
      TF_RETURN_IF_ERROR(mutation->Apply());
    }
  }

  // Remove useless node
  utils::Mutation* mutation = graph_view.GetMutationBuilder();
  for (int i = 0; i < num_nodes; ++i) {
    if (nodes_to_delete[i]) {
      mutation->RemoveNode(graph_view.GetNode(i));
    }
  }
  TF_RETURN_IF_ERROR(mutation->Apply());
  *output = *graph_view.graph();

  VLOG(3) << "After Dice fusion rewrites: " << output->DebugString();
#endif // AVX512F

  return Status::OK();
}

void DiceFusion::Feedback(Cluster* cluster, const GrapplerItem& item,
                          const GraphDef& optimize_output, double result) {
  // Nothing to do for DiceFusion.
}

}  // namespace grappler
}  // namespace tensorflow