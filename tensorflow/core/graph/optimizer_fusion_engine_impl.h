#ifndef TENSORFLOW_CORE_GRAPH_OPTIMIZER_FUSION_ENGINE_IMPL_H_
#define TENSORFLOW_CORE_GRAPH_OPTIMIZER_FUSION_ENGINE_IMPL_H_

#include <map>
#include <vector>
#include "tensorflow/core/graph/template_base.h"

namespace tensorflow {
class Edge;
class Graph;
class Node;
class OptimizerFusionImpl {
 public:
  explicit OptimizerFusionImpl(Graph* g, TemplateBase* t);
  bool Optimize();

private:
  bool VisitMatchedNodes();
  bool CheckOutputs(const Node* node,
                    const TempNode* temp_node);
  bool CheckInputs(const Node* node,
                   const TempNode* temp_node);
  bool CheckMatchedNodeInSameFrame();

private:
  Graph* g_;
  TemplateBase* t_;
  std::map<const std::string, TempNode> temp_node_map_;
  std::vector<const Edge*> fused_op_inputs_;
  std::vector<const Edge*> fused_op_deps_inputs_;
  std::vector<std::vector<const Edge*>> fused_op_outputs_;
  std::map<std::string, MatchedNode> matched_node_map_;
  int num_matched_;
  // for dynamic outputs of templates
  bool use_dynamic_output_keys_;
  bool use_dynamic_input_keys_;
  int dynamic_output_port_cur_;
  int dynamic_input_port_cur_;
  std::vector<std::vector<const Edge*>> fused_op_outputs_dynamic_;
  std::vector<const Edge*> fused_op_input_dynamic_;
  std::map<const Node *, std::string> node_frame_map_;
};

}

#endif // TENSORFLOW_CORE_GRAPH_OPTIMIZER_FUSION_ENGINE_IMPL_H_
