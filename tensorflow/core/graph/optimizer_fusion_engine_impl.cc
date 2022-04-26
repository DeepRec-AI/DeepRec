#include <algorithm>
#include <tuple>
#include <queue>
#include "tensorflow/core/graph/optimizer_fusion_engine_impl.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/graph.pb.h"

namespace tensorflow {

inline void sortInEdges(const EdgeSet& in, 
    std::vector<std::tuple<int, const Edge*> >& out) {
  for(auto* e : in) {
    out.emplace_back(std::make_tuple(e->dst_input(), e));
  }
  std::sort(out.begin(), out.end());
}

OptimizerFusionImpl::OptimizerFusionImpl(Graph* g, TemplateBase* t)
    : g_(g), t_(t), num_matched_(0) {
  for (auto node : t_->temp_nodes_) {
    temp_node_map_.emplace(node.key, node);
  }
  fused_op_inputs_.resize(t_->num_inputs_);
  fused_op_outputs_.resize(t_->num_outputs_);
  use_dynamic_output_keys_ = false;
  use_dynamic_input_keys_ = false;

  std::unordered_set<Node *> enter_nodes;
  for (Node *node : g->nodes()) {
    node_frame_map_[node] = "";
    if (node->IsEnter()) {
      enter_nodes.insert(node);
    }
  }

  std::unordered_set<Node *> has_visited;
  for (Node *node : enter_nodes) {
    const std::string frame_name = node->def().attr().at("frame_name").s();
    std::queue<Node *> q;
    q.push(node);
    while (!q.empty()) {
      Node *n = q.front();
      q.pop();
      has_visited.insert(n);
      node_frame_map_[n] = frame_name;
      for (auto e : n->out_edges()) {
        Node *dst = e->dst();
        if (has_visited.find(dst) == has_visited.end() && 
            (!dst->IsExit() || !dst->IsNextIteration())) {
          q.push(dst);
        }
      }
    }
  }
}

bool OptimizerFusionImpl::VisitMatchedNodes() {
  bool all_visited = false;
  while (!all_visited) {
    all_visited = true;
    for (auto iter = matched_node_map_.begin();
         iter != matched_node_map_.end(); ++iter) {
      if (iter->second.visited) {
        continue;
      }
      all_visited = false;
      // check dynamic inputs
      auto search_itr = 
        t_->nodes_dynamic_iedges_.find(temp_node_map_[iter->first].key);
      if (search_itr != t_->nodes_dynamic_iedges_.end()) {
        // inputs of this node is marked as dynamic 
        if (!t_->CheckDynamicInputs(iter->second.node, 
              &temp_node_map_[iter->first], search_itr->second, 
              fused_op_inputs_, temp_node_map_, matched_node_map_)) {
          return false;
        }
      } else {
        // standard input checking
        if (!CheckInputs(iter->second.node, &temp_node_map_[iter->first])) {
          return false;
        }
      }
      // check dynamic outputs
      search_itr = 
        t_->nodes_dynamic_oedges_.find(temp_node_map_[iter->first].key);
      if (search_itr != t_->nodes_dynamic_oedges_.end()) {
        // outputs of this node is marked as dynamic
        if (!t_->CheckDynamicOutputs(iter->second.node, 
              &temp_node_map_[iter->first], search_itr->second, 
              fused_op_outputs_, temp_node_map_, matched_node_map_)) {
          return false;
        }
      } else {
        // standard output checking
        if (!CheckOutputs(iter->second.node, &temp_node_map_[iter->first])) {
          return false;
        }
      }
      iter->second.visited = true;
    }
  }
  return true;
}
bool OptimizerFusionImpl::CheckOutputs(const Node* node,
                                   const TempNode* temp_node) {
  std::map<int, std::vector<const Edge*>> oedge_map;
  for (auto* oedge : node->out_edges()) {
    int output_port = oedge->src_output();
    oedge_map[output_port].push_back(oedge);
  }
  for (auto iter = oedge_map.begin(); iter != oedge_map.end(); ++iter) {
    int output_port = use_dynamic_output_keys_ ? dynamic_output_port_cur_ : 
      iter->first;
    dynamic_output_port_cur_ = output_port;
    std::vector<const Edge*> oedges = iter->second;
    std::vector<std::string> output_keys;
    if (output_port == -1) {
      output_keys = temp_node->deps_outputs;
    } else {
      output_keys = temp_node->outputs[output_port];
    }
    bool outgoing_port = false;
    for (auto& output_key : output_keys) {
      if (IsAllNum(output_key.c_str())) {
        fused_op_outputs_[atoi(output_key.c_str())] = iter->second;
        if (oedges.size() > 0) {
          oedges.erase(oedges.begin());
        }
        outgoing_port = true;
      } else if (output_key == "*") {
        // a case of dynamic outputs
        fused_op_outputs_dynamic_.push_back(iter->second);
        use_dynamic_output_keys_ = true;
        outgoing_port = true;
      } else {
        const TempNode temp_output_node = temp_node_map_[output_key];
        bool found = false;
        auto node_it = matched_node_map_.find(temp_output_node.key);
        if (node_it != matched_node_map_.end()) {
          for (auto oedge_it = oedges.begin();
               oedge_it != oedges.end(); ++oedge_it) {
            const Edge* oedge = *oedge_it;
            const Node* output_node = oedge->dst();
            if (output_node == node_it->second.node) {
              found = true;
              oedges.erase(oedge_it);
              break;
            }
          }
        } else {
          for (auto oedge_it = oedges.begin(); oedge_it != oedges.end();
               ++oedge_it) {
            const Edge* oedge = *oedge_it;
            const Node* output_node = oedge->dst();
            if (output_node->type_string() == temp_output_node.op) {
              MatchedNode matched_node(output_node);
              matched_node_map_.emplace(temp_output_node.key, matched_node);
              found = true;
              oedges.erase(oedge_it);
              break;
            }
          }
        }
        if (!found) {
          VLOG(2) << "Cant' find:" << temp_output_node.key
                  << ", op type:" << temp_output_node.op;
          return false;
        }
      }
    }  // end for each consumer
    if (!outgoing_port && oedges.size() > 0 &&
        node->type_string() != "Const") {  // There's no cost to duplicate Const
      // has more consumers than the pattern
      return false;
    }
  }  // end for each output_port
  use_dynamic_output_keys_ = false;
  return true;
}

bool OptimizerFusionImpl::CheckInputs(const Node* node,
                                  const TempNode* temp_node) {
  // require a sorting of in_edges by ascending order
  std::vector<std::tuple<int, const Edge*> > sorting_in_edges;
  sortInEdges(node->in_edges(), sorting_in_edges);
  std::set<std::string> visited_control_deps;
  std::vector<std::string> temp_deps_input_keys = temp_node->deps_inputs;
  auto deps_input_it = temp_deps_input_keys.begin();

  for (auto pair : sorting_in_edges) {
    auto* iedge = std::get<1>(pair);
    // added for dynamic input edges
    int input_port = use_dynamic_input_keys_ ? dynamic_input_port_cur_ : 
      iedge->dst_input();
    dynamic_input_port_cur_ = input_port; 
    if (input_port < 0) {
      if (node->type_string() == "Const"
          || (node->type_string() == "Identity"
             && temp_deps_input_keys.empty())) {
        // TODO(minmin) not 100% sure about the safty of ignoring control
        // input of Const node. Best to avoid Const node in the Template
        VLOG(2) << "unexpected here:" << node->DebugString();
        continue;
      }
    }
    if (input_port >= (int)temp_node->inputs.size() ) {
      LOG(FATAL) << "Please verify Template's node ("
                 << node->type_string() << ") definition"
                 << ", node inputs:" << node->in_edges().size()
                 << ", template node inputs:" << temp_node->inputs.size()
                 << " mismatch.";
    }
    const Node* input_node = iedge->src();
    // control dependency node
    if (input_port == -1) {
      bool found = false;
        if (temp_deps_input_keys.empty()) {
          VLOG(2) << "temp_deps_input_keys is empty"
                  << ", and node type is:" << node->type_string();
          continue;
        }
        auto input_key = *deps_input_it;
        ++deps_input_it;
        if (IsAllNum(input_key.c_str())) {
          fused_op_deps_inputs_.emplace_back(iedge);
          continue;
        } else {
          TempNode temp_input_node = temp_node_map_[input_key];
          if (input_node->type_string() == temp_input_node.op
              && visited_control_deps.end() == visited_control_deps.find(input_key)) {
            visited_control_deps.emplace(input_key);
            auto it = matched_node_map_.find(temp_input_node.key);
            if (it != matched_node_map_.end()) {
              if (input_node != it->second.node) {
                VLOG(2) << "port = -1 duplicate input:" << input_node->name()
                  << ", previous node:" << it->second.node->name();
                return false;
              }
            } else {
              MatchedNode matched_node(input_node);
              matched_node_map_.insert(
                  std::make_pair(temp_input_node.key, matched_node));
            }
            found = true;
            break;
          }
          if (found) {
            continue;
          } else {
            VLOG(2) << "port = -1 not found input:" << input_node->name();
            return false;
          }
        }
    } else {
      std::string temp_input_key = temp_node->inputs[input_port];
      if (IsAllNum(temp_input_key.c_str())){
        fused_op_inputs_[atoi(temp_input_key.c_str())] = iedge;
        continue;
      }
      // added for dynamic input edges
      if (temp_input_key == "*") {
        // add to dynamic input edges
        fused_op_input_dynamic_.push_back(iedge);
        use_dynamic_input_keys_ = true;
        continue;
      } else {
        // turn off dynamic input when ever a non-dynamic edge appears
        // the dynamic template keys must be after the static template keys
        use_dynamic_input_keys_ = false;
      }

      const TempNode temp_input_node = temp_node_map_[temp_input_key];
      if (input_node->type_string() == temp_input_node.op) {
        auto it = matched_node_map_.find(temp_input_node.key);
        if (it != matched_node_map_.end()) {
          if (input_node != it->second.node) {
            VLOG(2) << "checkInput:" << temp_input_key
                    << ", input_node:" << input_node->name();
            return false;
          }
        } else {
          MatchedNode matched_node(input_node);
          matched_node_map_.insert(
              std::make_pair(temp_input_node.key, matched_node));
        }
        continue;
      }
      return false;
    }
  }
  use_dynamic_input_keys_ = false;
  return true;
}

bool OptimizerFusionImpl::CheckMatchedNodeInSameFrame() {
  // TODO: only op in default frame can be fused
  const Node *first_key_node = matched_node_map_[t_->first_key_].node;
  std::string frame_name = node_frame_map_[first_key_node];
  if (frame_name != "")
    return false;
  for (auto matched_node_it : matched_node_map_) {
    const Node * node = std::get<1>(matched_node_it).node;
    if (node_frame_map_[node] != frame_name)
      return false;
  }

  return true;
}

bool OptimizerFusionImpl::Optimize() {
  bool changed = false;
  // TODO(minmin) check Template consistency before really optimizing
  for (Node* node : g_->nodes()) {
    if (node->type_string() == temp_node_map_[t_->first_key_].op) {
      matched_node_map_.clear();
      t_->node_to_temp_key_.clear();
      fused_op_deps_inputs_.clear();
      fused_op_input_dynamic_.clear();
      fused_op_outputs_dynamic_.clear();
      fused_op_inputs_.resize(t_->num_inputs_);
      fused_op_outputs_.resize(t_->num_outputs_);
      for (int i = 0; i < fused_op_inputs_.size(); ++i) {
        fused_op_inputs_[i] = nullptr;
      }
      for (int i = 0; i < fused_op_outputs_.size(); ++i) {
        fused_op_outputs_[i].clear();
      }
      VLOG(2) << "try to match: " << t_->name() << " " << node->name();
      VLOG(2) << "First Matched: " << node->name()
        << ", t->first_key:" << t_->first_key_
        << ", t->first_value:" << temp_node_map_[t_->first_key_].key;
      // check dynamic inputs
      auto search_itr = 
        t_->nodes_dynamic_iedges_.find(temp_node_map_[t_->first_key_].key);
      if (search_itr != t_->nodes_dynamic_iedges_.end()) {
        if (!t_->CheckDynamicInputs(node, &temp_node_map_[t_->first_key_], 
              search_itr->second, fused_op_inputs_, temp_node_map_, 
              matched_node_map_)) {
          continue;
        }
      } else {
        if (!CheckInputs(node, &temp_node_map_[t_->first_key_])) {
          continue;
        }
      }
      // check dynamic outputs
      search_itr = 
        t_->nodes_dynamic_oedges_.find(temp_node_map_[t_->first_key_].key);
      if (search_itr != t_->nodes_dynamic_oedges_.end()) {
        if (!t_->CheckDynamicOutputs(node, &temp_node_map_[t_->first_key_], 
            search_itr->second, fused_op_outputs_, temp_node_map_, 
            matched_node_map_)) {
          continue;
        }
      } else {
        if (!CheckOutputs(node, &temp_node_map_[t_->first_key_])) {
          continue;
        }
      }
      MatchedNode matched_node(node, true);
      matched_node_map_.insert(
          std::make_pair(t_->first_key_, matched_node));
      if (!VisitMatchedNodes()) {
        VLOG(2) << "VisitMatchedNodes failed";
        continue;
      }
      // double check the matched nodes
      if (matched_node_map_.size() != temp_node_map_.size()) {
        VLOG(2) << "Failed double check the matched nodes "
                << matched_node_map_.size() << " != "
                << temp_node_map_.size();
        continue;
      }

      // double check the matched nodes are in same frame
      if (!CheckMatchedNodeInSameFrame()) {
        VLOG(2) << "Failed double check the matched nodes, they are not in same frame";
        continue;
      }
      // double check the matched inputs
      bool passed = true;
      for (int i = 0; i < t_->num_inputs_; ++i) {
        if (fused_op_inputs_[i] == nullptr) {
          passed = false;
          VLOG(2) << "failed check inputs";
          continue;
        }
      }
      if (!passed) {
        VLOG(2) << "Failed double check the matched inputs";
        continue;
      }

      if (fused_op_input_dynamic_.size() > 0) {
        // append dynamic in edges
        fused_op_inputs_.reserve(fused_op_inputs_.size() + 
            fused_op_input_dynamic_.size());
        fused_op_inputs_.insert(fused_op_inputs_.end(), 
            fused_op_input_dynamic_.begin(),
            fused_op_input_dynamic_.end());
      }

      // double check the matched outputs
      for (int i = 0; i < t_->num_outputs_; ++i) {
        if (fused_op_outputs_[i].empty()) {
          passed = false;
          continue;
        }
      }
      if (!passed) {
        VLOG(2) << "Failed double check the matched outputs";
        continue;
      }

      ++num_matched_;
      VLOG(2) << "Matched: " << num_matched_;
      for (auto iter = matched_node_map_.begin();
           iter != matched_node_map_.end(); ++iter) {
        VLOG(2) << "  " << iter->second.node->name();
      }

      std::string fused_op_name = strings::StrCat("fused_op_", num_matched_);
      if (fused_op_outputs_dynamic_.size() > 0) {
        // append dynamic out edges
        fused_op_outputs_.reserve(fused_op_outputs_.size() + 
            fused_op_outputs_dynamic_.size());
        fused_op_outputs_.insert(fused_op_outputs_.end(), 
            fused_op_outputs_dynamic_.begin(),
            fused_op_outputs_dynamic_.end());
      }

      bool subgraph_replaced = false;
      if (t_->num_deps_inputs_ > 0) {
        subgraph_replaced = t_->add_subgraph(matched_node_map_,
          fused_op_name, g_, fused_op_inputs_, fused_op_deps_inputs_, 
          fused_op_outputs_);
      } else {
        subgraph_replaced = t_->add_subgraph(matched_node_map_,
          fused_op_name, g_, fused_op_inputs_, fused_op_outputs_);
      }

      VLOG(2) << "subgraph_replace:" << subgraph_replaced;

      if (!subgraph_replaced && t_->fused_op_ != "") {
        NodeDef* fused_def = new NodeDef();
        fused_def->set_op(t_->fused_op_);
        fused_def->set_name(fused_op_name);
        for (int i = 0; i < t_->num_inputs_; ++i) {
          const Edge* iedge = fused_op_inputs_[i];
          std::string input_name = strings::StrCat(iedge->src()->def().name(),
                                  ":", iedge->src_output());
          fused_def->add_input(input_name);
        }
        Status status;
        Node* fused_op = g_->AddNode(*fused_def, &status);
        if (status != Status::OK()) {
          VLOG(2) << status.error_message();
          continue;
        }
        for (int i = 0; i < t_->num_inputs_; ++i) {
          const Edge* iedge = fused_op_inputs_[i];
          g_->AddEdge(iedge->src(), iedge->src_output(), fused_op, i);
          g_->RemoveEdge(iedge);
        }
        for (int i = 0; i < t_->num_outputs_; ++i) {
          for (auto* oedge : fused_op_outputs_[i]) {
            g_->AddEdge(fused_op, i, oedge->dst(), oedge->dst_input());
            g_->RemoveEdge(oedge);
          }
        }
      }
      changed = true;
    }
  }
  VLOG(2) << "num_matched " << num_matched_;

  return changed;
}

} // tensorflow
