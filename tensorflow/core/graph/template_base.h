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

#ifndef TENSORFLOW_GRAPH_TEMPLATE_BASE_H_
#define TENSORFLOW_GRAPH_TEMPLATE_BASE_H_

#include <string>
#include <vector>
#include <map>
#include <utility>
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/node_def.pb.h"
namespace tensorflow {

// helper to check whether a template node is an input or output node  
inline bool ScanUnsignInteger(const char **str) {
  const char* begin = *str;
  while (**str != '\0'&&**str<='9'&&**str>='0') {
    ++(*str);
  }
  return *str > begin;
}

inline bool ScanInteger(const char **str) {
  if (**str == '+' || **str == '-') {
    ++(*str);
  }
  return ScanUnsignInteger(str);
}

inline bool IsAllNum(const char* str) {
  if (str == nullptr) return false;

  bool numeric = ScanInteger(&str);
  if (*str == '.') {
    ++str;
    numeric = ScanUnsignInteger(&str)||numeric;
  }
  if (*str == 'e' || *str == 'E') {
    ++str;
    numeric = numeric&&ScanInteger(&str);
  }
  return numeric &&*str == '\0';
}

struct TempNode {
    std::string key;
    std::string op;
    std::vector<std::string> inputs;
    std::vector<std::vector<std::string>> outputs;
    std::vector<std::string> deps_inputs;
    std::vector<std::string> deps_outputs;
};

struct MatchedNode{
    explicit MatchedNode(const Node* n = nullptr, bool v = false) {
      node = n;
      visited = v;
      dy_offset = 1;
    }
    const Node* node;
    bool visited;
    // to specify the offset of dynamically replicated
    // graph nodes to the original one, start from 1 and has an 
    // increment after each replication
    int dy_offset;
};

// a bundle of output edges attached to the 
// same port and utilized by CheckDynamicInputsImpl and 
// CheckDynamicOutputsImpl
class OutEdges {
  public: 
    OutEdges(): remainEdgeVal(0) {}

    // add a new edge
    void append(const Edge* item) {
      oedges.push_back(item);
      remainEdgeVal++;
    }

    // mapping a graph node_port to a tempalte node port
    // node_port: a graph node port (outputs) 
    // node_max_port: the maximum port of the node (graph) 
    // start_port: the template port index where the dynamic output edge starts  
    // end_port: the template port index where the dynamic output edge ends  
    // return: the port index on template   
    int getTemplatePort(int node_port, int node_max_port, 
                        int start_port, int end_port) {
      int output_port = -1;
      if (node_port >= start_port && node_port <= (node_max_port + end_port)) {
          output_port = start_port;
      } else {
          int dynamic_node_num = node_max_port + end_port - start_port + 1;
          if (dynamic_node_num >= 1) {
              dynamic_node_num--;
          }
          output_port = (node_port < start_port) ? node_port : 
            (node_port - dynamic_node_num);
      }
      return output_port;
    }  

    // if a template node has an existing matching, checking endpoints of 
    // all edges; if edge endpoint matched, decreases the remained edge number
    // by one. Otherwise, if dy_mode is 1 (parallel), add a new node to both of
    // temp_node_map and matched_node_map
    // return true if any of its output edge has a node matched
    bool checkMatchedNode(const TempNode& temp_output_node, 
                      MatchedNode& matched,
                      std::map<const std::string, TempNode>& temp_node_map, 
                      std::map<std::string, MatchedNode>& matched_node_map,
                      std::map<std::string, std::string>& node_to_temp_key,
                      int dy_mode ) {
      bool ret = false;
      for (auto oedge_it = oedges.begin();
          oedge_it != oedges.end(); ++oedge_it) {
        const Edge* oedge = *oedge_it;
        const Node* output_node = oedge->dst();
        if (output_node == matched.node ||
            output_node->type_string() == "ShapeN") {
            ret = true;
            remainEdgeVal--;
        } else if (dy_mode == 1) {
          std::string expand_key = temp_output_node.key + "_" 
            + std::to_string(matched.dy_offset);
          temp_node_map.emplace(expand_key, temp_output_node);
          MatchedNode dy_input_node(output_node);
          matched_node_map.emplace(expand_key, dy_input_node);
          node_to_temp_key.emplace(output_node->name(), expand_key);
          matched.dy_offset++;
          ret = true;
          remainEdgeVal--;
        }
      }
      return ret;
    }

    // if a template node is not in matched_node_map, trying to emplace it
    // return true if any of its output edge has a node being added
    bool addNewNode(const TempNode& temp_output_node, 
                    const std::string& output_key,
                    std::map<const std::string, TempNode>& temp_node_map, 
                    std::map<std::string, MatchedNode>& matched_node_map, 
                    std::map<std::string, std::string>& node_to_temp_key) {
      bool ret = false;
      int64 template_id = 1;
      for (auto oedge_it = oedges.begin(); 
          oedge_it != oedges.end(); ++oedge_it) {
        const Edge* oedge = *oedge_it;
        const Node* output_node = oedge->dst();
        if (output_node->type_string() == temp_output_node.op) {
          if (ret) {
            std::string expand_key = output_key 
              + "_" + std::to_string(template_id); 
            // add a new template node
            temp_node_map.emplace(expand_key, temp_output_node);
            // add a new matched node 
            MatchedNode matched_node(output_node);
            matched_node_map.emplace(expand_key, 
                matched_node);
            node_to_temp_key.emplace(output_node->name(), expand_key);
            template_id++;
          } else {
            MatchedNode matched_node(output_node);
            matched_node_map.emplace(temp_output_node.key, matched_node);
            node_to_temp_key.emplace(output_node->name(), temp_output_node.key);
            ret = true;
          }
          remainEdgeVal--;
        } else {
          if (output_node->type_string() == "ShapeN") {
            remainEdgeVal--;
          }
        }
      }
      return ret;
    }

    int size() { return oedges.size(); } 
    int remainEdge() {return remainEdgeVal;}
    std::vector<const Edge*> get() { return oedges; }

  private:
    std::vector<const Edge*> oedges;
    int remainEdgeVal;
};

class TemplateBase {
 public:
  std::vector<TempNode> temp_nodes_;

  std::string first_key_;
  int num_inputs_;
  int num_outputs_;
  int num_deps_inputs_ = 0;

  std::string fused_op_;
  // store nodes that has a dynamic number of in-edges from the same src node
  std::map<std::string, int> nodes_dynamic_iedges_;
  // store nodes that has a dynamic number of out-edges to the same dst node
  std::map<std::string, int> nodes_dynamic_oedges_;
  // store mapping from the name of an added node to its key in template
  std::map<std::string, std::string> node_to_temp_key_;

  virtual const string name() {
    return "TemplateBase";
  }

  virtual bool add_subgraph(std::map<std::string, MatchedNode>& nodes,
                            std::string name_prefix, Graph* g,
                            std::vector<const Edge*>& inputs,
                            std::vector<std::vector<const Edge*>>& outputs) {
    return false;
  }

  virtual bool add_subgraph(std::map<std::string, MatchedNode>& nodes,
                            std::string name_prefix, Graph* g,
                            std::vector<const Edge*>& inputs,
                            std::vector<const Edge*>& deps_inputs,
                            std::vector<std::vector<const Edge*>>& outputs) {
    return false;
  }

  // check dynamic inputs 
  // node: the target node in graph  
  // temp_node: the target node in template  
  // dy_mode: a flag to control dynamic node/edges processing
  // (1) dy_mode == 0, all dynamic input edges share the same endpoint
  // (2) dy_mode == 1, all dynamic input edges have their own distinct endpoint
  // (3) dy_mode == 2, no dynamic checking applied (override flag)  
  virtual bool CheckDynamicInputs(
      const Node* node, const TempNode* temp_node, int dy_mode, 
      std::vector<const Edge*>& fused_op_inputs, 
      std::map<const std::string, TempNode>& temp_node_map,
      std::map<std::string, MatchedNode>& matched_node_map) = 0; 

  // check dynamic outputs 
  // node: the target node in graph  
  // temp_node: the target node in template  
  // dy_mode: a flag to control dynamic node/edges processing
  // (1) dy_mode == 0, all dynamic output edges share the same endpoint
  // (2) dy_mode == 1, all dynamic output edges have their own distinct endpoint
  // (3) dy_mode == 2, no dynamic checking applied (override flag)  
  virtual bool CheckDynamicOutputs(
      const Node* node, const TempNode* temp_node, int dy_mode, 
      std::vector<std::vector<const Edge*>>& fused_op_outputs, 
      std::map<const std::string, TempNode>& temp_node_map,
      std::map<std::string, MatchedNode>& matched_node_map) = 0; 

  virtual bool CheckDynamicInputsImpl(
      const Node* node, const TempNode* temp_node, int dy_mode, 
      std::vector<const Edge*>& fused_op_inputs, 
      std::map<const std::string, TempNode>& temp_node_map,
      std::map<std::string, MatchedNode>& matched_node_map, 
      int32 start_port, int32 end_port) {
    // find the maximal input port 
    int max_port = 0;
    for (auto* iedge : node->in_edges()) {
      max_port = (iedge->dst_input() > max_port) ? 
        iedge->dst_input(): max_port;
    }
    for (auto* iedge : node->in_edges()) {
      int input_port = iedge->dst_input();
      if (input_port < 0) {
        // filter out NoOp edges 
        continue;
      }
      const Node* input_node = iedge->src();
      // find the src node key in template 
      std::string temp_input_key = "";
      auto iter = node_to_temp_key_.find(input_node->name());
      if (iter != node_to_temp_key_.end()) {
        temp_input_key = iter->second;
      } else {
        if (dy_mode == 2) {
          // switch off dynamic checking
          temp_input_key = temp_node->inputs[input_port];
        } else if (input_port >= start_port 
            && input_port < (max_port+1 + end_port)) {
          // input_port is within the dynamic checking range
          temp_input_key =  (input_port == start_port ) ? 
            temp_node->inputs[start_port] : (temp_node->inputs[start_port] 
                + "_" + std::to_string(input_port));
          if (input_port != start_port) {
            temp_node_map.emplace(temp_input_key, 
                temp_node_map[temp_node->inputs[start_port]]);
          }
        } else {
          // fallback to standard input edge/node 
          int dynamic_node_num = max_port+1 + end_port - start_port;
          if (dynamic_node_num >= 1) {
            dynamic_node_num--;
          }
          temp_input_key = (input_port < start_port) ? 
            temp_node->inputs[input_port] 
            : temp_node->inputs[input_port - dynamic_node_num];
        }
      }

      // from temp_input_key to temp node 
      if (IsAllNum(temp_input_key.c_str())) {
        fused_op_inputs[atoi(temp_input_key.c_str())] = iedge;
        continue;
      }

      // added for dynamic input edges
      if (temp_input_key == "*") {
        // add to dynamic input edges
        fused_op_inputs.push_back(iedge);
        continue;
      } 

      const TempNode temp_input_node = temp_node_map[temp_input_key];
      if (input_node->type_string() == temp_input_node.op) {
        auto it = matched_node_map.find(temp_input_key);
        if (it != matched_node_map.end()) {
          // double check the returned node of matched_node_map
          // with the input_node 
          if (input_node != it->second.node) {
            if (dy_mode == 1) {
              // add a new node to both of temp_node_map 
              // and matched_node_map
              std::string expand_key = temp_input_node.key 
                + "_" + std::to_string(it->second.dy_offset);
              temp_node_map.emplace(expand_key, temp_input_node);
              MatchedNode dy_input_node(input_node);
              matched_node_map.emplace(expand_key, dy_input_node);
              node_to_temp_key_.emplace(input_node->name(), expand_key);
              it->second.dy_offset++;
            } else {
              return false;
            }
          }
        } else {
          // add new item to matched_node_map
          MatchedNode matched_node(input_node);
          matched_node_map.emplace(temp_input_key, matched_node);
          node_to_temp_key_.emplace(input_node->name(), temp_input_key);
        }
        continue;
      }
      return false;
    }
    return true;
  }

  virtual bool CheckDynamicOutputsImpl(
      const Node* node, const TempNode* temp_node, int dy_mode, 
      std::vector<std::vector<const Edge*>>& fused_op_outputs, 
      std::map<const std::string, TempNode>& temp_node_map,
      std::map<std::string, MatchedNode>& matched_node_map,
      int32 start_port, int32 end_port) {
    // port to OutEdges
    std::map<int, OutEdges> oedge_map;
    int max_port = 0;
    // find the maximum port index
    for (auto* oedge : node->out_edges()) {
      int output_port = oedge->src_output();
      max_port = (output_port > max_port) ? output_port : max_port;
      if (output_port >= 0) {
        oedge_map[output_port].append(oedge);
      }
    }
    // looping over all ports (one port to one tensor)
    for (auto iter = oedge_map.begin(); iter != oedge_map.end(); ++iter) {
      auto oedges = iter->second;
      // obtain the corresponding output port on template
      int output_port = (dy_mode == 2) ? iter->first 
        : oedges.getTemplatePort(iter->first, max_port, 
          start_port, end_port);
      if (output_port < 0) {
        return false;
      }

      const auto& output_keys = temp_node->outputs[output_port];
      bool outgoing_port = false;

      for (auto& output_key : output_keys) {
        // check whether or not an input/output node
        if (IsAllNum(output_key.c_str())) {
          fused_op_outputs[atoi(output_key.c_str())] = oedges.get();
          outgoing_port = true;
        } else {
          const TempNode temp_output_node = temp_node_map[output_key];
          bool found = false;
          auto node_it = matched_node_map.find(temp_output_node.key);
          if (node_it != matched_node_map.end()) {
            // looping over all output edges bind to the same port
            found = oedges.checkMatchedNode(temp_output_node, 
                node_it->second, temp_node_map, matched_node_map, 
                node_to_temp_key_, dy_mode); 
          } else {
            // add new nodes
            found = oedges.addNewNode(temp_output_node, output_key, 
                temp_node_map, matched_node_map, node_to_temp_key_);
          }
          if (!found) {
            return false;
          }
        }
      }  // end for each consumer

      if (!outgoing_port && oedges.remainEdge() >0 && 
          node->type_string() != "Const") {  
        // There's no cost to duplicate Const
        // has more consumers than the pattern
        return false;
      }
    }  // end for each output_port
    return true;
  }

  TemplateBase() : first_key_(""), num_inputs_(0),
                   num_outputs_(0), fused_op_("") {}

 protected:
  // helper functions for constructing new subgraph
  void add_input(NodeDef& ndef, const Edge* iedge) {
    std::string input_name = strings::StrCat(iedge->src()->def().name(),
                            ":", iedge->src_output());
    ndef.add_input(input_name);
  }

  void copy_attr(NodeDef& dst, const NodeDef& src) {
    auto attr = src.attr();
    for (auto it = attr.begin(); it != attr.end(); ++it) {
      dst.mutable_attr()->insert({it->first, it->second});
    }
  }

  void add_iedge(Graph* g, Node* dst, int dst_input,
      const Edge* ori_edge, bool remove = true) {
    g->AddEdge(ori_edge->src(), ori_edge->src_output(), dst, dst_input);
    if (remove) {
      g->RemoveEdge(ori_edge);
    }
  }

  void add_oedges(Graph* g, Node* src, int src_output,
      std::vector<const Edge*>& ori_edges) {
    for (auto* ori_edge : ori_edges) {
      if (ori_edge != nullptr && ori_edge->dst() != nullptr) {
        g->AddEdge(src, src_output, ori_edge->dst(), ori_edge->dst_input());
        g->RemoveEdge(ori_edge);
      }
    }
  }
  void remove_oedges(Graph* g, std::vector<const Edge*>& ori_edges) {
    for (auto* ori_edge : ori_edges) {
      g->RemoveEdge(ori_edge);
    }
  }
};
}  // namespace tensorflow
#endif  // TENSORFLOW_GRAPH_TEMPLATE_BASE_H_
