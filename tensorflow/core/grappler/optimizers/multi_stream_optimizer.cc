/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <queue>
#include "tensorflow/core/grappler/optimizers/multi_stream_optimizer.h"

#include "tensorflow/core/common_runtime/scoped_allocator.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {
namespace grappler {

namespace {
std::vector<StringPiece> SplitString(const string& str,
                                     const char delim) {
  std::vector<StringPiece> result;
  StringPiece text(str);
  auto f = text.find(delim);
  while (f != StringPiece::npos) {
    StringPiece token = text.substr(0, f);
    result.emplace_back(token);
    text.remove_prefix(f + 1);
    f = text.find(delim);
  }
  result.push_back(text);
  
  return result;
}
}

MultiStreamOptimizer::MultiStreamOptimizer(
    const MultiStreamOptions& opt) : opt_(opt){
}

Status MultiStreamOptimizer::MarkEmbeddingGraphNodes(
    const std::vector<NodeDef*> start_nodes,
    std::unordered_map<std::string, std::vector<NodeDef*>> output_edges,
    GraphDef* optimized_graph) {
  int curr_stream_idx = 0;
  for (auto n : start_nodes) {
    tensorflow::AttrValue stream_idx_attr;
    stream_idx_attr.set_i(curr_stream_idx);
    (*n->mutable_attr())["_stream_id"] = stream_idx_attr;

    std::queue<NodeDef*> q;
    q.push(n);
    while (!q.empty()) {
      NodeDef* curr_node = q.front();
      q.pop();
      auto attr = n->attr().at("_stream_id");
      // check all output edges
      for (auto out_node : output_edges[curr_node->name()]) {
        if (out_node->name().find("/embedding_lookup_sparse") !=
            std::string::npos) {
          (*out_node->mutable_attr())["_stream_id"] = attr;
          q.push(out_node);
        }
      }
    }
    ++curr_stream_idx;
    curr_stream_idx %= opt_.multi_stream_num();
  }

  return Status::OK();
}

Status MultiStreamOptimizer::SplitEmbeddingGraph(
    const GrapplerItem& item, GraphDef* optimized_graph) {
  *optimized_graph = item.graph;
  // Find the embedding subgraphs:
  // step1: find GatherV2 ops,
  // step2: get all feature columns unique scope name,
  // step3: split out all embedding lookup subgraphs.
  std::unordered_map<std::string, NodeDef*> name_to_node;
  for (const NodeDef& node : optimized_graph->node()) {
    name_to_node[node.name()] = const_cast<NodeDef*>(&node);
  }
  std::unordered_set<std::string> embedding_gather_name_prefix;
  for (const NodeDef& node : optimized_graph->node()) {
    // Match pattern:
    // "VariableV2 -> Identity -> GatherV2"
    // "KvHandle -> ResourceGatherV2"
    if (node.op() == "GatherV2" || node.op() == "KvResourceGather") {
      // e.g. the input is a tensor like /xx/xx/reshape:1
      if (name_to_node.find(node.input()[0]) == name_to_node.end()) {
        continue;
      }
      NodeDef* input0 = name_to_node[node.input()[0]];
      if ((input0->op() == "Identity" &&
               name_to_node[input0->input()[0]]->op() == "VariableV2") ||
          input0->op() == "KvVarHandleOp") {
        // "user_define/input_layer/xx_embedding/xx_embedding_weights/embedding_lookup_sparse/GatherV2"
        // The common embedding prefix is 'user_define/input_layer/xx_embedding'
        std::vector<StringPiece> tokens = SplitString(node.name(), '/');
        std::string common_prefix("");
        for (int i = 0; i < tokens.size() - 3; ++i) {
          common_prefix += std::string(tokens[i].data(), tokens[i].size());
          common_prefix += "/";
        }
        embedding_gather_name_prefix.insert(common_prefix);
      }
    }
  }

  // Assign stream_id to each subgraph
  int stream_id = 0;
  std::unordered_map<std::string, int> name_to_streamid;
  for (auto prefix : embedding_gather_name_prefix) {
    name_to_streamid[prefix] = stream_id % opt_.multi_stream_num();
    stream_id++;
  }

  // Split out all embedding lookup graphs
  for (const NodeDef& node : optimized_graph->node()) {
    for (auto prefix : embedding_gather_name_prefix) {
      if (node.name().find(prefix) != std::string::npos) {
        tensorflow::AttrValue stream_id_attr;
        stream_id_attr.set_i(name_to_streamid[prefix]);
        (const_cast<NodeDef*>(&node))->mutable_attr()->insert(
            AttrValueMap::value_type("_stream_id", stream_id_attr));
        break;
      }
    }
  }

  return Status::OK();
}

Status MultiStreamOptimizer::Optimize(
    Cluster* cluster, const GrapplerItem& item,
    GraphDef* optimized_graph) {
  *optimized_graph = item.graph;

  if (opt_.partition_policy() ==
      MultiStreamPartitionPolicy::NO_PARTITION) {
    // nothing
  } else if (opt_.partition_policy() ==
      MultiStreamPartitionPolicy::USER_DEFINED_PARTITION) {
    // TODO
  } else if (opt_.partition_policy() ==
      MultiStreamPartitionPolicy::EMBEDDING_GRAPH_PARTITION) {
    return SplitEmbeddingGraph(item, optimized_graph);
  } else if (opt_.partition_policy() ==
             MultiStreamPartitionPolicy::FULL_GRAPH_PARTITION) {
    // TODO
  }

  return Status::OK();
}

}  // namespace grappler
}  // namespace tensorflow
