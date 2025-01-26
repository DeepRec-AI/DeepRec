#include "tensorflow/core/graph/stream_subgraph.h"

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(GenerateNodeStreamId, TestGraph) {
  Graph graph(OpRegistry::Global());
  std::vector<Node*> nodes;
  nodes.push_back(graph.source_node());
  nodes.push_back(graph.sink_node());
  for (auto edge : graph.source_node()->out_edges()) {
    graph.RemoveEdge(edge);
  }
  for (int i = 0; i < 5; ++i) {
    Node* node;
    TF_CHECK_OK(NodeBuilder(strings::StrCat("v", i+1), "NoOp").Finalize(&graph, &node));
    nodes.push_back(node);
  }

  graph.AddEdge(nodes[0], 0, nodes[1], 0);
  graph.AddEdge(nodes[0], 1, nodes[2], 0);
  graph.AddEdge(nodes[0], 2, nodes[3], 0);
  graph.AddEdge(nodes[0], 3, nodes[5], 0);

  graph.AddEdge(nodes[1], 0, nodes[4], 0);

  graph.AddEdge(nodes[2], 0, nodes[4], 1);
  graph.AddEdge(nodes[2], 1, nodes[6], 0);
  graph.AddEdge(nodes[2], 2, nodes[5], 1);

  graph.AddEdge(nodes[3], 0, nodes[5], 2);

  graph.AddEdge(nodes[4], 0, nodes[6], 1);

  auto mapping = stream_subgraph::GenerateNodeStreamId(&graph);

  EXPECT_EQ(mapping.size(), 7);
  EXPECT_EQ(mapping[0], mapping[1]);
  EXPECT_EQ(mapping[1], mapping[4]);
  EXPECT_EQ(mapping[4], mapping[6]);
  EXPECT_EQ(mapping[2], mapping[5]);
  for (int i = 0; i < mapping.size(); ++i) {
    VLOG(2) << i+1 << ": " << mapping[i];
  }
}

}  // namespace
}  // namespace tensorflow
