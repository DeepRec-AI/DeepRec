/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "odl_processor/core/util/utils.h"
#include "odl_processor/core/graph_optimizer.h"
#include "odl_processor/core/util/utils.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "gtest/gtest.h"

namespace tensorflow {
namespace odl_processor {

TEST(GraphOptimizerTest, StaticShapeCluteringStrategy0) {
  GraphDef graph_def;
  NodeDef* n0 = graph_def.add_node();
  n0->set_name("A");
  n0->set_op("A");
  AttrValue value0;
  tensorflow::TensorShapeProto tshape0;
  tshape0.add_dim()->set_size(-1);
  *value0.mutable_shape() = tshape0;
  (*n0->mutable_attr())["_output_shapes"] = value0;

  NodeDef* n1 = graph_def.add_node();
  n1->set_name("B");
  n1->set_op("B");
  AttrValue value1;
  tensorflow::TensorShapeProto tshape1;
  tshape1.add_dim()->set_size(1);
  *value1.mutable_shape() = tshape1;
  (*n1->mutable_attr())["_output_shapes"] = value1;

  NodeDef* n2 = graph_def.add_node();
  n2->set_name("C");
  n2->set_op("C");
  AttrValue value2;
  tensorflow::TensorShapeProto tshape2;
  tshape2.add_dim()->set_size(1);
  *value2.mutable_shape() = tshape2;
  (*n2->mutable_attr())["_output_shapes"] = value2;

  NodeDef* n3 = graph_def.add_node();
  n3->set_name("D");
  n3->set_op("D");
  AttrValue value3;
  tensorflow::TensorShapeProto tshape3;
  tshape3.add_dim()->set_size(1);
  *value3.mutable_shape() = tshape3;
  (*n3->mutable_attr())["_output_shapes"] = value3;

  n2->add_input("A");
  n2->add_input("B");
  n3->add_input("C");

  StaticShapeCluteringStrategy sscs;
  std::vector<std::string> inputs;
  inputs.push_back("A");
  inputs.push_back("B");
  std::vector<std::string> outputs;
  outputs.push_back("D");
  ClusteredGraphInfo clustered_graph_info;
 
  sscs.Run(graph_def, inputs, outputs, &clustered_graph_info);

  // graphdef1
  GraphDef gdef1 = clustered_graph_info.tf_subgraph;
  EXPECT_EQ(gdef1.node().size(), 3);
  int count = 3;
  for (auto node : gdef1.node()) {
    if (node.name() == "A" ||
        node.name() == "B" ||
        node.name() == "C") --count;
  }
  EXPECT_EQ(count, 0);

  // graphdef2
  GraphDef gdef2 = clustered_graph_info.iree_subgraph;
  EXPECT_EQ(gdef2.node().size(), 1);
  count = 1;
  for (auto node : gdef2.node()) {
    if (node.name() == "D") --count;
  }
  EXPECT_EQ(count, 0);
}

TEST(GraphOptimizerTest, StaticShapeCluteringStrategy1) {
  GraphDef graph_def;
  NodeDef* n0 = graph_def.add_node();
  n0->set_name("A");
  n0->set_op("A");
  AttrValue value0;
  tensorflow::TensorShapeProto tshape0;
  tshape0.add_dim()->set_size(1);
  *value0.mutable_shape() = tshape0;
  (*n0->mutable_attr())["_output_shapes"] = value0;

  NodeDef* n1 = graph_def.add_node();
  n1->set_name("B");
  n1->set_op("B");
  AttrValue value1;
  tensorflow::TensorShapeProto tshape1;
  tshape1.add_dim()->set_size(1);
  *value1.mutable_shape() = tshape1;
  (*n1->mutable_attr())["_output_shapes"] = value1;

  NodeDef* n2 = graph_def.add_node();
  n2->set_name("C");
  n2->set_op("C");
  AttrValue value2;
  tensorflow::TensorShapeProto tshape2;
  tshape2.add_dim()->set_size(1);
  *value2.mutable_shape() = tshape2;
  (*n2->mutable_attr())["_output_shapes"] = value2;

  NodeDef* n3 = graph_def.add_node();
  n3->set_name("D");
  n3->set_op("D");
  AttrValue value3;
  tensorflow::TensorShapeProto tshape3;
  tshape3.add_dim()->set_size(1);
  *value3.mutable_shape() = tshape3;
  (*n3->mutable_attr())["_output_shapes"] = value3;

  n2->add_input("A");
  n2->add_input("B");
  n3->add_input("C");

  StaticShapeCluteringStrategy sscs;
  std::vector<std::string> inputs;
  inputs.push_back("A");
  inputs.push_back("B");
  std::vector<std::string> outputs;
  outputs.push_back("D");
  ClusteredGraphInfo clustered_graph_info;
 
  sscs.Run(graph_def, inputs, outputs, &clustered_graph_info);

  // graphdef1
  GraphDef gdef1 = clustered_graph_info.tf_subgraph;
  EXPECT_EQ(gdef1.node().size(), 0);

  // graphdef2
  GraphDef gdef2 = clustered_graph_info.iree_subgraph;
  EXPECT_EQ(gdef2.node().size(), 4);
  int count = 4;
  for (auto node : gdef2.node()) {
    if (node.name() == "A" ||
        node.name() == "B" ||
        node.name() == "C" ||
        node.name() == "D") --count;
  }
  EXPECT_EQ(count, 0);
}

TEST(GraphOptimizerTest, StaticShapeCluteringStrategyBlackOps) {
  GraphDef graph_def;
  NodeDef* n0 = graph_def.add_node();
  n0->set_name("A");
  n0->set_op("A");
  AttrValue value0;
  tensorflow::TensorShapeProto tshape0;
  tshape0.add_dim()->set_size(1);
  *value0.mutable_shape() = tshape0;
  (*n0->mutable_attr())["_output_shapes"] = value0;

  NodeDef* n1 = graph_def.add_node();
  n1->set_name("B");
  n1->set_op("B");
  AttrValue value1;
  tensorflow::TensorShapeProto tshape1;
  tshape1.add_dim()->set_size(1);
  *value1.mutable_shape() = tshape1;
  (*n1->mutable_attr())["_output_shapes"] = value1;

  NodeDef* n2 = graph_def.add_node();
  n2->set_name("C");
  n2->set_op("SparseSegmentSum");
  AttrValue value2;
  tensorflow::TensorShapeProto tshape2;
  tshape2.add_dim()->set_size(1);
  *value2.mutable_shape() = tshape2;
  (*n2->mutable_attr())["_output_shapes"] = value2;

  NodeDef* n3 = graph_def.add_node();
  n3->set_name("D");
  n3->set_op("D");
  AttrValue value3;
  tensorflow::TensorShapeProto tshape3;
  tshape3.add_dim()->set_size(1);
  *value3.mutable_shape() = tshape3;
  (*n3->mutable_attr())["_output_shapes"] = value3;

  n2->add_input("A");
  n2->add_input("B");
  n3->add_input("C");

  StaticShapeCluteringStrategy sscs;
  std::vector<std::string> inputs;
  inputs.push_back("A");
  inputs.push_back("B");
  std::vector<std::string> outputs;
  outputs.push_back("D");
  ClusteredGraphInfo clustered_graph_info;
 
  sscs.Run(graph_def, inputs, outputs, &clustered_graph_info);

  // graphdef1
  GraphDef gdef1 = clustered_graph_info.tf_subgraph;
  EXPECT_EQ(gdef1.node().size(), 3);
  int count = 3;
  for (auto node : gdef1.node()) {
    if (node.name() == "A" ||
        node.name() == "B" ||
        node.name() == "C") --count;
  }
  EXPECT_EQ(count, 0);

  // graphdef2
  GraphDef gdef2 = clustered_graph_info.iree_subgraph;
  EXPECT_EQ(gdef2.node().size(), 1);
  count = 1;
  for (auto node : gdef2.node()) {
    if (node.name() == "D") --count;
  }
  EXPECT_EQ(count, 0);
}

TEST(GraphOptimizerTest, StaticShapeCluteringStrategyControlEdge) {
  GraphDef graph_def;
  NodeDef* n0 = graph_def.add_node();
  n0->set_name("A");
  n0->set_op("A");
  AttrValue value0;
  tensorflow::TensorShapeProto tshape0;
  tshape0.add_dim()->set_size(1);
  *value0.mutable_shape() = tshape0;
  (*n0->mutable_attr())["_output_shapes"] = value0;

  NodeDef* n1 = graph_def.add_node();
  n1->set_name("B");
  n1->set_op("B");
  AttrValue value1;
  tensorflow::TensorShapeProto tshape1;
  tshape1.add_dim()->set_size(1);
  *value1.mutable_shape() = tshape1;
  (*n1->mutable_attr())["_output_shapes"] = value1;

  NodeDef* n2 = graph_def.add_node();
  n2->set_name("C");
  n2->set_op("C");
  AttrValue value2;
  tensorflow::TensorShapeProto tshape2;
  tshape2.add_dim()->set_size(1);
  *value2.mutable_shape() = tshape2;
  (*n2->mutable_attr())["_output_shapes"] = value2;

  NodeDef* n3 = graph_def.add_node();
  n3->set_name("D");
  n3->set_op("D");
  AttrValue value3;
  tensorflow::TensorShapeProto tshape3;
  tshape3.add_dim()->set_size(1);
  *value3.mutable_shape() = tshape3;
  (*n3->mutable_attr())["_output_shapes"] = value3;

  n2->add_input("^A");
  n2->add_input("B");
  n3->add_input("C");

  StaticShapeCluteringStrategy sscs;
  std::vector<std::string> inputs;
  inputs.push_back("A");
  inputs.push_back("B");
  std::vector<std::string> outputs;
  outputs.push_back("D");
  ClusteredGraphInfo clustered_graph_info;
 
  sscs.Run(graph_def, inputs, outputs, &clustered_graph_info);

  // graphdef1
  GraphDef gdef1 = clustered_graph_info.tf_subgraph;
  EXPECT_EQ(gdef1.node().size(), 3);
  int count = 3;
  for (auto node : gdef1.node()) {
    if (node.name() == "A" ||
        node.name() == "B" ||
        node.name() == "C") --count;
  }
  EXPECT_EQ(count, 0);

  // graphdef2
  GraphDef gdef2 = clustered_graph_info.iree_subgraph;
  EXPECT_EQ(gdef2.node().size(), 1);
  count = 1;
  for (auto node : gdef2.node()) {
    if (node.name() == "D") --count;
  }
  EXPECT_EQ(count, 0);
}

} // namespace odl_processor
} // namespace tensorflow
