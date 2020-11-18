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

#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "gtest/gtest.h"

namespace tensorflow {
namespace processor {

TEST(UtilsTest, HasDynamicShapeOutput0) {
  GraphDef graph_def;
  NodeDef* n0 = graph_def.add_node();
  n0->set_name("A");
  n0->set_op("A");
  AttrValue value0;
  tensorflow::TensorShapeProto tshape0;
  tshape0.add_dim()->set_size(-1);
  *value0.mutable_shape() = tshape0;
  (*n0->mutable_attr())["_output_shapes"] = value0;

  EXPECT_TRUE(HasDynamicShapeOutput(n0));
}

TEST(UtilsTest, HasDynamicShapeOutput1) {
  GraphDef graph_def;
  NodeDef* n0 = graph_def.add_node();
  n0->set_name("A");
  n0->set_op("A");
  AttrValue value0;
  TensorShapeProto* tshape0 = value0.mutable_list()->add_shape();
  tshape0->add_dim()->set_size(-1);
  TensorShapeProto* tshape1 = value0.mutable_list()->add_shape();
  tshape1->add_dim()->set_size(3);
  tshape1->add_dim()->set_size(5);
  (*n0->mutable_attr())["_output_shapes"] = value0;

  EXPECT_TRUE(HasDynamicShapeOutput(n0));
}

TEST(UtilsTest, HasDynamicShapeOutput2) {
  GraphDef graph_def;
  NodeDef* n0 = graph_def.add_node();
  n0->set_name("A");
  n0->set_op("A");
  AttrValue value0;
  tensorflow::TensorShapeProto tshape0;
  tshape0.add_dim()->set_size(1);
  *value0.mutable_shape() = tshape0;
  (*n0->mutable_attr())["_output_shapes"] = value0;

  EXPECT_FALSE(HasDynamicShapeOutput(n0));
}

TEST(UtilsTest, HasDynamicShapeOutput3) {
  GraphDef graph_def;
  NodeDef* n0 = graph_def.add_node();
  n0->set_name("A");
  n0->set_op("A");
  AttrValue value0;
  TensorShapeProto* tshape0 = value0.mutable_list()->add_shape();
  tshape0->add_dim()->set_size(1);
  TensorShapeProto* tshape1 = value0.mutable_list()->add_shape();
  tshape1->add_dim()->set_size(3);
  tshape1->add_dim()->set_size(5);
  (*n0->mutable_attr())["_output_shapes"] = value0;

  EXPECT_FALSE(HasDynamicShapeOutput(n0));
}

TEST(UtilsTest, GetNodesHasDynamicShapeMap0) {
  GraphDef graph_def;
  NodeDef* n0 = graph_def.add_node();
  n0->set_name("A");
  n0->set_op("A");
  AttrValue value0;
  TensorShapeProto* tshape0 = value0.mutable_list()->add_shape();
  tshape0->add_dim()->set_size(1);
  TensorShapeProto* tshape1 = value0.mutable_list()->add_shape();
  tshape1->add_dim()->set_size(3);
  tshape1->add_dim()->set_size(5);
  (*n0->mutable_attr())["_output_shapes"] = value0;

  auto m = GetNodesHasDynamicShapeMap(graph_def);
  EXPECT_FALSE(m["A"]);
}

TEST(UtilsTest, GetNodesHasDynamicShapeMap1) {
  GraphDef graph_def;
  NodeDef* n0 = graph_def.add_node();
  n0->set_name("A");
  n0->set_op("A");
  AttrValue value0;
  tensorflow::TensorShapeProto tshape0;
  tshape0.add_dim()->set_size(-1);
  *value0.mutable_shape() = tshape0;
  (*n0->mutable_attr())["_output_shapes"] = value0;

  auto m = GetNodesHasDynamicShapeMap(graph_def);
  EXPECT_TRUE(m["A"]);
}

TEST(UtilsTest, GetNodesHasDynamicShapeMap2) {
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

  auto m = GetNodesHasDynamicShapeMap(graph_def);
  EXPECT_TRUE(m["A"]);
  EXPECT_FALSE(m["B"]);
  EXPECT_TRUE(m["C"]);
  EXPECT_FALSE(m["D"]);
}

} // namespace processor
} // namespace tensorflow
