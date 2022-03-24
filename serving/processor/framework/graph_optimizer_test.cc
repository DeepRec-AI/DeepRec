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

#include "serving/processor/framework/util/utils.h"
#include "serving/processor/framework/graph_optimizer.h"
#include "serving/processor/framework/util/utils.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "gtest/gtest.h"

namespace tensorflow {
namespace processor {

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

TEST(GraphOptimizerTest, GetDynamicAndStaticMetaGraphDef) {
  /*
    Testing case:
            A       B      C
          /  \      |     / 
         /    D     E    /
        /      \    |   /
        G       \ --|--/
                    F
     A: dynamic output
     B: dynamic output
     C: static output
     D: static output
     E: static output
     F: static output
     G: static output

     signature_def {
       key: "serving_default"
       value {
         inputs {
           key: "in_A0"
           value {
             name: "A:0"
             dtype: DT_FLOAT
             tensor_shape {
               dim {  
                 size: -1
               }      
             }      
           }      
         }     
         inputs {
           key: "in_A1"
           value {
             name: "A:1"
             dtype: DT_FLOAT
             tensor_shape {
               dim {  
                 size: 1
               }      
             }      
           }      
         }      
         inputs {
           key: "in_B"
           value {
             name: "B:0"
             dtype: DT_FLOAT
             tensor_shape {
               dim {  
                 size: -1
               }      
             }      
           }      
         }      
         inputs {
           key: "in_C"
           value {
             name: "C:0"
             dtype: DT_FLOAT
             tensor_shape {
               dim {  
                 size: 1
               }      
             }      
           }      
         }      

         outputs {
           key: "out_G"
           value {
             name: "G:0"
             dtype: DT_FLOAT
             tensor_shape {
               dim {  
                 size: 1
               }      
               dim {  
                 size: 1
               }      
             }      
           }      
         }      
         outputs {
           key: "out_F"
           value {
             name: "F:0"
             dtype: DT_FLOAT
             tensor_shape {
               dim {  
                 size: 1
               }      
             }      
           }      
         }      
         method_name: "tensorflow/serving/predict"
       }
     }

     result:

     => dynamic grpah
            A       B     
          /  \      |     
         /    D     E   
        G      \    |
 
     => static graph
                           C
          /  \      |     / 
         /    \     |    /
        /      \    |   /
                \ --|--/
                    F
    
     => dynamic graph signature def

     signature_def {
       key: "serving_default"
       value {
         inputs {
           key: "in_A0"
           value {
             name: "A:0"
             dtype: DT_FLOAT
             tensor_shape {
               dim {  
                 size: -1
               }      
             }      
           }      
         }     
         inputs {
           key: "in_A1"
           value {
             name: "A:1"
             dtype: DT_FLOAT
             tensor_shape {
               dim {  
                 size: 1
               }      
             }      
           }      
         }      
         inputs {
           key: "in_B"
           value {
             name: "B:0"
             dtype: DT_FLOAT
             tensor_shape {
               dim {  
                 size: -1
               }      
             }      
           }      
         }      

         outputs {
           key: "out_G"
           value {
             name: "G:0"
             dtype: DT_FLOAT
             tensor_shape {
               dim {  
                 size: 1
               }      
               dim {  
                 size: 1
               }      
             }      
           }      
         }      
         outputs {
           key: "out_D"
           value {
             name: "D:0"
             dtype: DT_FLOAT
             tensor_shape {
               dim {  
                 size: 1
               }      
             }      
           }      
         }      
         outputs {
           key: "out_E"
           value {
             name: "E:0"
             dtype: DT_FLOAT
             tensor_shape {
               dim {  
                 size: 1
               }      
             }      
           }      
         }      
         method_name: "tensorflow/serving/predict"
       }
     }

     => static graph signature def

     signature_def {
       key: "serving_default"
       value {
         inputs {
           key: "in_C"
           value {
             name: "C:0"
             dtype: DT_FLOAT
             tensor_shape {
               dim {  
                 size: 1
               }      
             }      
           }      
         }     
         intputs {
           key: "out_D"
           value {
             name: "D:0"
             dtype: DT_FLOAT
             tensor_shape {
               dim {  
                 size: 1
               }      
             }      
           }      
         }      
         inputs {
           key: "out_E"
           value {
             name: "E:0"
             dtype: DT_FLOAT
             tensor_shape {
               dim {  
                 size: 1
               }      
             }      
           }      
         }      
         outputs {
           key: "out_F"
           value {
             name: "F:0"
             dtype: DT_FLOAT
             tensor_shape {
               dim {  
                 size: 1
               }      
             }      
           }      
         }      
         method_name: "tensorflow/serving/predict"
       }
     }
  */
  REGISTER_OP("A")
      .Output("output1: float")
      .Output("output2: float")
      .Attr("T: type");
  REGISTER_OP("B")
      .Output("output1: float")
      .Attr("T: type");
  REGISTER_OP("C")
      .Output("output1: float")
      .Attr("T: type");
  REGISTER_OP("D")
      .Input("input1: float")
      .Output("output1: float")
      .Attr("T: type");
  REGISTER_OP("E")
      .Input("input1: float")
      .Output("output1: float")
      .Attr("T: type");
  REGISTER_OP("F")
      .Input("input1: float")
      .Input("input2: float")
      .Input("input3: float")
      .Output("output1: float")
      .Attr("T: type");
  REGISTER_OP("G")
      .Input("input1: float")
      .Output("output1: float")
      .Output("output2: float")
      .Attr("T: type");

  GraphDef graph_def;
  AttrValue values_type_attr;
  SetAttrValue(DT_FLOAT, &values_type_attr);

  NodeDef* n0 = graph_def.add_node();
  n0->set_name("A");
  n0->set_op("A");
  (*n0->mutable_attr())["T"] = values_type_attr;
  AttrValue value0;
  tensorflow::TensorShapeProto* tshape0 =
      value0.mutable_list()->add_shape();
  tshape0->add_dim()->set_size(-1);
  tensorflow::TensorShapeProto* tshape01 =
      value0.mutable_list()->add_shape();
  tshape01->add_dim()->set_size(1);
  (*n0->mutable_attr())["_output_shapes"] = value0;

  NodeDef* n1 = graph_def.add_node();
  n1->set_name("B");
  n1->set_op("B");
  (*n1->mutable_attr())["T"] = values_type_attr;
  AttrValue value1;
  tensorflow::TensorShapeProto tshape1;
  tshape1.add_dim()->set_size(-1);
  *value1.mutable_shape() = tshape1;
  (*n1->mutable_attr())["_output_shapes"] = value1;

  NodeDef* n2 = graph_def.add_node();
  n2->set_name("C");
  n2->set_op("C");
  (*n2->mutable_attr())["T"] = values_type_attr;
  AttrValue value2;
  tensorflow::TensorShapeProto tshape2;
  tshape2.add_dim()->set_size(1);
  *value2.mutable_shape() = tshape2;
  (*n2->mutable_attr())["_output_shapes"] = value2;

  NodeDef* n3 = graph_def.add_node();
  n3->set_name("D");
  n3->set_op("D");
  (*n3->mutable_attr())["T"] = values_type_attr;
  AttrValue value3;
  tensorflow::TensorShapeProto tshape3;
  tshape3.add_dim()->set_size(1);
  *value3.mutable_shape() = tshape3;
  (*n3->mutable_attr())["_output_shapes"] = value3;

  NodeDef* n4 = graph_def.add_node();
  n4->set_name("E");
  n4->set_op("E");
  (*n4->mutable_attr())["T"] = values_type_attr;
  AttrValue value4;
  tensorflow::TensorShapeProto tshape4;
  tshape4.add_dim()->set_size(1);
  *value4.mutable_shape() = tshape4;
  (*n4->mutable_attr())["_output_shapes"] = value4;

  NodeDef* n5 = graph_def.add_node();
  n5->set_name("F");
  n5->set_op("F");
  (*n5->mutable_attr())["T"] = values_type_attr;
  AttrValue value5;
  tensorflow::TensorShapeProto tshape5;
  tshape5.add_dim()->set_size(1);
  *value5.mutable_shape() = tshape5;
  (*n5->mutable_attr())["_output_shapes"] = value5;

  NodeDef* n6 = graph_def.add_node();
  n6->set_name("G");
  n6->set_op("G");
  (*n6->mutable_attr())["T"] = values_type_attr;
  AttrValue value6;
  tensorflow::TensorShapeProto* tshape6 =
      value6.mutable_list()->add_shape();
  tshape6->add_dim()->set_size(1);
  tensorflow::TensorShapeProto* tshape61 =
      value6.mutable_list()->add_shape();
  tshape61->add_dim()->set_size(1);
  (*n6->mutable_attr())["_output_shapes"] = value6;

  n3->add_input("A:1");
  n6->add_input("A");
  n4->add_input("B");
  n5->add_input("C");
  n5->add_input("D");
  n5->add_input("E");

  MetaGraphDef mgdef;
  *mgdef.mutable_graph_def() = graph_def;
  auto sdef_map = mgdef.mutable_signature_def();
  SignatureDef sdef;

  TensorInfo tinfo_A0;
  tinfo_A0.set_name("A:0");
  tinfo_A0.set_dtype(DT_FLOAT);
  tinfo_A0.mutable_tensor_shape()->add_dim()->set_size(-1);
  (*sdef.mutable_inputs())["in_A0"] = tinfo_A0;

  TensorInfo tinfo_A1;
  tinfo_A1.set_name("A:1");
  tinfo_A1.set_dtype(DT_FLOAT);
  tinfo_A1.mutable_tensor_shape()->add_dim()->set_size(1);
  (*sdef.mutable_inputs())["in_A1"] = tinfo_A1;

  TensorInfo tinfo_B;
  tinfo_B.set_name("B:0");
  tinfo_B.set_dtype(DT_FLOAT);
  tinfo_B.mutable_tensor_shape()->add_dim()->set_size(-1);
  (*sdef.mutable_inputs())["in_B"] = tinfo_B;

  TensorInfo tinfo_C;
  tinfo_C.set_name("C:0");
  tinfo_C.set_dtype(DT_FLOAT);
  tinfo_C.mutable_tensor_shape()->add_dim()->set_size(1);
  (*sdef.mutable_inputs())["in_C"] = tinfo_C;

  TensorInfo tinfo_F;
  tinfo_F.set_name("F:0");
  tinfo_F.set_dtype(DT_FLOAT);
  tinfo_F.mutable_tensor_shape()->add_dim()->set_size(1);
  (*sdef.mutable_outputs())["out_F"] = tinfo_F;

  TensorInfo tinfo_G0;
  tinfo_G0.set_name("G:0");
  tinfo_G0.set_dtype(DT_FLOAT);
  tinfo_G0.mutable_tensor_shape()->add_dim()->set_size(1);
  (*sdef.mutable_outputs())["out_G0"] = tinfo_G0;

  TensorInfo tinfo_G1;
  tinfo_G1.set_name("G:1");
  tinfo_G1.set_dtype(DT_FLOAT);
  tinfo_G1.mutable_tensor_shape()->add_dim()->set_size(1);
  (*sdef.mutable_outputs())["out_G1"] = tinfo_G1;

  (*sdef_map)["serving_default"] = sdef;

  std::map<std::string, SignatureDef> dynamic_sdef;
  std::map<std::string, SignatureDef> static_sdef;
  StaticShapeCluteringStrategy sscs;
  sscs.GetDynamicAndStaticSignatureDef(mgdef, &dynamic_sdef, &static_sdef);

  EXPECT_TRUE(dynamic_sdef.find("serving_default") != dynamic_sdef.end());
  EXPECT_TRUE(static_sdef.find("serving_default") != static_sdef.end());

  auto& dyn_input_map = dynamic_sdef["serving_default"].inputs();
  auto& dyn_output_map = dynamic_sdef["serving_default"].outputs();
  auto& sta_input_map = static_sdef["serving_default"].inputs();
  auto& sta_output_map = static_sdef["serving_default"].outputs();

  EXPECT_TRUE(dyn_input_map.find("in_A0") != dyn_input_map.end());
  EXPECT_TRUE(dyn_input_map.find("in_A1") != dyn_input_map.end());
  EXPECT_TRUE(dyn_input_map.find("in_B") != dyn_input_map.end());
  EXPECT_TRUE(dyn_output_map.find("out_G0") != dyn_output_map.end());
  EXPECT_TRUE(dyn_output_map.find("out_G1") != dyn_output_map.end());
  EXPECT_TRUE(dyn_output_map.find("dynamic_sig_outputs_D_0") != dyn_output_map.end());
  EXPECT_TRUE(dyn_output_map.find("dynamic_sig_outputs_E_0") != dyn_output_map.end());

  EXPECT_TRUE(sta_input_map.find("in_C") != sta_input_map.end());
  EXPECT_TRUE(sta_input_map.find("static_sig_inputs_D_0") != sta_input_map.end());
  EXPECT_TRUE(sta_input_map.find("static_sig_inputs_E_0") != sta_input_map.end());
  EXPECT_TRUE(sta_output_map.find("out_F") != sta_output_map.end());
}

namespace {

void TestFeatureNameAttr(NodeDef& n) {
  EXPECT_TRUE(n.attr().find("feature_name") != n.attr().end());
  EXPECT_TRUE(n.attr().find("feature_name_to_id") != n.attr().end());
  EXPECT_TRUE(*(n.mutable_attr()->at("feature_name").mutable_s()) == "var_0");
  EXPECT_TRUE(n.mutable_attr()->at("feature_name_to_id").i() == 0);
}

}

TEST(GraphOptimizerTest, SavedModelOptimize0) {
  GraphDef graph_def;

  NodeDef* n_save_const_0 = graph_def.add_node();
  n_save_const_0->set_name("save/Const");
  n_save_const_0->set_op("Const");
  (*n_save_const_0->mutable_attr())["dtype"].set_type(DT_STRING);
 
  NodeDef* n_tensor_name_0 = graph_def.add_node();
  n_tensor_name_0->set_name("tensor_names_0");
  n_tensor_name_0->set_op("Const");
  (*n_tensor_name_0->mutable_attr())["dtype"].set_type(DT_STRING);

  NodeDef* n_shape_slice_0 = graph_def.add_node();
  n_shape_slice_0->set_name("shape_and_slices_0");
  n_shape_slice_0->set_op("Const");
  (*n_shape_slice_0->mutable_attr())["dtype"].set_type(DT_STRING);

  NodeDef* n_restore_0 = graph_def.add_node();
  n_restore_0->set_name("RestoreV2_0");
  n_restore_0->set_op("RestoreV2");
  DataTypeVector dtypes0;
  dtypes0.push_back(DT_INT64);
  AttrValue attr_value0;
  SetAttrValue(dtypes0, &attr_value0);
  (*n_restore_0->mutable_attr())["dtypes"] = attr_value0;
  n_restore_0->add_input("save/Const");
  n_restore_0->add_input("tensor_names_0");
  n_restore_0->add_input("shape_and_slices_0");

  NodeDef* n_tensor_name_1 = graph_def.add_node();
  n_tensor_name_1->set_name("tensor_names_1");
  n_tensor_name_1->set_op("Const");
  (*n_tensor_name_1->mutable_attr())["dtype"].set_type(DT_STRING);

  NodeDef* n_shape_slice_1 = graph_def.add_node();
  n_shape_slice_1->set_name("shape_and_slices_1");
  n_shape_slice_1->set_op("Const");
  (*n_shape_slice_1->mutable_attr())["dtype"].set_type(DT_STRING);

  NodeDef* n_restore_1 = graph_def.add_node();
  n_restore_1->set_name("RestoreV2_1");
  n_restore_1->set_op("RestoreV2");
  DataTypeVector dtypes1;
  dtypes1.push_back(DT_FLOAT);
  AttrValue attr_value1;
  SetAttrValue(dtypes1, &attr_value1);
  (*n_restore_1->mutable_attr())["dtypes"] = attr_value1;
  n_restore_1->add_input("save/Const");
  n_restore_1->add_input("tensor_names_1");
  n_restore_1->add_input("shape_and_slices_1");

  NodeDef* n_var_0 = graph_def.add_node();
  n_var_0->set_name("var_0");
  n_var_0->set_op("KvVarHandleOp");
  AttrValue value_shape;
  tensorflow::TensorShapeProto tshape_proto;
  tshape_proto.add_dim()->set_size(1);
  *value_shape.mutable_shape() = tshape_proto;
  (*n_var_0->mutable_attr())["shape"] = value_shape;


  // KvResourceImportV2

  NodeDef* n_prefix_const_0 = graph_def.add_node();
  n_prefix_const_0->set_name("prefix/Const");
  n_prefix_const_0->set_op("Const");
  (*n_prefix_const_0->mutable_attr())["dtype"].set_type(DT_STRING);
  auto* attr = n_prefix_const_0->mutable_attr();
  Tensor const_tensor(DT_STRING, TensorShape({1}));
  const_tensor.vec<std::string>()(0) = "AAAAAAAAAAAAAAAAAAAAA";
  const_tensor.AsProtoTensorContent((*attr)["value"].mutable_tensor());

  NodeDef* n_value_const_0 = graph_def.add_node();
  n_value_const_0->set_name("value/Const");
  n_value_const_0->set_op("Const");
  (*n_value_const_0->mutable_attr())["dtype"].set_type(DT_FLOAT);

  NodeDef* n_tsname_const_0 = graph_def.add_node();
  n_tsname_const_0->set_name("tsname/Const");
  n_tsname_const_0->set_op("Const");
  (*n_tsname_const_0->mutable_attr())["dtype"].set_type(DT_STRING);

  NodeDef* n_ek_const_0 = graph_def.add_node();
  n_ek_const_0->set_name("empty_key/Const");
  n_ek_const_0->set_op("Const");
  (*n_ek_const_0->mutable_attr())["dtype"].set_type(DT_INT64);

  NodeDef* n_slot_const_0 = graph_def.add_node();
  n_slot_const_0->set_name("slotnum/Const");
  n_slot_const_0->set_op("Const");
  (*n_slot_const_0->mutable_attr())["dtype"].set_type(DT_INT64);

  NodeDef* n_lookup_import_0 = graph_def.add_node();
  n_lookup_import_0->set_name("KvResourceImportV2_0");
  n_lookup_import_0->set_op("KvResourceImportV2");
  (*n_lookup_import_0->mutable_attr())["Tkeys"].set_type(DT_INT64);
  (*n_lookup_import_0->mutable_attr())["dtype"].set_type(DT_FLOAT);
  AttrValue value0;
  tensorflow::TensorShapeProto tshape0;
  tshape0.add_dim()->set_size(-1);
  *value0.mutable_shape() = tshape0;
  (*n_lookup_import_0->mutable_attr())["shape"] = value0;
  n_lookup_import_0->add_input("prefix/Const");
  n_lookup_import_0->add_input("var_0");
  n_lookup_import_0->add_input("var_0");
  n_lookup_import_0->add_input("value/Const");
  n_lookup_import_0->add_input("tsname/Const");
  n_lookup_import_0->add_input("empty_key/Const");
  n_lookup_import_0->add_input("slotnum/Const");
  
  NodeDef* n_restore_shard = graph_def.add_node();
  n_restore_shard->set_name("save/restore_shard");
  n_restore_shard->set_op("NoOp");
  n_restore_shard->add_input("^KvResourceImportV2_0");

  NodeDef* n_restore_all = graph_def.add_node();
  n_restore_all->set_name("save/restore_all");
  n_restore_all->set_op("NoOp");
  n_restore_all->add_input("^save/restore_shard");

  // KvResourceGather

  REGISTER_OP("FakeUnique")
      .Output("y: int64")
      .Output("idx: int32");
 
  NodeDef* n_default_const_0 = graph_def.add_node();
  n_default_const_0->set_name("default/Const");
  n_default_const_0->set_op("Const");
  (*n_default_const_0->mutable_attr())["dtype"].set_type(DT_FLOAT);

  NodeDef* n_unique_0 = graph_def.add_node();
  n_unique_0->set_name("Uniue");
  n_unique_0->set_op("FakeUnique");
 
  NodeDef* n_lookup_find_0 = graph_def.add_node();
  n_lookup_find_0->set_name("KvResourceGather_0");
  n_lookup_find_0->set_op("KvResourceGather");
  (*n_lookup_find_0->mutable_attr())["Tkeys"].set_type(DT_INT64);
  (*n_lookup_find_0->mutable_attr())["dtype"].set_type(DT_FLOAT);
  n_lookup_find_0->add_input("var_0");
  n_lookup_find_0->add_input("Uniue");
  n_lookup_find_0->add_input("default/Const");

  // Fake signature def

  SaverDef saver_def;
  saver_def.set_restore_op_name("save/restore_all");

  SavedModelBundle saved_model_bundle;
  *(saved_model_bundle.meta_graph_def.mutable_graph_def()) = graph_def;
  *(saved_model_bundle.meta_graph_def.mutable_saver_def()) = saver_def;
  auto sdef_map = saved_model_bundle.meta_graph_def.mutable_signature_def();

  SignatureDef sdef;
  TensorInfo tinfo_A0;
  tinfo_A0.set_name("A:0");
  tinfo_A0.set_dtype(DT_FLOAT);
  tinfo_A0.mutable_tensor_shape()->add_dim()->set_size(-1);
  (*sdef.mutable_inputs())["in_A0"] = tinfo_A0;
  TensorInfo tinfo_F;
  tinfo_F.set_name("F:0");
  tinfo_F.set_dtype(DT_FLOAT);
  tinfo_F.mutable_tensor_shape()->add_dim()->set_size(1);
  (*sdef.mutable_outputs())["out_F"] = tinfo_F;
  (*sdef_map)["serving_default"] = sdef;

  SignatureDef sdef2;
  TensorInfo tinfo_A02;
  tinfo_A02.set_name("A:0");
  tinfo_A02.set_dtype(DT_FLOAT);
  tinfo_A02.mutable_tensor_shape()->add_dim()->set_size(-1);
  (*sdef2.mutable_inputs())["in_A0"] = tinfo_A0;
  TensorInfo tinfo_F2;
  tinfo_F2.set_name("F:0");
  tinfo_F2.set_dtype(DT_FLOAT);
  tinfo_F2.mutable_tensor_shape()->add_dim()->set_size(1);
  (*sdef2.mutable_outputs())["out_F"] = tinfo_F2;
  (*sdef_map)["serving_x"] = sdef2;

  // testing all ...

  GraphOptimizerOption option;
  SavedModelOptimizer opt("serving_x",
                          &saved_model_bundle.meta_graph_def,
                          option);
  opt.Optimize();

  std::unordered_map<std::string, NodeDef> nodes;
  GraphDef new_graph_def = saved_model_bundle.meta_graph_def.graph_def();
  for (auto n : new_graph_def.node()) {
    EXPECT_TRUE(nodes.find(n.name()) == nodes.end());
    nodes[n.name()] = n;
  }

  EXPECT_TRUE(nodes.find("save/restore_all/Kv_all") != nodes.end());
  EXPECT_TRUE(nodes.find("save/restore_all/Dense_all") != nodes.end());

  EXPECT_TRUE(nodes.find("KvResourceImportV2_0") != nodes.end());
  auto n = nodes["KvResourceImportV2_0"];
  EXPECT_TRUE(n.op() == "KvImport");
  TestFeatureNameAttr(n);

  EXPECT_TRUE(nodes.find("KvResourceGather_0") != nodes.end());
  n = nodes["KvResourceGather_0"];
  EXPECT_TRUE(n.op() == "KvLookup");
  TestFeatureNameAttr(n);

  EXPECT_TRUE(saved_model_bundle.meta_graph_def.signature_def().size() == 2);
  for (auto sdef : saved_model_bundle.meta_graph_def.signature_def()) {
    EXPECT_TRUE(sdef.first == "serving_x" || sdef.first == GetInitDefKey());
  }

  std::string init_op_name;
  for (auto sdef : saved_model_bundle.meta_graph_def.signature_def()) {
    if (sdef.first == tensorflow::processor::GetInitDefKey()) {
      TensorInfo tinfo;
      for (auto output : sdef.second.outputs()) {
        if (output.first == "init_op") {
          tinfo = output.second;
          break;
        }
      }
      init_op_name = tinfo.name();
      int offset = init_op_name.find(":");
      init_op_name = init_op_name.substr(0, offset);
      break;
    }
  }
  EXPECT_TRUE(init_op_name == "GlobalODL/KvInit");
}

/*
              KvVarHandleOp
       Assign  /        \    ...
           \  /          \    /
         KvResource   KvResource
          ImportV2      Gather
             |              \
             |               \
        NoOp(save/restore)   Identity
   
                   ||
                   ||
                   \/

              KvVarHandleOp
       Assign  /        \    ...
           \  /          \    /
         KvResource     KvResource
          ImportV2        Gather
           /   \              \
          /     \              \
       NoOp     NoOp         Identity  
(restore_delta) (restore_all) 

*/
TEST(GraphOptimizerTest, NativeGraphOptimizerOptimize0) {
  GraphDef graph_def;

  NodeDef* n_prefix_const_0 = graph_def.add_node();
  n_prefix_const_0->set_name("prefix/Const");
  n_prefix_const_0->set_op("Const");
  (*n_prefix_const_0->mutable_attr())["dtype"].set_type(DT_STRING);
  auto* attr = n_prefix_const_0->mutable_attr();
  Tensor const_tensor(DT_STRING, TensorShape({1}));
  const_tensor.vec<std::string>()(0) = "AAAAAAAAAAAAAAAAAAAAA";
  const_tensor.AsProtoTensorContent((*attr)["value"].mutable_tensor());


  NodeDef* n_value_const_0 = graph_def.add_node();
  n_value_const_0->set_name("value/Const");
  n_value_const_0->set_op("Const");
  (*n_value_const_0->mutable_attr())["dtype"].set_type(DT_FLOAT);

  NodeDef* n_tsname_const_0 = graph_def.add_node();
  n_tsname_const_0->set_name("tsname/Const");
  n_tsname_const_0->set_op("Const");
  (*n_tsname_const_0->mutable_attr())["dtype"].set_type(DT_STRING);
  auto* attr1 = n_tsname_const_0->mutable_attr();
  Tensor const_tensor1(DT_STRING, TensorShape({1}));
  const_tensor1.vec<std::string>()(0) = "BBBBBBBBBBBBBBBBBBBB";
  const_tensor1.AsProtoTensorContent((*attr1)["value"].mutable_tensor());

  NodeDef* n_ek_const_0 = graph_def.add_node();
  n_ek_const_0->set_name("empty_key/Const");
  n_ek_const_0->set_op("Const");
  (*n_ek_const_0->mutable_attr())["dtype"].set_type(DT_INT64);
 
  NodeDef* n_slot_const_0 = graph_def.add_node();
  n_slot_const_0->set_name("slotnum0/Const");
  n_slot_const_0->set_op("Const");
  (*n_slot_const_0->mutable_attr())["dtype"].set_type(DT_INT64);
 
  // KvVarHandle
  NodeDef* n_var_0 = graph_def.add_node();
  n_var_0->set_name("var_0");
  n_var_0->set_op("KvVarHandleOp");
  AttrValue value_shape;
  tensorflow::TensorShapeProto tshape_proto;
  tshape_proto.add_dim()->set_size(1);
  *value_shape.mutable_shape() = tshape_proto;
  (*n_var_0->mutable_attr())["shape"] = value_shape;
  tensorflow::AttrValue container_attr;
  *container_attr.mutable_s() = "A";
  (*n_var_0->mutable_attr())["container"] = container_attr;
  tensorflow::AttrValue shared_name_attr;
  *shared_name_attr.mutable_s() = "B/feature";
  (*n_var_0->mutable_attr())["shared_name"] = shared_name_attr;
  (*n_var_0->mutable_attr())["dtype"].set_type(DT_FLOAT);
  (*n_var_0->mutable_attr())["Tkeys"].set_type(DT_INT64);
  AttrValue var_value0;
  tensorflow::TensorShapeProto var_shape0;
  var_shape0.add_dim()->set_size(1);
  *var_value0.mutable_shape() = var_shape0;
  (*n_var_0->mutable_attr())["shape"] = var_value0;

  // KvResourceImportV2
  NodeDef* n_lookup_import_0 = graph_def.add_node();
  n_lookup_import_0->set_name("KvResourceImportV2_0");
  n_lookup_import_0->set_op("KvResourceImportV2");
  (*n_lookup_import_0->mutable_attr())["Tkeys"].set_type(DT_INT64);
  (*n_lookup_import_0->mutable_attr())["dtype"].set_type(DT_FLOAT);
  AttrValue value0;
  tensorflow::TensorShapeProto tshape0;
  tshape0.add_dim()->set_size(-1);
  *value0.mutable_shape() = tshape0;
  (*n_lookup_import_0->mutable_attr())["shape"] = value0;
  n_lookup_import_0->add_input("prefix/Const");
  n_lookup_import_0->add_input("var_0");
  n_lookup_import_0->add_input("var_0");
  n_lookup_import_0->add_input("value/Const");
  n_lookup_import_0->add_input("tsname/Const");
  n_lookup_import_0->add_input("empty_key/Const");
  n_lookup_import_0->add_input("slotnum0/Const");
  n_lookup_import_0->add_input("^prefix/Const");

  REGISTER_OP("FakeAssign")
      .Output("x: int32");
  NodeDef* fake_assign_0 = graph_def.add_node();
  fake_assign_0->set_name("FakeAssign");
  fake_assign_0->set_op("FakeAssign");
 
  // NoOp
  NodeDef* n_noop_0 = graph_def.add_node(); 
  n_noop_0->set_name("save/restore");
  n_noop_0->set_op("NoOp");
  n_noop_0->add_input("^KvResourceImportV2_0");
  n_noop_0->add_input("^FakeAssign");

  NodeDef* n_restore_all = graph_def.add_node();
  n_restore_all->set_name("save/restore_all");
  n_restore_all->set_op("NoOp");
  n_restore_all->add_input("^save/restore");
 
  // KvResourceGather
  NodeDef* n_default_const_0 = graph_def.add_node();
  n_default_const_0->set_name("default/Const");
  n_default_const_0->set_op("Const");
  (*n_default_const_0->mutable_attr())["dtype"].set_type(DT_FLOAT);

  NodeDef* n_unique_0 = graph_def.add_node();
  n_unique_0->set_name("Uniue");
  n_unique_0->set_op("FakeUnique");
 
  NodeDef* n_lookup_find_0 = graph_def.add_node();
  n_lookup_find_0->set_name("KvResourceGather_0");
  n_lookup_find_0->set_op("KvResourceGather");
  (*n_lookup_find_0->mutable_attr())["Tkeys"].set_type(DT_INT64);
  (*n_lookup_find_0->mutable_attr())["dtype"].set_type(DT_FLOAT);
  n_lookup_find_0->add_input("var_0");
  n_lookup_find_0->add_input("Uniue");
  n_lookup_find_0->add_input("default/Const");
  n_lookup_find_0->add_input("^var_0");

  SaverDef saver_def;
  saver_def.set_restore_op_name("save/restore_all");

  // Identity
  NodeDef* n_identity_0 = graph_def.add_node();
  n_identity_0->set_name("lookup/Identity");
  n_identity_0->set_op("Identity");
  (*n_identity_0->mutable_attr())["T"].set_type(DT_FLOAT);
  n_identity_0->add_input("KvResourceGather_0");

  SavedModelBundle saved_model_bundle;
  *(saved_model_bundle.meta_graph_def.mutable_graph_def()) = graph_def;
  *(saved_model_bundle.meta_graph_def.mutable_saver_def()) = saver_def;

  GraphOptimizerOption option;
  option.native_tf_mode = true;
  SavedModelOptimizer opt("serving_default",
                          &saved_model_bundle.meta_graph_def,
                          option);
  opt.Optimize();

  GraphDef new_graph_def = saved_model_bundle.meta_graph_def.graph_def();
  std::unordered_map<std::string, NodeDef> nodes;
  int node_count = 0;
  for (auto n : new_graph_def.node()) {
    nodes[n.name()] = n;
    ++node_count;
  }
  EXPECT_TRUE(node_count == 14);
}

/*
                KvVarHandleOp
         ......  /        \   unique default_value
             \  /          \    /   / 
            KvResource     KvResource
    assign   ImportV2        Gather
       \        |               \
        \       |                \
        NoOp(save/restore_shard) Identity
                |
       NoOp(save/restore_all)

                   ||
                   ||
                   \/

  ==> curr_id = 1, total_instance = 3

              KvVarHandleOp/part_1
          ......  /        \   unique default_value
              \  /          \    /    /
  assign   KvResource      part_1/KvResource
     \ \    ImportV2         Gather
      \ \    /   \              \
       \ -- /--\  \              \
        NoOp     NoOp         Identity  
(restore_shard) (restore_delta) 
         |
 NoOp(save/restore_all)

*/
TEST(GraphOptimizerTest, ShardEmbeddingOptimizer1) {
  GraphDef graph_def;

  NodeDef* n_prefix_const_0 = graph_def.add_node();
  n_prefix_const_0->set_name("prefix/Const");
  n_prefix_const_0->set_op("Const");
  (*n_prefix_const_0->mutable_attr())["dtype"].set_type(DT_STRING);
  auto* attr = n_prefix_const_0->mutable_attr();
  Tensor const_tensor(DT_STRING, TensorShape({1}));
  const_tensor.vec<std::string>()(0) = "AAAAAAAAAAAAAAAAAAAAA";
  const_tensor.AsProtoTensorContent((*attr)["value"].mutable_tensor());

  NodeDef* n_value_const_0 = graph_def.add_node();
  n_value_const_0->set_name("value/Const");
  n_value_const_0->set_op("Const");
  (*n_value_const_0->mutable_attr())["dtype"].set_type(DT_FLOAT);

  NodeDef* n_tsname_const_0 = graph_def.add_node();
  n_tsname_const_0->set_name("tsname/Const");
  n_tsname_const_0->set_op("Const");
  (*n_tsname_const_0->mutable_attr())["dtype"].set_type(DT_STRING);
  auto* attr1 = n_tsname_const_0->mutable_attr();
  Tensor const_tensor1(DT_STRING, TensorShape({1}));
  const_tensor1.vec<std::string>()(0) = "BBBBBBBBBBBBBBBBBBBB";
  const_tensor1.AsProtoTensorContent((*attr1)["value"].mutable_tensor());

  NodeDef* n_ek_const_0 = graph_def.add_node();
  n_ek_const_0->set_name("empty_key/Const");
  n_ek_const_0->set_op("Const");
  (*n_ek_const_0->mutable_attr())["dtype"].set_type(DT_INT64);

  // KvVarHandle
  NodeDef* n_var_0 = graph_def.add_node();
  n_var_0->set_name("var_0");
  n_var_0->set_op("KvVarHandleOp");
  AttrValue value_shape;
  tensorflow::TensorShapeProto tshape_proto;
  tshape_proto.add_dim()->set_size(1);
  *value_shape.mutable_shape() = tshape_proto;
  (*n_var_0->mutable_attr())["shape"] = value_shape;
  tensorflow::AttrValue container_attr;
  *container_attr.mutable_s() = "A";
  (*n_var_0->mutable_attr())["container"] = container_attr;
  tensorflow::AttrValue shared_name_attr;
  *shared_name_attr.mutable_s() = "B/feature";
  (*n_var_0->mutable_attr())["shared_name"] = shared_name_attr;
  (*n_var_0->mutable_attr())["dtype"].set_type(DT_FLOAT);
  (*n_var_0->mutable_attr())["Tkeys"].set_type(DT_INT64);
  AttrValue var_value0;
  tensorflow::TensorShapeProto var_shape0;
  var_shape0.add_dim()->set_size(1);
  *var_value0.mutable_shape() = var_shape0;
  (*n_var_0->mutable_attr())["shape"] = var_value0;

  NodeDef* n_slot_const_0 = graph_def.add_node();
  n_slot_const_0->set_name("slotnum0/Const");
  n_slot_const_0->set_op("Const");
  (*n_slot_const_0->mutable_attr())["dtype"].set_type(DT_INT64);

  // KvResourceImportV2
  NodeDef* n_lookup_import_0 = graph_def.add_node();
  n_lookup_import_0->set_name("KvResourceImportV2_0");
  n_lookup_import_0->set_op("KvResourceImportV2");
  (*n_lookup_import_0->mutable_attr())["Tkeys"].set_type(DT_INT64);
  (*n_lookup_import_0->mutable_attr())["dtype"].set_type(DT_FLOAT);
  AttrValue value0;
  tensorflow::TensorShapeProto tshape0;
  tshape0.add_dim()->set_size(-1);
  *value0.mutable_shape() = tshape0;
  (*n_lookup_import_0->mutable_attr())["shape"] = value0;
  n_lookup_import_0->add_input("prefix/Const");
  n_lookup_import_0->add_input("var_0");
  n_lookup_import_0->add_input("var_0");
  n_lookup_import_0->add_input("value/Const");
  n_lookup_import_0->add_input("tsname/Const");
  n_lookup_import_0->add_input("empty_key/Const");
  n_lookup_import_0->add_input("slotnum0/Const");
  n_lookup_import_0->add_input("^prefix/Const");

  REGISTER_OP("MyFakeAssign")
      .Output("x: int32");
  NodeDef* fake_assign_0 = graph_def.add_node();
  fake_assign_0->set_name("FakeAssign");
  fake_assign_0->set_op("MyFakeAssign");
 
  // restore_shard/NoOp
  NodeDef* n_noop_0 = graph_def.add_node(); 
  n_noop_0->set_name("save/restore");
  n_noop_0->set_op("NoOp");
  n_noop_0->add_input("^KvResourceImportV2_0");
  n_noop_0->add_input("^FakeAssign");

  // restore_all/NoOp
  NodeDef* n_restore_all = graph_def.add_node();
  n_restore_all->set_name("save/restore_all");
  n_restore_all->set_op("NoOp");
  n_restore_all->add_input("^save/restore");

   // KvResourceGather
  NodeDef* n_default_const_0 = graph_def.add_node();
  n_default_const_0->set_name("default/Const");
  n_default_const_0->set_op("Const");
  (*n_default_const_0->mutable_attr())["dtype"].set_type(DT_FLOAT);

  NodeDef* n_unique_0 = graph_def.add_node();
  n_unique_0->set_name("Uniue");
  n_unique_0->set_op("FakeUnique");
 
  NodeDef* n_lookup_find_0 = graph_def.add_node();
  n_lookup_find_0->set_name("KvResourceGather_0");
  n_lookup_find_0->set_op("KvResourceGather");
  (*n_lookup_find_0->mutable_attr())["Tkeys"].set_type(DT_INT64);
  (*n_lookup_find_0->mutable_attr())["dtype"].set_type(DT_FLOAT);
  n_lookup_find_0->add_input("var_0");
  n_lookup_find_0->add_input("Uniue");
  n_lookup_find_0->add_input("default/Const");
  n_lookup_find_0->add_input("^var_0");

  // Identity
  NodeDef* n_identity_0 = graph_def.add_node();
  n_identity_0->set_name("lookup/Identity");
  n_identity_0->set_op("Identity");
  (*n_identity_0->mutable_attr())["T"].set_type(DT_FLOAT);
  n_identity_0->add_input("KvResourceGather_0");

  SaverDef saver_def;
  saver_def.set_restore_op_name("save/restore_all");

  SavedModelBundle saved_model_bundle;
  *(saved_model_bundle.meta_graph_def.mutable_graph_def()) = graph_def;
  *(saved_model_bundle.meta_graph_def.mutable_saver_def()) = saver_def;

  GraphOptimizerOption option;
  option.native_tf_mode = true;
  // shard user embedding or not
  option.shard_embedding = true;
  // which user embeddings will be sharded
  option.shard_embedding_names.push_back("var_0");
  // current instance partition id
  option.partition_id = 1;
  option.shard_instance_count = 3;

  SavedModelOptimizer opt("serving_default",
                          &saved_model_bundle.meta_graph_def,
                          option);
  opt.Optimize();

  GraphDef new_graph_def = saved_model_bundle.meta_graph_def.graph_def();

  std::unordered_map<std::string, NodeDef> nodes;
  int node_count = 0;
  for (auto n : new_graph_def.node()) {
    nodes[n.name()] = n;
    ++node_count;
  }

  EXPECT_TRUE(node_count == 16);
  EXPECT_TRUE(nodes.find("var_0/odl_var_part/part_1/KvResourceImportV2") != nodes.end());
  EXPECT_TRUE(nodes.find("save/restore_all/Kv_incr_all") != nodes.end());
  EXPECT_TRUE(nodes.find("var_0/odl_var_part/part_1/tensor_names") != nodes.end());
  EXPECT_TRUE(nodes.find("var_0/odl_var_part/part_1") != nodes.end());
 
  EXPECT_TRUE(1);
}

/*
                                     unique
                                       |
        --------------------------------------------------------
        |                                                      |  
        |               KvVarHandleOp/    KvVarHandleOp/       |
    ...  ...               part_0            part_1        ...   ...    
     \    /                  / \             /    \          \   /
 -- dynamic                 /   \           /      \        dynamic--------------------      
 |  patition default_value /     \         /        \      patition:1  default_value  |
 |     \       |          /       \       /          \        /          /            |
 |       KvResourceGather   KvResource  KvResource     KvResourceGather_1             |
 |             |             ImportV2   ImportV2_1             |                      |
 |          Identity             \          /              Identity_1                 |
 |             |                  \        /                   |                      |
 |             |          NoOp(save/restore_shard)             |                      |
 |             |                      |                        |                      |
 |             |            NoOp(save/restore_all)             |                      |
 |             |                                               |                      |
 |             |                                               |                      |
 |             |                                               |                      |
 |----------------------> ParallelDynamicStitch <--------------------------------------
                                     |
                                  Reshape
                                     |
                                  AAAAAAA

                                    ||
                                    ||
                                    \/

  ==> curr_id = 0, total_instance = 3  (partition 0)
  ==> curr_id = 1, total_instance = 3  (partition 1)
  ==> curr_id = 2, total_instance = 3  (partition 2)

              KvVarHandleOp/part_2
          ......  /        \   unique default_value
              \  /          \    /    /
           KvResource      part_2/KvResource
           ImportV2_2         Gather
             /   \              \
            /     \              \
        NoOp     NoOp            part_2/Identity  
(restore_shard) (restore_delta)        |
         |                          AAAAAAA
 NoOp(save/restore_all)

*/
TEST(GraphOptimizerTest, ShardEmbeddingOptimizer2) {
  GraphDef graph_def;

  NodeDef* n_unique_0 = graph_def.add_node();
  n_unique_0->set_name("Unique");
  n_unique_0->set_op("Unique");
  (*n_unique_0->mutable_attr())["T"].set_type(DT_INT64);
  (*n_unique_0->mutable_attr())["out_idx"].set_type(DT_INT32);

  NodeDef* shape_const_0 = graph_def.add_node();
  shape_const_0->set_name("const_shape");
  shape_const_0->set_op("Const");
  (*shape_const_0->mutable_attr())["dtype"].set_type(DT_INT32);
  auto* shape_const_0_attr = shape_const_0->mutable_attr();
  Tensor shape_const_tensor(DT_INT32, TensorShape({1}));
  shape_const_tensor.vec<int>()(0) = -1;
  shape_const_tensor.AsProtoTensorContent((*shape_const_0_attr)["value"].mutable_tensor());

  NodeDef* reshape_0 = graph_def.add_node();
  reshape_0->set_name("unique_reshape");
  reshape_0->set_op("Reshape");
  (*reshape_0->mutable_attr())["T"].set_type(DT_INT64);
  (*reshape_0->mutable_attr())["Tshape"].set_type(DT_INT32);
  reshape_0->add_input("Unique");
  reshape_0->add_input("const_shape");

  NodeDef* size_0 = graph_def.add_node();
  size_0->set_name("unique_size");
  size_0->set_op("Size");
  (*size_0->mutable_attr())["T"].set_type(DT_INT64);
  (*size_0->mutable_attr())["out_type"].set_type(DT_INT32);
  size_0->add_input("unique_reshape");

  NodeDef* start_const_0 = graph_def.add_node();
  start_const_0->set_name("const_start");
  start_const_0->set_op("Const");
  (*start_const_0->mutable_attr())["dtype"].set_type(DT_INT32);
  auto* start_const_0_attr = start_const_0->mutable_attr();
  Tensor start_const_tensor(DT_INT32, TensorShape({}));
  start_const_tensor.scalar<int>()() = 0;
  start_const_tensor.AsProtoTensorContent((*start_const_0_attr)["value"].mutable_tensor());

  NodeDef* delta_const_0 = graph_def.add_node();
  delta_const_0->set_name("const_delta");
  delta_const_0->set_op("Const");
  (*delta_const_0->mutable_attr())["dtype"].set_type(DT_INT32);
  auto* delta_const_0_attr = delta_const_0->mutable_attr();
  Tensor delta_const_tensor(DT_INT32, TensorShape({}));
  delta_const_tensor.scalar<int>()() = 1;
  delta_const_tensor.AsProtoTensorContent((*delta_const_0_attr)["value"].mutable_tensor());

  NodeDef* range_0 = graph_def.add_node();
  range_0->set_name("range");
  range_0->set_op("Range");
  (*range_0->mutable_attr())["Tidx"].set_type(DT_INT32);
  range_0->add_input("const_start");
  range_0->add_input("unique_size");
  range_0->add_input("const_delta");

  NodeDef* cast_1 = graph_def.add_node();
  cast_1->set_name("cast1");
  cast_1->set_op("Cast");
  (*cast_1->mutable_attr())["DstT"].set_type(DT_INT64);
  (*cast_1->mutable_attr())["SrcT"].set_type(DT_INT32);
  (*cast_1->mutable_attr())["Truncate"].set_b(false);
  cast_1->add_input("range");

  NodeDef* cast_0 = graph_def.add_node();
  cast_0->set_name("cast");
  cast_0->set_op("Cast");
  (*cast_0->mutable_attr())["DstT"].set_type(DT_INT32);
  (*cast_0->mutable_attr())["SrcT"].set_type(DT_INT64);
  (*cast_0->mutable_attr())["Truncate"].set_b(false);

  // create DynamicPartition
  NodeDef* dynamic_partition_0 = graph_def.add_node();
  dynamic_partition_0->set_name("dynamic_partition");
  dynamic_partition_0->set_op("DynamicPartition");
  (*dynamic_partition_0->mutable_attr())["T"].set_type(DT_INT64);
  (*dynamic_partition_0->mutable_attr())["num_partitions"].set_i(2);
  dynamic_partition_0->add_input("cast1");
  dynamic_partition_0->add_input("cast");

  NodeDef* dynamic_partition_cast = graph_def.add_node();
  dynamic_partition_cast->set_name("dynamic_partition_cast");
  dynamic_partition_cast->set_op("Cast");
  (*dynamic_partition_cast->mutable_attr())["DstT"].set_type(DT_INT32);
  (*dynamic_partition_cast->mutable_attr())["SrcT"].set_type(DT_INT64);
  (*dynamic_partition_cast->mutable_attr())["Truncate"].set_b(false);
  dynamic_partition_cast->add_input("dynamic_partition");

  NodeDef* dynamic_partition_cast1 = graph_def.add_node();
  dynamic_partition_cast1->set_name("dynamic_partition_cast_2");
  dynamic_partition_cast1->set_op("Cast");
  (*dynamic_partition_cast1->mutable_attr())["DstT"].set_type(DT_INT32);
  (*dynamic_partition_cast1->mutable_attr())["SrcT"].set_type(DT_INT64);
  (*dynamic_partition_cast1->mutable_attr())["Truncate"].set_b(false);
  dynamic_partition_cast1->add_input("dynamic_partition:1");

  // ----------------------------------------------
 
  NodeDef* n_prefix_const_0 = graph_def.add_node();
  n_prefix_const_0->set_name("prefix/Const");
  n_prefix_const_0->set_op("Const");
  (*n_prefix_const_0->mutable_attr())["dtype"].set_type(DT_STRING);
  auto* attr = n_prefix_const_0->mutable_attr();
  Tensor const_tensor(DT_STRING, TensorShape({1}));
  const_tensor.vec<std::string>()(0) = "AAAAAAAAAAAAAAAAAAAAA/part_0";
  const_tensor.AsProtoTensorContent((*attr)["value"].mutable_tensor());

  NodeDef* n_value_const_0 = graph_def.add_node();
  n_value_const_0->set_name("value/Const");
  n_value_const_0->set_op("Const");
  (*n_value_const_0->mutable_attr())["dtype"].set_type(DT_FLOAT);

  NodeDef* n_tsname_const_0 = graph_def.add_node();
  n_tsname_const_0->set_name("tsname/Const");
  n_tsname_const_0->set_op("Const");
  (*n_tsname_const_0->mutable_attr())["dtype"].set_type(DT_STRING);
  auto* attr1 = n_tsname_const_0->mutable_attr();
  Tensor const_tensor1(DT_STRING, TensorShape({1}));
  const_tensor1.vec<std::string>()(0) = "BBBBBBBBBBBBBBBBBBBB/part_0";
  const_tensor1.AsProtoTensorContent((*attr1)["value"].mutable_tensor());

  NodeDef* n_ek_const_0 = graph_def.add_node();
  n_ek_const_0->set_name("empty_key/Const");
  n_ek_const_0->set_op("Const");
  (*n_ek_const_0->mutable_attr())["dtype"].set_type(DT_INT64);

  // KvVarHandle/part-0
  NodeDef* n_var_0 = graph_def.add_node();
  n_var_0->set_name("var_0/part_0");
  n_var_0->set_op("KvVarHandleOp");
  AttrValue value_shape;
  tensorflow::TensorShapeProto tshape_proto;
  tshape_proto.add_dim()->set_size(1);
  *value_shape.mutable_shape() = tshape_proto;
  (*n_var_0->mutable_attr())["shape"] = value_shape;
  tensorflow::AttrValue container_attr;
  *container_attr.mutable_s() = "A";
  (*n_var_0->mutable_attr())["container"] = container_attr;
  tensorflow::AttrValue shared_name_attr;
  *shared_name_attr.mutable_s() = "var_0/part_0";
  (*n_var_0->mutable_attr())["shared_name"] = shared_name_attr;
  (*n_var_0->mutable_attr())["dtype"].set_type(DT_FLOAT);
  (*n_var_0->mutable_attr())["Tkeys"].set_type(DT_INT64);
  AttrValue var_value0;
  tensorflow::TensorShapeProto var_shape0;
  var_shape0.add_dim()->set_size(1);
  *var_value0.mutable_shape() = var_shape0;
  (*n_var_0->mutable_attr())["shape"] = var_value0;

  NodeDef* n_slot_const_0 = graph_def.add_node();
  n_slot_const_0->set_name("slotnum0/Const");
  n_slot_const_0->set_op("Const");
  (*n_slot_const_0->mutable_attr())["dtype"].set_type(DT_INT64);

  // part_0/KvResourceImportV2
  NodeDef* n_lookup_import_0 = graph_def.add_node();
  n_lookup_import_0->set_name("part_0/KvResourceImportV2");
  n_lookup_import_0->set_op("KvResourceImportV2");
  (*n_lookup_import_0->mutable_attr())["Tkeys"].set_type(DT_INT64);
  (*n_lookup_import_0->mutable_attr())["dtype"].set_type(DT_FLOAT);
  AttrValue value0;
  tensorflow::TensorShapeProto tshape0;
  tshape0.add_dim()->set_size(-1);
  *value0.mutable_shape() = tshape0;
  (*n_lookup_import_0->mutable_attr())["shape"] = value0;
  n_lookup_import_0->add_input("prefix/Const");
  n_lookup_import_0->add_input("var_0/part_0");
  n_lookup_import_0->add_input("var_0/part_0");
  n_lookup_import_0->add_input("value/Const");
  n_lookup_import_0->add_input("tsname/Const");
  n_lookup_import_0->add_input("empty_key/Const");
  n_lookup_import_0->add_input("slotnum0/Const");
  n_lookup_import_0->add_input("^prefix/Const");

  REGISTER_OP("MyFakeAssign2")
      .Output("x: int32");
  NodeDef* fake_assign_0 = graph_def.add_node();
  fake_assign_0->set_name("FakeAssign");
  fake_assign_0->set_op("MyFakeAssign2");

  // part_0/KvResourceGather
  NodeDef* n_default_const_0 = graph_def.add_node();
  n_default_const_0->set_name("default/Const");
  n_default_const_0->set_op("Const");
  (*n_default_const_0->mutable_attr())["dtype"].set_type(DT_FLOAT);

  NodeDef* n_lookup_find_0 = graph_def.add_node();
  n_lookup_find_0->set_name("part_0/KvResourceGather");
  n_lookup_find_0->set_op("KvResourceGather");
  (*n_lookup_find_0->mutable_attr())["Tkeys"].set_type(DT_INT64);
  (*n_lookup_find_0->mutable_attr())["dtype"].set_type(DT_FLOAT);
  n_lookup_find_0->add_input("var_0/part_0");
  n_lookup_find_0->add_input("dynamic_partition");
  n_lookup_find_0->add_input("default/Const");
  n_lookup_find_0->add_input("^var_0/part_0");

  // Identity
  NodeDef* n_identity_0 = graph_def.add_node();
  n_identity_0->set_name("lookup/Identity");
  n_identity_0->set_op("Identity");
  (*n_identity_0->mutable_attr())["T"].set_type(DT_FLOAT);
  n_identity_0->add_input("part_0/KvResourceGather");

  // ----------------------------------------------

  NodeDef* n_value_const_1 = graph_def.add_node();
  n_value_const_1->set_name("value/Const_1");
  n_value_const_1->set_op("Const");
  (*n_value_const_1->mutable_attr())["dtype"].set_type(DT_FLOAT);

  NodeDef* n_tsname_const_1 = graph_def.add_node();
  n_tsname_const_1->set_name("tsname/Const_1");
  n_tsname_const_1->set_op("Const");
  (*n_tsname_const_1->mutable_attr())["dtype"].set_type(DT_STRING);
  auto* attr1_1 = n_tsname_const_1->mutable_attr();
  Tensor const_tensor1_1(DT_STRING, TensorShape({1}));
  const_tensor1_1.vec<std::string>()(0) = "BBBBBBBBBBBBBBBBBBBB/part_1";
  const_tensor1_1.AsProtoTensorContent((*attr1_1)["value"].mutable_tensor());

  NodeDef* n_ek_const_1 = graph_def.add_node();
  n_ek_const_1->set_name("empty_key/Const_1");
  n_ek_const_1->set_op("Const");
  (*n_ek_const_1->mutable_attr())["dtype"].set_type(DT_INT64);

  // KvVarHandle/part-1
  NodeDef* n_var_1 = graph_def.add_node();
  n_var_1->set_name("var_0/part_1");
  n_var_1->set_op("KvVarHandleOp");
  AttrValue value_shape_1;
  tensorflow::TensorShapeProto tshape_proto_1;
  tshape_proto_1.add_dim()->set_size(1);
  *value_shape_1.mutable_shape() = tshape_proto_1;
  (*n_var_1->mutable_attr())["shape"] = value_shape_1;
  tensorflow::AttrValue container_attr_1;
  *container_attr_1.mutable_s() = "A";
  (*n_var_1->mutable_attr())["container"] = container_attr_1;
  tensorflow::AttrValue shared_name_attr_1;
  *shared_name_attr_1.mutable_s() = "var_0/part_1";
  (*n_var_1->mutable_attr())["shared_name"] = shared_name_attr_1;
  (*n_var_1->mutable_attr())["dtype"].set_type(DT_FLOAT);
  (*n_var_1->mutable_attr())["Tkeys"].set_type(DT_INT64);
  AttrValue var_value0_1;
  tensorflow::TensorShapeProto var_shape0_1;
  var_shape0_1.add_dim()->set_size(1);
  *var_value0_1.mutable_shape() = var_shape0_1;
  (*n_var_1->mutable_attr())["shape"] = var_value0_1;

  NodeDef* n_slot_const_1 = graph_def.add_node();
  n_slot_const_1->set_name("slotnum1/Const");
  n_slot_const_1->set_op("Const");
  (*n_slot_const_1->mutable_attr())["dtype"].set_type(DT_INT64);

  // part_1/KvResourceImportV2
  NodeDef* n_lookup_import_1 = graph_def.add_node();
  n_lookup_import_1->set_name("part_1/KvResourceImportV2");
  n_lookup_import_1->set_op("KvResourceImportV2");
  (*n_lookup_import_1->mutable_attr())["Tkeys"].set_type(DT_INT64);
  (*n_lookup_import_1->mutable_attr())["dtype"].set_type(DT_FLOAT);
  AttrValue value0_1;
  tensorflow::TensorShapeProto tshape0_1;
  tshape0_1.add_dim()->set_size(-1);
  *value0_1.mutable_shape() = tshape0_1;
  (*n_lookup_import_1->mutable_attr())["shape"] = value0_1;
  n_lookup_import_1->add_input("prefix/Const");
  n_lookup_import_1->add_input("var_0/part_1");
  n_lookup_import_1->add_input("var_0/part_1");
  n_lookup_import_1->add_input("value/Const_1");
  n_lookup_import_1->add_input("tsname/Const_1");
  n_lookup_import_1->add_input("empty_key/Const_1");
  n_lookup_import_1->add_input("slotnum1/Const");
  n_lookup_import_1->add_input("^prefix/Const");

  // part_1/KvResourceGather
  NodeDef* n_lookup_find_1 = graph_def.add_node();
  n_lookup_find_1->set_name("part_1/KvResourceGather");
  n_lookup_find_1->set_op("KvResourceGather");
  (*n_lookup_find_1->mutable_attr())["Tkeys"].set_type(DT_INT64);
  (*n_lookup_find_1->mutable_attr())["dtype"].set_type(DT_FLOAT);
  n_lookup_find_1->add_input("var_0/part_1");
  n_lookup_find_1->add_input("dynamic_partition:1");
  n_lookup_find_1->add_input("default/Const");
  n_lookup_find_1->add_input("^var_0/part_1");

  // Identity
  NodeDef* n_identity_1 = graph_def.add_node();
  n_identity_1->set_name("lookup/Identity_1");
  n_identity_1->set_op("Identity");
  (*n_identity_1->mutable_attr())["T"].set_type(DT_FLOAT);
  n_identity_1->add_input("part_1/KvResourceGather");

  // ----------------------------------------------

  NodeDef* n_dynamic_stitch = graph_def.add_node(); 
  n_dynamic_stitch->set_name("dynamic_stitch");
  n_dynamic_stitch->set_op("ParallelDynamicStitch");
  (*n_dynamic_stitch->mutable_attr())["N"].set_i(2);
  (*n_dynamic_stitch->mutable_attr())["T"].set_type(DT_FLOAT);
  n_dynamic_stitch->add_input("dynamic_partition_cast");
  n_dynamic_stitch->add_input("dynamic_partition_cast_2");
  n_dynamic_stitch->add_input("lookup/Identity");
  n_dynamic_stitch->add_input("lookup/Identity_1");

  NodeDef* ds_reshape = graph_def.add_node();
  ds_reshape->set_name("ds_reshape");
  ds_reshape->set_op("Reshape");
  (*ds_reshape->mutable_attr())["T"].set_type(DT_FLOAT);
  (*ds_reshape->mutable_attr())["Tshape"].set_type(DT_INT32);
  ds_reshape->add_input("dynamic_stitch");

  NodeDef* n_identity_x = graph_def.add_node();
  n_identity_x->set_name("Identity_x");
  n_identity_x->set_op("Identity");
  (*n_identity_x->mutable_attr())["T"].set_type(DT_FLOAT);
  n_identity_x->add_input("ds_reshape");

  // ----------------------------------------------

  // restore_shard/NoOp
  NodeDef* n_noop_0 = graph_def.add_node(); 
  n_noop_0->set_name("save/restore");
  n_noop_0->set_op("NoOp");
  n_noop_0->add_input("^part_0/KvResourceImportV2");
  n_noop_0->add_input("^part_1/KvResourceImportV2");
  n_noop_0->add_input("^FakeAssign");

  // restore_all/NoOp
  NodeDef* n_restore_all = graph_def.add_node();
  n_restore_all->set_name("save/restore_all");
  n_restore_all->set_op("NoOp");
  n_restore_all->add_input("^save/restore");

  SaverDef saver_def;
  saver_def.set_restore_op_name("save/restore_all");

  // TEST: option.partition_id = 0,1,2
  for (int id = 0; id < 3; ++id)
  {
    SavedModelBundle saved_model_bundle;
    *(saved_model_bundle.meta_graph_def.mutable_graph_def()) = graph_def;
    *(saved_model_bundle.meta_graph_def.mutable_saver_def()) = saver_def;

    GraphOptimizerOption option;
    option.native_tf_mode = true;
    // shard user embedding or not
    option.shard_embedding = true;
    // which user embeddings will be sharded
    option.shard_embedding_names.push_back("var_0");
    // current instance partition id
    option.partition_id = id;
    option.shard_instance_count = 3;

    SavedModelOptimizer opt("serving_default",
                            &saved_model_bundle.meta_graph_def,
                            option);
    opt.Optimize();

    GraphDef new_graph_def = saved_model_bundle.meta_graph_def.graph_def();

    std::unordered_map<std::string, NodeDef> nodes;
    int node_count = 0;
    for (auto n : new_graph_def.node()) {
      nodes[n.name()] = n;
      ++node_count;
    }

    EXPECT_TRUE(node_count == 33);
    EXPECT_TRUE(nodes.find("var_0/part_0") == nodes.end());
    EXPECT_TRUE(nodes.find("var_0/part_1") == nodes.end());
    EXPECT_TRUE(nodes.find("part_0/KvResourceImportV2") == nodes.end());
    EXPECT_TRUE(nodes.find("part_1/KvResourceImportV2") == nodes.end());
    EXPECT_TRUE(nodes.find("part_0/KvResourceGather") == nodes.end());
    EXPECT_TRUE(nodes.find("part_1/KvResourceGather") == nodes.end());
    EXPECT_TRUE(nodes.find("dynamic_stitch") == nodes.end());

    EXPECT_TRUE(nodes.find("var_0/odl_var_part/part_"+std::to_string(id)) != nodes.end());
    EXPECT_TRUE(nodes.find("var_0/odl_var_part/part_"+std::to_string(id)+"/KvResourceGather") != nodes.end());
    EXPECT_TRUE(nodes.find("var_0/odl_var_part/part_"+std::to_string(id)+"/Identity") != nodes.end());
    EXPECT_TRUE(nodes.find("var_0/odl_var_part/part_"+std::to_string(id)+"/KvResourceImportV2") != nodes.end());
    EXPECT_TRUE(nodes.find("save/restore_all/Kv_incr_all") != nodes.end());
    EXPECT_TRUE(nodes.find("var_0/odl_var_part/part_"+std::to_string(id)+"/default_value") != nodes.end());
  }

  EXPECT_TRUE(1);
}

} // namespace processor
} // namespace tensorflow
