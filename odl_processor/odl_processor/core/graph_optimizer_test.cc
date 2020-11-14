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

} // namespace odl_processor
} // namespace tensorflow
