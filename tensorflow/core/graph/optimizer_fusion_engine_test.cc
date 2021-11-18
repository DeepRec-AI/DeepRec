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

#include "tensorflow/core/graph/optimizer_fusion_engine.h"
#include "tensorflow/core/graph/optimizer_fusion_engine_impl.h"

#include <utility>
#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
using namespace grappler;
namespace {

static void InitGraph(const std::string& s, Graph* graph) {
  GraphDef graph_def;

  auto parser = protobuf::TextFormat::Parser();
  //  parser.AllowRelaxedWhitespace(true);
  CHECK(parser.MergeFromString(s, &graph_def)) << s;
  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, graph));
}

class OptimizerFusionTest : public ::testing::Test {
 public:
  OptimizerFusionTest() : graph_(OpRegistry::Global()) {}

  void InitGraph(const std::string& s) {
    ::tensorflow::InitGraph(s, &graph_);
    original_ = CanonicalGraphString(&graph_);
  }

  NodeDef* AddNode(const string& name, const string& op,
      const std::vector<string>& inputs,
      const std::vector<std::pair<string, AttrValue>>& attributes,
      GraphDef* graph) const {
    NodeDef* node = graph->add_node();
    node->set_name(name);
    node->set_op(op);
    for (const string& input : inputs) {
      node->add_input(input);
    }
    for (auto attr : attributes) {
      (*node->mutable_attr())[attr.first] = attr.second;
    }
    return node;
  }

  static bool IncludeNode(const Node* n) { return n->IsOp(); }

  static std::string EdgeId(const Node* n, int index) {
    if (index == 0) {
      return n->name();
    } else if (index == Graph::kControlSlot) {
      return strings::StrCat(n->name(), ":control");
    } else {
      return strings::StrCat(n->name(), ":", index);
    }
  }

  std::string CanonicalGraphString(Graph* g) {
    std::vector<std::string> nodes;
    std::vector<std::string> edges;
    for (const Node* n : g->nodes()) {
      if (IncludeNode(n)) {
        nodes.push_back(strings::StrCat(n->name(), "(", n->type_string(), ")"));
      }
    }
    for (const Edge* e : g->edges()) {
      if (IncludeNode(e->src()) && IncludeNode(e->dst())) {
        edges.push_back(strings::StrCat(EdgeId(e->src(), e->src_output()), "->",
                                        EdgeId(e->dst(), e->dst_input())));
      }
    }
    // Canonicalize
    std::sort(nodes.begin(), nodes.end());
    std::sort(edges.begin(), edges.end());
    return strings::StrCat(str_util::Join(nodes, ";"), "|",
                           str_util::Join(edges, ";"));
  }

  std::string DoFusion(std::function<bool(const Node*)> consider_fn = nullptr) {
    std::string before = CanonicalGraphString(&graph_);
    LOG(ERROR) << "Before rewrites: " << before;

    OptimizeFusion(&graph_);

    std::string result = CanonicalGraphString(&graph_);
    LOG(ERROR) << "After rewrites:  " << result;
    return result;
  }

  const std::string& OriginalGraph() const { return original_; }

  Graph graph_;
  std::string original_;
};

REGISTER_OP("Input").Output("o: float").SetIsStateful();
REGISTER_OP("InputInt64").Output("o: int64").SetIsStateful();
REGISTER_OP("Output").Output("o: float");

TEST_F(OptimizerFusionTest, test_input_is_control_dependency_edge) {
  GraphDef test_graph;
  AttrValue type;
  type.set_type(DT_FLOAT);
  AddNode("begin1", "Const", {}, {{"dtype", type}}, &test_graph);
  AddNode("s1", "Identity", 
      {AsControlDependency("begin1")}, 
      {{"T", type}}, &test_graph);
  AddNode("s2", "Identity", {"s1"}, {{"T", type}}, &test_graph);

  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, test_graph, &graph_));

  class TemplateControlDeps : public TemplateBase {
   public:
    TemplateControlDeps() {
      const TempNode n0 = {
        .key = "begin1",
        .op = "Const",
        .inputs = {},
        .outputs = {{}},
        .deps_inputs = {},
        .deps_outputs = {"s1"}
      };
      temp_nodes_.emplace_back(n0);
      const TempNode n2 = {
        .key = "s1",
        .op = "Identity",
        .inputs = {},
        .outputs = {{"0"}},
        .deps_inputs = {"begin1", "end1"}
      };
      temp_nodes_.emplace_back(n2);

      first_key_ = "begin1";
      num_inputs_ = 0;
      num_outputs_ = 1;
    }

    bool add_subgraph(std::map<std::string, MatchedNode>& nodes,
        std::string name_prefix, Graph* g,
        std::vector<const Edge*>& inputs,
        std::vector<std::vector<const Edge*>>& outputs) override {
      NodeDef fused_def;
      fused_def.set_op("Output");
      fused_def.set_name("fused_op");
      Status status;
      Node* fused_node = g->AddNode(fused_def, &status);
      add_oedges(g, fused_node, 0, outputs[0]);
      return true;
    }

    bool CheckDynamicInputs(
        const Node* node, const TempNode* temp_node, int dy_mode,
        std::vector<const Edge*>& fused_op_inputs,
        std::map<const std::string, TempNode>& temp_node_map,
        std::map<std::string, MatchedNode>& matched_node_map) override {
      return false;
    }

    bool CheckDynamicOutputs(
        const Node* node, const TempNode* temp_node, int dy_mode,
        std::vector<std::vector<const Edge*>>& fused_op_outputs,
        std::map<const std::string, TempNode>& temp_node_map,
        std::map<std::string, MatchedNode>& matched_node_map) override {
      return false;
    }
  };

  TemplateControlDeps tdeps;
  OptimizerFusionImpl of(&graph_, &tdeps);
  EXPECT_TRUE(of.Optimize());
}

// Note that the "rules" in these tests are not meant to be logically correct
/*
TEST_F(OptimizerFusionTest, LSTMMatched) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'BiasAdd'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }"
      "node { name: 'E' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT32 } } }"
      "node { name: 'F' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 4} }"
      " input: ['E', 'D'] }"
      "node { name: 'G' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:0'] }"
      "node { name: 'H' op: 'Tanh'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:1'] }"
      "node { name: 'I' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'J' op: 'Add'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:2', 'I'] }"
      "node { name: 'K' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['J'] }"
      "node { name: 'L' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:3'] }"
      "node { name: 'M' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['G', 'H'] }"
      "node { name: 'N' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'K'] }"
      "node { name: 'O' op: 'Add'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['N', 'M'] }"
      "node { name: 'P' op: 'Tanh'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['O'] }"
      "node { name: 'Q' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['P', 'L'] }"
      "node { name: 'R' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['O'] }"
      "node { name: 'S' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['Q'] }");
  EXPECT_EQ(
      DoFusion(),
      "A(Input);B(Input);C(Input);D(BiasAdd);E(Const);F(Split);G(Sigmoid);H(Tanh);I(Const);J(Add);K(Sigmoid);L(Sigmoid);M(Mul);N(Mul);O(Add);P(Tanh);Q(Mul);R(Identity);S(Identity);fused_op_1_lstm_elewise(FusedLSTMElementWise)|A->fused_op_1_lstm_elewise;B->fused_op_1_lstm_elewise:1;C->fused_op_1_lstm_elewise:2;D->F:1;E->F;F->G;F:1->H;F:2->J;F:3->L;G->M;H->M:1;I->J:1;J->K;K->N:1;L->Q:1;M->O:1;N->O;P->Q;fused_op_1_lstm_elewise->P;fused_op_1_lstm_elewise->R;fused_op_1_lstm_elewise:1->S");
}*/

TEST_F(OptimizerFusionTest, LSTMControlOutputs) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'BiasAdd'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }"
      "node { name: 'E' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT32 } } }"
      "node { name: 'F' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 4} }"
      " input: ['E', 'D'] }"
      "node { name: 'G' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:0'] }"
      "node { name: 'H' op: 'Tanh'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:1'] }"
      "node { name: 'I' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'J' op: 'Add'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:2', 'I'] }"
      "node { name: 'K' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['J'] }"
      "node { name: 'L' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:3'] }"
      "node { name: 'M' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['G', 'H'] }"
      "node { name: 'N' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'K'] }"
      "node { name: 'O' op: 'Add'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['N', 'M'] }"
      "node { name: 'P' op: 'Tanh'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['O'] }"
      "node { name: 'Q' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['P', 'L'] }"
      "node { name: 'R' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['O'] }"
      "node { name: 'S' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['Q', '^Q'] }");
  EXPECT_EQ(
      DoFusion(),
      "A(Input);B(Input);C(Input);D(BiasAdd);E(Const);F(Split);G(Sigmoid);H(Tanh);I(Const);J(Add);K(Sigmoid);L(Sigmoid);M(Mul);N(Mul);O(Add);P(Tanh);Q(Mul);R(Identity);S(Identity)|A->D;B->N;C->D:1;D->F:1;E->F;F->G;F:1->H;F:2->J;F:3->L;G->M;H->M:1;I->J:1;J->K;K->N:1;L->Q:1;M->O:1;N->O;O->P;O->R;P->Q;Q->S;Q:control->S:control");
}

TEST_F(OptimizerFusionTest, LSTMControlInputs) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'BiasAdd'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C', '^A'] }"
      "node { name: 'E' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT32 } } }"
      "node { name: 'F' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 4} }"
      " input: ['E', 'D'] }"
      "node { name: 'G' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:0'] }"
      "node { name: 'H' op: 'Tanh'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:1'] }"
      "node { name: 'I' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'J' op: 'Add'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:2', 'I'] }"
      "node { name: 'K' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['J'] }"
      "node { name: 'L' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:3'] }"
      "node { name: 'M' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['G', 'H'] }"
      "node { name: 'N' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'K'] }"
      "node { name: 'O' op: 'Add'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['N', 'M'] }"
      "node { name: 'P' op: 'Tanh'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['O'] }"
      "node { name: 'Q' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['P', 'L'] }"
      "node { name: 'R' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['O'] }"
      "node { name: 'S' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['Q'] }");
  EXPECT_EQ(
      DoFusion(),
      "A(Input);B(Input);C(Input);D(BiasAdd);E(Const);F(Split);G(Sigmoid);H(Tanh);I(Const);J(Add);K(Sigmoid);L(Sigmoid);M(Mul);N(Mul);O(Add);P(Tanh);Q(Mul);R(Identity);S(Identity);fused_op_1_lstm_elewise(FusedLSTMElementWise)|A->fused_op_1_lstm_elewise;A:control->D:control;B->fused_op_1_lstm_elewise:1;C->fused_op_1_lstm_elewise:2;D->F:1;E->F;F->G;F:1->H;F:2->J;F:3->L;G->M;H->M:1;I->J:1;J->K;K->N:1;L->Q:1;M->O:1;N->O;P->Q;fused_op_1_lstm_elewise->P;fused_op_1_lstm_elewise->R;fused_op_1_lstm_elewise:1->S");
}

TEST_F(OptimizerFusionTest, LSTMConstControlInputs) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'BiasAdd'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }"
      "node { name: 'E' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT32 } } "
      " input: ['^A'] }"
      "node { name: 'F' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 4} }"
      " input: ['E', 'D'] }"
      "node { name: 'G' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:0'] }"
      "node { name: 'H' op: 'Tanh'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:1'] }"
      "node { name: 'I' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'J' op: 'Add'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:2', 'I'] }"
      "node { name: 'K' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['J'] }"
      "node { name: 'L' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:3'] }"
      "node { name: 'M' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['G', 'H'] }"
      "node { name: 'N' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'K'] }"
      "node { name: 'O' op: 'Add'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['N', 'M'] }"
      "node { name: 'P' op: 'Tanh'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['O'] }"
      "node { name: 'Q' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['P', 'L'] }"
      "node { name: 'R' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['O'] }"
      "node { name: 'S' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['Q'] }");
  EXPECT_EQ(
      DoFusion(),
      "A(Input);B(Input);C(Input);D(BiasAdd);E(Const);F(Split);G(Sigmoid);H(Tanh);I(Const);J(Add);K(Sigmoid);L(Sigmoid);M(Mul);N(Mul);O(Add);P(Tanh);Q(Mul);R(Identity);S(Identity);fused_op_1_lstm_elewise(FusedLSTMElementWise)|A->fused_op_1_lstm_elewise;A:control->E:control;B->fused_op_1_lstm_elewise:1;C->fused_op_1_lstm_elewise:2;D->F:1;E->F;F->G;F:1->H;F:2->J;F:3->L;G->M;H->M:1;I->J:1;J->K;K->N:1;L->Q:1;M->O:1;N->O;P->Q;fused_op_1_lstm_elewise->P;fused_op_1_lstm_elewise->R;fused_op_1_lstm_elewise:1->S");
}

TEST_F(OptimizerFusionTest, LSTMMoreConsumers1) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'BiasAdd'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }"
      "node { name: 'E' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT32 } } }"
      "node { name: 'F' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 4} }"
      " input: ['E', 'D'] }"
      "node { name: 'G' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:0'] }"
      "node { name: 'H' op: 'Tanh'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:1'] }"
      "node { name: 'I' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'J' op: 'Add'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:2', 'I'] }"
      "node { name: 'K' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['J'] }"
      "node { name: 'L' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:3'] }"
      "node { name: 'M' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['G', 'H'] }"
      "node { name: 'N' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'K'] }"
      "node { name: 'O' op: 'Add'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['N', 'M'] }"
      "node { name: 'P' op: 'Tanh'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['O'] }"
      "node { name: 'Q' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['P', 'L'] }"
      "node { name: 'R' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['O'] }"
      "node { name: 'S' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['Q'] }"
      "node { name: 'T' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['O'] }");
  EXPECT_EQ(
      DoFusion(),
      "A(Input);B(Input);C(Input);D(BiasAdd);E(Const);F(Split);G(Sigmoid);H(Tanh);I(Const);J(Add);K(Sigmoid);L(Sigmoid);M(Mul);N(Mul);O(Add);P(Tanh);Q(Mul);R(Identity);S(Identity);T(Identity);fused_op_1_lstm_elewise(FusedLSTMElementWise)|A->fused_op_1_lstm_elewise;B->fused_op_1_lstm_elewise:1;C->fused_op_1_lstm_elewise:2;D->F:1;E->F;F->G;F:1->H;F:2->J;F:3->L;G->M;H->M:1;I->J:1;J->K;K->N:1;L->Q:1;M->O:1;N->O;P->Q;fused_op_1_lstm_elewise->P;fused_op_1_lstm_elewise->R;fused_op_1_lstm_elewise->T;fused_op_1_lstm_elewise:1->S");
}

TEST_F(OptimizerFusionTest, LSTMMoreConsumers2) {
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Input'}"
      "node { name: 'D' op: 'BiasAdd'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'C'] }"
      "node { name: 'E' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT32 } } }"
      "node { name: 'F' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 4} }"
      " input: ['E', 'D'] }"
      "node { name: 'G' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:0'] }"
      "node { name: 'H' op: 'Tanh'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:1'] }"
      "node { name: 'I' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'J' op: 'Add'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:2', 'I'] }"
      "node { name: 'K' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['J'] }"
      "node { name: 'L' op: 'Sigmoid'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['F:3'] }"
      "node { name: 'M' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['G', 'H'] }"
      "node { name: 'N' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['B', 'K'] }"
      "node { name: 'O' op: 'Add'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['N', 'M'] }"
      "node { name: 'P' op: 'Tanh'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['O'] }"
      "node { name: 'Q' op: 'Mul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['P', 'L'] }"
      "node { name: 'R' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['O'] }"
      "node { name: 'S' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['Q'] }"
      "node { name: 'T' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['P'] }");
  EXPECT_EQ(
      DoFusion(),
      "A(Input);B(Input);C(Input);D(BiasAdd);E(Const);F(Split);G(Sigmoid);H(Tanh);I(Const);J(Add);K(Sigmoid);L(Sigmoid);M(Mul);N(Mul);O(Add);P(Tanh);Q(Mul);R(Identity);S(Identity);T(Identity)|A->D;B->N;C->D:1;D->F:1;E->F;F->G;F:1->H;F:2->J;F:3->L;G->M;H->M:1;I->J:1;J->K;K->N:1;L->Q:1;M->O:1;N->O;O->P;O->R;P->Q;P->T;Q->S");
}

TEST_F(OptimizerFusionTest, SparseInnerFlattenFuse) {
  InitGraph(
      "node { name: 'A' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"

      "node { name: 'B' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"

      "node { name: 'C' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"

      "node { name: 'D' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"

      "node { name: 'E' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"

      "node { name: 'F' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"

      "node { name: 'G' op: 'InputInt64' }"

      "node { name: 'H' op: 'StridedSlice'"
      " attr { key: 'T' value { type: DT_INT64 } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " attr { key: 'begin_mask' value { i: 0 } }"
      " attr { key: 'ellipsis_mask' value { i: 0 } }"
      " attr { key: 'end_mask' value { i: 0 } }"
      " attr { key: 'new_axis_mask' value { i: 0 } }"
      " attr { key: 'shrink_axis_mask' value { i: 1 } }"
      " input: ['G', 'A', 'B', 'C'] }"

      "node { name: 'I' op: 'StridedSlice'"
      " attr { key: 'T' value { type: DT_INT64 } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " attr { key: 'begin_mask' value { i: 0 } }"
      " attr { key: 'ellipsis_mask' value { i: 0 } }"
      " attr { key: 'end_mask' value { i: 0 } }"
      " attr { key: 'new_axis_mask' value { i: 0 } }"
      " attr { key: 'shrink_axis_mask' value { i: 1 } }"
      " input: ['G', 'D', 'E', 'F'] }"

      "node { name: 'J' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"

      "node { name: 'K' op: 'Prod'"
      " attr { key: 'T' value { type: DT_INT64 } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'keep_dims' value { b: false } }"
      " input: ['I', 'J'] }"

      "node { name: 'L' op: 'Pack'"
      " attr { key: 'T' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 1 } }"
      " attr { key: 'axis' value {i: 0 } }"
      " input: ['K'] }"

      "node { name: 'N' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"

      "node { name: 'M' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_INT64 } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 2 } }"
      " input: ['H', 'L', 'N'] }"

      "node { name: 'P' op: 'InputInt64' }"

      "node { name: 'O' op: 'SparseReshape'"
      " input: ['P', 'G', 'M'] }"

      "node { name: 'R' op: 'Identity'"
      " attr { key: 'T' value { type: DT_INT64 } }"
      " input: ['O'] }"

      "node { name: 'S' op: 'Identity'"
      " attr { key: 'T' value { type: DT_INT64 } }"
      " input: ['O'] }");

  EXPECT_EQ(
      DoFusion(),
      "A(Const);B(Const);C(Const);D(Const);E(Const);F(Const);G(InputInt64);H(StridedSlice);I(StridedSlice);J(Const);K(Prod);L(Pack);M(ConcatV2);N(Const);O(SparseReshape);P(InputInt64);R(Identity);S(Identity)|A->H:1;B->H:2;C->H:3;D->I:1;E->I:2;F->I:3;G->H;G->I;G->O:1;H->M;I->K;J->K:1;K->L;L->M:1;M->O:2;N->M:2;O->R;O->S;P->O");
}

#ifndef GOOGLE_CUDA
TEST_F(OptimizerFusionTest, MSBatchMatMulFuse2Heads) {
  InitGraph(
      "node { name: 'A' op: 'Input'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'B' op: 'Input'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'C' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_BOOL } } }"
      "node { name: 'D' op: 'Input'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'E' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT32 } } }"
      "node { name: 'F' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"
      "node { name: 'G' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 2} }"
      " input: ['E', 'A'] }"
      "node { name: 'H' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 2} }"
      " input: ['E', 'B'] }"
      "node { name: 'I' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 2 } }"
      " input: ['G:0', 'G:1','F'] }"
      "node { name: 'J' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 2 } }"
      " input: ['H:0', 'H:1','F'] }"
      "node { name: 'K' op: 'BatchMatMul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'adj_x' value { type: DT_BOOL } }"
      " attr { key: 'adj_y' value { type: DT_BOOL } }"
      " input: ['I', 'J'] }"
      "node { name: 'L' op: 'Select'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'K', 'D'] }"
      "node { name: 'M' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['L'] }");

  EXPECT_EQ(
      DoFusion(),"A(Input);B(Input);C(Const);D(Input);E(Const);F(Const);G(Split);H(Split);I(ConcatV2);J(ConcatV2);K(BatchMatMul);L(Select);M(Identity);fused_op_1_msbatchmatmul(MSBatchMatMul)|A->fused_op_1_msbatchmatmul;B->fused_op_1_msbatchmatmul:1;C->fused_op_1_msbatchmatmul:2;D->fused_op_1_msbatchmatmul:3;E->G;E->H;F->I:2;F->J:2;G->I;G:1->I:1;H->J;H:1->J:1;I->K;J->K:1;K->L:1;fused_op_1_msbatchmatmul->M");
}

TEST_F(OptimizerFusionTest, MSBatchMatMulFuse4Heads) {
  InitGraph(
      "node { name: 'A' op: 'Input'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'B' op: 'Input'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'C' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_BOOL } } }"
      "node { name: 'D' op: 'Input'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'E' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT32 } } }"
      "node { name: 'F' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"
      "node { name: 'G' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 4} }"
      " input: ['E', 'A'] }"
      "node { name: 'H' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 4} }"
      " input: ['E', 'B'] }"
      "node { name: 'I' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 4 } }"
      " input: ['G:0', 'G:1', 'G:2','G:3', 'F'] }"
      "node { name: 'J' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 4 } }"
      " input: ['H:0', 'H:1', 'H:2','H:3', 'F'] }"
      "node { name: 'K' op: 'BatchMatMul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'adj_x' value { type: DT_BOOL } }"
      " attr { key: 'adj_y' value { type: DT_BOOL } }"
      " input: ['I', 'J'] }"
      "node { name: 'L' op: 'Select'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['C', 'K', 'D'] }"
      "node { name: 'M' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['L'] }");

  EXPECT_EQ(
      DoFusion(),"A(Input);B(Input);C(Const);D(Input);E(Const);F(Const);G(Split);H(Split);I(ConcatV2);J(ConcatV2);K(BatchMatMul);L(Select);M(Identity);fused_op_1_msbatchmatmul(MSBatchMatMul)|A->fused_op_1_msbatchmatmul;B->fused_op_1_msbatchmatmul:1;C->fused_op_1_msbatchmatmul:2;D->fused_op_1_msbatchmatmul:3;E->G;E->H;F->I:4;F->J:4;G->I;G:1->I:1;G:2->I:2;G:3->I:3;H->J;H:1->J:1;H:2->J:2;H:3->J:3;I->K;J->K:1;K->L:1;fused_op_1_msbatchmatmul->M");
}

TEST_F(OptimizerFusionTest, MSBatchMatMulGradFuse2Heads) {
  InitGraph(
      "node { name: 'A' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_BOOL } } }"
      "node { name: 'B' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'C' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'D' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'E' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'F' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"
      "node { name: 'G' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"
      "node { name: 'H' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT32 } } }"
      "node { name: 'I' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"
      "node { name: 'J' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"

      "node { name: 'K_0' op: 'Select'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B', 'C'] }"

      "node { name: 'L_0' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 2} }"
      " input: ['H', 'D'] }"

      "node { name: 'M_0' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 2 } }"
      " input: ['L_0:0', 'L_0:1', 'I'] }"

      "node { name: 'N_0' op: 'BatchMatMul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'adj_x' value { type: DT_BOOL } }"
      " attr { key: 'adj_y' value { type: DT_BOOL } }"
      " input: ['K_0', 'M_0'] }"
     
      "node { name: 'L_1' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 2} }"
      " input: ['H', 'E'] }"

      "node { name: 'M_1' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 2 } }"
      " input: ['L_1:0','L_1:1', 'J'] }"

      "node { name: 'N_1' op: 'BatchMatMul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'adj_x' value { type: DT_BOOL } }"
      " attr { key: 'adj_y' value { type: DT_BOOL } }"
      " input: ['K_0', 'M_1'] }"

      "node { name: 'O_0' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['N_0'] }"

      "node { name: 'O_1' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['N_1'] }"

      "node { name: 'P_0' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['O_0','F','G'] }"

      "node { name: 'P_0_1' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['O_0','F','G'] }"

      "node { name: 'P_1' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['O_1','F','G'] }"

      "node { name: 'P_1_1' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['O_1','F','G'] }"

      "node { name: 'Q_0' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 2 } }"
      " input: ['P_0', 'P_0_1', 'I'] }"

      "node { name: 'Q_1' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 2 } }"
      " input: ['P_1', 'P_1_1', 'I'] }"
      
      "node { name: 'R_0' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['Q_0'] }"

      "node { name: 'R_1' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['Q_1'] }");

  EXPECT_EQ(
      DoFusion(),"A(Const);B(Const);C(Const);D(Const);E(Const);F(Const);G(Const);H(Const);I(Const);J(Const);K_0(Select);L_0(Split);L_1(Split);M_0(ConcatV2);M_1(ConcatV2);N_0(BatchMatMul);N_1(BatchMatMul);O_0(Identity);O_1(Identity);P_0(Slice);P_0_1(Slice);P_1(Slice);P_1_1(Slice);Q_0(ConcatV2);Q_1(ConcatV2);R_0(Identity);R_1(Identity);fused_op_1_msbatchmatmulgrad(MSBatchMatMulGrad)|A->fused_op_1_msbatchmatmulgrad:3;B->fused_op_1_msbatchmatmulgrad;C->fused_op_1_msbatchmatmulgrad:4;D->L_0:1;D->fused_op_1_msbatchmatmulgrad:1;E->L_1:1;E->fused_op_1_msbatchmatmulgrad:2;F->P_0:1;F->P_0_1:1;F->P_1:1;F->P_1_1:1;G->P_0:2;G->P_0_1:2;G->P_1:2;G->P_1_1:2;H->L_0;H->L_1;I->M_0:2;I->Q_0:2;I->Q_1:2;J->M_1:2;K_0->N_0;K_0->N_1;L_0->M_0;L_0:1->M_0:1;L_1->M_1;L_1:1->M_1:1;M_0->N_0:1;M_1->N_1:1;N_0->O_0;N_1->O_1;O_0->P_0;O_0->P_0_1;O_1->P_1;O_1->P_1_1;P_0->Q_0;P_0_1->Q_0:1;P_1->Q_1;P_1_1->Q_1:1;fused_op_1_msbatchmatmulgrad->R_1;fused_op_1_msbatchmatmulgrad:1->R_0");
}

/*
TEST_F(OptimizerFusionTest, MSBatchMatMulGradFuse4Heads) {
  InitGraph(
      "node { name: 'A' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_BOOL } } }"
      "node { name: 'B' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'C' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'D' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'E' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'F' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"
      "node { name: 'G' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"
      "node { name: 'H' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT32 } } }"
      "node { name: 'I' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"
      "node { name: 'J' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"

      "node { name: 'K_0' op: 'Select'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B', 'C'] }"

      "node { name: 'L_0' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 4} }"
      " input: ['H', 'D'] }"

      "node { name: 'M_0' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 4 } }"
      " input: ['L_0:0', 'L_0:1', 'L_0:2', 'L_0:3', 'I'] }"

      "node { name: 'N_0' op: 'BatchMatMul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'adj_x' value { type: DT_BOOL } }"
      " attr { key: 'adj_y' value { type: DT_BOOL } }"
      " input: ['K_0', 'M_0'] }"
     
      "node { name: 'L_1' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 4} }"
      " input: ['H', 'E'] }"

      "node { name: 'M_1' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 4 } }"
      " input: ['L_1:0','L_1:1','L_1:2','L_1:3', 'J'] }"

      "node { name: 'N_1' op: 'BatchMatMul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'adj_x' value { type: DT_BOOL } }"
      " attr { key: 'adj_y' value { type: DT_BOOL } }"
      " input: ['K_0', 'M_1'] }"

      "node { name: 'O_0' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['N_0'] }"

      "node { name: 'O_1' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['N_1'] }"

      "node { name: 'P_0' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['O_0','F','G'] }"

      "node { name: 'P_0_1' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['O_0','F','G'] }"

      "node { name: 'P_0_2' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['O_0','F','G'] }"

      "node { name: 'P_0_3' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['O_0','F','G'] }"

      "node { name: 'P_1' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['O_1','F','G'] }"

      "node { name: 'P_1_1' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['O_1','F','G'] }"

      "node { name: 'P_1_2' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['O_1','F','G'] }"

      "node { name: 'P_1_3' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['O_1','F','G'] }"
      
      "node { name: 'Q_0' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 4 } }"
      " input: ['P_0', 'P_0_1', 'P_0_2', 'P_0_3', 'I'] }"

      "node { name: 'Q_1' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 4 } }"
      " input: ['P_1', 'P_1_1', 'P_1_2', 'P_1_3', 'I'] }"
      
      "node { name: 'R_0' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['Q_0'] }"

      "node { name: 'R_1' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['Q_1'] }");

  EXPECT_EQ(
      DoFusion(),"A(Const);B(Const);C(Const);D(Const);E(Const);F(Const);G(Const);H(Const);I(Const);J(Const);K_0(Select);L_0(Split);L_1(Split);M_0(ConcatV2);M_1(ConcatV2);N_0(BatchMatMul);N_1(BatchMatMul);O_0(Identity);O_1(Identity);P_0(Slice);P_0_1(Slice);P_0_2(Slice);P_0_3(Slice);P_1(Slice);P_1_1(Slice);P_1_2(Slice);P_1_3(Slice);Q_0(ConcatV2);Q_1(ConcatV2);R_0(Identity);R_1(Identity);fused_op_1_msbatchmatmulgrad(MSBatchMatMulGrad)|A->fused_op_1_msbatchmatmulgrad:3;B->fused_op_1_msbatchmatmulgrad;C->fused_op_1_msbatchmatmulgrad:4;D->L_0:1;D->fused_op_1_msbatchmatmulgrad:1;E->L_1:1;E->fused_op_1_msbatchmatmulgrad:2;F->P_0:1;F->P_0_1:1;F->P_0_2:1;F->P_0_3:1;F->P_1:1;F->P_1_1:1;F->P_1_2:1;F->P_1_3:1;G->P_0:2;G->P_0_1:2;G->P_0_2:2;G->P_0_3:2;G->P_1:2;G->P_1_1:2;G->P_1_2:2;G->P_1_3:2;H->L_0;H->L_1;I->M_0:4;I->Q_0:4;I->Q_1:4;J->M_1:4;K_0->N_0;K_0->N_1;L_0->M_0;L_0:1->M_0:1;L_0:2->M_0:2;L_0:3->M_0:3;L_1->M_1;L_1:1->M_1:1;L_1:2->M_1:2;L_1:3->M_1:3;M_0->N_0:1;M_1->N_1:1;N_0->O_0;N_1->O_1;O_0->P_0;O_0->P_0_1;O_0->P_0_2;O_0->P_0_3;O_1->P_1;O_1->P_1_1;O_1->P_1_2;O_1->P_1_3;P_0->Q_0;P_0_1->Q_0:1;P_0_2->Q_0:2;P_0_3->Q_0:3;P_1->Q_1;P_1_1->Q_1:1;P_1_2->Q_1:2;P_1_3->Q_1:3;fused_op_1_msbatchmatmulgrad->R_1;fused_op_1_msbatchmatmulgrad:1->R_0");
}*/

/*
TEST_F(OptimizerFusionTest, StackBatchMatMulFuse2Heads) {
  InitGraph(
      "node { name: 'A' op: 'Input'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'B' op: 'Input'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'C' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT32 } } }"
      "node { name: 'D' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"

      "node { name: 'E' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 2} }"
      " input: ['C', 'B'] }"

      "node { name: 'F' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 2 } }"
      " input: ['E:0', 'E:1', 'D'] }"

      "node { name: 'G' op: 'BatchMatMul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'adj_x' value { type: DT_BOOL } }"
      " attr { key: 'adj_y' value { type: DT_BOOL } }"
      " input: ['A', 'F'] }"

       "node { name: 'H' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 2} }"
      " input: ['C', 'G'] }"

      "node { name: 'I' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 2 } }"
      " input: ['H:0', 'H:1', 'D'] }"

      "node { name: 'J' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['I'] }");

  EXPECT_EQ(
      DoFusion(),"A(Input);B(Input);C(Const);D(Const);E(Split);F(ConcatV2);G(BatchMatMul);H(Split);I(ConcatV2);J(Identity);fused_op_1_stackbatchmatmul(StackBatchMatMul)|A->fused_op_1_stackbatchmatmul;B->fused_op_1_stackbatchmatmul:1;C->E;C->H;D->F:2;D->I:2;E->F;E:1->F:1;F->G:1;G->H:1;H->I;H:1->I:1;fused_op_1_stackbatchmatmul->J");
}*/

/*
TEST_F(OptimizerFusionTest, StackBatchMatMulFuse4Heads) {
  InitGraph(
      "node { name: 'A' op: 'Input'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'B' op: 'Input'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'C' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT32 } } }"
      "node { name: 'D' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"

      "node { name: 'E' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 4} }"
      " input: ['C', 'B'] }"

      "node { name: 'F' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 4 } }"
      " input: ['E:0','E:1','E:2', 'E:3','D'] }"

      "node { name: 'G' op: 'BatchMatMul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'adj_x' value { type: DT_BOOL } }"
      " attr { key: 'adj_y' value { type: DT_BOOL } }"
      " input: ['A', 'F'] }"

       "node { name: 'H' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 4} }"
      " input: ['C', 'G'] }"

      "node { name: 'I' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 4 } }"
      " input: ['H:0', 'H:1','H:2', 'H:3','D'] }"

      "node { name: 'J' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['I'] }");

  EXPECT_EQ(
      DoFusion(),"A(Input);B(Input);C(Const);D(Const);E(Split);F(ConcatV2);G(BatchMatMul);H(Split);I(ConcatV2);J(Identity);fused_op_1_stackbatchmatmul(StackBatchMatMul)|A->fused_op_1_stackbatchmatmul;B->fused_op_1_stackbatchmatmul:1;C->E;C->H;D->F:4;D->I:4;E->F;E:1->F:1;E:2->F:2;E:3->F:3;F->G:1;G->H:1;H->I;H:1->I:1;H:2->I:2;H:3->I:3;fused_op_1_stackbatchmatmul->J");
}*/

/*
TEST_F(OptimizerFusionTest, StackBatchMatMulGradFuse2Heads) {
  InitGraph(
      "node { name: 'A' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'B' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'C' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'D' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"
      "node { name: 'E' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"
      "node { name: 'F' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT32 } } }"
      "node { name: 'G' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"
      "node { name: 'H' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT32 } } }"

      "node { name: 'I_0' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['A','D','E'] }"

      "node { name: 'I_0_1' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['A','D','E'] }"

      "node { name: 'J_0' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['I_0'] }"

      "node { name: 'J_0_1' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['I_0_1'] }"

      "node { name: 'K_0' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 2 } }"
      " input: ['J_0', 'J_0_1', 'G'] }"

      "node { name: 'L_0' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 2} }"
      " input: ['H', 'C'] }"

      "node { name: 'K_1' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 2 } }"
      " input: ['L_0:0', 'L_0:1', 'G'] }"

      "node { name: 'M_0' op: 'BatchMatMul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'adj_x' value { type: DT_BOOL } }"
      " attr { key: 'adj_y' value { type: DT_BOOL } }"
      " input: ['K_0', 'K_1'] }"

      "node { name: 'M_1' op: 'BatchMatMul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'adj_x' value { type: DT_BOOL } }"
      " attr { key: 'adj_y' value { type: DT_BOOL } }"
      " input: ['B', 'K_0'] }"

      "node { name: 'J_1' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['M_1'] }"

      "node { name: 'I_1' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['J_1', 'D','E'] }"

      "node { name: 'I_1_1' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['J_1', 'D','E'] }"

      "node { name: 'J_2' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['I_1'] }"

      "node { name: 'J_2_1' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['I_1_1'] }"

      "node { name: 'K_2' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 2 } }"
      " input: ['J_2', 'J_2_1', 'G'] }"

      "node { name: 'N_0' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['M_0'] }"

      "node { name: 'N_1' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['K_2'] }"

      "node { name: 'N_2' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['K_1'] }"
      );

  EXPECT_EQ(
      DoFusion(),"A(Const);B(Const);C(Const);D(Const);E(Const);F(Const);G(Const);H(Const);I_0(Slice);I_0_1(Slice);I_1(Slice);I_1_1(Slice);J_0(Identity);J_0_1(Identity);J_1(Identity);J_2(Identity);J_2_1(Identity);K_0(ConcatV2);K_1(ConcatV2);K_2(ConcatV2);L_0(Split);M_0(BatchMatMul);M_1(BatchMatMul);N_0(Identity);N_1(Identity);N_2(Identity);fused_op_1_stackbatchmatmulgrad(StackBatchMatMulGrad)|A->I_0;A->fused_op_1_stackbatchmatmulgrad;B->M_1;B->fused_op_1_stackbatchmatmulgrad:1;C->L_0:1;C->fused_op_1_stackbatchmatmulgrad:2;D->I_0:1;D->I_0_1:1;D->I_1:1;D->I_1_1:1;E->I_0:2;E->I_0_1:2;E->I_1:2;E->I_1_1:2;G->K_0:2;G->K_1:2;G->K_2:2;H->L_0;I_0->J_0;I_0_1->J_0_1;I_1->J_2;I_1_1->J_2_1;J_0->K_0;J_0_1->K_0:1;J_1->I_1;J_1->I_1_1;J_2->K_2;J_2_1->K_2:1;K_0->M_0;K_0->M_1:1;K_1->M_0:1;K_1->N_2;L_0->K_1;L_0:1->K_1:1;M_1->J_1;fused_op_1_stackbatchmatmulgrad->N_0;fused_op_1_stackbatchmatmulgrad:1->N_1");
}*/

/*
TEST_F(OptimizerFusionTest, StackBatchMatMulGradFuse4Heads) {
  InitGraph(
      "node { name: 'A' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'B' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'C' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_FLOAT } } }"
      "node { name: 'D' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"
      "node { name: 'E' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"
      "node { name: 'F' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT32 } } }"
      "node { name: 'G' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT64 } } }"
      "node { name: 'H' op: 'Const'"
      " attr { key: 'dtype' value { type: DT_INT32 } } }"

      "node { name: 'I_0' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['A','D','E'] }"

      "node { name: 'I_0_1' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['A','D','E'] }"

      "node { name: 'I_0_2' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['A','D','E'] }"

      "node { name: 'I_0_3' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['A','D','E'] }"

      "node { name: 'J_0' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['I_0'] }"

      "node { name: 'J_0_1' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['I_0_1'] }"

      "node { name: 'J_0_2' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['I_0_2'] }"

      "node { name: 'J_0_3' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['I_0_3'] }"

      "node { name: 'K_0' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 4 } }"
      " input: ['J_0', 'J_0_1', 'J_0_2', 'J_0_3', 'G'] }"

      "node { name: 'L_0' op: 'Split'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'num_split' value { i: 4} }"
      " input: ['H', 'C'] }"

      "node { name: 'K_1' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 4 } }"
      " input: ['L_0:0', 'L_0:1', 'L_0:2', 'L_0:3', 'G'] }"

      "node { name: 'M_0' op: 'BatchMatMul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'adj_x' value { type: DT_BOOL } }"
      " attr { key: 'adj_y' value { type: DT_BOOL } }"
      " input: ['K_0', 'K_1'] }"

      "node { name: 'M_1' op: 'BatchMatMul'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'adj_x' value { type: DT_BOOL } }"
      " attr { key: 'adj_y' value { type: DT_BOOL } }"
      " input: ['B', 'K_0'] }"

      "node { name: 'J_1' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['M_1'] }"

      "node { name: 'I_1' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['J_1', 'D','E'] }"

      "node { name: 'I_1_1' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['J_1', 'D','E'] }"

      "node { name: 'I_1_2' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['J_1', 'D','E'] }"

      "node { name: 'I_1_3' op: 'Slice'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Index' value { type: DT_INT64 } }"
      " input: ['J_1', 'D','E'] }"

      "node { name: 'J_2' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['I_1'] }"

      "node { name: 'J_2_1' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['I_1_1'] }"

      "node { name: 'J_2_2' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['I_1_2'] }"

      "node { name: 'J_2_3' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['I_1_3'] }"

      "node { name: 'K_2' op: 'ConcatV2'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " attr { key: 'Tidx' value { type: DT_INT64 } }"
      " attr { key: 'N' value { i: 4 } }"
      " input: ['J_2', 'J_2_1', 'J_2_2', 'J_2_3', 'G'] }"

      "node { name: 'N_0' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['M_0'] }"

      "node { name: 'N_1' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['K_2'] }"

      "node { name: 'N_2' op: 'Identity'"
      " attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['K_1'] }"
      );

  EXPECT_EQ(
      DoFusion(),"A(Const);B(Const);C(Const);D(Const);E(Const);F(Const);G(Const);H(Const);I_0(Slice);I_0_1(Slice);I_0_2(Slice);I_0_3(Slice);I_1(Slice);I_1_1(Slice);I_1_2(Slice);I_1_3(Slice);J_0(Identity);J_0_1(Identity);J_0_2(Identity);J_0_3(Identity);J_1(Identity);J_2(Identity);J_2_1(Identity);J_2_2(Identity);J_2_3(Identity);K_0(ConcatV2);K_1(ConcatV2);K_2(ConcatV2);L_0(Split);M_0(BatchMatMul);M_1(BatchMatMul);N_0(Identity);N_1(Identity);N_2(Identity);fused_op_1_stackbatchmatmulgrad(StackBatchMatMulGrad)|A->I_0;A->I_0_1;A->I_0_2;A->fused_op_1_stackbatchmatmulgrad;B->M_1;B->fused_op_1_stackbatchmatmulgrad:1;C->L_0:1;C->fused_op_1_stackbatchmatmulgrad:2;D->I_0:1;D->I_0_1:1;D->I_0_2:1;D->I_0_3:1;D->I_1:1;D->I_1_1:1;D->I_1_2:1;D->I_1_3:1;E->I_0:2;E->I_0_1:2;E->I_0_2:2;E->I_0_3:2;E->I_1:2;E->I_1_1:2;E->I_1_2:2;E->I_1_3:2;G->K_0:4;G->K_1:4;G->K_2:4;H->L_0;I_0->J_0;I_0_1->J_0_1;I_0_2->J_0_2;I_0_3->J_0_3;I_1->J_2;I_1_1->J_2_1;I_1_2->J_2_2;I_1_3->J_2_3;J_0->K_0;J_0_1->K_0:1;J_0_2->K_0:2;J_0_3->K_0:3;J_1->I_1;J_1->I_1_1;J_1->I_1_2;J_1->I_1_3;J_2->K_2;J_2_1->K_2:1;J_2_2->K_2:2;J_2_3->K_2:3;K_0->M_0;K_0->M_1:1;K_1->M_0:1;K_1->N_2;L_0->K_1;L_0:1->K_1:1;L_0:2->K_1:2;L_0:3->K_1:3;M_1->J_1;fused_op_1_stackbatchmatmulgrad->N_0;fused_op_1_stackbatchmatmulgrad:1->N_1");
}*/
#endif

}  // namespace
}  // namespace tensorflow
