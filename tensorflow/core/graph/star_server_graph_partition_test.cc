#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/star_server_graph_partition.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/device_name_utils.h"

using namespace std;

namespace tensorflow {

static string SplitByWorker(const Node* node) {
  string task;
  string device;
  CHECK(DeviceNameUtils::SplitDeviceName(node->assigned_device_name(), &task,
          &device))
    << "node: " << node->name()
    << " dev: " << node->assigned_device_name();
  return task;
}

class DistGraphPartitionTest : public ::testing::Test {
public:
  DistGraphPartitionTest()
    : in_(Scope::NewRootScope().ExitOnError()),
      scope_worker_(Scope::NewRootScope().ExitOnError().WithDevice(
              "/job:worker/replica:0/task:0/cpu:0")),
      scope_ps1_(Scope::NewRootScope().ExitOnError().WithDevice(
              "/job:ps/replica:0/task:1/cpu:0")),
      scope_ps2_(Scope::NewRootScope().ExitOnError().WithDevice(
              "/job:ps/replica:0/task:2/cpu:0"))
  {

    popts_.node_to_loc = SplitByWorker;
    popts_.new_name = [this](const string& prefix) {
      return this->g_->NewName(prefix); 
    };
    popts_.get_incarnation = [](const string& name) {
      return (name[0] - 'A') + 100;
    };

  }

  shared_ptr<Graph> ConstructGraph() {
    TF_EXPECT_OK(in_.ToGraphDef(&in_graph_def_));
    GraphConstructorOptions opts;
    opts.expect_device_spec = true;
    g_.reset(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(ConvertGraphDefToGraph(opts, in_graph_def_, g_.get()));
    return g_;
  }

protected:
  Scope in_;
  GraphDef in_graph_def_;
  Scope scope_worker_;
  Scope scope_ps1_;
  Scope scope_ps2_;
  PartitionOptions popts_;
  shared_ptr<Graph> g_;
};

REGISTER_OP("FloatInput")
.Output("o: float")
.SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("BoolInput")
.Output("o: bool")
.SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("Combine")
.Input("a: float")
.Input("b: float")
.Output("o: float")
.SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("FakeIdentity")
.Input("a: float")
.Output("o: float")
.SetShapeFn(shape_inference::ScalarShape);

Output ConstructOp(const Scope& scope, const string& op_type,
                   const string &device,
                   const gtl::ArraySlice<Input>& inputs,
                   const gtl::ArraySlice<Node*>& control_inputs = {}) {

  if (op_type == "VariableV2") {
    auto v = ops::Variable(scope, {}, DT_FLOAT);
    v.node()->set_assigned_device_name(device);
    return v;
  }
  if (!scope.ok()) {
      return Output();
  }
  const string unique_name = scope.GetUniqueNameForOp(op_type);
  auto builder =
    NodeBuilder(unique_name, op_type, scope.graph()->op_registry());
  builder.Device(device);
  for (auto const& input : inputs) {
    builder.Input(ops::NodeOut(input.node(), input.index()));
  }
  if (!control_inputs.empty()) {
    builder.ControlInputs(control_inputs);
  }
  scope.UpdateBuilder(&builder);
  Node* ret;
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) {
      return Output();
  }
  scope.UpdateStatus(scope.DoShapeInference(ret));
  return Output(ret);
}

Output FloatInput(const Scope& scope, const string &device) {
  return ConstructOp(scope, "FloatInput", device, {});
}

Output BoolInput(const Scope& scope, const string &device) {
  return ConstructOp(scope, "BoolInput", device, {});
}

Output Combine(const Scope& scope, const string &device, Input a, Input b) {
  return ConstructOp(scope, "Combine", device, {a, b});
}

TEST_F(DistGraphPartitionTest, testSplitSubGraph) {
  string worker_device = "/job:worker/replica:0/task:0/cpu:0";
  string ps1_device = "/job:ps/replica:0/task:1/cpu:0";
  string ps2_device = "/job:ps/replica:0/task:2/cpu:0";
  setenv("TASK_INDEX","0",1);

  auto a = FloatInput(in_.WithOpName("A"), ps1_device);
  auto b = FloatInput(in_.WithOpName("B"), ps2_device);
  auto c = Combine(in_.WithOpName("C"), worker_device, a, b);

  shared_ptr<Graph> g = ConstructGraph();
  SubGraph worker_sub_graph;
  vector<SubGraph> ps_sub_graphs;

  TrainGraphPartitioner gp(popts_, g.get(), false, false);
  Status s = gp.SplitGraph(&worker_sub_graph, &ps_sub_graphs);
  ASSERT_TRUE(s.ok());
  ASSERT_EQ((size_t)1, worker_sub_graph.GetNodes().size());
  ASSERT_EQ((size_t)2, ps_sub_graphs.size());
  ASSERT_EQ((size_t)1, ps_sub_graphs[0].GetNodes().size());
  ASSERT_EQ((size_t)1, ps_sub_graphs[1].GetNodes().size());
  ASSERT_EQ("C", worker_sub_graph.GetNodes()[0]->name());
  ASSERT_TRUE(ps_sub_graphs[0].GetNodes()[0]->name() !=
              ps_sub_graphs[1].GetNodes()[0]->name());
}

TEST_F(DistGraphPartitionTest, testCompleteVariableWithOnePS) {
  string worker_device = "/job:worker/replica:0/task:0/cpu:0";
  string ps1_device = "/job:ps/replica:0/task:1/cpu:0";
  string ps2_device = "/job:ps/replica:0/task:2/cpu:0";
  setenv("TASK_INDEX","0",1);

  auto v = ops::Variable(in_, {}, DT_FLOAT);
  v.node()->set_assigned_device_name(ps1_device);
  auto a = FloatInput(in_.WithOpName("A"), ps1_device);
  auto b = FloatInput(in_.WithOpName("B"), ps1_device);
  auto c = Combine(in_.WithOpName("C"), ps1_device, a, v);
  auto d = Combine(in_.WithOpName("D"), ps1_device, b, v);
  auto e = FloatInput(in_.WithOpName("E"), worker_device);

  shared_ptr<Graph> g = ConstructGraph();
  SubGraph worker_sub_graph;
  vector<SubGraph> ps_sub_graphs;

  TrainGraphPartitioner gp(popts_, g.get(), false, true);
  Status s = gp.SplitGraph(&worker_sub_graph, &ps_sub_graphs, false);
  TF_ASSERT_OK(s);
  ASSERT_EQ((size_t)1, worker_sub_graph.GetNodes().size());
  cout << "debug:ps_sub_graph node size:" << ps_sub_graphs[0].GetNodes().size() << endl;
  ASSERT_EQ((size_t)2, ps_sub_graphs.size());
  ASSERT_EQ((size_t)3, ps_sub_graphs[0].GetNodes().size());
  ASSERT_EQ((size_t)3, ps_sub_graphs[1].GetNodes().size());
  bool flag = false;
  for(auto &node : ps_sub_graphs[0].GetNodes()) {
    if (node->def().op() == "VariableV2") flag = true;
  }
  ASSERT_TRUE(flag);
  flag = false;
  for(auto &node : ps_sub_graphs[1].GetNodes()) {
    if (node->def().op() == "VariableV2") flag = true;
  }
  ASSERT_TRUE(flag);
}

int HasVariableCount(const SubGraph &sub_graph) {
  int c = 0;
  for (const Node* node : sub_graph.GetNodes()) {
    if (node->def().op() == "VariableV2") {
      c ++;
    }
  }
  return c;
}

TEST_F(DistGraphPartitionTest, testCompleteVariableWithCrossPS) {
  string worker_device = "/job:worker/replica:0/task:0/cpu:0";
  string ps1_device = "/job:ps/replica:0/task:1/cpu:0";
  string ps2_device = "/job:ps/replica:0/task:2/cpu:0";
  setenv("TASK_INDEX","0",1);

  auto v = ops::Variable(in_, {}, DT_FLOAT);
  v.node()->set_assigned_device_name(ps1_device);
  auto a = FloatInput(in_.WithOpName("A"), ps1_device);
  auto b = FloatInput(in_.WithOpName("B"), ps2_device);
  auto c = Combine(in_.WithOpName("C"), ps1_device, a, v);
  auto d = Combine(in_.WithOpName("D"), ps2_device, b, v);
  auto e = FloatInput(in_.WithOpName("E"), worker_device);

  shared_ptr<Graph> g = ConstructGraph();
  SubGraph worker_sub_graph;
  vector<SubGraph> ps_sub_graphs;

  TrainGraphPartitioner gp(popts_, g.get(), false, false);
  Status s = gp.SplitGraph(&worker_sub_graph, &ps_sub_graphs, false);
  ASSERT_TRUE(s.ok());
  ASSERT_EQ((size_t)1, worker_sub_graph.GetNodes().size());
  ASSERT_EQ((size_t)3, ps_sub_graphs.size());

  for (const SubGraph &ps_graph : ps_sub_graphs) {
    int c = HasVariableCount(ps_graph);
    if (ps_graph.GetNodes().size() == 1) {
      ASSERT_EQ(1, c);
      ASSERT_TRUE(ps_graph.IsOnlyVariable());
      continue;
    }

    if (c == 0) {
      ASSERT_EQ((size_t)2, ps_graph.GetNodes().size());
    } else {
      ASSERT_EQ((size_t)3, ps_graph.GetNodes().size());
    }
  }
}

TEST_F(DistGraphPartitionTest, testCompleteTwoVariableWithOnePS) {
  string worker_device = "/job:worker/replica:0/task:0/cpu:0";
  string ps1_device = "/job:ps/replica:0/task:1/cpu:0";
  setenv("TASK_INDEX","0",1);

  auto v1 = ConstructOp(in_.WithOpName("V1"), "VariableV2", ps1_device, {});
  auto v2 = ConstructOp(in_.WithOpName("V2"), "VariableV2", ps1_device, {});
  auto a = ConstructOp(in_.WithOpName("A"), "FakeIdentity", ps1_device, {v1});
  auto b = ConstructOp(in_.WithOpName("B"), "FakeIdentity", ps1_device, {v2});
  auto e = FloatInput(in_.WithOpName("E"), worker_device);

  shared_ptr<Graph> g = ConstructGraph();
  SubGraph worker_sub_graph;
  vector<SubGraph> ps_sub_graphs;

  TrainGraphPartitioner gp(popts_, g.get(), false, true);
  Status s = gp.SplitGraph(&worker_sub_graph, &ps_sub_graphs, false);
  ASSERT_TRUE(s.ok());
  ASSERT_EQ((size_t)1, worker_sub_graph.GetNodes().size());
  ASSERT_EQ((size_t)2, ps_sub_graphs.size());
  ASSERT_EQ((size_t)2, ps_sub_graphs[0].GetNodes().size());
  ASSERT_EQ((size_t)2, ps_sub_graphs[1].GetNodes().size());
  ASSERT_EQ(1, HasVariableCount(ps_sub_graphs[0]));
  ASSERT_EQ(1, HasVariableCount(ps_sub_graphs[1]));
}

TEST_F(DistGraphPartitionTest, testMergePsGraph) {
  string worker_device = "/job:worker/replica:0/task:0/cpu:0";
  string ps1_device = "/job:ps/replica:0/task:1/cpu:0";
  setenv("TASK_INDEX","0",1);

  auto v1 = ConstructOp(in_.WithOpName("V1"), "VariableV2", ps1_device, {});
  auto v2 = ConstructOp(in_.WithOpName("V2"), "VariableV2", ps1_device, {});
  auto a = ConstructOp(in_.WithOpName("A"), "FakeIdentity", ps1_device, {v1});
  auto b = ConstructOp(in_.WithOpName("B"), "FakeIdentity", ps1_device, {v2});
  auto c = FloatInput(in_.WithOpName("C"), worker_device);

  shared_ptr<Graph> g = ConstructGraph();
  SubGraph worker_sub_graph;
  vector<SubGraph> ps_sub_graphs;

  TrainGraphPartitioner gp(popts_, g.get(), false, false);
  Status s = gp.SplitGraph(&worker_sub_graph, &ps_sub_graphs);
  TF_ASSERT_OK(s);
  ASSERT_EQ((size_t)1, worker_sub_graph.GetNodes().size());
  ASSERT_EQ((size_t)1, ps_sub_graphs.size());
  ASSERT_EQ(2, HasVariableCount(ps_sub_graphs[0]));
}

TEST_F(DistGraphPartitionTest, testMergePsGraphWithVariableDepandency) {
  string worker_device = "/job:worker/replica:0/task:0/cpu:0";
  string ps1_device = "/job:ps/replica:0/task:1/cpu:0";
  setenv("TASK_INDEX","0",1);

  auto v1 = ConstructOp(in_.WithOpName("V1"), "VariableV2", ps1_device, {});
  auto v2 = ConstructOp(in_.WithOpName("V2"), "VariableV2", ps1_device, {});
  auto a = ConstructOp(in_.WithOpName("A"), "FakeIdentity", ps1_device, {v1});
  auto b = ConstructOp(in_.WithOpName("B"), "FakeIdentity", worker_device, {a});
  auto c = ConstructOp(in_.WithOpName("C"), "Combine", ps1_device, {b, v2});
  auto d = ConstructOp(in_.WithOpName("D"), "Combine", worker_device, {b, c});

  shared_ptr<Graph> g = ConstructGraph();
  SubGraph worker_sub_graph;
  vector<SubGraph> ps_sub_graphs;

  TrainGraphPartitioner gp(popts_, g.get(), false, true);
  Status s = gp.SplitGraph(&worker_sub_graph, &ps_sub_graphs);
  TF_ASSERT_OK(s);
  ASSERT_EQ((size_t)2, worker_sub_graph.GetNodes().size());
  ASSERT_EQ((size_t)2, ps_sub_graphs.size());
}

TEST_F(DistGraphPartitionTest, testMergePsGraphWithOnlyVariable) {
  string worker_device = "/job:worker/replica:0/task:1/cpu:0";
  string ps1_device = "/job:ps/replica:0/task:1/cpu:0";
  setenv("TASK_INDEX","1",1);

  auto v1 = ops::Variable(in_, {}, DT_FLOAT);
  v1.node()->set_assigned_device_name(ps1_device);
  auto v2 = ops::Variable(in_, {}, DT_FLOAT);
  v2.node()->set_assigned_device_name(ps1_device);
  auto a = ConstructOp(in_.WithOpName("A"), "FakeIdentity", ps1_device, {v1});
  auto b = ConstructOp(in_.WithOpName("B"), "FakeIdentity", worker_device, {v2});

  shared_ptr<Graph> g = ConstructGraph();
  SubGraph worker_sub_graph;
  vector<SubGraph> ps_sub_graphs;

  TrainGraphPartitioner gp(popts_, g.get(), false, false);
  Status s = gp.SplitGraph(&worker_sub_graph, &ps_sub_graphs);
  TF_ASSERT_OK(s);
  ASSERT_EQ((size_t)1, worker_sub_graph.GetNodes().size());
  ASSERT_EQ((size_t)2, ps_sub_graphs.size());
}

TEST_F(DistGraphPartitionTest, testMergePsGraphWithLoopOnWorker) {
  string worker_device = "/job:worker/replica:0/task:1/cpu:0";
  string ps1_device = "/job:ps/replica:0/task:1/cpu:0";
  setenv("TASK_INDEX","1",1);

  using namespace ::tensorflow::ops;
  auto a1 = BoolInput(in_.WithOpName("A1"), ps1_device);
  ::tensorflow::ops::internal::Enter a2(in_.WithOpName("A2"), a1, "foo");
  a2.node()->set_assigned_device_name(worker_device);
  auto a3 = ::tensorflow::ops::Merge(in_.WithOpName("A3"),
                                     {a2, Input("A5", 0, DT_BOOL)})
            .output;
  a3.node()->set_assigned_device_name(worker_device);
  auto l = LoopCond(in_.WithOpName("A4"), a3);
  l.node()->set_assigned_device_name(worker_device);

  auto b1 = Identity(in_.WithOpName("B1"), a3);
  b1.node()->set_assigned_device_name(worker_device);

  auto n = NextIteration(in_.WithOpName("A5"), b1);
  n.node()->set_assigned_device_name(worker_device);

  shared_ptr<Graph> g = ConstructGraph();
  SubGraph worker_sub_graph;
  vector<SubGraph> ps_sub_graphs;

  TrainGraphPartitioner gp(popts_, g.get(), true, false);
  Status s = gp.SplitGraph(&worker_sub_graph, &ps_sub_graphs);
  TF_ASSERT_OK(s);
}


TEST_F(DistGraphPartitionTest, testCompleteSubGraphsSimple) {
  string worker_device = "/job:worker/replica:0/task:1/cpu:0";
  string ps1_device = "/job:ps/replica:0/task:1/cpu:0";
  setenv("TASK_INDEX","1",1);

  auto a = FloatInput(in_.WithOpName("A"), worker_device);
  auto b = ConstructOp(in_.WithOpName("B"), "FakeIdentity", ps1_device, {a});
  auto c = ConstructOp(in_.WithOpName("C"), "FakeIdentity", worker_device, {b});

  shared_ptr<Graph> g = ConstructGraph();
  SubGraph worker_sub_graph;
  vector<SubGraph> ps_sub_graphs;

  TrainGraphPartitioner gp(popts_, g.get(), true, true);
  Status s = gp.SplitGraph(&worker_sub_graph, &ps_sub_graphs);
  TF_ASSERT_OK(s);
  ASSERT_EQ((size_t)2, worker_sub_graph.GetNodes().size());
  ASSERT_EQ((size_t)1, ps_sub_graphs.size());
  s = gp.CompleteSubGraphs(&ps_sub_graphs);
  TF_ASSERT_OK(s);
  ASSERT_EQ((size_t)1, ps_sub_graphs[0].GetNodes().size());
  auto graph_def = ps_sub_graphs[0].GetGraphDef();
  ASSERT_EQ(3, graph_def.node_size());
  ASSERT_EQ((size_t)1, ps_sub_graphs[0].GetInputEdges().size());
  ASSERT_EQ((size_t)1, ps_sub_graphs[0].GetOutputEdges().size());
}

TEST_F(DistGraphPartitionTest, testCompleteSubGraphsWithCrossPS) {
  string worker_device = "/job:worker/replica:0/task:1/cpu:0";
  string ps1_device = "/job:ps/replica:0/task:1/cpu:0";
  string ps2_device = "/job:ps/replica:0/task:2/cpu:0";
  setenv("TASK_INDEX","1",1);

  auto a = FloatInput(in_.WithOpName("A"), ps1_device);
  auto b = ConstructOp(in_.WithOpName("B"), "FakeIdentity", ps2_device, {a});
  auto c = FloatInput(in_.WithOpName("C"), worker_device);

  shared_ptr<Graph> g = ConstructGraph();
  SubGraph worker_sub_graph;
  vector<SubGraph> ps_sub_graphs;

  TrainGraphPartitioner gp(popts_, g.get(), true, false);
  Status s = gp.SplitGraph(&worker_sub_graph, &ps_sub_graphs);
  TF_ASSERT_OK(s);
  ASSERT_EQ((size_t)1, worker_sub_graph.GetNodes().size());
  ASSERT_EQ((size_t)2, ps_sub_graphs.size());
  s = gp.CompleteSubGraphs(&ps_sub_graphs);
  TF_ASSERT_OK(s);
  for (auto &ps_graph : ps_sub_graphs) {
    ASSERT_EQ((size_t)1, ps_graph.GetNodes().size());
    auto graph_def = ps_graph.GetGraphDef();
    ASSERT_EQ(2, graph_def.node_size());
    bool flag = false;
    if(ps_graph.GetLoc() == "/job:ps/replica:0/task:1") {
      for(int i = 0; i < graph_def.node_size(); i++) {
        if(graph_def.node(i).op() == "_Send") flag = true;
      }
    }
    if(ps_graph.GetLoc() == "/job:ps/replica:0/task:2") {
      for(int i = 0; i < graph_def.node_size(); i++) {
        if(graph_def.node(i).op() == "_Recv") flag = true;
      }
    }
    ASSERT_TRUE(flag);
  }
}

TEST_F(DistGraphPartitionTest, testCompleteSubGraphsWithControlInput) {
  string ps_device = "/job:ps/replica:0/task:1/cpu:0";
  string worker_device = "/job:worker/replica:0/task:2/cpu:0";
  setenv("TASK_INDEX","2",1);

  auto a = FloatInput(in_.WithOpName("A"), worker_device);
  auto b = ConstructOp(in_.WithOpName("B"), "FloatInput", ps_device, {}, {a.node()});
  auto c = ConstructOp(in_.WithOpName("C"), "FloatInput", worker_device, {}, {b.node()});

  shared_ptr<Graph> g = ConstructGraph();
  SubGraph worker_sub_graph;
  vector<SubGraph> ps_sub_graphs;

  TrainGraphPartitioner gp(popts_, g.get(), true, true);
  Status s = gp.SplitGraph(&worker_sub_graph, &ps_sub_graphs);
  TF_ASSERT_OK(s);
  ASSERT_EQ((size_t)2, worker_sub_graph.GetNodes().size());
  ASSERT_EQ((size_t)1, ps_sub_graphs.size());
  s = gp.CompleteSubGraphs(&ps_sub_graphs);
  TF_ASSERT_OK(s);
  ASSERT_EQ((size_t)1, ps_sub_graphs[0].GetInputEdges().size());
  ASSERT_EQ("", ps_sub_graphs[0].GetInputEdges()[0].first);
  ASSERT_EQ((size_t)1, ps_sub_graphs[0].GetOutputEdges().size());
  ASSERT_EQ("", ps_sub_graphs[0].GetOutputEdges()[0].first);
}

TEST_F(DistGraphPartitionTest, testCompleteWorkerSimple) {
  string ps_device = "/job:ps/replica:0/task:1/cpu:0";
  string worker_device = "/job:worker/replica:0/task:2/cpu:0";

  setenv("TASK_INDEX","2",1);

  auto a = FloatInput(in_.WithOpName("A"), worker_device);
  auto b = ConstructOp(in_.WithOpName("B"), "FakeIdentity", ps_device, {a});
  auto c = ConstructOp(in_.WithOpName("C"), "FakeIdentity", worker_device, {b});

  shared_ptr<Graph> g = ConstructGraph();
  SubGraph worker_sub_graph;
  vector<SubGraph> ps_sub_graphs;

  TrainGraphPartitioner gp(popts_, g.get(), true, false);
  Status s = gp.SplitGraph(&worker_sub_graph, &ps_sub_graphs);
  TF_ASSERT_OK(s);
  ASSERT_EQ((size_t)2, worker_sub_graph.GetNodes().size());
  ASSERT_EQ((size_t)1, ps_sub_graphs.size());
  s = gp.CompleteSubGraphs(&ps_sub_graphs);
  TF_ASSERT_OK(s);
  s = gp.CompleteMainGraph(ps_sub_graphs, &worker_sub_graph);
  TF_ASSERT_OK(s);
  ASSERT_EQ((size_t)3, worker_sub_graph.GetGraphDef().node_size());
}

TEST_F(DistGraphPartitionTest, testCompleteWorkerCrossPS) {
  string ps1_device = "/job:ps/replica:0/task:1/cpu:0";
  string ps2_device = "/job:ps/replica:0/task:2/cpu:0";
  string worker_device = "/job:worker/replica:0/task:2/cpu:0";
  setenv("TASK_INDEX","2",1);

  auto a = FloatInput(in_.WithOpName("A"), ps1_device);
  auto b = ConstructOp(in_.WithOpName("B"), "FloatInput", ps2_device, {}, {a.node()});
  auto c = ConstructOp(in_.WithOpName("C"), "FloatInput", worker_device, {}, {b.node()});

  shared_ptr<Graph> g = ConstructGraph();
  SubGraph worker_sub_graph;
  vector<SubGraph> ps_sub_graphs;

  TrainGraphPartitioner gp(popts_, g.get(), true, true);
  Status s = gp.SplitGraph(&worker_sub_graph, &ps_sub_graphs);
  TF_ASSERT_OK(s);
  ASSERT_EQ((size_t)1, worker_sub_graph.GetNodes().size());
  ASSERT_EQ((size_t)2, ps_sub_graphs.size());
  s = gp.CompleteSubGraphs(&ps_sub_graphs);
  TF_ASSERT_OK(s);
  s = gp.CompleteMainGraph(ps_sub_graphs, &worker_sub_graph);
  TF_ASSERT_OK(s);
  auto graph_def = worker_sub_graph.GetGraphDef();
  ASSERT_EQ((size_t)4, graph_def.node_size());
  bool flag = true;
  for(int i = 0; i < graph_def.node_size(); i++) {
    auto node_def = graph_def.node(i);
    if(node_def.op() == "NoOp") {
      flag = true;
      ASSERT_EQ((size_t)1, node_def.input_size());
      cout << node_def.input(0) << endl;
      ASSERT_TRUE(node_def.input(0).find("^run_graph_job_ps_replica_0_task_1") !=
                  string::npos);
    }
  }
  ASSERT_TRUE(flag);
}

}  // namespace tensorflow

