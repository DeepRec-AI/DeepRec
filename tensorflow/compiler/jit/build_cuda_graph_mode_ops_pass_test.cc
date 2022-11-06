/* Copyright 2022 The DeepRec Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/build_cuda_graph_mode_ops_pass.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/encapsulate_cuda_graph_mode_subgraphs_pass.h"
#include "tensorflow/compiler/jit/node_matchers.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace cugraphtest {

class BuildCgmodeOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CHECK(DeviceFactory::AddDevices(
              SessionOptions(), "/job:localhost/replica:0/task:0", &devices_)
              .ok());
  }

 private:
  std::vector<std::unique_ptr<Device>> devices_;
};

using ::tensorflow::testing::FindNodeByName;
using ::tensorflow::testing::matchers::Attr;
using ::tensorflow::testing::matchers::CtrlDeps;
using ::tensorflow::testing::matchers::Inputs;
using ::tensorflow::testing::matchers::NodeWith;
using ::tensorflow::testing::matchers::Op;
using ::tensorflow::testing::matchers::Out;
using ::testing::_;

Status BuildCgmodeOps(const Scope& s, const FunctionDefLibrary& fdef_lib,
                      std::unique_ptr<Graph>* result) {
  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  TF_RETURN_IF_ERROR(s.ToGraph(graph.get()));
  FunctionLibraryDefinition flib_def(graph->op_registry(), fdef_lib);

  // Assign all nodes to the GPU device.
  static const char* kGpuDevice = "/job:localhost/replica:0/task:0/cpu:0";
  for (Node* n : graph->nodes()) {
    if (n->requested_device().empty()) {
      n->set_assigned_device_name(kGpuDevice);
    } else {
      n->set_assigned_device_name(n->requested_device());
    }
  }

  FixupSourceAndSinkEdges(graph.get());

  SessionOptions session_options;
  session_options.config.mutable_gpu_options()->set_cuda_graph_enable_jit(true);
  GraphOptimizationPassOptions opt_options;
  opt_options.session_options = &session_options;
  opt_options.flib_def = &flib_def;
  opt_options.graph = &graph;
  BuildCgmodeOpsPass pass;
  TF_RETURN_IF_ERROR(pass.Run(opt_options));
  VLOG(1) << graph->ToGraphDefDebug().DebugString();
  *result = std::move(graph);
  return Status::OK();
}

Status MakeCgmodeCompiledKernel(Graph* graph, const string& callee_name,
                                const string& node_name, int num_constant_args,
                                int num_resource_args, Node** result) {
  NodeDef call_node;
  call_node.set_name(node_name);
  call_node.set_op(callee_name);
  AddNodeAttr(kCgmodeCompiledAttr, true, &call_node);
  AddNodeAttr(kCgmodeNumConstantArgsAttr, num_constant_args, &call_node);
  AddNodeAttr(kCgmodeNumResourceArgsAttr, num_resource_args, &call_node);
  Status s;
  *result = graph->AddNode(call_node, &s);
  return s;
}

Status MakeCgmodeCompiledKernel(Graph* graph, const string& callee_name,
                                const string& node_name, Node** result) {
  return MakeCgmodeCompiledKernel(graph, callee_name, node_name,
                                  /*num_constant_args=*/0,
                                  /*num_resource_args=*/0, result);
}

Node* MakeWrite(const Scope& scope, Output value_to_write, const string& id) {
  Output var_handle = ops::VarHandleOp(scope.WithOpName("Var_" + id), DT_FLOAT,
                                       TensorShape({}));
  ops::AssignVariableOp assign_op(scope.WithOpName("Assignee_" + id),
                                  var_handle, value_to_write);
  return assign_op.operation.node();
}

Node* MakeWrite(const Scope& scope, const string& id) {
  return MakeWrite(
      scope, ops::Const(scope.WithOpName("ValueToAssign" + id), 1.0f), id);
}

FunctionDefLibrary CreateFunctionDefLibWithConstFunction(const string& name) {
  FunctionDefLibrary fdef_lib;
  FunctionDef func = FunctionDefHelper::Create(
      /*function_name=*/name, /*in_def=*/{}, /*out_def=*/{"out: float"},
      /*attr_def*/
      {}, /*node_def=*/{FunctionDefHelper::Const("one", 1.0f)},
      /*ret_def=*/{{"out", "out:output:0"}});
  *fdef_lib.add_function() = std::move(func);
  return fdef_lib;
}

TEST_F(BuildCgmodeOpsTest, OnDevice) {
  const char* kCgmodeDeviceName = "/job:worker/replica:0/task:0/device:GPU:0";
  Scope root =
      Scope::NewRootScope().WithDevice(kCgmodeDeviceName).ExitOnError();

  FunctionDefLibrary fdef_lib =
      CreateFunctionDefLibWithConstFunction("cluster_0");
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(fdef_lib));

  Node* call;
  TF_ASSERT_OK(MakeCgmodeCompiledKernel(root.graph(), "cluster_0", "C", &call));
  call->set_requested_device(kCgmodeDeviceName);
  TF_ASSERT_OK(root.DoShapeInference(call));

  Node* write_op = MakeWrite(root, Output(call), "write_result");

  std::unique_ptr<Graph> graph;
  TF_ASSERT_OK(BuildCgmodeOps(root, fdef_lib, &graph));

  auto cgmode_op =
      NodeWith(Op("_CgmodeRun"), Inputs(Out(NodeWith(Op("_CgmodeCompile")))));
  auto assign_var =
      NodeWith(Op("AssignVariableOp"), Inputs(Out(NodeWith()), Out(cgmode_op)));

  Node* write_op_new = FindNodeByName(graph.get(), write_op->name());
  ASSERT_NE(write_op_new, nullptr);
  EXPECT_THAT(write_op_new, assign_var);
}
}  // namespace cugraphtest
}  // namespace tensorflow
