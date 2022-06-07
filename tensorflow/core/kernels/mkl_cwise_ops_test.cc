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

#ifdef INTEL_MKL

#include "absl/strings/match.h"
#include "dnnl.hpp"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/math_ops_internal.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/mkl_util.h"

// Compare performance of default Tensorflow cwise kernels (Eigen) with
// OneDNN kernels on CPU.
// Before running these benchmarks configure OpenMP environment variables:
//   export KMP_BLOCKTIME=0
//   export OMP_NUM_THREADS=${num_threads}
//   export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
//
// Then you could run below command to test OneDNN kernels performance:
// $bazel run --config opt --config=mkl
// //tensorflow/core/kernels/mkl:mkl_cwise_ops_test \
//   --  --benchmarks=..

namespace tensorflow {

// --------------------------------------------------------------------------//
//  Test OneDNN cwise kernels accuracy with Eigen kernels                       //
// --------------------------------------------------------------------------//
static const uint8 dummy_tensor[] = {0, 0, 0, 0, 0, 0, 0, 0};
static const TensorShape dummy_shape({8});

using GraphRunner =
    std::function<void(const Tensor& input0, const Tensor& input1,
                       const string& op_name, Tensor* output)>;

template <typename T>
class CommonTestUtilities : public OpsTestBase {
 public:
  void PerformConversion(DataType dtype, const Tensor& tensor,
                         const Tensor& mkl_meta_tensor, Tensor* output) {
    // Create an OneDNN to TF conversion node and execute it
    TF_EXPECT_OK(NodeDefBuilder("mkl_to_tf_op", "_MklToTf")
                     .Input(FakeInput(dtype))     // Input
                     .Input(FakeInput(DT_UINT8))  // OneDNN second tensor
                     .Attr("T", dtype)
                     .Attr("_kernel", "MklLayoutDependentOp")
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    AddInputFromArray<T>(tensor.shape(), tensor.flat<T>());
    AddInputFromArray<uint8>(mkl_meta_tensor.shape(),
                             mkl_meta_tensor.flat<uint8>());
    TF_ASSERT_OK(RunOpKernel());

    *output = *GetOutput(0);
  }

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the output Tensor.
  static void RunAndFetch(const tensorflow::Scope& root, const string& fetch,
                          Tensor* output) {
    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    // We really want to make sure that graph executed exactly as we passed it
    // to the session, so we disable various optimizations.
    tensorflow::SessionOptions session_options;

    // Disable common runtime constant folding.
    session_options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_opt_level(OptimizerOptions::L0);

    // Disable Grappler optimizations for tests.
    tensorflow::RewriterConfig* cfg =
        session_options.config.mutable_graph_options()
            ->mutable_rewrite_options();
    cfg->set_constant_folding(tensorflow::RewriterConfig::OFF);
    cfg->set_layout_optimizer(tensorflow::RewriterConfig::OFF);
    cfg->set_remapping(tensorflow::RewriterConfig::OFF);

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(session_options));

    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> output_tensors;
    TF_ASSERT_OK(session->Run({}, {fetch}, {}, &output_tensors));

    *output = output_tensors[0];
  }

  void TestBody() {}

  static void VerifyTensorsClose(const GraphRunner& run,
                                 const GraphRunner& run_mkl,
                                 const string& op_name,
                                 const TensorShape& input0_shape,
                                 const TensorShape& input1_shape) {
    float atol = 1e-6, rtol = 1e-6;
    DataType dtype = DataTypeToEnum<T>::v();
    Tensor input0(dtype, input0_shape);
    Tensor input1(dtype, input1_shape);
    input0.flat<T>() = input0.flat<T>().setRandom();
    input1.flat<T>() = input1.flat<T>().setRandom();

    Tensor output;
    Tensor mkl_output;
    run(input0, input1, op_name, &output);
    run_mkl(input0, input1, op_name, &mkl_output);

    ASSERT_EQ(output.dtype(), mkl_output.dtype());
    ASSERT_EQ(output.shape(), mkl_output.shape());
    if (dtype == DT_BFLOAT16) {
      rtol = 1e-2;
      atol = 1e-2;
    }
    test::ExpectClose(output, mkl_output, atol, rtol);
  }
};

template <typename T>
class CwiseOpsTest : public OpsTestBase {
 protected:
  void VerifyCwiseOps(const string& op_name, const TensorShape& input0_shape,
                      const TensorShape& input1_shape) {
    const GraphRunner run = [this](const Tensor& input0, const Tensor& input1,
                                   const string& op_name, Tensor* output) {
      auto root = tensorflow::Scope::NewRootScope();
      auto input0_op =
          ops::Const(root.WithOpName("input0"), Input::Initializer(input0));
      auto input1_op =
          ops::Const(root.WithOpName("input1"), Input::Initializer(input1));
      Output cwise_op;
      if (op_name == "Add") {
        cwise_op =
            ops::Add(root.WithOpName(strings::StrCat("Default_", op_name)),
                     input0_op, input1_op);
      } else if (op_name == "AddV2") {
        cwise_op =
            ops::AddV2(root.WithOpName(strings::StrCat("Default_", op_name)),
                       input0_op, input1_op);
      } else if (op_name == "Sub") {
        cwise_op =
            ops::Sub(root.WithOpName(strings::StrCat("Default_", op_name)),
                     input0_op, input1_op);
      } else if (op_name == "Maximum") {
        cwise_op =
            ops::Maximum(root.WithOpName(strings::StrCat("Default_", op_name)),
                         input0_op, input1_op);
      } else if (op_name == "SquaredDifference") {
        cwise_op = ops::SquaredDifference(
            root.WithOpName(strings::StrCat("Default_", op_name)), input0_op,
            input1_op);
      }
      auto output_op = ops::Identity(root.WithOpName("output"), cwise_op);

      CommonTestUtilities<T>::RunAndFetch(root, "output", output);
    };

    const GraphRunner run_mkl = [this](const Tensor& input0,
                                       const Tensor& input1,
                                       const string& op_name, Tensor* output) {
      DataType dtype = DataTypeToEnum<T>::v();

      TF_EXPECT_OK(NodeDefBuilder(strings::StrCat("Mkl_", op_name),
                                  strings::StrCat("_Mkl", op_name))
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(DT_UINT8))
                       .Input(FakeInput(DT_UINT8))
                       .Attr("T", dtype)
                       .Attr("_kernel", "MklLayoutDependentOp")
                       .Finalize(node_def()));
      TF_EXPECT_OK(InitOp());

      AddInputFromArray<T>(input0.shape(), input0.flat<T>());
      AddInputFromArray<T>(input1.shape(), input1.flat<T>());
      AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
      AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
      TF_ASSERT_OK(RunOpKernel());

      CommonTestUtilities<T> test_util;
      test_util.PerformConversion(dtype, *GetOutput(0), *GetOutput(1), output);
    };
    CommonTestUtilities<T>::VerifyTensorsClose(run, run_mkl, op_name,
                                               input0_shape, input1_shape);
  }
};

TYPED_TEST_SUITE_P(CwiseOpsTest);

#define VERIFY_CWISE_OPS(name, shape0, shape1) \
  this->VerifyCwiseOps(#name, shape0, shape1)

#define SHAPE(...) TensorShape({__VA_ARGS__})

#define TEST_CASE(name, case_name, shape_info_0, shape_info_1) \
  TYPED_TEST_P(CwiseOpsTest, case_name) {                      \
    VERIFY_CWISE_OPS(name, shape_info_0, shape_info_1);        \
  }

#define TEST_NAME(name, shape_info) name##shape_info
#define X_TEST_NAME(name, shape_info) TEST_NAME(name, shape_info)

#define SHAPE_NAME_2D(A0, B0, A1, B1) OpTest_##A0##_##B0##_##A1##_##B1
#define TEST_NAME_2D(name, A0, B0, A1, B1) \
  X_TEST_NAME(name, SHAPE_NAME_2D(A0, B0, A1, B1))
#define TEST_CASE_2D(name, A0, B0, A1, B1)                           \
  TEST_CASE(name, TEST_NAME_2D(name, A0, B0, A1, B1), SHAPE(A0, B0), \
            SHAPE(A1, B1))

#define SHAPE_NAME_3D(A0, B0, C0, A1, B1, C1) \
  OpTest_##A0##_##B0##_##C0##_##A1##_##B1##_##C1
#define TEST_NAME_3D(name, A0, B0, C0, A1, B1, C1) \
  X_TEST_NAME(name, SHAPE_NAME_3D(A0, B0, C0, A1, B1, C1))
#define TEST_CASE_3D(name, A0, B0, C0, A1, B1, C1)            \
  TEST_CASE(name, TEST_NAME_3D(name, A0, B0, C0, A1, B1, C1), \
            SHAPE(A0, B0, C0), SHAPE(A1, B1, C1))

#define SHAPE_NAME_4D(A0, B0, C0, D0, A1, B1, C1, D1) \
  OpTest_##A0##_##B0##_##C0##_##D0##_##A1##_##B1##_##C1##_##D1
#define TEST_NAME_4D(name, A0, B0, C0, D0, A1, B1, C1, D1) \
  X_TEST_NAME(name, SHAPE_NAME_4D(A0, B0, C0, D0, A1, B1, C1, D1))
#define TEST_CASE_4D(name, A0, B0, C0, D0, A1, B1, C1, D1)            \
  TEST_CASE(name, TEST_NAME_4D(name, A0, B0, C0, D0, A1, B1, C1, D1), \
            SHAPE(A0, B0, C0, D0), SHAPE(A1, B1, C1, D1))

#define TEST_ALL_SHAPE(name)                 \
  TEST_CASE_2D(name, 1, 3, 3, 3)             \
  TEST_CASE_2D(name, 3, 3, 3, 3)             \
  TEST_CASE_3D(name, 1, 3, 3, 2, 3, 3)       \
  TEST_CASE_3D(name, 2, 3, 3, 2, 3, 3)       \
  TEST_CASE_4D(name, 1, 2, 3, 3, 4, 2, 3, 3) \
  TEST_CASE_4D(name, 4, 2, 3, 3, 4, 2, 3, 3) \
  TEST_CASE_4D(name, 4, 1, 3, 3, 4, 5, 3, 3) \
  TEST_CASE_4D(name, 4, 5, 3, 3, 4, 1, 3, 3)

#define ALL_SHAPE_TEST_NAME(name)                                 \
  TEST_NAME_2D(name, 1, 3, 3, 3), TEST_NAME_2D(name, 3, 3, 3, 3), \
      TEST_NAME_3D(name, 1, 3, 3, 2, 3, 3),                       \
      TEST_NAME_3D(name, 2, 3, 3, 2, 3, 3),                       \
      TEST_NAME_4D(name, 1, 2, 3, 3, 4, 2, 3, 3),                 \
      TEST_NAME_4D(name, 4, 2, 3, 3, 4, 2, 3, 3),                 \
      TEST_NAME_4D(name, 4, 1, 3, 3, 4, 5, 3, 3),                 \
      TEST_NAME_4D(name, 4, 5, 3, 3, 4, 1, 3, 3)

TEST_ALL_SHAPE(Add)
TEST_ALL_SHAPE(AddV2)
TEST_ALL_SHAPE(Sub)
TEST_ALL_SHAPE(Maximum)

// Note, Eigen version does not support bfloat16
// TEST_ALL_SHAPE(SquaredDifference)

#define REGISTER_TEST(name, ...) \
  REGISTER_TYPED_TEST_SUITE_P(CwiseOpsTest, __VA_ARGS__);

REGISTER_TEST(CwiseOpsTest, ALL_SHAPE_TEST_NAME(Add),
              ALL_SHAPE_TEST_NAME(AddV2), ALL_SHAPE_TEST_NAME(Sub),
              ALL_SHAPE_TEST_NAME(Maximum));

using CwiseOpsDataTypes = ::testing::Types<float, bfloat16>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, CwiseOpsTest, CwiseOpsDataTypes);

// --------------------------------------------------------------------------//
// Test OneDNN element-wise kernels performance with Eigen                      //
// --------------------------------------------------------------------------//
template <typename T>
static Graph* Cwise(const string& op_name, const string& kind,
                    const TensorShape& shape0, const TensorShape& shape1) {
  auto* graph = new Graph(OpRegistry::Global());
  const string node_name = kind + "_" + op_name;
  const bool isDefault = (kind == "Default");

  DataType dtype = DataTypeToEnum<T>::v();

  auto init_input = [&](const TensorShape& shape, const std::string& name) {
    DataType dtype = DataTypeToEnum<T>::v();
    Tensor input_t(dtype, shape);
    input_t.flat<T>().setRandom();
    return test::graph::Constant(graph, input_t, name);
  };

  Node* input0 = init_input(shape0, "input0");
  Node* input1 = init_input(shape1, "input1");

  Node* not_mkl_shape =
      test::graph::Constant(graph, GetMklMetaTensor(), "not_mkl");

  auto nodeBuilder = NodeBuilder(graph->NewName(node_name), isDefault ? op_name : "_Mkl" + op_name)
                    .Input(input0)
                    .Input(input1)
                    .Attr("T", dtype);

  isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
                                       .Input(not_mkl_shape)
                                       .Attr("_kernel", "MklLayoutDependentOp");

  // MKL forward op.
  TF_CHECK_OK(nodeBuilder.Finalize(graph, nullptr));
  return graph;
}

#define BM_Cwise_Base(op, kind, name, shape_info_0, shape_info_1, T, DEVICE, NTH)     \
  static void BM_##kind##_##name##_##NTH(int iters) {                                 \
    int64 num_computed_elements =                                                     \
        shape_info_0.num_elements() > shape_info_1.num_elements()                     \
            ? shape_info_0.num_elements()                                             \
            : shape_info_1.num_elements();                                            \
    int64 flops_per_iter = num_computed_elements;                                     \
    testing::UseRealTime();                                                           \
    SessionOptions opts;                                                              \
    opts.config.set_intra_op_parallelism_threads(NTH);                                \
    testing::ItemsProcessed(static_cast<int64>(iters) * flops_per_iter);              \
    test::Benchmark(#DEVICE, Cwise<T>(#op, #kind, shape_info_0, shape_info_1), &opts) \
        .Run(iters);                                                                  \
  }                                                                                   \
  BENCHMARK(BM_##kind##_##name##_##NTH);                                              \

#define BM_Cwise_kind(op, name, shape_info_0, shape_info_1, T, DEVICE, NTH)     \
  BM_Cwise_Base(op, Default, name, shape_info_0, shape_info_1, T, DEVICE, NTH); \
  BM_Cwise_Base(op, Mkl, name, shape_info_0, shape_info_1, T, DEVICE, NTH);     \

#define BM_Cwise_NTH(op, name, shape_info_0, shape_info_1, T, DEVICE) \
  BM_Cwise_kind(op, name, shape_info_0, shape_info_1, T, DEVICE, 1);  \
  BM_Cwise_kind(op, name, shape_info_0, shape_info_1, T, DEVICE, 4);  \
  BM_Cwise_kind(op, name, shape_info_0, shape_info_1, T, DEVICE, 8);  \

#define BM_Cwise_2D(op, A0, B0, A1, B1, T, DEVICE)                  \
  BM_Cwise_NTH(op, op##_##DEVICE##_##A0##x##B0##_##A1##x##B1##_##T, \
           TensorShape({A0, B0}), TensorShape({A1, B1}), T, DEVICE) \

#define BM_2D(op, A0, B0, A1, B1, T, DEVICE)  \
  BM_Cwise_2D(op, A0, B0, A1, B1, T, DEVICE); \

#define BM_Cwise_4D(op, A0, B0, C0, D0, A1, B1, C1, D1, T, DEVICE)                     \
  BM_Cwise_NTH(                                                                        \
      op, op##_##DEVICE##_##A0##x##B0##x##C0##x##D0##_##A1##x##B1##x##C1##x##D1##_##T, \
      TensorShape({A0, B0, C0, D0}), TensorShape({A1, B1, C1, D1}), T, DEVICE)         \

#define BM_4D(op, A0, B0, C0, D0, A1, B1, C1, D1, T, DEVICE)  \
  BM_Cwise_4D(op, A0, B0, C0, D0, A1, B1, C1, D1, T, DEVICE); \

#define TEST_ALL_SIZES(op, T)                          \
  BM_2D(op, 1, 384, 384, 384, T, cpu);                 \
  BM_2D(op, 384, 384, 384, 384, T, cpu);               \
  BM_4D(op, 1, 12, 384, 384, 1, 1, 384, 384, T, cpu);  \
  BM_4D(op, 1, 12, 384, 384, 1, 12, 384, 384, T, cpu); \

TEST_ALL_SIZES(Add, float)
TEST_ALL_SIZES(Add, bfloat16)
TEST_ALL_SIZES(Mul, float)
TEST_ALL_SIZES(Mul, bfloat16)
}  // namespace tensorflow

#endif  // INTEL_MKL
