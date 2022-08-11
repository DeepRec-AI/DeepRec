/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#ifdef INTEL_MKL
#include <omp.h>
#endif

#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

class BiasGradTest : public OpsTestBase {
 protected:

  void CreateOp(DataType dtype) {
    TF_ASSERT_OK(NodeDefBuilder("op", "BiasAddGrad")
                     .Input(FakeInput(dtype))
                     .Finalize(node_def()));
  }
};

TEST_F(BiasGradTest, Brick_int64) {
  CreateOp(DT_INT64);
  TF_ASSERT_OK(InitOp());
  std::vector<int64> in(131072 * 6, 1);
  std::vector<int64> out(131072, 6);

  // input
  AddInputFromArray<int64>(TensorShape({6, 131072}), in);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_INT64, {131072});
  test::FillValues<int64>(&expected, out);
  test::ExpectTensorEqual<int64>(expected, *GetOutput(0));
}

TEST_F(BiasGradTest, small_int64) {
  CreateOp(DT_INT64);
  TF_ASSERT_OK(InitOp());
  std::vector<int64> in(6 * 6, 1);
  std::vector<int64> out(6, 6);

  // input
  AddInputFromArray<int64>(TensorShape({6, 6}), in);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_INT64, {6});
  test::FillValues<int64>(&expected, out);
  test::ExpectTensorEqual<int64>(expected, *GetOutput(0));
}

TEST_F(BiasGradTest, Bar_int64) {
  CreateOp(DT_INT64);
  TF_ASSERT_OK(InitOp());
  std::vector<int64> in(131072 * 6, 1);
  std::vector<int64> out(6, 131072);

  // input
  AddInputFromArray<int64>(TensorShape({131072, 6}), in);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_INT64, {6});
  test::FillValues<int64>(&expected, out);
  test::ExpectTensorEqual<int64>(expected, *GetOutput(0));
}

TEST_F(BiasGradTest, Brick_float) {
  CreateOp(DT_FLOAT);
  TF_ASSERT_OK(InitOp());
  std::vector<float> in(131072 * 6, 1.0);
  std::vector<float> out(131072, 6.0);

  // input
  AddInputFromArray<float>(TensorShape({6, 131072}), in);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_FLOAT, {131072});
  test::FillValues<float>(&expected, out);
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(BiasGradTest, Brick_Long_float) {
  CreateOp(DT_FLOAT);
  TF_ASSERT_OK(InitOp());
  std::vector<float> in(131072 * 97, 1.0);
  std::vector<float> out(131072, 97.0);

  // input
  AddInputFromArray<float>(TensorShape({97, 131072}), in);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_FLOAT, {131072});
  test::FillValues<float>(&expected, out);
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(BiasGradTest, Line_float) {
  CreateOp(DT_FLOAT);
  TF_ASSERT_OK(InitOp());
  std::vector<float> in(262144, 1.0);
  std::vector<float> out(1, 262144.0);

  // input
  AddInputFromArray<float>(TensorShape({262144, 1}), in);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_FLOAT, {1});
  test::FillValues<float>(&expected, out);
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(BiasGradTest, TwoColumn_float) {
  CreateOp(DT_FLOAT);
  TF_ASSERT_OK(InitOp());
  std::vector<float> in(131072 * 2, 1.0);
  std::vector<float> out(2, 131072);

  // input
  AddInputFromArray<float>(TensorShape({131072, 2}), in);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_FLOAT, {2});
  test::FillValues<float>(&expected, out);
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}


TEST_F(BiasGradTest, Brick_double) {
  CreateOp(DT_DOUBLE);
  TF_ASSERT_OK(InitOp());
  std::vector<double> in(131072 * 6, 1.0);
  std::vector<double> out(131072, 6.0);

  // input
  AddInputFromArray<double>(TensorShape({6, 131072}), in);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_DOUBLE, {131072});
  test::FillValues<double>(&expected, out);
  test::ExpectTensorEqual<double>(expected, *GetOutput(0));
}

TEST_F(BiasGradTest, Line_double) {
  CreateOp(DT_DOUBLE);
  TF_ASSERT_OK(InitOp());
  std::vector<double> in(262144, 1.0);
  std::vector<double> out(1, 262144.0);

  // input
  AddInputFromArray<double>(TensorShape({262144, 1}), in);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_DOUBLE, {1});
  test::FillValues<double>(&expected, out);
  test::ExpectTensorEqual<double>(expected, *GetOutput(0));
}

TEST_F(BiasGradTest, Brick_half) {
  CreateOp(DT_HALF);
  TF_ASSERT_OK(InitOp());
  std::vector<Eigen::half> in(131072 * 6, static_cast<Eigen::half>(1.0));
  std::vector<Eigen::half> out(131072, static_cast<Eigen::half>(6.0));

  // input
  AddInputFromArray<Eigen::half>(TensorShape({6, 131072}), in);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_HALF, {131072});
  test::FillValues<Eigen::half>(&expected, out);
  test::ExpectTensorEqual<Eigen::half>(expected, *GetOutput(0));
}

TEST_F(BiasGradTest, Bar_half) {
  CreateOp(DT_HALF);
  TF_ASSERT_OK(InitOp());
  std::vector<Eigen::half> in(131072 * 6, static_cast<Eigen::half>(1.0));
  std::vector<Eigen::half> out(6, static_cast<Eigen::half>(131072.0));

  // input
  AddInputFromArray<Eigen::half>(TensorShape({131072, 6}), in);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_HALF, {6});
  test::FillValues<Eigen::half>(&expected, out);
  test::ExpectTensorEqual<Eigen::half>(expected, *GetOutput(0));
}

TEST_F(BiasGradTest, Line_half) {
  CreateOp(DT_HALF);
  TF_ASSERT_OK(InitOp());
  std::vector<Eigen::half> in(131072 * 2, static_cast<Eigen::half>(1.0));
  std::vector<Eigen::half> out(2, static_cast<Eigen::half>(131072.0));

  // input
  AddInputFromArray<Eigen::half>(TensorShape({131072, 2}), in);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_HALF, {2});
  test::FillValues<Eigen::half>(&expected, out);
  test::ExpectTensorEqual<Eigen::half>(expected, *GetOutput(0));
}

// Benchmarks
static void BM_BiasGradFloat(int iters, int rows, int cols,
                             int num_threads, const string& label) {
  tensorflow::testing::StopTiming();
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  thread::ThreadPool threadpool(Env::Default(), "test", num_threads);
  Eigen::ThreadPoolDevice eigen_cpu_device(threadpool.AsEigenThreadPool(), num_threads);
  device->set_eigen_cpu_device(&eigen_cpu_device);

#ifdef INTEL_MKL
  omp_set_num_threads(num_threads);
#endif

  gtl::InlinedVector<TensorValue, 4> inputs;
  TensorShape shape1({rows, cols});
  Tensor input1(DT_FLOAT, shape1);
  test::FillIota<float>(&input1, 1.0);
  inputs.push_back({nullptr, &input1});

  // BiasGrading op.
  NodeDef bias_grad_node_def;
  Status status = NodeDefBuilder("bias_grad_op", "BiasAddGrad")
                      .Input(FakeInput(DT_FLOAT))
                      .Finalize(&bias_grad_node_def);
  TF_CHECK_OK(status);
  std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                              cpu_allocator(), bias_grad_node_def,
                                              TF_GRAPH_DEF_VERSION, &status));
  TF_CHECK_OK(status);
  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &inputs;
  params.op_kernel = op.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(&params, &attrs);

  std::unique_ptr<OpKernelContext> bias_grad_context(new OpKernelContext(&params));

  op->Compute(bias_grad_context.get());
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    delete bias_grad_context->release_output(0).tensor;
    op->Compute(bias_grad_context.get());
  }
  tensorflow::testing::StopTiming();
  testing::ItemsProcessed(input1.NumElements() * iters);
  testing::SetLabel(label);
}

// M: input_rows
// N: input_cols
#define BM_BiasGrad(M, N, TH, LABEL)                         \
  static void BM_BiasGradFloat_##M##_##N##_##TH(int iters) { \
    BM_BiasGradFloat(iters, M, N, TH, LABEL);                \
  }                                                          \
  BENCHMARK(BM_BiasGradFloat_##M##_##N##_##TH)

BM_BiasGrad(33, 33, 1, "bias_grad_small_square ");
BM_BiasGrad(5, 257, 1, "bias_grad_small_brick  ");
BM_BiasGrad(257, 5, 1, "bias_grad_small_bar    ");
BM_BiasGrad(1025, 1, 1, "bias_grad_small_line   ");
BM_BiasGrad(513, 513, 1, "bias_grad_medium_square");
BM_BiasGrad(33, 8193, 1, "bias_grad_medium_brick ");
BM_BiasGrad(8193, 33, 1, "bias_grad_medium_bar   ");
BM_BiasGrad(262145, 1, 1, "bias_grad_medium_line  ");
BM_BiasGrad(4097, 4097, 1, "bias_grad_large_square ");
BM_BiasGrad(129, 131073, 1, "bias_grad_large_brick  ");
BM_BiasGrad(131073, 129, 1, "bias_grad_large_bar    ");
BM_BiasGrad(2097153, 9, 1, "bias_grad_large_line   ");
BM_BiasGrad(4194305, 5, 1, "bias_grad_large_line   ");
BM_BiasGrad(8388609, 3, 1, "bias_grad_large_line   ");
BM_BiasGrad(16777217, 1, 1, "bias_grad_large_line   ");
BM_BiasGrad(33, 33, 4, "bias_grad_small_square ");
BM_BiasGrad(5, 257, 4, "bias_grad_small_brick  ");
BM_BiasGrad(257, 5, 4, "bias_grad_small_bar    ");
BM_BiasGrad(1025, 1, 4, "bias_grad_small_line   ");
BM_BiasGrad(513, 513, 4, "bias_grad_medium_square");
BM_BiasGrad(33, 8193, 4, "bias_grad_medium_brick ");
BM_BiasGrad(8193, 33, 4, "bias_grad_medium_bar   ");
BM_BiasGrad(262145, 1, 4, "bias_grad_medium_line  ");
BM_BiasGrad(4097, 4097, 4, "bias_grad_large_square ");
BM_BiasGrad(129, 131073, 4, "bias_grad_large_brick  ");
BM_BiasGrad(131073, 129, 4, "bias_grad_large_bar    ");
BM_BiasGrad(2097153, 9, 4, "bias_grad_large_line   ");
BM_BiasGrad(4194305, 5, 4, "bias_grad_large_line   ");
BM_BiasGrad(8388609, 3, 4, "bias_grad_large_line   ");
BM_BiasGrad(16777217, 1, 4, "bias_grad_large_line   ");
BM_BiasGrad(33, 33, 8, "bias_grad_small_square ");
BM_BiasGrad(5, 257, 8, "bias_grad_small_brick  ");
BM_BiasGrad(257, 5, 8, "bias_grad_small_bar    ");
BM_BiasGrad(1025, 1, 8, "bias_grad_small_line   ");
BM_BiasGrad(513, 513, 8, "bias_grad_medium_square");
BM_BiasGrad(33, 8193, 8, "bias_grad_medium_brick ");
BM_BiasGrad(8193, 33, 8, "bias_grad_medium_bar   ");
BM_BiasGrad(262145, 1, 8, "bias_grad_medium_line  ");
BM_BiasGrad(4097, 4097, 8, "bias_grad_large_square ");
BM_BiasGrad(129, 131073, 8, "bias_grad_large_brick  ");
BM_BiasGrad(131073, 129, 8, "bias_grad_large_bar    ");
BM_BiasGrad(2097153, 9, 8, "bias_grad_large_line   ");
BM_BiasGrad(4194305, 5, 8, "bias_grad_large_line   ");
BM_BiasGrad(8388609, 3, 8, "bias_grad_large_line   ");
BM_BiasGrad(16777217, 1, 8, "bias_grad_large_line   ");


static void BM_BiasGradHalf(int iters, int rows, int cols,
                            int num_threads, const string& label) {
  tensorflow::testing::StopTiming();
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  thread::ThreadPool threadpool(Env::Default(), "test", num_threads);
  Eigen::ThreadPoolDevice eigen_cpu_device(threadpool.AsEigenThreadPool(), num_threads);
  device->set_eigen_cpu_device(&eigen_cpu_device);

#ifdef INTEL_MKL
  omp_set_num_threads(num_threads);
#endif

  gtl::InlinedVector<TensorValue, 4> inputs;
  TensorShape shape1({rows, cols});
  Tensor input1(DT_HALF, shape1);
  std::vector<Eigen::half> in(rows * cols, static_cast<Eigen::half>(1));
  test::FillValues<Eigen::half>(&input1, in);
  inputs.push_back({nullptr, &input1});

  // BiasGrading op.
  NodeDef bias_grad_node_def;
  Status status = NodeDefBuilder("bias_grad_op", "BiasAddGrad")
                      .Input(FakeInput(DT_HALF))
                      .Finalize(&bias_grad_node_def);
  TF_CHECK_OK(status);
  std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                              cpu_allocator(), bias_grad_node_def,
                                              TF_GRAPH_DEF_VERSION, &status));
  TF_CHECK_OK(status);
  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &inputs;
  params.op_kernel = op.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(&params, &attrs);

  std::unique_ptr<OpKernelContext> bias_grad_context(new OpKernelContext(&params));

  op->Compute(bias_grad_context.get());
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    delete bias_grad_context->release_output(0).tensor;
    op->Compute(bias_grad_context.get());
  }
  tensorflow::testing::StopTiming();
  testing::ItemsProcessed(input1.NumElements() * iters);
  testing::SetLabel(label);
}

// M: input_rows
// N: input_cols
#define BM_BiasGradHalf(M, N, TH, LABEL)                         \
  static void BM_BiasGradHalf_##M##_##N##_##TH(int iters) { \
    BM_BiasGradHalf(iters, M, N, TH, LABEL);                \
  }                                                          \
  BENCHMARK(BM_BiasGradHalf_##M##_##N##_##TH)

BM_BiasGradHalf(33, 33, 1, "bias_grad_small_square ");
BM_BiasGradHalf(5, 257, 1, "bias_grad_small_brick  ");
BM_BiasGradHalf(257, 5, 1, "bias_grad_small_bar    ");
BM_BiasGradHalf(1025, 1, 1, "bias_grad_small_line   ");
BM_BiasGradHalf(513, 513, 1, "bias_grad_medium_square");
BM_BiasGradHalf(33, 8193, 1, "bias_grad_medium_brick ");
BM_BiasGradHalf(8193, 33, 1, "bias_grad_medium_bar   ");
BM_BiasGradHalf(262145, 1, 1, "bias_grad_medium_line  ");
BM_BiasGradHalf(4097, 4097, 1, "bias_grad_large_square ");
BM_BiasGradHalf(129, 131073, 1, "bias_grad_large_brick  ");
BM_BiasGradHalf(131073, 129, 1, "bias_grad_large_bar    ");
BM_BiasGradHalf(2097153, 9, 1, "bias_grad_large_line   ");
BM_BiasGradHalf(4194305, 5, 1, "bias_grad_large_line   ");
BM_BiasGradHalf(8388609, 3, 1, "bias_grad_large_line   ");
BM_BiasGradHalf(16777217, 1, 1, "bias_grad_large_line   ");
BM_BiasGradHalf(33, 33, 4, "bias_grad_small_square ");
BM_BiasGradHalf(5, 257, 4, "bias_grad_small_brick  ");
BM_BiasGradHalf(257, 5, 4, "bias_grad_small_bar    ");
BM_BiasGradHalf(1025, 1, 4, "bias_grad_small_line   ");
BM_BiasGradHalf(513, 513, 4, "bias_grad_medium_square");
BM_BiasGradHalf(33, 8193, 4, "bias_grad_medium_brick ");
BM_BiasGradHalf(8193, 33, 4, "bias_grad_medium_bar   ");
BM_BiasGradHalf(262145, 1, 4, "bias_grad_medium_line  ");
BM_BiasGradHalf(4097, 4097, 4, "bias_grad_large_square ");
BM_BiasGradHalf(129, 131073, 4, "bias_grad_large_brick  ");
BM_BiasGradHalf(131073, 129, 4, "bias_grad_large_bar    ");
BM_BiasGradHalf(2097153, 9, 4, "bias_grad_large_line   ");
BM_BiasGradHalf(4194305, 5, 4, "bias_grad_large_line   ");
BM_BiasGradHalf(8388609, 3, 4, "bias_grad_large_line   ");
BM_BiasGradHalf(16777217, 1, 4, "bias_grad_large_line   ");
BM_BiasGradHalf(33, 33, 8, "bias_grad_small_square ");
BM_BiasGradHalf(5, 257, 8, "bias_grad_small_brick  ");
BM_BiasGradHalf(257, 5, 8, "bias_grad_small_bar    ");
BM_BiasGradHalf(1025, 1, 8, "bias_grad_small_line   ");
BM_BiasGradHalf(513, 513, 8, "bias_grad_medium_square");
BM_BiasGradHalf(33, 8193, 8, "bias_grad_medium_brick ");
BM_BiasGradHalf(8193, 33, 8, "bias_grad_medium_bar   ");
BM_BiasGradHalf(262145, 1, 8, "bias_grad_medium_line  ");
BM_BiasGradHalf(4097, 4097, 8, "bias_grad_large_square ");
BM_BiasGradHalf(129, 131073, 8, "bias_grad_large_brick  ");
BM_BiasGradHalf(131073, 129, 8, "bias_grad_large_bar    ");
BM_BiasGradHalf(2097153, 9, 8, "bias_grad_large_line   ");
BM_BiasGradHalf(4194305, 5, 8, "bias_grad_large_line   ");
BM_BiasGradHalf(8388609, 3, 8, "bias_grad_large_line   ");
BM_BiasGradHalf(16777217, 1, 8, "bias_grad_large_line   ");

}  // namespace tensorflow
