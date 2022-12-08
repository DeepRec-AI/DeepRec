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

#include <functional>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

class SparseSegmentMeanTest : public OpsTestBase {
 protected:
  void CreateOp(DataType dtype, DataType indices_dtype) {
    TF_ASSERT_OK(NodeDefBuilder("op", "SparseSegmentMean")
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(indices_dtype))
                     .Input(FakeInput(DT_INT32))
                     .Finalize(node_def()));
  }
};

class SparseSegmentSqrtNTest : public OpsTestBase {
 protected:
  void CreateOp(DataType dtype) {
    TF_ASSERT_OK(NodeDefBuilder("op", "SparseSegmentSqrtN")
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Finalize(node_def()));
  }
};

class SparseSegmentSumTest : public OpsTestBase {
 protected:
  void CreateOp(DataType dtype) {
    TF_ASSERT_OK(NodeDefBuilder("op", "SparseSegmentSum")
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Finalize(node_def()));
  }
};

#if GOOGLE_CUDA
TEST_F(SparseSegmentSumTest, gpu_float32) {
  CreateOp(DT_FLOAT);
  SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
  TF_ASSERT_OK(InitOp());
  std::vector<float> input(262144 * 6);
  std::vector<int32> indices(131072);
  std::vector<int32> segment_ids(131072);
  std::vector<float> out(65536 * 6);
  for (int i = 0; i < 262144 * 6; ++i) {
    input[i] = static_cast<float>(i / 6);
  }
  for (int i = 0; i < 131072; ++i) {
    indices[i] = i * 2;
    segment_ids[i] = i / 2;
  }
  for (int i = 0; i < 65536 * 6; ++i) {
    out[i] = static_cast<float>((i / 6) * 4 + (i / 6) * 4 + 2);
  }

  // input
  AddInputFromArray<float>(TensorShape({262144, 6}), input);
  AddInputFromArray<int32>(TensorShape({131072}), indices);
  AddInputFromArray<int32>(TensorShape({131072}), segment_ids);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_FLOAT, TensorShape{65536, 6});
  test::FillValues<float>(&expected, out);
  Tensor* result_ptr = GetOutput(0);
  TF_EXPECT_OK(device_->Sync());
  test::ExpectTensorEqual<float>(expected, *result_ptr);
}
#endif

TEST_F(SparseSegmentMeanTest, Normal_float32) {
  CreateOp(DT_FLOAT, DT_INT32);
  TF_ASSERT_OK(InitOp());
  std::vector<float> input(262144 * 6);
  std::vector<int32> indices(131072);
  std::vector<int32> segment_ids(131072);
  std::vector<float> out(65536 * 6);
  for (int i = 0; i < 262144 * 6; ++i) {
    input[i] = static_cast<float>(i / 6);
  }
  for (int i = 0; i < 131072; ++i) {
    indices[i] = i * 2;
    segment_ids[i] = i / 2;
  }
  for (int i = 0; i < 65536 * 6; ++i) {
    out[i] = static_cast<float>((i / 6) * 4 + (i / 6) * 4 + 2) / 2.0f;
  }

  // input
  AddInputFromArray<float>(TensorShape({262144, 6}), input);
  AddInputFromArray<int32>(TensorShape({131072}), indices);
  AddInputFromArray<int32>(TensorShape({131072}), segment_ids);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_FLOAT, TensorShape{65536, 6});
  test::FillValues<float>(&expected, out);
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseSegmentSqrtNTest, Normal_float32) {
  CreateOp(DT_FLOAT);
  TF_ASSERT_OK(InitOp());
  std::vector<float> input(262144 * 6);
  std::vector<int32> indices(131072);
  std::vector<int32> segment_ids(131072);
  std::vector<float> out(65536 * 6);
  for (int i = 0; i < 262144 * 6; ++i) {
    input[i] = static_cast<float>(i / 6);
  }
  for (int i = 0; i < 131072; ++i) {
    indices[i] = i * 2;
    segment_ids[i] = i / 2;
  }
  for (int i = 0; i < 65536 * 6; ++i) {
    out[i] = static_cast<float>((i / 6) * 4 + (i / 6) * 4 + 2) / sqrt(2.0f);
  }

  // input
  AddInputFromArray<float>(TensorShape({262144, 6}), input);
  AddInputFromArray<int32>(TensorShape({131072}), indices);
  AddInputFromArray<int32>(TensorShape({131072}), segment_ids);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_FLOAT, TensorShape{65536, 6});
  test::FillValues<float>(&expected, out);
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

#if GOOGLE_CUDA
TEST_F(SparseSegmentMeanTest, gpu_float32) {
  CreateOp(DT_FLOAT, DT_INT32);
  SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
  TF_ASSERT_OK(InitOp());
  std::vector<float> input(262144 * 6);
  std::vector<int32> indices(131072);
  std::vector<int32> segment_ids(131072);
  std::vector<float> out(65536 * 6);
  for (int i = 0; i < 262144 * 6; ++i) {
    input[i] = static_cast<float>(i / 6);
  }
  for (int i = 0; i < 131072; ++i) {
    indices[i] = i * 2;
    segment_ids[i] = i / 2;
  }
  for (int i = 0; i < 65536 * 6; ++i) {
    out[i] = static_cast<float>((i / 6) * 4 + (i / 6) * 4 + 2) / 2.0f;
  }

  // input
  AddInputFromArray<float>(TensorShape({262144, 6}), input);
  AddInputFromArray<int32>(TensorShape({131072}), indices);
  AddInputFromArray<int32>(TensorShape({131072}), segment_ids);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* result_ptr = GetOutput(0); 
  TF_EXPECT_OK(device_->Sync());

  Tensor expected(DT_FLOAT, TensorShape{65536, 6});
  test::FillValues<float>(&expected, out);
  test::ExpectTensorEqual<float>(expected, *result_ptr);
}

TEST_F(SparseSegmentSqrtNTest, gpu_float32) {
  CreateOp(DT_FLOAT);
  SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
  TF_ASSERT_OK(InitOp());
  std::vector<float> input(262144 * 6);
  std::vector<int32> indices(131072);
  std::vector<int32> segment_ids(131072);
  std::vector<float> out(65536 * 6);
  for (int i = 0; i < 262144 * 6; ++i) {
    input[i] = static_cast<float>(i / 6);
  }
  for (int i = 0; i < 131072; ++i) {
    indices[i] = i * 2;
    segment_ids[i] = i / 2;
  }
  for (int i = 0; i < 65536 * 6; ++i) {
    out[i] = static_cast<float>((i / 6) * 4 + (i / 6) * 4 + 2) / sqrt(2.0f);
  }

  // input
  AddInputFromArray<float>(TensorShape({262144, 6}), input);
  AddInputFromArray<int32>(TensorShape({131072}), indices);
  AddInputFromArray<int32>(TensorShape({131072}), segment_ids);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* result_ptr = GetOutput(0); 
  TF_EXPECT_OK(device_->Sync());

  Tensor expected(DT_FLOAT, TensorShape{65536, 6});
  test::FillValues<float>(&expected, out);
  test::ExpectTensorEqual<float>(expected, *result_ptr);
}
#endif

TEST_F(SparseSegmentMeanTest, Normal_double64_int64) {
  CreateOp(DT_DOUBLE, DT_INT64);
  TF_ASSERT_OK(InitOp());
  std::vector<double> input(262144 * 6);
  std::vector<int64> indices(131072);
  std::vector<int32> segment_ids(131072);
  std::vector<double> out(65536 * 6);
  for (int i = 0; i < 262144 * 6; ++i) {
    input[i] = static_cast<double>(i / 6);
  }
  for (int i = 0; i < 131072; ++i) {
    indices[i] = i * 2;
    segment_ids[i] = i / 2;
  }
  for (int i = 0; i < 65536 * 6; ++i) {
    out[i] = static_cast<double>((i / 6) * 4 + (i / 6) * 4 + 2) / 2.0;
  }

  // input
  AddInputFromArray<double>(TensorShape({262144, 6}), input);
  AddInputFromArray<int32>(TensorShape({131072}), indices);
  AddInputFromArray<int32>(TensorShape({131072}), segment_ids);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_DOUBLE, TensorShape{65536, 6});
  test::FillValues<double>(&expected, out);
  test::ExpectTensorEqual<double>(expected, *GetOutput(0));
}

TEST_F(SparseSegmentMeanTest, Normal_double64_int32) {
  CreateOp(DT_DOUBLE, DT_INT32);
  TF_ASSERT_OK(InitOp());
  std::vector<double> input(262144 * 6);
  std::vector<int32> indices(131072);
  std::vector<int32> segment_ids(131072);
  std::vector<double> out(65536 * 6);
  for (int i = 0; i < 262144 * 6; ++i) {
    input[i] = static_cast<double>(i / 6);
  }
  for (int i = 0; i < 131072; ++i) {
    indices[i] = i * 2;
    segment_ids[i] = i / 2;
  }
  for (int i = 0; i < 65536 * 6; ++i) {
    out[i] = static_cast<double>((i / 6) * 4 + (i / 6) * 4 + 2) / 2.0;
  }

  // input
  AddInputFromArray<double>(TensorShape({262144, 6}), input);
  AddInputFromArray<int32>(TensorShape({131072}), indices);
  AddInputFromArray<int32>(TensorShape({131072}), segment_ids);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_DOUBLE, TensorShape{65536, 6});
  test::FillValues<double>(&expected, out);
  test::ExpectTensorEqual<double>(expected, *GetOutput(0));
}


class SparseSegmentMeanGradTest : public OpsTestBase {
 protected:
  void CreateOp(DataType dtype) {
    TF_ASSERT_OK(NodeDefBuilder("op", "SparseSegmentMeanGrad")
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Finalize(node_def()));
  }
};

class SparseSegmentSqrtNGradTest : public OpsTestBase {
 protected:
  void CreateOp(DataType dtype) {
    TF_ASSERT_OK(NodeDefBuilder("op", "SparseSegmentSqrtNGrad")
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Finalize(node_def()));
  }
};

#if GOOGLE_CUDA
TEST_F(SparseSegmentMeanGradTest, gpu_double64) {
  CreateOp(DT_DOUBLE);
  SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
  TF_ASSERT_OK(InitOp());
  std::vector<double> grad(65536 * 6);
  std::vector<int32> indices(131072);
  std::vector<int32> segment_ids(131072);
  std::vector<int32> dim0(1, 262144);
  std::vector<double> out(262144 * 6);

  for (int i = 0; i < 65536 * 6; ++i) {
    grad[i] = static_cast<double>(i / 6);
  }
  for (int i = 0; i < 131072; ++i) {
    indices[i] = i * 2;
    segment_ids[i] = i / 2;
  }
  for (int i = 0; i < 262144 * 6; ++i) {
    if ((i / 6) % 2 == 0) {
      out[i] = static_cast<double>(i / 24) / 2.0f;
    } else {
      out[i] = static_cast<double>(0);
    }
  }

  // input
  AddInputFromArray<double>(TensorShape({65536, 6}), grad);
  AddInputFromArray<int32>(TensorShape({131072}), indices);
  AddInputFromArray<int32>(TensorShape({131072}), segment_ids);
  AddInputFromArray<int32>(TensorShape({}), dim0);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* result_ptr = GetOutput(0); 
  TF_EXPECT_OK(device_->Sync());

  Tensor expected(DT_DOUBLE, TensorShape{262144, 6});
  test::FillValues<double>(&expected, out);
  test::ExpectTensorEqual<double>(expected, *result_ptr);
}

TEST_F(SparseSegmentMeanGradTest, gpu_float32) {
  CreateOp(DT_FLOAT);
  SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
  TF_ASSERT_OK(InitOp());
  std::vector<float> grad(65536 * 6);
  std::vector<int32> indices(131072);
  std::vector<int32> segment_ids(131072);
  std::vector<int32> dim0(1, 262144);
  std::vector<float> out(262144 * 6);

  for (int i = 0; i < 65536 * 6; ++i) {
    grad[i] = static_cast<float>(i / 6);
  }
  for (int i = 0; i < 131072; ++i) {
    indices[i] = i * 2;
    segment_ids[i] = i / 2;
  }
  for (int i = 0; i < 262144 * 6; ++i) {
    if ((i / 6) % 2 == 0) {
      out[i] = static_cast<float>(i / 24) / 2.0f;
    } else {
      out[i] = static_cast<float>(0);
    }
  }
  // input
  AddInputFromArray<float>(TensorShape({65536, 6}), grad);
  AddInputFromArray<int32>(TensorShape({131072}), indices);
  AddInputFromArray<int32>(TensorShape({131072}), segment_ids);
  AddInputFromArray<int32>(TensorShape({}), dim0);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* result_ptr = GetOutput(0); 
  TF_EXPECT_OK(device_->Sync());

  Tensor expected(DT_FLOAT, TensorShape{262144, 6});
  test::FillValues<float>(&expected, out);
  test::ExpectTensorEqual<float>(expected, *result_ptr);
}

TEST_F(SparseSegmentSqrtNGradTest, gpu_double64) {
  CreateOp(DT_DOUBLE);
  SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
  TF_ASSERT_OK(InitOp());
  std::vector<double> grad(65536 * 6);
  std::vector<int32> indices(131072);
  std::vector<int32> segment_ids(131072);
  std::vector<int32> dim0(1, 262144);
  std::vector<double> out(262144 * 6);

  for (int i = 0; i < 65536 * 6; ++i) {
    grad[i] = static_cast<double>(i / 6);
  }
  for (int i = 0; i < 131072; ++i) {
    indices[i] = i * 2;
    segment_ids[i] = i / 2;
  }
  for (int i = 0; i < 262144 * 6; ++i) {
    if ((i / 6) % 2 == 0) {
      out[i] = static_cast<double>(i / 24) / sqrt(static_cast<double>(2.0f));
    } else {
      out[i] = static_cast<double>(0);
    }
  }
  // input
  AddInputFromArray<double>(TensorShape({65536, 6}), grad);
  AddInputFromArray<int32>(TensorShape({131072}), indices);
  AddInputFromArray<int32>(TensorShape({131072}), segment_ids);
  AddInputFromArray<int32>(TensorShape({}), dim0);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* result_ptr = GetOutput(0); 
  TF_EXPECT_OK(device_->Sync());

  Tensor expected(DT_DOUBLE, TensorShape{262144, 6});
  test::FillValues<double>(&expected, out);
  test::ExpectTensorEqual<double>(expected, *result_ptr);
}

TEST_F(SparseSegmentSqrtNGradTest, gpu_float32) {
  CreateOp(DT_FLOAT);
  SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
  TF_ASSERT_OK(InitOp());
  std::vector<float> grad(65536 * 6);
  std::vector<int32> indices(131072);
  std::vector<int32> segment_ids(131072);
  std::vector<int32> dim0(1, 262144);
  std::vector<float> out(262144 * 6);

  for (int i = 0; i < 65536 * 6; ++i) {
    grad[i] = static_cast<float>(i / 6);
  }
  for (int i = 0; i < 131072; ++i) {
    indices[i] = i * 2;
    segment_ids[i] = i / 2;
  }
  for (int i = 0; i < 262144 * 6; ++i) {
    if ((i / 6) % 2 == 0) {
      out[i] = static_cast<float>(i / 24) / sqrt(2.0f);
    } else {
      out[i] = static_cast<float>(0);
    }
  }
  // input
  AddInputFromArray<float>(TensorShape({65536, 6}), grad);
  AddInputFromArray<int32>(TensorShape({131072}), indices);
  AddInputFromArray<int32>(TensorShape({131072}), segment_ids);
  AddInputFromArray<int32>(TensorShape({}), dim0);

  TF_ASSERT_OK(RunOpKernel());
  Tensor* result_ptr = GetOutput(0); 
  TF_EXPECT_OK(device_->Sync());

  Tensor expected(DT_FLOAT, TensorShape{262144, 6});
  test::FillValues<float>(&expected, out);
  test::ExpectTensorEqual<float>(expected, *result_ptr);
}
#endif

TEST_F(SparseSegmentMeanGradTest, Normal_double64) {
  CreateOp(DT_DOUBLE);
  TF_ASSERT_OK(InitOp());
  std::vector<double> grad(65536 * 6);
  std::vector<int32> indices(131072);
  std::vector<int32> segment_ids(131072);
  std::vector<int32> dim0(1, 262144);
  std::vector<double> out(262144 * 6);

  for (int i = 0; i < 65536 * 6; ++i) {
    grad[i] = static_cast<double>(i / 6);
  }
  for (int i = 0; i < 131072; ++i) {
    indices[i] = i * 2;
    segment_ids[i] = i / 2;
  }
  for (int i = 0; i < 262144 * 6; ++i) {
    if ((i / 6) % 2 == 0) {
      out[i] = static_cast<double>(i / 24) / 2.0f;
    } else {
      out[i] = static_cast<double>(0);
    }
  }

  // input
  AddInputFromArray<double>(TensorShape({65536, 6}), grad);
  AddInputFromArray<int32>(TensorShape({131072}), indices);
  AddInputFromArray<int32>(TensorShape({131072}), segment_ids);
  AddInputFromArray<int32>(TensorShape({}), dim0);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_DOUBLE, TensorShape{262144, 6});
  test::FillValues<double>(&expected, out);
  test::ExpectTensorEqual<double>(expected, *GetOutput(0));
}

TEST_F(SparseSegmentMeanGradTest, Normal_float32) {
  CreateOp(DT_FLOAT);
  TF_ASSERT_OK(InitOp());
  std::vector<float> grad(65536 * 6);
  std::vector<int32> indices(131072);
  std::vector<int32> segment_ids(131072);
  std::vector<int32> dim0(1, 262144);
  std::vector<float> out(262144 * 6);

  for (int i = 0; i < 65536 * 6; ++i) {
    grad[i] = static_cast<float>(i / 6);
  }
  for (int i = 0; i < 131072; ++i) {
    indices[i] = i * 2;
    segment_ids[i] = i / 2;
  }
  for (int i = 0; i < 262144 * 6; ++i) {
    if ((i / 6) % 2 == 0) {
      out[i] = static_cast<float>(i / 24) / 2.0f;
    } else {
      out[i] = static_cast<float>(0);
    }
  }

  // input
  AddInputFromArray<float>(TensorShape({65536, 6}), grad);
  AddInputFromArray<int32>(TensorShape({131072}), indices);
  AddInputFromArray<int32>(TensorShape({131072}), segment_ids);
  AddInputFromArray<int32>(TensorShape({}), dim0);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_FLOAT, TensorShape{262144, 6});
  test::FillValues<float>(&expected, out);
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}


template <typename Index>
static void BM_SegmentReduction(int iters, const string& reduction,
                                Index num_rows, Index num_cols,
                                Index segment_size) {
  testing::StopTiming();
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  // Create inputs
  gtl::InlinedVector<TensorValue, 4> reduction_inputs;
  TensorShape shape1({num_rows, num_cols});
  Tensor input1(DT_FLOAT, shape1);
  reduction_inputs.push_back({nullptr, &input1});

  TensorShape shape2({num_rows});
  Tensor input2(DataTypeToEnum<Index>::v(), shape2);
  test::FillFn<Index>(&input2, [&num_rows, &segment_size](Index i) -> Index {
    return std::min(i / segment_size, num_rows - 1);
  });
  reduction_inputs.push_back({nullptr, &input2});

  NodeDef reduction_node_def;
  TF_CHECK_OK(NodeDefBuilder(reduction, reduction)
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DataTypeToEnum<Index>::v()))
                  .Finalize(&reduction_node_def));
  Status status;
  std::unique_ptr<OpKernel> reduction_op(
      CreateOpKernel(DEVICE_CPU, device.get(), cpu_allocator(),
                     reduction_node_def, TF_GRAPH_DEF_VERSION, &status));
  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &reduction_inputs;
  params.op_kernel = reduction_op.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(&params, &attrs);

  std::unique_ptr<OpKernelContext> reduction_context(
      new OpKernelContext(&params));

  reduction_op->Compute(reduction_context.get());
  TF_CHECK_OK(reduction_context->status());
  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    delete reduction_context->release_output(0).tensor;
    reduction_op->Compute(reduction_context.get());
  }
  int64 bytes_per_iter =
      static_cast<int64>(num_rows * num_cols * sizeof(float));
  testing::BytesProcessed(bytes_per_iter * iters);
}

#define BM_Reduce(O, R, C, S)                                      \
  static void BM_Reduce_##O##_##R##_##C##_##S##_int32(int iters) { \
    BM_SegmentReduction<int32>(iters, #O, R, C, S);                \
  }                                                                \
  static void BM_Reduce_##O##_##R##_##C##_##S##_int64(int iters) { \
    BM_SegmentReduction<int64>(iters, #O, R, C, S);                \
  }                                                                \
  BENCHMARK(BM_Reduce_##O##_##R##_##C##_##S##_int32);              \
  BENCHMARK(BM_Reduce_##O##_##R##_##C##_##S##_int64);

#define BM_Reduce_Arg(R, C, S)    \
  BM_Reduce(SegmentSum, R, C, S); \
  BM_Reduce(SegmentMean, R, C, S);

BM_Reduce_Arg(64, 32, 1);
BM_Reduce_Arg(4096, 128, 1);
BM_Reduce_Arg(16, 8, 2);
BM_Reduce_Arg(64, 32, 2);
BM_Reduce_Arg(4096, 32, 2);
BM_Reduce_Arg(4096, 128, 2);

static void SparseSegmentMeanGradHelper(int iters, float uniqueness,
                                        int size, int nth) {
  testing::StopTiming();
  Graph* g = new Graph(OpRegistry::Global());
  CHECK_LE(uniqueness, 1.0);
  CHECK_GT(uniqueness, 0.0);

  const int kNumIndices = size;
  Tensor indices(DT_INT32, TensorShape({kNumIndices}));
  auto indices_flat = indices.flat<int32>();
  Tensor segments(DT_INT32, TensorShape({kNumIndices}));
  auto segments_flat = segments.flat<int32>();

  int kUniqueIndices = uniqueness * kNumIndices;
  Tensor output_dim0(DT_INT32, TensorShape({}));
  output_dim0.scalar<int32>()() = kUniqueIndices;

  for (int i = 0; i < kNumIndices; ++i) {
    indices_flat(i) = (i * 31) % kUniqueIndices;
    segments_flat(i) = i * .8;
  }

  const int kDim1 = segments_flat(kNumIndices - 1) + 1;
  const int kDim2 = 128;
  Tensor input(DT_FLOAT, TensorShape({kDim1, kDim2}));
  input.flat<float>().setRandom();

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "SparseSegmentMeanGrad")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, indices))
                  .Input(test::graph::Constant(g, segments))
                  .Input(test::graph::Constant(g, output_dim0))
                  .Attr("T", DT_FLOAT)
                  .Finalize(g, &node));

  testing::UseRealTime();
  testing::BytesProcessed(static_cast<int64>(iters) * (kDim1 * kDim2) *
                          sizeof(float));
  SessionOptions opts;
  opts.config.set_intra_op_parallelism_threads(nth);
  testing::StartTiming();
  test::Benchmark("cpu", g, &opts).Run(iters);
}

#define BM_SparseSegmentMeanGrad(LEVEL, RATE, NTH)                          \
static void BM_SparseSegmentMeanGrad_##LEVEL##_##NTH(int iters, int size) { \
  return SparseSegmentMeanGradHelper(iters, RATE, size, NTH);               \
}                                                                           \
BENCHMARK(BM_SparseSegmentMeanGrad_##LEVEL##_##NTH)->Arg(1000)->Arg(300000);

BM_SparseSegmentMeanGrad(Low, 1.0, 1);
BM_SparseSegmentMeanGrad(Med, 0.6, 1);
BM_SparseSegmentMeanGrad(High, 0.01, 1);
BM_SparseSegmentMeanGrad(Low, 1.0, 4);
BM_SparseSegmentMeanGrad(Med, 0.6, 4);
BM_SparseSegmentMeanGrad(High, 0.01, 4);
BM_SparseSegmentMeanGrad(Low, 1.0, 8);
BM_SparseSegmentMeanGrad(Med, 0.6, 8);
BM_SparseSegmentMeanGrad(High, 0.01, 8);

}  // namespace tensorflow