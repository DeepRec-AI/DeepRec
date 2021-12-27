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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {

enum class Device { CPU, GPU };

class FusedEmbeddingSparsePostLookUpGradOpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, int num_partitions, DataType dtype,
                          const std::string& combiner, const float max_norm,
                          const int default_id) {
    if (device == Device::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }

    TF_EXPECT_OK(NodeDefBuilder("fused_embedding__sparse_post_look_up_grad",
                                "FusedEmbeddingSparsePostLookUpGrad")
                     .Attr("T", dtype)
                     .Attr("num_partitions", num_partitions)
                     .Attr("partition_axis", 0)
                     .Attr("combiner", combiner)
                     .Attr("max_norm", max_norm)
                     .Attr("default_id", default_id)
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(FusedEmbeddingSparsePostLookUpGradOpTest,
       Partition2_Mean_MaxNorm100_Float) {
  const int nnz = 10;
  const int batch_size = 4;
  const int emb_vector_dim = 8;
  const int entries = 8;

  MakeOpAndSetDevice(Device::GPU, 2, DT_FLOAT, "mean", 100.0, -1);

  // top_grad
  AddInputFromArray<float>(
      TensorShape({batch_size, emb_vector_dim}),
      {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0,
       11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
       22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0});

  // emb_shards
  AddInputFromArray<float>(
      TensorShape({6, emb_vector_dim}),
      {8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 24.0, 25.0, 26.0, 27.0,
       28.0, 29.0, 30.0, 31.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
       32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 32.0, 33.0, 34.0, 35.0,
       36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0});
  AddInputFromArray<float>(
      TensorShape({4, emb_vector_dim}),
      {56.0,  57.0,  58.0,  59.0,  60.0,  61.0,  62.0,  63.0,
       96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0,
       96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0,
       120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0});

  // sp_values: 3, 1, 4, 5, 7, 3, 12, 12, 15, 4
  // partitioned_values: 1, 3, 3, 4, 4, 5 and 7, 12, 12, 15
  // partitioned_indices
  AddInputFromArray<int64>(TensorShape({6, 2}),
                           {0, 5, 0, 1, 2, 1, 1, 2, 3, 6, 1, 1});
  AddInputFromArray<int64>(TensorShape({4, 2}), {1, 7, 2, 4, 2, 7, 3, 0});

  // feature_nums
  AddInputFromArray<int>(TensorShape({batch_size}), {2, 3, 3, 2});

  // row_empty_and_invalid_flags
  AddInputFromArray<int>(TensorShape({batch_size + nnz}),
                         {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor grad_shards_1(allocator(), DT_FLOAT,
                         TensorShape({6, emb_vector_dim}));
    test::FillValues<float>(
        &grad_shards_1,
        {0.00000000,  0.50000000,  1.00000000,  1.50000000,  2.00000000,
         2.50000000,  3.00000000,  3.50000000,  0.00000000,  0.50000000,
         1.00000000,  1.50000000,  2.00000000,  2.50000000,  3.00000000,
         3.50000000,  5.33333349,  5.66666651,  6.00000000,  6.33333349,
         6.66666651,  7.00000000,  7.33333349,  7.66666651,  2.65028572,
         2.98157120,  3.31285667,  3.64414287,  3.97542834,  4.30671406,
         4.63799953,  4.96928549,  11.92628479, 12.42321396, 12.92014217,
         13.41707039, 13.91399956, 14.41092777, 14.90785599, 15.40478516,
         2.16437674,  2.43492365,  2.70547056,  2.97601795,  3.24656487,
         3.51711202,  3.78765893,  4.05820608});
    test::ExpectTensorNear<float>(grad_shards_1, *GetOutput(0), 1e-4);
  }

  {
    Tensor grad_shards_2(allocator(), DT_FLOAT,
                         TensorShape({4, emb_vector_dim}));
    test::FillValues<float>(
        &grad_shards_2,
        {1.58337951, 1.78130186, 1.97922409, 2.17714667, 2.37506914, 2.57299161,
         2.77091384, 2.96883631, 1.89459133, 2.01300311, 2.13141513, 2.24982715,
         2.36823893, 2.48665094, 2.60506320, 2.72347474, 1.89459133, 2.01300311,
         2.13141513, 2.24982715, 2.36823893, 2.48665094, 2.60506320, 2.72347474,
         3.43474555, 3.57786012, 3.72097445, 3.86408877, 4.00720310, 4.15031767,
         4.29343224, 4.43654633});
    test::ExpectTensorNear<float>(grad_shards_2, *GetOutput(1), 1e-4);
  }
}

TEST_F(FusedEmbeddingSparsePostLookUpGradOpTest,
       Partition2_SUM_Float_No_Default) {
  const int nnz = 3;
  const int batch_size = 3;
  const int emb_vector_dim = 4;
  const int entries = 8;

  MakeOpAndSetDevice(Device::GPU, 2, DT_FLOAT, "sum", -1.0, -1);

  // top_grad
  AddInputFromArray<float>(
      TensorShape({batch_size, emb_vector_dim}),
      {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0});

  // emb_shards
  AddInputFromArray<float>(TensorShape({2, emb_vector_dim}),
                           {8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0});
  AddInputFromArray<float>(TensorShape({2, emb_vector_dim}),
                           {56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0});

  // partitioned_indices
  AddInputFromArray<int64>(TensorShape({2, 2}), {0, 0, 0, 5});
  AddInputFromArray<int64>(TensorShape({2, 2}), {1, 4, 2, 0});

  // feature_nums
  AddInputFromArray<int>(TensorShape({batch_size}), {2, 1, 1});

  // row_empty_and_invalid_flags
  AddInputFromArray<int>(TensorShape({batch_size + nnz}), {0, 0, 1, 1, 1, 1});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor grad_shards_1(allocator(), DT_FLOAT,
                         TensorShape({2, emb_vector_dim}));
    test::FillValues<float>(&grad_shards_1,
                            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    test::ExpectTensorNear<float>(grad_shards_1, *GetOutput(0), 1e-4);
  }

  {
    Tensor grad_shards_2(allocator(), DT_FLOAT,
                         TensorShape({2, emb_vector_dim}));
    test::FillValues<float>(&grad_shards_2,
                            {2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0});
    test::ExpectTensorNear<float>(grad_shards_2, *GetOutput(1), 1e-4);
  }
}

TEST_F(FusedEmbeddingSparsePostLookUpGradOpTest,
       Partition2_SUM_Float_Default_0) {
  const int nnz = 3;
  const int batch_size = 3;
  const int emb_vector_dim = 4;
  const int entries = 8;

  MakeOpAndSetDevice(Device::GPU, 2, DT_FLOAT, "sum", -1.0, 0);

  // top_grad
  AddInputFromArray<float>(
      TensorShape({batch_size, emb_vector_dim}),
      {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0});

  // emb_shards
  AddInputFromArray<float>(TensorShape({2, emb_vector_dim}),
                           {8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0});
  AddInputFromArray<float>(TensorShape({2, emb_vector_dim}),
                           {56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0});

  // partitioned_indices
  AddInputFromArray<int64>(TensorShape({2, 2}), {0, 0, 0, 5});
  AddInputFromArray<int64>(TensorShape({2, 2}), {1, 4, 2, 0});

  // feature_nums
  AddInputFromArray<int>(TensorShape({batch_size}), {2, 1, 1});

  // row_empty_and_invalid_flags
  AddInputFromArray<int>(TensorShape({batch_size + nnz}), {0, 0, 1, 1, 1, 1});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor grad_shards_1(allocator(), DT_FLOAT,
                         TensorShape({2, emb_vector_dim}));
    test::FillValues<float>(&grad_shards_1,
                            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    test::ExpectTensorNear<float>(grad_shards_1, *GetOutput(0), 1e-4);
  }

  {
    Tensor grad_shards_2(allocator(), DT_FLOAT,
                         TensorShape({2, emb_vector_dim}));
    test::FillValues<float>(&grad_shards_2,
                            {2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0});
    test::ExpectTensorNear<float>(grad_shards_2, *GetOutput(1), 1e-4);
  }
}

}  // namespace
}  // namespace tensorflow