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

#include <string>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {

enum class Device { CPU, GPU };

class FusedEmbeddingSparsePreLookUpOpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, const int num_partitions,
                          const bool fill_empty_row,
                          const bool prune_invalid_id, const int default_id,
                          const std::string& partition_strategy) {
    if (device == Device::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }

    TF_EXPECT_OK(NodeDefBuilder("fused_embedding_sparse_pre_look_up",
                                "FusedEmbeddingSparsePreLookUp")
                     .Attr("num_partitions", num_partitions)
                     .Attr("partition_axis", 0)
                     .Attr("fill_empty_row", fill_empty_row)
                     .Attr("prune_invalid_id", prune_invalid_id)
                     .Attr("default_id", default_id)
                     .Attr("partition_strategy", partition_strategy)
                     .Input(FakeInput(num_partitions, DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(FusedEmbeddingSparsePreLookUpOpTest, OnlyTestPartition3Div) {
  MakeOpAndSetDevice(Device::GPU, 3, false, false, -1, std::string("div"));
  // partition_shapes 0
  AddInputFromArray<int64>(TensorShape({2}), {6, 16});
  // partition_shapes 1
  AddInputFromArray<int64>(TensorShape({2}), {3, 16});
  // partition_shapes 2
  AddInputFromArray<int64>(TensorShape({2}), {7, 16});
  // sp_values
  AddInputFromArray<int64>(TensorShape({12}),
                           {1, 5, 3, 6, 12, 14, 15, 0, 5, 5, 11, 7});
  // sp_indices
  AddInputFromArray<int64>(TensorShape({12, 2}),
                           {2,  3, 4,  6, 1, 6, 12, 12, 12, 12, 11, 5,
                            15, 0, 11, 6, 7, 9, 11, 8,  12, 13, 13, 0});
  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {16, 16});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  // partitioned_values and partition_permutations 0
  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({4}));
    test::FillValues<int64>(&expected_values, {1, 5, 3, 0});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));

    Tensor expected_permutes(allocator(), DT_INT64, TensorShape({4}));
    test::FillValues<int64>(&expected_permutes, {0, 1, 2, 7});
    test::ExpectTensorEqual<int64>(expected_permutes, *GetOutput(3));
  }
  // partitioned_values and partition_permutations 1
  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({2}));
    test::FillValues<int64>(&expected_values, {0, 1});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(1));

    Tensor expected_permutes(allocator(), DT_INT64, TensorShape({2}));
    test::FillValues<int64>(&expected_permutes, {3, 9});
    test::ExpectTensorEqual<int64>(expected_permutes, *GetOutput(4));
  }
  // partitioned_values and partition_permutations 2
  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({4}));
    test::FillValues<int64>(&expected_values, {3, 5, 6, 2});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(2));

    Tensor expected_permutes(allocator(), DT_INT64, TensorShape({4}));
    test::FillValues<int64>(&expected_permutes, {4, 5, 6, 8});
    test::ExpectTensorEqual<int64>(expected_permutes, *GetOutput(5));
  }

  // indices_before_unique
  Tensor expected_indices_before_unique(allocator(), DT_INT64,
                                        TensorShape({12, 2}));
  test::FillValues<int64>(&expected_indices_before_unique,
                          {2,  3, 4,  6, 1, 6, 12, 12, 12, 12, 11, 5,
                           15, 0, 11, 6, 7, 9, 11, 8,  12, 13, 13, 0});
  test::ExpectTensorEqual<int64>(expected_indices_before_unique, *GetOutput(7));

  // unique_idxs
  Tensor expected_unique_idxs(allocator(), DT_INT64, TensorShape({12}));
  test::FillValues<int64>(&expected_unique_idxs,
                          {0, 1, 2, 3, 4, 5, 6, 7, 1, 1, 8, 9});
  test::ExpectTensorEqual<int64>(expected_unique_idxs, *GetOutput(8));

  // unique_counts
  Tensor expected_unique_counts(allocator(), DT_INT64, TensorShape({10}));
  test::FillValues<int64>(&expected_unique_counts,
                          {1, 3, 1, 1, 1, 1, 1, 1, 1, 1});
  test::ExpectTensorEqual<int64>(expected_unique_counts, *GetOutput(9));

  // idx_of_input_to_unique
  Tensor expected_idx_of_input_to_unique(allocator(), DT_INT64,
                                         TensorShape({12}));
  test::FillValues<int64>(&expected_idx_of_input_to_unique,
                          {0, 1, 8, 9, 2, 3, 4, 5, 6, 7, 10, 11});
  test::ExpectTensorEqual<int64>(expected_idx_of_input_to_unique,
                                 *GetOutput(10));

  // unique_offsets
  Tensor expected_unique_offsets(allocator(), DT_INT64, TensorShape({10}));
  test::FillValues<int64>(&expected_unique_offsets,
                          {0, 1, 4, 5, 6, 7, 8, 9, 10, 11});
  test::ExpectTensorEqual<int64>(expected_unique_offsets, *GetOutput(11));
}

TEST_F(FusedEmbeddingSparsePreLookUpOpTest, OnlyTestUnique) {
  MakeOpAndSetDevice(Device::GPU, 1, false, false, -1, std::string("div"));
  // partition_shapes 0
  AddInputFromArray<int64>(TensorShape({2}), {20, 8});
  // sp_values
  AddInputFromArray<int64>(TensorShape({20}), {3, 5, 3, 4, 1, 4, 9, 8, 6, 3,
                                               5, 7, 8, 8, 4, 6, 4, 2, 5, 6});
  // sp_indices, whatever the content
  AddInputFromArray<int64>(
      TensorShape({20, 2}),
      {3, 5, 3, 4, 1, 4, 9, 8, 6, 3, 5, 7, 8, 8, 4, 6, 4, 2, 5, 6,
       3, 5, 3, 4, 1, 4, 9, 8, 6, 3, 5, 7, 8, 8, 4, 6, 4, 2, 5, 6});
  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {16, 16});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  // partitioned_values and partition_permutations 0
  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({9}));
    test::FillValues<int64>(&expected_values, {3, 5, 4, 1, 9, 8, 6, 7, 2});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));

    Tensor expected_permutes(allocator(), DT_INT64, TensorShape({9}));
    test::FillValues<int64>(&expected_permutes, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    test::ExpectTensorEqual<int64>(expected_permutes, *GetOutput(1));
  }

  // indices_before_unique
  Tensor expected_indices_before_unique(allocator(), DT_INT64,
                                        TensorShape({20, 2}));
  test::FillValues<int64>(
      &expected_indices_before_unique,
      {3, 5, 3, 4, 1, 4, 9, 8, 6, 3, 5, 7, 8, 8, 4, 6, 4, 2, 5, 6,
       3, 5, 3, 4, 1, 4, 9, 8, 6, 3, 5, 7, 8, 8, 4, 6, 4, 2, 5, 6});
  test::ExpectTensorEqual<int64>(expected_indices_before_unique, *GetOutput(3));

  // unique_idxs
  Tensor expected_unique_idxs(allocator(), DT_INT64, TensorShape({20}));
  test::FillValues<int64>(
      &expected_unique_idxs,
      {0, 1, 0, 2, 3, 2, 4, 5, 6, 0, 1, 7, 5, 5, 2, 6, 2, 8, 1, 6});
  test::ExpectTensorEqual<int64>(expected_unique_idxs, *GetOutput(4));

  // unique_counts
  Tensor expected_unique_counts(allocator(), DT_INT64, TensorShape({9}));
  test::FillValues<int64>(&expected_unique_counts, {3, 3, 4, 1, 1, 3, 3, 1, 1});
  test::ExpectTensorEqual<int64>(expected_unique_counts, *GetOutput(5));

  // idx_of_input_to_unique
  Tensor expected_idx_of_input_to_unique(allocator(), DT_INT64,
                                         TensorShape({20}));
  test::FillValues<int64>(
      &expected_idx_of_input_to_unique,
      {0, 2, 9, 1, 10, 18, 3, 5, 14, 16, 4, 6, 7, 12, 13, 8, 15, 19, 11, 17});
  test::ExpectTensorEqual<int64>(expected_idx_of_input_to_unique,
                                 *GetOutput(6));

  // unique_offsets
  Tensor expected_unique_offsets(allocator(), DT_INT64, TensorShape({9}));
  test::FillValues<int64>(&expected_unique_offsets,
                          {0, 3, 6, 10, 11, 12, 15, 18, 19});
  test::ExpectTensorEqual<int64>(expected_unique_offsets, *GetOutput(7));
}

TEST_F(FusedEmbeddingSparsePreLookUpOpTest, OnlyTestFillEmpty) {
  MakeOpAndSetDevice(Device::GPU, 1, true, false, -1, std::string("div"));
  // partition_shapes 0
  AddInputFromArray<int64>(TensorShape({2}), {99999, 8});

  // sp_values
  AddInputFromArray<int64>(TensorShape({10}),
                           {10, 4, 3, 2, 5, 13, 14, 9, 6, 1});

  // sp_indices
  AddInputFromArray<int64>(
      TensorShape({10, 2}),
      {0, 0, 0, 4, 1, 2, 3, 0, 3, 4, 4, 0, 5, 2, 6, 0, 6, 1, 6, 7});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {7, 8});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  Tensor expected_values(allocator(), DT_INT64, TensorShape({11}));
  test::FillValues<int64>(&expected_values,
                          {10, 4, 3, 2, 5, 13, 14, 9, 6, 1, 0});
  test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));

  Tensor expected_indices_before_unique(allocator(), DT_INT64,
                                        TensorShape({11, 2}));
  test::FillValues<int64>(
      &expected_indices_before_unique,
      {0, 0, 0, 4, 1, 2, 3, 0, 3, 4, 4, 0, 5, 2, 6, 0, 6, 1, 6, 7, 2, 0});
  test::ExpectTensorEqual<int64>(expected_indices_before_unique, *GetOutput(3));
}

TEST_F(FusedEmbeddingSparsePreLookUpOpTest, OnlyTestPruneInvalid) {
  MakeOpAndSetDevice(Device::GPU, 1, false, true, -1, std::string("div"));
  // partition_shapes 0
  AddInputFromArray<int64>(TensorShape({2}), {20, 8});
  // sp_values
  AddInputFromArray<int64>(TensorShape({10}),
                           {10, 4, -3, 2, 5, 13, 14, -9, 6, 1});

  // sp_indices
  AddInputFromArray<int64>(
      TensorShape({10, 2}),
      {0, 0, 0, 4, 1, 2, 3, 0, 3, 4, 4, 0, 5, 2, 6, 0, 6, 1, 6, 7});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {7, 8});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  Tensor expected_values(allocator(), DT_INT64, TensorShape({8}));
  test::FillValues<int64>(&expected_values, {10, 4, 2, 5, 13, 14, 6, 1});
  test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));

  Tensor expected_indices_before_unique(allocator(), DT_INT64,
                                        TensorShape({8, 2}));
  test::FillValues<int64>(&expected_indices_before_unique,
                          {0, 0, 0, 4, 3, 0, 3, 4, 4, 0, 5, 2, 6, 1, 6, 7});
  test::ExpectTensorEqual<int64>(expected_indices_before_unique, *GetOutput(3));
}

TEST_F(FusedEmbeddingSparsePreLookUpOpTest, PruneInvalidAndFillEmptyDeault7) {
  MakeOpAndSetDevice(Device::GPU, 1, true, true, 7, std::string("div"));
  // partition_shapes 0
  AddInputFromArray<int64>(TensorShape({2}), {20, 8});
  // sp_values
  AddInputFromArray<int64>(TensorShape({10}),
                           {10, 4, 3, 2, 5, 13, 14, -9, 6, 1});

  // sp_indices
  AddInputFromArray<int64>(
      TensorShape({10, 2}),
      {0, 0, 0, 4, 1, 2, 3, 0, 3, 4, 4, 0, 5, 2, 6, 0, 6, 1, 6, 7});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {7, 8});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  Tensor expected_values(allocator(), DT_INT64, TensorShape({10}));
  test::FillValues<int64>(&expected_values, {10, 4, 3, 2, 5, 13, 14, 6, 1, 7});
  test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));

  Tensor expected_indices_before_unique(allocator(), DT_INT64,
                                        TensorShape({10, 2}));
  test::FillValues<int64>(
      &expected_indices_before_unique,
      {0, 0, 0, 4, 1, 2, 3, 0, 3, 4, 4, 0, 5, 2, 6, 1, 6, 7, 2, 0});
  test::ExpectTensorEqual<int64>(expected_indices_before_unique, *GetOutput(3));
}

TEST_F(FusedEmbeddingSparsePreLookUpOpTest, TestEverthingDiv) {
  MakeOpAndSetDevice(Device::GPU, 4, true, true, 3, std::string("div"));
  // partition_shapes 0
  AddInputFromArray<int64>(TensorShape({2}), {3, 16});
  // partition_shapes 1
  AddInputFromArray<int64>(TensorShape({2}), {4, 16});
  // partition_shapes 2
  AddInputFromArray<int64>(TensorShape({2}), {6, 16});
  // partition_shapes 3
  AddInputFromArray<int64>(TensorShape({2}), {5, 16});

  // for div, the parition interval is 0-2, 3-6, 7-12, 13-17

  // sp_values
  AddInputFromArray<int64>(TensorShape({16}), {-2, 1, 1, 3, -2, 4, 10, 2, 6, 14,
                                               12, 16, -1, -100, 12, 6});
  // sp_indices
  AddInputFromArray<int64>(
      TensorShape({16, 2}),
      {0, 3, 0, 4, 2,  1, 3,  2, 3,  6, 5,  1, 5,  4, 6,  7,
       8, 3, 9, 4, 11, 2, 11, 3, 11, 5, 12, 3, 13, 4, 13, 7});
  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {14, 16});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  /*
    after fill_empty and prune_invalid:
    1, 1, 3, 4, 10, 2, 6, 14, 12, 16, 12, 6,  3, 3, 3,  3, 3
    ---------------------------------------------------------
    0| 2| 3| 5| 5|  6| 8| 9|  11| 11| 13| 13| 1| 4| 7| 10| 12
    4| 1| 2| 1| 4|  7| 3| 4|   2|  3|  4|  7| 0| 0| 0|  0|  0
    ---------------------------------------------------------
    0, 1, 2, 3, 4,  5, 6, 7,   8,  9,  10, 11,12,13,14, 15, 16


    after unique:
    1, 3, 4, 10, 2, 6, 14 ,12 ,16
    -----------------------------
    0, 1, 2, 3,  4, 5, 6,  7,  8
  */

  // partitioned_values and partition_permutations 0
  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({2}));
    test::FillValues<int64>(&expected_values, {1, 2});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));

    Tensor expected_permutes(allocator(), DT_INT64, TensorShape({2}));
    test::FillValues<int64>(&expected_permutes, {0, 4});
    test::ExpectTensorEqual<int64>(expected_permutes, *GetOutput(4));
  }
  // partitioned_values and partition_permutations 1
  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({3}));
    test::FillValues<int64>(&expected_values, {0, 1, 3});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(1));

    Tensor expected_permutes(allocator(), DT_INT64, TensorShape({3}));
    test::FillValues<int64>(&expected_permutes, {1, 2, 5});
    test::ExpectTensorEqual<int64>(expected_permutes, *GetOutput(5));
  }

  // partitioned_values and partition_permutations 2
  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({2}));
    test::FillValues<int64>(&expected_values, {3, 5});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(2));

    Tensor expected_permutes(allocator(), DT_INT64, TensorShape({2}));
    test::FillValues<int64>(&expected_permutes, {3, 7});
    test::ExpectTensorEqual<int64>(expected_permutes, *GetOutput(6));
  }
  // partitioned_values and partition_permutations 3
  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({2}));
    test::FillValues<int64>(&expected_values, {1, 3});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(3));

    Tensor expected_permutes(allocator(), DT_INT64, TensorShape({2}));
    test::FillValues<int64>(&expected_permutes, {6, 8});
    test::ExpectTensorEqual<int64>(expected_permutes, *GetOutput(7));
  }

  // indices_before_unique
  Tensor expected_indices_before_unique(allocator(), DT_INT64,
                                        TensorShape({17, 2}));
  test::FillValues<int64>(
      &expected_indices_before_unique,
      {0, 4,  2, 1,  3, 2,  5, 1, 5, 4, 6, 7, 8, 3,  9, 4,  11,
       2, 11, 3, 13, 4, 13, 7, 1, 0, 4, 0, 7, 0, 10, 0, 12, 0});
  test::ExpectTensorEqual<int64>(expected_indices_before_unique, *GetOutput(9));

  // unique_idxs
  Tensor expected_unique_idxs(allocator(), DT_INT64, TensorShape({17}));
  test::FillValues<int64>(&expected_unique_idxs,
                          {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 5, 1, 1, 1, 1, 1});
  test::ExpectTensorEqual<int64>(expected_unique_idxs, *GetOutput(10));

  // unique_counts
  Tensor expected_unique_counts(allocator(), DT_INT64, TensorShape({9}));
  test::FillValues<int64>(&expected_unique_counts, {2, 6, 1, 1, 1, 2, 1, 2, 1});
  test::ExpectTensorEqual<int64>(expected_unique_counts, *GetOutput(11));

  // idx_of_input_to_unique
  Tensor expected_idx_of_input_to_unique(allocator(), DT_INT64,
                                         TensorShape({17}));
  test::FillValues<int64>(
      &expected_idx_of_input_to_unique,
      {0, 1, 2, 12, 13, 14, 15, 16, 3, 4, 5, 6, 11, 7, 8, 10, 9});
  test::ExpectTensorEqual<int64>(expected_idx_of_input_to_unique,
                                 *GetOutput(12));

  // unique_offsets
  Tensor expected_unique_offsets(allocator(), DT_INT64, TensorShape({9}));
  test::FillValues<int64>(&expected_unique_offsets,
                          {0, 2, 8, 9, 10, 11, 13, 14, 16});
  test::ExpectTensorEqual<int64>(expected_unique_offsets, *GetOutput(13));
}

}  // namespace
}  // namespace tensorflow