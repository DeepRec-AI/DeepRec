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

#include <stdio.h>

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
#include "tensorflow/cc/ops/standard_ops.h"
namespace tensorflow {
namespace {

enum class Device { CPU, GPU };

class FusedEmbeddingSparsePreLookUpOpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, const int num_partitions,
                          const bool fill_empty_row,
                          const bool prune_invalid_id, const int default_id,
                          const string partition_strategy = "div") {
    if (device == Device::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }

    TF_EXPECT_OK(NodeDefBuilder("FusedEmbeddingSparsePreLookUp",
                                "FusedEmbeddingSparsePreLookUp")
                     .Attr("num_partitions", num_partitions)
                     .Attr("partition_strategy", partition_strategy)
                     .Attr("partition_axis", 0)
                     .Attr("fill_empty_row", fill_empty_row)
                     .Attr("prune_invalid_id", prune_invalid_id)
                     .Attr("default_id", default_id)
                     .Input(FakeInput(num_partitions, DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(FusedEmbeddingSparsePreLookUpOpTest, Partition3_Int64) {
  MakeOpAndSetDevice(Device::CPU, 3, false, false, -1);
  // partition_shapes 0
  AddInputFromArray<int64>(TensorShape({2}), {6, 16});
  // partition_shapes 1
  AddInputFromArray<int64>(TensorShape({2}), {3, 16});
  // partition_shapes 2
  AddInputFromArray<int64>(TensorShape({2}), {7, 16});
  // sp_values
  AddInputFromArray<int64>(TensorShape({12}),
                           {1, 5, 3, 6, 12, 14,
                           15, 0, 5, 5, 11, 7});
  // sp_indices
  AddInputFromArray<int64>(TensorShape({12, 2}),
                           {2,  3, 4,  6, 1, 6, 12, 12, 12, 12, 11, 5,
                            15, 0, 11, 6, 7, 9, 11, 8,  12, 13, 13, 0});
  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {16, 16});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({6}));
    test::FillValues<int64>(&expected_values, {1, 5, 3, 0, 5, 5});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));

    Tensor expected_indices(allocator(), DT_INT64, TensorShape({6, 2}));
    test::FillValues<int64>(&expected_indices,
                            {2, 3, 4, 6, 1, 6, 11, 6, 7, 9, 11, 8});
    test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(3));
  }

  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({2}));
    test::FillValues<int64>(&expected_values, {0, 1});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(1));
    Tensor expected_indices(allocator(), DT_INT64, TensorShape({2, 2}));
    test::FillValues<int64>(&expected_indices, {12, 12, 13, 0});
    test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(4));
  }

  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({4}));
    test::FillValues<int64>(&expected_values, {1, 3, 4, 0});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(2));

    Tensor expected_indices(allocator(), DT_INT64, TensorShape({4, 2}));
    test::FillValues<int64>(&expected_indices, {12, 12, 11, 5, 15, 0, 12, 13});
    test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(5));
  }
}

TEST_F(FusedEmbeddingSparsePreLookUpOpTest, Partition2_Fill_Empty) {
  MakeOpAndSetDevice(Device::CPU, 2, true, false, -1);
  // partition_shapes 0
  AddInputFromArray<int64>(TensorShape({2}), {5, 8});
  // partition_shapes 1
  AddInputFromArray<int64>(TensorShape({2}), {5, 8}); 

  // sp_values
  AddInputFromArray<int64>(TensorShape({10}),
                           {0, 4, 3, -2, 5, 
                           -3, -4, 9, -6, 2});

  // sp_indices
  AddInputFromArray<int64>(
      TensorShape({10, 2}),
      {0, 0, 0, 4, 1, 2, 3, 0, 3, 4,
       4, 0, 5, 2, 6, 0, 6, 1, 6, 7});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {7, 8});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({9}));
    test::FillValues<int64>(&expected_values, {0, 4, 3, -2, -3, -4, -6, 2, 0});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));

    Tensor expected_indices(allocator(), DT_INT64, TensorShape({9, 2}));
    test::FillValues<int64>(&expected_indices, {0, 0, 0, 4, 1, 2, 3, 0,
                                                4, 0, 5, 2, 6, 1, 6, 7, 2, 0});
    test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(2));
  }

  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({2}));
    test::FillValues<int64>(&expected_values, {0, 4});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(1));
    Tensor expected_indices(allocator(), DT_INT64, TensorShape({2, 2}));
    test::FillValues<int64>(&expected_indices, {3, 4, 6, 0});
    test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(3));
  }
}

TEST_F(FusedEmbeddingSparsePreLookUpOpTest,
       Partition2_Fill_Empty_Prune_Invalid) {
  MakeOpAndSetDevice(Device::CPU, 2, true, true, -1);
  // partition_shapes 0
  AddInputFromArray<int64>(TensorShape({2}), {5, 8});
  // partition_shapes 1
  AddInputFromArray<int64>(TensorShape({2}), {5, 8});

  // sp_values
  AddInputFromArray<int64>(TensorShape({10}),
                           {0, 4, 3, -2, 5,
                           -3, -4, 9, -6, 2});

  // sp_indices
  AddInputFromArray<int64>(
      TensorShape({10, 2}),
      {0, 0, 0, 4, 1, 2, 3, 0, 3, 4, 
       4, 0, 5, 2, 6, 0, 6, 1, 6, 7});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {7, 8});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({7}));
    test::FillValues<int64>(&expected_values, {0, 4, 3, 2, 0, 0, 0});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));

    Tensor expected_indices(allocator(), DT_INT64, TensorShape({7, 2}));
    test::FillValues<int64>(&expected_indices,
                            {0, 0, 0, 4, 1, 2, 6, 7, 2, 0, 4, 0, 5, 0});
    test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(2));
  }

  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({2}));
    test::FillValues<int64>(&expected_values, {0, 4});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(1));
    Tensor expected_indices(allocator(), DT_INT64, TensorShape({2, 2}));
    test::FillValues<int64>(&expected_indices, {3, 4, 6, 0});
    test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(3));
  }
}

TEST_F(FusedEmbeddingSparsePreLookUpOpTest,
       Partition2_Fill_Empty_Prune_Invalid_Default_7) {
  MakeOpAndSetDevice(Device::CPU, 2, true, true, 7);
  // partition_shapes 0
  AddInputFromArray<int64>(TensorShape({2}), {5, 8});
  // partition_shapes 1
  AddInputFromArray<int64>(TensorShape({2}), {5, 8});

  // sp_values
  AddInputFromArray<int64>(TensorShape({10}),
                           {0, 4, 3, -2, 5,
                           -3, -4, 9, -6, 2});

  // sp_indices
  AddInputFromArray<int64>(
      TensorShape({10, 2}),
      {0, 0, 0, 4, 1, 2, 3, 0, 3, 4, 
       4, 0, 5, 2, 6, 0, 6, 1, 6, 7});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {7, 8});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({4}));
    test::FillValues<int64>(&expected_values, {0, 4, 3, 2});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));

    Tensor expected_indices(allocator(), DT_INT64, TensorShape({4, 2}));
    test::FillValues<int64>(&expected_indices,
                            {0, 0, 0, 4, 1, 2, 6, 7});
    test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(2));
  }

  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({5}));
    test::FillValues<int64>(&expected_values, {0, 4, 2, 2, 2});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(1));
    Tensor expected_indices(allocator(), DT_INT64, TensorShape({5, 2}));
    test::FillValues<int64>(&expected_indices, {3, 4, 6, 0, 2, 0, 4, 0, 5, 0});
    test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(3));
  }
}

TEST_F(FusedEmbeddingSparsePreLookUpOpTest,
       Partition2_Prune_Invalid_Default_3) {
  MakeOpAndSetDevice(Device::CPU, 2, false, true, 3);
  // partition_shapes 0
  AddInputFromArray<int64>(TensorShape({2}), {5, 8});
  // partition_shapes 1
  AddInputFromArray<int64>(TensorShape({2}), {5, 8});

  // sp_values
  AddInputFromArray<int64>(TensorShape({10}),
                           {0, 4, 3, -2, 5,
                           -3, -4, 9, -6, 2});

  // sp_indices
  AddInputFromArray<int64>(
      TensorShape({10, 2}),
      {0, 0, 0, 4, 1, 2, 3, 0, 3, 4, 
       4, 0, 5, 2, 6, 0, 6, 1, 6, 7});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {7, 8});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({4}));
    test::FillValues<int64>(&expected_values, {0, 4, 3, 2});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));

    Tensor expected_indices(allocator(), DT_INT64, TensorShape({4, 2}));
    test::FillValues<int64>(&expected_indices,
                            {0, 0, 0, 4, 1, 2, 6, 7});
    test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(2));
  }

  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({2}));
    test::FillValues<int64>(&expected_values, {0, 4});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(1));
    Tensor expected_indices(allocator(), DT_INT64, TensorShape({2, 2}));
    test::FillValues<int64>(&expected_indices, {3, 4, 6, 0});
    test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(3));
  }
}

TEST_F(FusedEmbeddingSparsePreLookUpOpTest, Partition1) {
  MakeOpAndSetDevice(Device::CPU, 1, false, false, -1);
  // partition_shapes 0
  AddInputFromArray<int64>(TensorShape({2}), {10, 8});

  // sp_values
  AddInputFromArray<int64>(TensorShape({10}),
                           {0, 4, 3, -2, 5, -3, -4, 9, -6, 2});

  // sp_indices
  AddInputFromArray<int64>(
      TensorShape({10, 2}),
      {0, 0, 0, 4, 1, 2, 3, 0, 3, 4, 4, 0, 5, 2, 6, 0, 6, 1, 6, 7});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {7, 8});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({10}));
    test::FillValues<int64>(&expected_values,
                            {0, 4, 3, -2, 5, -3, -4, 9, -6, 2});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));

    Tensor expected_indices(allocator(), DT_INT64, TensorShape({10, 2}));
    test::FillValues<int64>(&expected_indices, {0, 0, 0, 4, 1, 2, 3, 0, 3, 4,
                                                4, 0, 5, 2, 6, 0, 6, 1, 6, 7});
    test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(1));
  }
}

TEST_F(FusedEmbeddingSparsePreLookUpOpTest,
       Partition1_Fill_Empty_Prune_Invalid_Default_3) {
  MakeOpAndSetDevice(Device::CPU, 1, true, true, 3);
  // partition_shapes 0
  AddInputFromArray<int64>(TensorShape({2}), {10, 8});

  // sp_values
  AddInputFromArray<int64>(TensorShape({10}),
                           {0, 4, 3, -2, 5, -3, -4, 9, -6, 2});

  // sp_indices
  AddInputFromArray<int64>(
      TensorShape({10, 2}),
      {0, 0, 0, 4, 1, 2, 3, 0, 3, 4, 4, 0, 5, 2, 6, 0, 6, 1, 6, 7});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {7, 8});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({9}));
    test::FillValues<int64>(&expected_values, {0, 4, 3, 5, 9, 2, 3, 3, 3});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));;

    Tensor expected_indices(allocator(), DT_INT64, TensorShape({9, 2}));
    test::FillValues<int64>(&expected_indices, {0, 0, 0, 4, 1, 2, 3, 4, 6, 0, 6,
                                                7, 2, 0, 4, 0, 5, 0});
    test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(1));
  }
}

TEST_F(FusedEmbeddingSparsePreLookUpOpTest, Partition3_Int64_Perfs) {

  int num_partitions = 4;
  int batch_size = 100000;
  int num_per_part = batch_size / num_partitions;
  int embed_dim = 32;
  int default_id = -1;

  std::vector<int64> sp_values;
  std::vector<int64> sp_indices;

  MakeOpAndSetDevice(Device::CPU, num_partitions, false, false, default_id);

  for(int i = 0; i < num_partitions; ++i){
    AddInputFromArray<int64>(TensorShape({2}), {num_per_part * embed_dim, embed_dim});
  }

  for(int i = 0; i < batch_size * embed_dim; ++i){
    sp_values.push_back(i);
  }

  for(int i = 0; i < batch_size; ++i){
    for(int j = 0; j < embed_dim; ++j){
      sp_indices.push_back(i);
      sp_indices.push_back(j);
    }
  }
  // sp_values
  AddInputFromArray<int64>(TensorShape({sp_values.size()}), sp_values);
  // sp_indices
  AddInputFromArray<int64>(TensorShape({sp_values.size(), 2}), sp_indices);
  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {batch_size, embed_dim});
  TF_ASSERT_OK(RunOpKernel());
}



//----------------------------------------------------------------------------//
// Performance benchmarks                                                     //
//----------------------------------------------------------------------------//

template <typename T>
void FillValues(Tensor* tensor, gtl::ArraySlice<T> vals) {
  auto flat = tensor->flat<T>();
  CHECK_EQ(flat.size(), vals.size());
  if (flat.size() > 0) {
    std::copy_n(vals.data(), vals.size(), flat.data());
  }
}

template <typename T>
void FillZerosValues(Tensor* tensor) {
  auto flat = tensor->flat<T>();
  for (int i = 0; i < flat.size(); ++i) {
    flat.data()[i] = 0.0;
  }
}

template <typename T>
void FillOnesValues(Tensor* tensor) {
  auto flat = tensor->flat<T>();
  float scale = std::rand()/((RAND_MAX + 1u)/6);
  for (int i = 0; i < flat.size(); ++i) {
    flat.data()[i] = 1.1 * scale;
  }
}

template <typename T>
void FillIndiceValues(Tensor* tensor, const int partitions, const int batch_size, const int entries) {
  auto flat = tensor->flat<T>();
  int k = 0;
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < entries; ++j) {
      flat.data()[k] = i + partitions;
      flat.data()[k+1] = j;
      k += 2;
    }
  }
}

template <typename T>
void PrintValues(Tensor* tensor) {
  auto flat = tensor->flat<T>();
  for (int i = 0; i < flat.size(); ++i) {
    std::cout << flat.data()[i] << ", ";
  }
  std::cout << std::endl;
}

template <typename T>
static Graph* EmbPreOp(const string& kind, int num_partitions, const std::string& combiner,
                      const float max_norm, const int default_id) {
  
  int batch_size = 100000;
  int num_per_part = batch_size / num_partitions;
  int embed_dim = 32;
  const string partition_strategy = "div";
  const bool fill_empty_row = false;
  const bool prune_invalid_id = false;

  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "FusedEmbeddingSparsePreLookUp" : "FusedEmbeddingSparsePreLookUp";

  std::vector<int64> sp_values;
  std::vector<int64> sp_indices;

  // partitioned_indices
  std::vector<NodeBuilder::NodeOut> partitioned_indices;
  partitioned_indices.reserve(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    Tensor sub_partitioned_indice(DT_INT64, TensorShape({2}));
    FillValues<int64>(&sub_partitioned_indice, {num_per_part * embed_dim, embed_dim});
    partitioned_indices.push_back(test::graph::Constant(g, sub_partitioned_indice));
  }

  for(int i = 0; i < batch_size * embed_dim; ++i){
    sp_values.push_back(i);
  }

  for(int i = 0; i < batch_size; ++i){
    for(int j = 0; j < embed_dim; ++j){
      sp_indices.push_back(i);
      sp_indices.push_back(j);
    }
  }

  // sp_values
  Tensor sp_values_t(DT_INT64, TensorShape({sp_values.size()}));
  FillValues<int64>(&sp_values_t, sp_values);

  // sp_indices
  Tensor sp_indices_t(DT_INT64, TensorShape({sp_values.size(), 2}));
  FillValues<int64>(&sp_indices_t, sp_indices);

  // sp_dense_shape
  Tensor sp_dense_shape_t(DT_INT64, TensorShape({2}));
  FillValues<int64>(&sp_dense_shape_t, {batch_size, embed_dim});

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Attr("num_partitions", num_partitions)
                    .Attr("partition_strategy", partition_strategy)
                    .Attr("partition_axis", 0)
                    .Attr("fill_empty_row", fill_empty_row)
                    .Attr("prune_invalid_id", prune_invalid_id)
                    .Attr("default_id", default_id)
                    .Input(partitioned_indices)
                    .Input(test::graph::Constant(g, sp_values_t))
                    .Input(test::graph::Constant(g, sp_indices_t))
                    .Input(test::graph::Constant(g, sp_dense_shape_t));
  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));
  return g;
}

#define BM_EMB_PRE_OP(kind, NP, C, T, DEVICE, NTH)                                \
  static void BM_EMB_PRE_OP##_##kind##_##NP##_##C##_##T##_##DEVICE##_##NTH(       \
      int iters) {                                                                \
    testing::UseRealTime();                                                       \
    SessionOptions opts;                                                          \
    opts.config.set_intra_op_parallelism_threads(NTH);                            \
    test::Benchmark(#DEVICE, EmbPreOp<T>(#kind, NP, #C, -1.0, -1), &opts).Run(iters); \
  }                                                                               \
  BENCHMARK(BM_EMB_PRE_OP##_##kind##_##NP##_##C##_##T##_##DEVICE##_##NTH);        \

#define BM_EMB_PRE_OP_kind(NP, C, NTH)            \
  BM_EMB_PRE_OP(OPT, NP, C, float, CPU, NTH);     \

#define BM_EMB_PRE_OP_NTH(NP, C) \
  BM_EMB_PRE_OP_kind(NP, C, 1);  \
  // BM_EMB_PRE_OP_kind(NP, C, 4);  \
  // BM_EMB_PRE_OP_kind(NP, C, 8);  \

BM_EMB_PRE_OP_NTH(4, sum);

}  // namespace
}  // namespace tensorflow