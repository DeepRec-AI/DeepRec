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

#include <algorithm>
#include <functional>
#include <memory>
#include <random>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class DynamicStitchOpTest : public OpsTestBase {
 protected:
  void MakeOp(int n, DataType dt) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "DynamicStitch")
                     .Input(FakeInput(n, DT_INT32))
                     .Input(FakeInput(n, dt))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

class DynamicStitchFastOpTest : public OpsTestBase {
 protected:
  void MakeOp(int n, DataType dt) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "DynamicStitchFast")
                     .Input(FakeInput(n, DT_INT32))
                     .Input(FakeInput(n, dt))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(DynamicStitchOpTest, Simple_OneD) {
  MakeOp(2, DT_FLOAT);

  // Feed and run
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 7});
  AddInputFromArray<int32>(TensorShape({5}), {1, 6, 2, 3, 5});
  AddInputFromArray<float>(TensorShape({3}), {0, 40, 70});
  AddInputFromArray<float>(TensorShape({5}), {10, 60, 20, 30, 50});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({8}));
  test::FillValues<float>(&expected, {0, 10, 20, 30, 40, 50, 60, 70});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(DynamicStitchFastOpTest, Simple_OneD) {
  MakeOp(2, DT_FLOAT);
  // Feed and run
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 7});
  AddInputFromArray<int32>(TensorShape({5}), {1, 6, 2, 3, 5});
  AddInputFromArray<float>(TensorShape({3}), {0, 40, 70});
  AddInputFromArray<float>(TensorShape({5}), {10, 60, 20, 30, 50});
  TF_ASSERT_OK(RunOpKernel());
  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({8}));
  test::FillValues<float>(&expected, {0, 10, 20, 30, 40, 50, 60, 70});
  Tensor* result_ptr = GetOutput(0);
  TF_EXPECT_OK(device_->Sync());
  test::ExpectTensorEqual<float>(expected, *result_ptr);
}

TEST_F(DynamicStitchOpTest, Simple_TwoD) {
  MakeOp(3, DT_FLOAT);

  // Feed and run
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 7});
  AddInputFromArray<int32>(TensorShape({2}), {1, 6});
  AddInputFromArray<int32>(TensorShape({3}), {2, 3, 5});
  AddInputFromArray<float>(TensorShape({3, 2}), {0, 1, 40, 41, 70, 71});
  AddInputFromArray<float>(TensorShape({2, 2}), {10, 11, 60, 61});
  AddInputFromArray<float>(TensorShape({3, 2}), {20, 21, 30, 31, 50, 51});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({8, 2}));
  test::FillValues<float>(&expected, {0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50,
                                      51, 60, 61, 70, 71});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(DynamicStitchFastOpTest, Simple_TwoD) {
  MakeOp(3, DT_FLOAT);
  // Feed and run
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 7});
  AddInputFromArray<int32>(TensorShape({2}), {1, 6});
  AddInputFromArray<int32>(TensorShape({3}), {2, 3, 5});
  AddInputFromArray<float>(TensorShape({3, 2}), {0, 1, 40, 41, 70, 71});
  AddInputFromArray<float>(TensorShape({2, 2}), {10, 11, 60, 61});
  AddInputFromArray<float>(TensorShape({3, 2}), {20, 21, 30, 31, 50, 51});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({8, 2}));
  test::FillValues<float>(&expected, {0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50,
                                      51, 60, 61, 70, 71});
  Tensor* result_ptr = GetOutput(0);
  TF_EXPECT_OK(device_->Sync());
  test::ExpectTensorEqual<float>(expected, *result_ptr);
}

TEST_F(DynamicStitchOpTest, Error_IndicesMultiDimensional) {
  MakeOp(2, DT_FLOAT);

  // Feed and run
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 7});
  AddInputFromArray<int32>(TensorShape({1, 5}), {1, 6, 2, 3, 5});
  AddInputFromArray<float>(TensorShape({3}), {0, 40, 70});
  AddInputFromArray<float>(TensorShape({5}), {10, 60, 20, 30, 50});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(),
      "data[1].shape = [5] does not start with indices[1].shape = [1,5]"))
      << s;
}

TEST_F(DynamicStitchOpTest, Error_DataNumDimsMismatch) {
  MakeOp(2, DT_FLOAT);

  // Feed and run
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 7});
  AddInputFromArray<int32>(TensorShape({5}), {1, 6, 2, 3, 5});
  AddInputFromArray<float>(TensorShape({3}), {0, 40, 70});
  AddInputFromArray<float>(TensorShape({1, 5}), {10, 60, 20, 30, 50});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(),
      "data[1].shape = [1,5] does not start with indices[1].shape = [5]"))
      << s;
}

TEST_F(DynamicStitchOpTest, Error_DataDimSizeMismatch) {
  MakeOp(2, DT_FLOAT);

  // Feed and run
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 5});
  AddInputFromArray<int32>(TensorShape({4}), {1, 6, 2, 3});
  AddInputFromArray<float>(TensorShape({3, 1}), {0, 40, 70});
  AddInputFromArray<float>(TensorShape({4, 2}),
                           {10, 11, 60, 61, 20, 21, 30, 31});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      absl::StrContains(s.ToString(),
                        "Need data[0].shape[1:] = data[1].shape[1:], got "
                        "data[0].shape = [3,1], data[1].shape = [4,2]"))
      << s;
}

TEST_F(DynamicStitchOpTest, Error_DataAndIndicesSizeMismatch) {
  MakeOp(2, DT_FLOAT);

  // Feed and run
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 7});
  AddInputFromArray<int32>(TensorShape({5}), {1, 6, 2, 3, 5});
  AddInputFromArray<float>(TensorShape({3}), {0, 40, 70});
  AddInputFromArray<float>(TensorShape({4}), {10, 60, 20, 30});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(),
      "data[1].shape = [4] does not start with indices[1].shape = [5]"))
      << s;
}

Node* DynamicStitchFastNode(Graph* g,
                            Node** data,
                            Node** indices,
                            int num_partitions) {
  std::vector<NodeBuilder::NodeOut> stitch_data_list;
  std::vector<NodeBuilder::NodeOut> stitch_indices_list;
  for(int i=0; i< num_partitions; i++) {
    NodeBuilder::NodeOut data_item(data[i]);
    NodeBuilder::NodeOut indices_item(indices[i]);
    stitch_data_list.emplace_back(data_item);
    stitch_indices_list.emplace_back(indices_item);
  }
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("stitch"), "DynamicStitchFast")
              .Input(stitch_indices_list)
              .Input(stitch_data_list)
              .Attr("N", num_partitions)
              .Finalize(g, &ret));
  return ret;
}

template <typename T>
static Graph* DynamicStitchFast(int num_partitions, int dim) {
  Graph* g = new Graph(OpRegistry::Global());
  // Always use a 128MB buffer.
  const int kRows = ((128 << 20) / sizeof(T)) / dim;
  // prepare data
  std::vector<Tensor*> data(num_partitions);
  for(int i=0; i<num_partitions; i++) {
    data[i] = new Tensor(DataTypeToEnum<T>::value,
                         TensorShape({kRows, dim}));
    data[i]->flat<T>().setRandom();
  }
  // prepare indices
  const int len = kRows*num_partitions;
  std::vector<Tensor*> indices(num_partitions);
  std::vector<int32> indices_val(len);
  for(int i=0; i<len; i++) {
      indices_val[i] = i;
  }
  // shuffling the indices
  auto rng = std::default_random_engine {};
  std::shuffle(std::begin(indices_val), std::end(indices_val), rng);
  for(int i=0; i<num_partitions; i++) {
    indices[i] = new Tensor(DT_INT32, TensorShape({kRows}));
    // initialization
    for(int j=0; j<kRows; ++j) {
      indices[i]->flat<int32>()(j) = indices_val[i*kRows+j];
    }
  }
  std::vector<Node*> data_nodes(num_partitions);
  std::vector<Node*> indices_nodes(num_partitions);
  for(int i=0; i<num_partitions; i++) {
    data_nodes[i] = test::graph::Constant(g, *(data[i]));
    indices_nodes[i] = test::graph::Constant(g, *(indices[i]));
  }
  DynamicStitchFastNode(g, &data_nodes[0], &indices_nodes[0], num_partitions);
  return g;
}

#define BM_DYNAMIC_STITCH_FAST(DEVICE, T, num)                          \
  static void BM_##DEVICE##_dystitch_fast_##T##_##num(int iters, int dim) {  \
    const int64 items = ((128 << 20) / sizeof(T)) * num;                \
    const int64 tot = static_cast<int64>(iters) * items;                \
    tensorflow::SessionOptions options;                                 \
    options.config.set_inter_op_parallelism_threads(16);                \
    options.config.set_intra_op_parallelism_threads(16);                \
    testing::ItemsProcessed(tot);                                       \
    testing::UseRealTime();                                             \
    test::Benchmark(#DEVICE, \
      DynamicStitchFast<T>(num, dim), &options).Run(iters);             \
  }                                                                     \
  BENCHMARK(BM_##DEVICE##_dystitch_fast_##T##_##num)->Arg(256)

// start benchmarking
BM_DYNAMIC_STITCH_FAST(cpu, float, 8);
BM_DYNAMIC_STITCH_FAST(gpu, float, 8);

}  // namespace
}  // namespace tensorflow
