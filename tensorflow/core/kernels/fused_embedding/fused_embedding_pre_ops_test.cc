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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
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

class FusedEmbeddingSparsePreLookUpOpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, const int num_partitions,
                          const bool fill_empty_row,
                          const bool prune_invalid_id, const int default_id) {
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
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(FusedEmbeddingSparsePreLookUpOpTest, Parition3_Int64) {
  MakeOpAndSetDevice(Device::GPU, 3, false, false, -1);
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
  AddInputFromArray<int64>(TensorShape({24}),
                           {2,  3, 4,  6, 1, 6, 12, 12, 12, 12, 11, 5,
                            15, 0, 11, 6, 7, 9, 11, 8,  12, 13, 13, 0});
  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {16, 16});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());
  {
    Tensor expected_values(allocator(), DT_INT64, TensorShape({6}));
    test::FillValues<int64>(&expected_values, {0, 1, 3, 5, 5, 5});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(0));

    Tensor expected_indices(allocator(), DT_INT64, TensorShape({6, 2}));
    test::FillValues<int64>(&expected_indices,
                            {11, 6, 2, 3, 1, 6, 4, 6, 7, 9, 11, 8});
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
    test::FillValues<int64>(&expected_values, {2, 3, 5, 6});
    test::ExpectTensorEqual<int64>(expected_values, *GetOutput(2));

    Tensor expected_indices(allocator(), DT_INT64, TensorShape({4, 2}));
    test::FillValues<int64>(&expected_indices, {12, 13, 12, 12, 11, 5, 15, 0});
    test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(5));
  }
}

}  // namespace
}  // namespace tensorflow