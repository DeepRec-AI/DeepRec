/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#include "tensorflow/core/kernels/incr_save_restore_ops.h"

namespace tensorflow {
namespace {

void doTestSplitParallelParts(int part_count, int min_part_size, int total_num,
                              std::vector<std::pair<int64, int64>> expect_parts)
{
  ParallelHashMap<int32> parallel_hashmap(min_part_size, part_count);

  std::vector<std::pair<int64, int64>> parts;
  parallel_hashmap.SplitParallelParts(total_num, part_count, parts);

  ASSERT_EQ(expect_parts.size(), parts.size());
  for (size_t i = 0; i < parts.size(); i++) {
    EXPECT_EQ(expect_parts[i].first, parts[i].first);
    EXPECT_EQ(expect_parts[i].second, parts[i].second);
  }
}

TEST(ParallelHashMapTest, TestSplitParallelParts) {
  doTestSplitParallelParts(4, 3, 0, {});
  doTestSplitParallelParts(4, 3, 1, {{0, 1}});
  doTestSplitParallelParts(4, 3, 8, {{0, 4}, {4, 8}});
  doTestSplitParallelParts(4, 3, 12, {{0, 3}, {3, 6}, {6, 9}, {9, 12}});
  doTestSplitParallelParts(4, 3, 13, {{0, 4}, {4, 7}, {7, 10}, {10, 13}});
  doTestSplitParallelParts(4, 3, 15, {{0, 4}, {4, 8}, {8, 12}, {12, 15}});
  doTestSplitParallelParts(4, 3, 16, {{0, 4}, {4, 8}, {8, 12}, {12, 16}});
  doTestSplitParallelParts(4, 3, 17, {{0, 5}, {5, 9}, {9, 13}, {13, 17}});
}

TEST(ParallelHashMapTest, TestUpdateAndSwap) {
  ParallelHashMap<int32> parallel_hashmap(2);
  Tensor t(DT_INT32, TensorShape({5}));
  test::FillValues<int32>(&t, {1, 2, 3, 2, 3});

  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  std::unique_ptr<OpKernelContext> context(new OpKernelContext(&params, 3));

  parallel_hashmap.Update(t, context.get());

  std::unordered_map<int32, uint64> out_indices;
  parallel_hashmap.Swap(out_indices);
  EXPECT_EQ(3, out_indices.size());
  EXPECT_EQ(1, out_indices[1]);
  EXPECT_EQ(2, out_indices[2]);
  EXPECT_EQ(2, out_indices[3]);
}

TEST(ParallelHashMapTest, TestGetKeys) {
  ParallelHashMap<int32> parallel_hashmap(2);
  Tensor t(DT_INT32, TensorShape({5}));
  test::FillValues<int32>(&t, {1, 2, 3, 2, 3});

  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  std::unique_ptr<OpKernelContext> context(new OpKernelContext(&params, 3));

  parallel_hashmap.Update(t, context.get());

  std::set<int32> keys;
  parallel_hashmap.GetKeys(keys);
  EXPECT_EQ(3, keys.size());
  EXPECT_TRUE(keys.find(1) != keys.end());
  EXPECT_TRUE(keys.find(2) != keys.end());
  EXPECT_TRUE(keys.find(3) != keys.end());
}

TEST(IndicesIncrRecorderTest, TestUpdateAndSwap) {
  Tensor t(DT_INT32, TensorShape({5}));
  test::FillValues<int32>(&t, {1, 2, 3, 2, 3});

  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  std::unique_ptr<OpKernelContext> context(new OpKernelContext(&params, 3));

  IndicesIncrRecorder<int32> recorder("test", 16, 2);
  recorder.UpdateGlobalVersion();
  recorder.UpdateIndices(t, context.get());

  std::unordered_map<int32, uint64> out_indices;
  recorder.SwapIndices(out_indices);
  EXPECT_EQ(3, out_indices.size());
  EXPECT_EQ(1, out_indices[1]);
  EXPECT_EQ(2, out_indices[2]);
  EXPECT_EQ(2, out_indices[3]);
}

TEST(DivSparsePartitionerTest, TestCalcGlobalOffset) {
  // part_count: 4, hash_bucket_size: 15
  // [0, 4), [4, 8), [8, 12), [12, 15)

  {
    DivSparsePartitioner p(4, 0, 15);
    EXPECT_EQ(0, p.CalcGlobalOffset(0));
    EXPECT_EQ(1, p.CalcGlobalOffset(1));
    EXPECT_EQ(2, p.CalcGlobalOffset(2));
    EXPECT_EQ(3, p.CalcGlobalOffset(3));
  }

  {
    DivSparsePartitioner p(4, 1, 15);
    EXPECT_EQ(4, p.CalcGlobalOffset(0));
    EXPECT_EQ(5, p.CalcGlobalOffset(1));
    EXPECT_EQ(6, p.CalcGlobalOffset(2));
    EXPECT_EQ(7, p.CalcGlobalOffset(3));
  }

  {
    DivSparsePartitioner p(4, 2, 15);
    EXPECT_EQ(8, p.CalcGlobalOffset(0));
    EXPECT_EQ(9, p.CalcGlobalOffset(1));
    EXPECT_EQ(10, p.CalcGlobalOffset(2));
    EXPECT_EQ(11, p.CalcGlobalOffset(3));
  }

  {
    DivSparsePartitioner p(4, 3, 15);
    EXPECT_EQ(12, p.CalcGlobalOffset(0));
    EXPECT_EQ(13, p.CalcGlobalOffset(1));
    EXPECT_EQ(14, p.CalcGlobalOffset(2));
  }
}

class CollectOpTest : public OpsTestBase {
 protected:
  void MakeOp(const string &config_str,
              const string &tensor_name,
              DataType ktype,
              const string &part_mode = "div",
              int64 part_idx = 0,
              int64 part_count = 0,
              int64 hash_bucket_size = 0)
  {
    TF_EXPECT_OK(NodeDefBuilder("collect_op", "CollectSparseIndices")
                 .Attr("tensor_name", tensor_name)
                 .Attr("config", config_str)
                 .Attr("part_idx", part_idx)
                 .Attr("part_count", part_count)
                 .Attr("hash_bucket_size", hash_bucket_size)
                 .Attr("part_mode", part_mode)
                 .Attr("ktype", ktype)
                 .Finalize(node_def()));

    TF_EXPECT_OK(InitOp());
  }

  template <typename KeyType>
  void CheckCollect() {
    string tensor_name = "test_tensor_name";
    DataType key_type = DataTypeToEnum<KeyType>::v();
    MakeOp("", tensor_name, key_type);

    // prepare context to run the op
    context_.reset(nullptr);

    params_.reset(new OpKernelContext::Params);
    params_.get()->device = device_;
    params_.get()->frame_iter = FrameAndIter(0, 0);
    params_.get()->inputs = &inputs_;
    params_.get()->op_kernel = kernel_.get();
    step_container_.reset(new ScopedStepContainer(0, [](const string&) {}));
    params_->step_container = step_container_.get();
    std::vector<AllocatorAttributes> attrs;
    test::SetOutputAttrs(params_.get(), &attrs);
    checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
    params_.get()->slice_reader_cache = &slice_reader_cache_wrapper;
    params_.get()->resource_manager = device_->resource_manager();

    context_.reset(new OpKernelContext(params_.get()));

    IndicesIncrRecorder<KeyType>* sparse_incr_res = nullptr;
    auto rm = device_->resource_manager();

    Status s = rm->LookupOrCreate<IndicesIncrRecorder<KeyType>>(
        "", tensor_name + "_sparse_incr", &sparse_incr_res,
        [this, tensor_name](IndicesIncrRecorder<KeyType>** ptr) {
          *ptr = new IndicesIncrRecorder<KeyType>(tensor_name);
          (*ptr)->UpdateGlobalVersion();
          return Status::OK();
        });
    ASSERT_TRUE(s.ok());

    Tensor indices(allocator(), key_type, TensorShape({5}));
    test::FillValues<KeyType>(&indices, {
        (KeyType)1, (KeyType)2, (KeyType)3, (KeyType)4, (KeyType)5});
    sparse_incr_res->UpdateIndices(indices, context_.get());

    device_->Compute(kernel_.get(), context_.get());

    Tensor output_keys = *GetOutput(0);
    Tensor output_global_keys = *GetOutput(1);
    EXPECT_EQ(5, output_keys.NumElements());
    EXPECT_EQ(5, output_global_keys.NumElements());
    test::ExpectTensorEqual<KeyType>(output_keys, output_global_keys);
  }
};

#define TEST_COLLECT(kt)                                                \
  TEST_F(CollectOpTest, TestCollect##_##kt) { CheckCollect<kt>(); }

TEST_COLLECT(int64);
TEST_COLLECT(int32);

}  // namespace
}  // namespace tensorflow
