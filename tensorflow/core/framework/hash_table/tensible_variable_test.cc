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

#include <utility>

#include "tensorflow/core/framework/hash_table/tensible_variable.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

#include "gmock/gmock.h"

namespace tensorflow {

namespace {

class MockProducer {
 public:
  MOCK_METHOD0(Produce, std::pair<Status, Tensor>());
};

TEST(TensibleVariable, Simple) {
  Status ok;
  Tensor a(DT_INT64, TensorShape({10})), b(DT_INT64, TensorShape({10}));
  std::vector<Tensor> tensors;
  for (int i = 0; i < 10; i++) {
    tensors.emplace_back(DT_INT64, TensorShape({5, 2}));
    auto f = tensors[i].flat<int64>();
    for (int j = 0; j < 10; j++) {
      f(j) = i * 10 + j;
    }
  }
  MockProducer producer;
  EXPECT_CALL(producer, Produce())
    .WillOnce(::testing::Return(std::make_pair(ok, tensors[0])))
    .WillOnce(::testing::Return(std::make_pair(ok, tensors[1])))
    .WillOnce(::testing::Return(std::make_pair(ok, tensors[2])))
    .WillOnce(::testing::Return(std::make_pair(ok, tensors[3])))
    .WillOnce(::testing::Return(std::make_pair(ok, tensors[4])))
    .WillOnce(::testing::Return(std::make_pair(ok, tensors[5])))
    .WillOnce(::testing::Return(std::make_pair(ok, tensors[6])))
    .WillOnce(::testing::Return(std::make_pair(ok, tensors[7])))
    .WillOnce(::testing::Return(std::make_pair(ok, tensors[8])))
    .WillOnce(::testing::Return(std::make_pair(ok, tensors[9])));
  Status rst_status;
  TensorGenerator* generator = new TensorGenerator(
  [&](TensorGenerator::Consumer consumer) {
    auto rst = producer.Produce();
    consumer(rst.first, rst.second);
  });
  auto consumer = [&](Status st) {
    rst_status = st;
  };
  TensibleVariable* tv = new TensibleVariable(
      generator, TensorShape({5, 2}), DT_INT64);
  generator->Unref();

  tv->Resize(4, consumer);
  TF_EXPECT_OK(rst_status);
  EXPECT_EQ(5, tv->Size());

  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(i * 2 + 0, tv->GetSlice<int64>(i)[0]);
    EXPECT_EQ(i * 2 + 1, tv->GetSlice<int64>(i)[1]);
  }

  tv->Resize(23, consumer);
  TF_EXPECT_OK(rst_status);
  EXPECT_EQ(25, tv->Size());

  for (int i = 0; i < 23; i++) {
    EXPECT_EQ(i * 2 + 0, tv->GetSlice<int64>(i)[0]);
    EXPECT_EQ(i * 2 + 1, tv->GetSlice<int64>(i)[1]);
  }

  EXPECT_EQ(5, tv->SegmentSize());
  EXPECT_EQ(TensorShape({5, 2}), tv->shape());
  EXPECT_EQ(DT_INT64, tv->dtype());
  EXPECT_EQ(2 * sizeof(int64), tv->SliceSize());

  tv->Unref();
}

}  // namespace

}  // namespace tensorflow
