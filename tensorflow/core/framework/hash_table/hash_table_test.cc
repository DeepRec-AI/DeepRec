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

#include "tensorflow/core/framework/hash_table/hash_table.h"
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

}

TEST(HashTable, Simple) {
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
    .WillOnce(::testing::Return(std::make_pair(ok, tensors[7])));
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
  HashTable ht(2, true, 7, 3);
  ht.AddTensible(tv, consumer);
  TF_ASSERT_OK(rst_status);
  tv->Unref();

  {
    int64 keys[8] = {105, 100, 101, 102, 103, 100, 101, 104};
    int64 ids[8];
    int64 result[8] = {0, 1, 5, 2, 3, 1, 5, 6};
    ht.GetIds(keys, nullptr, ids, 8, nullptr, nullptr, consumer, false);

    for (int i = 0; i < 8; i++) {
      EXPECT_EQ(result[i], ids[i]);
    }
    EXPECT_EQ(10, tv->Size());
  }

  {
    int64 keys[9] = {105 + 64, 100 + 64, 101 + 64, 102 + 64, 103 + 64, 100 + 64, 101 + 64, 104 + 64, 106 + 64};
    int64 ids[9];
    int64 result[9] = {4, 8, 7, 9, 10, 8, 7, 13, 11};
    ht.GetIds(keys, nullptr, ids, 9, nullptr, nullptr, consumer, false);

    for (int i = 0; i < 9; i++) {
      EXPECT_EQ(result[i], ids[i]);
    }
    EXPECT_EQ(15, tv->Size());
  }
}

}  // namespace tensorflow


