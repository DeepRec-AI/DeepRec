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

#include "tensorflow/core/framework/hash_table/tensor_generator.h"
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


TEST(TensorGenerator, Simple) {
  Status ok;
  Status err = errors::InvalidArgument("error");
  Tensor a(DT_INT64, TensorShape({10})), b(DT_INT64, TensorShape({10}));
  MockProducer producer;
  EXPECT_CALL(producer, Produce())
    .WillOnce(::testing::Return(std::make_pair(ok, a)))
    .WillOnce(::testing::Return(std::make_pair(ok, a)))
    .WillOnce(::testing::Return(std::make_pair(ok, a)))
    .WillOnce(::testing::Return(std::make_pair(ok, b)))
    .WillOnce(::testing::Return(std::make_pair(ok, b)))
    .WillOnce(::testing::Return(std::make_pair(ok, b)))
    .WillOnce(::testing::Return(std::make_pair(ok, b)))
    .WillOnce(::testing::Return(std::make_pair(err, a)))
    .WillOnce(::testing::Return(std::make_pair(ok, b)))
    .WillOnce(::testing::Return(std::make_pair(ok, b)))
    .WillOnce(::testing::Return(std::make_pair(ok, b)))
    .WillOnce(::testing::Return(std::make_pair(ok, b)))
    .WillOnce(::testing::Return(std::make_pair(ok, b)))
    .WillOnce(::testing::Return(std::make_pair(ok, b)));
  Status rst_status;
  Tensor rst_tensor;
  TensorGenerator* generator = new TensorGenerator(
  [&](TensorGenerator::Consumer consumer) {
    auto rst = producer.Produce();
    consumer(rst.first, rst.second);
  });
  auto consumer = [&](Status st, const Tensor& t) {
    rst_status = st;
    rst_tensor = t;
  };
  generator->GetNextTensor(consumer);
  EXPECT_EQ(ok, rst_status);
  EXPECT_TRUE(rst_tensor.SharesBufferWith(a));
  generator->GetNextTensor(consumer);
  EXPECT_EQ(ok, rst_status);
  EXPECT_TRUE(rst_tensor.SharesBufferWith(a));
  generator->GetNextTensor(consumer);
  EXPECT_EQ(ok, rst_status);
  EXPECT_TRUE(rst_tensor.SharesBufferWith(a));
  generator->GetNextTensor(consumer);
  EXPECT_EQ(ok, rst_status);
  EXPECT_TRUE(rst_tensor.SharesBufferWith(b));
  generator->GetNextTensor(consumer);
  EXPECT_EQ(ok, rst_status);
  EXPECT_TRUE(rst_tensor.SharesBufferWith(b));
  generator->GetNextTensor(consumer);
  EXPECT_EQ(ok, rst_status);
  EXPECT_TRUE(rst_tensor.SharesBufferWith(b));
  generator->GetNextTensor(consumer);
  EXPECT_EQ(ok, rst_status);
  EXPECT_TRUE(rst_tensor.SharesBufferWith(b));
  generator->GetNextTensor(consumer);
  EXPECT_EQ(err, rst_status);
  EXPECT_TRUE(rst_tensor.SharesBufferWith(a));
  EXPECT_TRUE(generator->RefCountIsOne());
  generator->GetNextTensor(consumer);
  EXPECT_EQ(ok, rst_status);
  EXPECT_TRUE(rst_tensor.SharesBufferWith(b));
  generator->Unref();
}

}  // namespace tensorflow

