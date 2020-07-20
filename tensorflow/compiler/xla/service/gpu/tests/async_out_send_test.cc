/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <unistd.h>
#include <memory>

#include "tensorflow/compiler/jit/kernels/async_io_rendezvous.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace {

tensorflow::AsyncIoRendezvous::DoneCallback make_recv_callback(void* host_ptr) {
  using namespace std::placeholders;
  return [host_ptr](const Status& s,
                    const tensorflow::AsyncIoRendezvous::TensorPayload& val) {
    TF_ASSERT_OK(s);
    TF_ASSERT_OK_AND_ASSIGN(auto* platform,
                            xla::PlatformUtil::GetDefaultPlatform());
    TF_ASSERT_OK_AND_ASSIGN(auto executors,
                            xla::PlatformUtil::GetStreamExecutors(platform));
    se::StreamExecutor* executor = executors.front();
    size_t size = ShapeUtil::ByteSizeOf(val.shape);
    executor->SynchronousMemcpyD2H(val.addr, size, host_ptr);
  };
}

class AsyncOutSendTest : public ClientLibraryTestBase {
 public:
  AsyncOutSendTest()
      : ClientLibraryTestBase(
            xla::PlatformUtil::GetDefaultPlatform().ValueOrDie()) {}

 protected:
  void TestAsyncOutSendRoundTrip(const Literal& literal, void* host_ptr) {
    string key = "_uniq_key_string";
    uint64 key_hash = tensorflow::AsyncIoRendezvous::GetRendezvousKeyHash(key);
    tensorflow::GetXlaAsyncIORendezvous()->InitializeRendezvousQueue(key_hash);
    XlaBuilder builder(TestName());
    auto const_op = ConstantLiteral(&builder, literal);
    AsyncOutSend(const_op, literal.shape(), key);
    auto s = Execute(&builder, {});
    EXPECT_TRUE(s.ok());

    // Verify the transffered result. Note that because we are sure that the
    // sender has finished execution, the done callback of RecvAsync will be
    // executed immediately.
    tensorflow::GetXlaAsyncIORendezvous()->RecvAsync(
        key_hash, make_recv_callback(host_ptr));
    tensorflow::GetXlaAsyncIORendezvous()->FinalizeRendezvousQueue(key_hash);
  }

  void TestAsyncOutSendRoundTripRecvSend(const Literal& literal,
                                         void* host_ptr) {
    string key = "_uniq_key_string";
    uint64 key_hash = tensorflow::AsyncIoRendezvous::GetRendezvousKeyHash(key);
    tensorflow::GetXlaAsyncIORendezvous()->InitializeRendezvousQueue(key_hash);
    XlaBuilder builder(TestName());
    auto const_op = ConstantLiteral(&builder, literal);
    tensorflow::GetXlaAsyncIORendezvous()->RecvAsync(
        key_hash, make_recv_callback(host_ptr));
    xla::Shape shape_with_layout =
        xla::ShapeUtil::MakeShapeWithDescendingLayout(
            literal.shape().element_type(), literal.shape().dimensions());
    AsyncOutSend(const_op, shape_with_layout, key);
    auto s = Execute(&builder, {});
    tensorflow::GetXlaAsyncIORendezvous()->FinalizeRendezvousQueue(key_hash);

    EXPECT_TRUE(s.ok());
  }
};

TEST_F(AsyncOutSendTest, SingleAsyncOutSendR0Bool) {
  bool result_val = false;
  void* host_ptr = &result_val;
  TestAsyncOutSendRoundTrip(LiteralUtil::CreateR0<bool>(true), host_ptr);
  EXPECT_TRUE(result_val == true);
}

TEST_F(AsyncOutSendTest, SingleAsyncOutSendR1U32) {
  std::vector<uint32> expected{1, 3, 5};
  std::vector<uint32> result_on_host(3);
  TestAsyncOutSendRoundTrip(LiteralUtil::CreateR1<uint32>(expected),
                            result_on_host.data());
  LiteralTestUtil::ExpectR1Equal<uint32>(
      expected, LiteralUtil::CreateR1<uint32>(result_on_host));
}

TEST_F(AsyncOutSendTest, SingleAsyncOutSendR3F32) {
  Array3D<float> expected({{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                           {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}});
  Array3D<float> result_on_host(2, 2, 3);
  TestAsyncOutSendRoundTrip(LiteralUtil::CreateR3FromArray3D(expected),
                            result_on_host.data());
  LiteralTestUtil::ExpectR3EqualArray3D<float>(
      expected, LiteralUtil::CreateR3FromArray3D(result_on_host));
}

TEST_F(AsyncOutSendTest, SingleAsyncOutSendR3F32_RecvSend) {
  Array3D<float> expected({{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                           {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}});
  Array3D<float> result_on_host(2, 2, 3);
  TestAsyncOutSendRoundTripRecvSend(
      LiteralUtil::CreateR3({{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                             {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}}),
      result_on_host.data());
  LiteralTestUtil::ExpectR3EqualArray3D<float>(
      expected, LiteralUtil::CreateR3FromArray3D(result_on_host));
}

// Tests that a large AsyncOutSend can be handled.
TEST_F(AsyncOutSendTest, LargeAsyncOutSend) {
  Array4D<float> input(80, 100, 8, 128);
  input.FillIota(1.0f);
  Array4D<float> output(80, 100, 8, 128);
  TestAsyncOutSendRoundTrip(LiteralUtil::CreateR4FromArray4D<float>(input),
                            output.data());
  EXPECT_TRUE(output.data()[0] == 1.0f);
  EXPECT_TRUE(output.data()[1000] == 1001.0f);
}

TEST_F(AsyncOutSendTest, LargeAsyncOutSend_RecvSend) {
  Array4D<float> input(80, 100, 8, 128);
  input.FillIota(1.0f);
  Array4D<float> output(80, 100, 8, 128);
  TestAsyncOutSendRoundTripRecvSend(
      LiteralUtil::CreateR4FromArray4D<float>(input), output.data());
  EXPECT_TRUE(output.data()[0] == 1.0f);
  EXPECT_TRUE(output.data()[1000] == 1001.0f);
}

TEST_F(AsyncOutSendTest, SingleAsyncOutSendR3F32DifferentLayout) {
  const Layout r3_dim0minor = LayoutUtil::MakeLayout({0, 1, 2});
  const Layout r3_dim0major = LayoutUtil::MakeLayout({2, 1, 0});
  Array3D<float> expected({{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}},
                           {{1.1f, 2.1f, 3.1f}, {6.1f, 3.5f, 2.8f}}});
  Array3D<float> result_on_host(2, 2, 3);

  // Row-major.
  TestAsyncOutSendRoundTrip(
      LiteralUtil::CreateR3FromArray3DWithLayout(expected, r3_dim0major),
      result_on_host.data());
  LiteralTestUtil::ExpectR3EqualArray3D<float>(
      expected,
      LiteralUtil::CreateR3FromArray3DWithLayout(result_on_host, r3_dim0major));

  // Column-major. We test it but in practice we do not use it.
  Array3D<float> expected_reshaped({{{1.0f, 1.1f, 4.0f}, {6.1f, 2.0f, 2.1f}},
                                    {{5.0f, 3.5f, 3.0f}, {3.1f, 6.0f, 2.8f}}});
  TestAsyncOutSendRoundTrip(
      LiteralUtil::CreateR3FromArray3DWithLayout(expected, r3_dim0minor),
      result_on_host.data());
  LiteralTestUtil::ExpectR3EqualArray3D<float>(
      expected_reshaped, LiteralUtil::CreateR3FromArray3D(result_on_host));
}

TEST_F(AsyncOutSendTest, SingleAsyncOutSendR4S32) {
  Array4D<int32> expected(
      {{{{1, -2}, {-4, 5}, {6, 7}}, {{8, 9}, {10, 11}, {12, 13}}},
       {{{10, 3}, {7, -2}, {3, 6}}, {{2, 5}, {-11, 5}, {-2, -5}}}});
  Array4D<int32> result_on_host(2, 2, 3, 2);
  TestAsyncOutSendRoundTrip(LiteralUtil::CreateR4FromArray4D(expected),
                            result_on_host.data());
  LiteralTestUtil::ExpectR4EqualArray4D<int32>(
      expected, LiteralUtil::CreateR4FromArray4D(result_on_host));
}
}  // namespace
}  // namespace xla
