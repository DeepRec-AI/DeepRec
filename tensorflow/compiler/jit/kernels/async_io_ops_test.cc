/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "tensorflow/compiler/jit/kernels/async_io_rendezvous.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

const string kTestTensorName = "__a_uniq_tensor_name";
const char* kTestDeviceName = "/job:worker/replica:0/task:0/device:GPU:0";

class AsyncOutOpTest : public OpsTestBase {};

TEST_F(AsyncOutOpTest, Recv) {
  SetDevice(DEVICE_GPU,
            std::unique_ptr<tensorflow::Device>(
                DeviceFactory::NewDevice("GPU", {}, kTestDeviceName)));

  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 1, 6, 2}));
  test::FillValues<float>(
      &expected, {-1.0f, -2.0f, -3.0f, -5.0f, -8.0f, -13.0f, -21.0f, -34.0f,
                  -55.0f, -89.0f, -143.0f, -231.0f});
  string key =
      AsyncIoRendezvous::GetRendezvousKey(kTestDeviceName, kTestTensorName);
  uint64 key_hash = AsyncIoRendezvous::GetRendezvousKeyHash(key);
  GetXlaAsyncIORendezvous()->InitializeRendezvousQueue(key_hash);
  AsyncIoRendezvous::TensorPayload val;
  val.tensor = expected;
  TF_ASSERT_OK(GetXlaAsyncIORendezvous()->Send(key_hash, val));

  TF_EXPECT_OK(NodeDefBuilder("async_out_recv", "_XlaAsyncOutRecv")
                   .Device(kTestDeviceName)
                   .Attr("T", DT_FLOAT)
                   .Attr("device_name", kTestDeviceName)
                   .Attr("tensor_name", kTestTensorName)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOpWithGraphVersion(8));
  TF_ASSERT_OK(RunOpKernel());
  GetXlaAsyncIORendezvous()->FinalizeRendezvousQueue(key_hash);

  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(AsyncOutOpTest, Send) {
  std::vector<float> golden{-1.0f, -2.0f, -3.0f, -5.0f, -8.0f, -13.0f};
  SetDevice(DEVICE_GPU,
            std::unique_ptr<tensorflow::Device>(
                DeviceFactory::NewDevice("GPU", {}, kTestDeviceName)));

  string key =
      AsyncIoRendezvous::GetRendezvousKey(kTestDeviceName, kTestTensorName);
  uint64 key_hash = AsyncIoRendezvous::GetRendezvousKeyHash(key);
  GetXlaAsyncIORendezvous()->InitializeRendezvousQueue(key_hash);

  TF_EXPECT_OK(NodeDefBuilder("async_out_send", "_XlaAsyncOutSend")
                   .Input(FakeInput(DT_FLOAT))
                   .Device(kTestDeviceName)
                   .Attr("device_name", kTestDeviceName)
                   .Attr("tensor_name", kTestTensorName)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOpWithGraphVersion(8));
  AddInputFromArray<float>(TensorShape{3, 1, 2}, golden);
  TF_ASSERT_OK(RunOpKernel());

  AsyncIoRendezvous::TensorPayload recv_val;
  GetXlaAsyncIORendezvous()->RecvAsync(
      key_hash,
      [&recv_val](const Status& s,
                  const tensorflow::AsyncIoRendezvous::TensorPayload& val) {
        TF_ASSERT_OK(s);
        recv_val = val;
      });
  GetXlaAsyncIORendezvous()->FinalizeRendezvousQueue(key_hash);

  Tensor expected(allocator(), DT_FLOAT, TensorShape({3, 1, 2}));
  test::FillValues<float>(&expected, golden);
  test::ExpectTensorEqual<float>(expected, recv_val.tensor);
}

}  // namespace
}  // namespace tensorflow
