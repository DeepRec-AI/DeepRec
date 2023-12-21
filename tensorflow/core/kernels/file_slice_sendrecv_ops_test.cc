/* Copyright 2023 The DeepRec Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace {
// Implement a trivial version of the Rendezvous interface, to avoid
// clouding the benchmark results with the time spent in the various
// implementations, and to avoid the duplicate-send or duplicate-recv
// errors that would arise from running either benchmark in a loop.
class DummyRendezvous : public Rendezvous {
  // Functions.
  Status Send(const ParsedKey& key, const Args& args, const Tensor& val,
              const bool is_dead) override {
    std::string key_str = { key.FullKey().data(), key.FullKey().size() };
    mutex_lock l(mu_);
    // consumer does not reach.
    if (kv_.count(key_str) == 0) {
      struct Var var;
      var.type = send;
      var.args = args;
      var.data = val;
      var.is_dead = is_dead;

      kv_[key_str] = var;
      return Status::OK();
    }

    auto var = kv_[key_str];
    CHECK_EQ(var.type, recv);
    var.done(Status::OK(), args, var.args, val, is_dead);
    kv_.erase(key_str);
    return Status::OK();
  }
  void RecvAsync(const ParsedKey& key, const Args& args,
                 DoneCallback done) override {
    std::string key_str = { key.FullKey().data(), key.FullKey().size() };

    mutex_lock l(mu_);
    // producer does not reach.
    if (kv_.count(key_str) == 0) {
      struct Var var;
      var.type = recv;
      var.args = args;
      var.done = done;

      kv_[key_str] = var;
      return;
    }

    // auto var = kv_[key_str];
    auto var =  kv_[key_str];
    CHECK_EQ(var.type, send);
    done(Status::OK(), var.args, args, var.data, var.is_dead);
    kv_.erase(key_str);
  }
  void StartAbort(const Status& status) override {}

 private:
  enum RendezvousType {
    send,
    recv
  };
  // Type define.
  struct Var {
    RendezvousType type;
    Args args;
    Tensor data;
    bool is_dead;
    DoneCallback done;
  };

  // Variables.
  mutex mu_;
  std::unordered_map<std::string, struct Var> kv_ GUARDED_BY(mu_);
};

//------------------------------------------------------------------------------
// Utils.
Node* FileSliceSend(Graph* g, Node* filename, const string& tensor,
                    const string& sender, const uint64 sender_incarnation,
                    const string& receiver, const int32 slice_size) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("FileSliceSend"), "_FileSliceSend")
              .Input(filename, 0)
              .Attr("tensor_name", tensor)
              .Attr("send_device", sender)
              .Attr("send_device_incarnation",
                    static_cast<int64>(sender_incarnation))
              .Attr("recv_device", receiver)
              .Attr("slice_size", slice_size)
              .Finalize(g, &ret));

  return ret;
}

Node* FileSliceRecv(Graph* g, const string& tensor, const string& sender,
                    const uint64 sender_incarnation, const string& receiver,
                    const string& recv_dir, const int32 slice_size,
                    const int64 timeout_ms) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("FileSliceRecv"), "_FileSliceRecv")
              .Attr("tensor_name", tensor)
              .Attr("send_device", sender)
              .Attr("send_device_incarnation",
                    static_cast<int64>(sender_incarnation))
              .Attr("recv_device", receiver)
              .Attr("recv_dir", recv_dir)
              .Attr("slice_size", slice_size)
              .Attr("timeout_ms", timeout_ms)
              .Finalize(g, &ret));

  return ret;
}

Node* SliceSend(Graph* g, Node* input, const string& tensor,
                const string& sender, const uint64 sender_incarnation,
                const string& receiver, const int32 slice_size) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "_SliceSend")
              .Input(input, 0)
              .Attr("tensor_name", tensor)
              .Attr("send_device", sender)
              .Attr("send_device_incarnation",
                    static_cast<int64>(sender_incarnation))
              .Attr("recv_device", receiver)
              .Attr("slice_size", slice_size)
              .Finalize(g, &ret));
  return ret;
}

Node* SliceRecv(Graph* g, const string& tensor, const string& sender,
                const uint64 sender_incarnation, const string& receiver,
                const int32 slice_size, const int64 timeout_ms) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "_SliceRecv")
              .Attr("tensor_type", DT_STRING)
              .Attr("tensor_name", tensor)
              .Attr("send_device", sender)
              .Attr("send_device_incarnation",
                    static_cast<int64>(sender_incarnation))
              .Attr("recv_device", receiver)
              .Attr("slice_size", slice_size)
              .Attr("timeout_ms", timeout_ms)
              .Finalize(g, &ret));
  return ret;
}

Node* ReadFile(Graph* g, Node* filename) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("ReadFile"), "ReadFile")
              .Input(filename, 0)
              .Finalize(g, &ret));

  return ret;
}

Node* WriteFile(Graph* g, Node* filename, Node* contents) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("WriteFile"), "WriteFile")
              .Input(filename, 0)
              .Input(contents, 0)
              .Finalize(g, &ret));

  return ret;
}

Node* Equal(Graph* g, Node* x, Node* y) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("Equal"), "Equal")
              .Input(x)
              .Input(y)
              .Finalize(g, &ret));
  return ret;
}

Node* Assert(Graph* g, Node* condition,
             std::vector<NodeBuilder::NodeOut>& data) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Assert")
              .Input(condition)
              .Input(data)
              .Finalize(g, &ret));
  return ret;
}

//------------------------------------------------------------------------------
// Graph Constructor.

static Graph* TransferFile(const std::string& test_type,
                           const int32 slice_size) {
  Graph* g = new Graph(OpRegistry::Global());
  const int64 timeout_ms = 5000;
  std::string recv_dir = "/tmp/FileSliceTransferTestRecv";
  std::string filename = "/tmp/FileSliceTransferTestSend/send_" + test_type;
  std::string contents = \
    "The quick brown fox jumps over the lazy dog."; // 44 chars.

  // send filename node.
  Tensor filename_t(DT_STRING, TensorShape({}));
  filename_t.scalar<tstring>().setConstant(filename);
  Node* filename_n = test::graph::Constant(g, filename_t);

  // contents node.
  Tensor contents_t(DT_STRING, TensorShape({}));
  contents_t.scalar<tstring>().setConstant(contents);
  Node* contents_n = test::graph::Constant(g, contents_t);

  Node* write_file_n = WriteFile(g, filename_n, contents_n);
  Node* send_n = \
    FileSliceSend(g, filename_n, test_type, "/cpu:0", 1, "/cpu:0", slice_size);
  g->AddControlEdge(write_file_n, send_n);

  Node* recv_n = FileSliceRecv(g, test_type, "/cpu:0", 1, "/cpu:0", recv_dir,
                               slice_size, timeout_ms);
  Node* read_file_n = ReadFile(g, recv_n);
  Node* equal_n = Equal(g, contents_n, read_file_n);

  std::vector<NodeBuilder::NodeOut> data_out;
  data_out.emplace_back(contents_n, 0);
  data_out.emplace_back(read_file_n, 0);
  Assert(g, equal_n, data_out);

  return g;
}

static Graph* FileSliceSendTransferFileToSliceRecv(const std::string& test_type,
                                                   const int32 slice_size) {
  Graph* g = new Graph(OpRegistry::Global());
  const int64 timeout_ms = 5000;
  std::string filename = "/tmp/FileSliceTransferTestSend/send_" + test_type;
  std::string contents = \
    "The quick brown fox jumps over the lazy dog."; // 44 chars.

  // send filename node.
  Tensor filename_t(DT_STRING, TensorShape({}));
  filename_t.scalar<tstring>().setConstant(filename);
  Node* filename_n = test::graph::Constant(g, filename_t);

  // contents node.
  Tensor contents_t(DT_STRING, TensorShape({}));
  contents_t.scalar<tstring>().setConstant(contents);
  Node* contents_n = test::graph::Constant(g, contents_t);

  Node* write_file_n = WriteFile(g, filename_n, contents_n);
  Node* send_n = \
    FileSliceSend(g, filename_n, test_type, "/cpu:0", 1, "/cpu:0", slice_size);
  g->AddControlEdge(write_file_n, send_n);

  Node* recv_n = \
    SliceRecv(g, test_type, "/cpu:0", 1, "/cpu:0", slice_size, timeout_ms);
  Node* equal_n = Equal(g, contents_n, recv_n);

  std::vector<NodeBuilder::NodeOut> data_out;
  data_out.emplace_back(contents_n, 0);
  data_out.emplace_back(recv_n, 0);
  Assert(g, equal_n, data_out);

  return g;
}

static Graph* SliceSendTransferFileToFileSliceRecv(const std::string& test_type,
                                                   const int32 slice_size) {
  Graph* g = new Graph(OpRegistry::Global());
  const int64 timeout_ms = 5000;
  std::string recv_dir = "/tmp/FileSliceTransferTestRecv";
  std::string contents = \
    "The quick brown fox jumps over the lazy dog."; // 44 chars.

  // contents node.
  Tensor contents_t(DT_STRING, TensorShape({}));
  contents_t.scalar<tstring>().setConstant(contents);
  Node* contents_n = test::graph::Constant(g, contents_t);

  Node* send_n = \
    SliceSend(g, contents_n, test_type, "/cpu:0", 1, "/cpu:0", slice_size);

  Node* recv_n = FileSliceRecv(g, test_type, "/cpu:0", 1, "/cpu:0", recv_dir,
                               slice_size, timeout_ms);
  Node* read_file_n = ReadFile(g, recv_n);
  Node* equal_n = Equal(g, contents_n, read_file_n);

  std::vector<NodeBuilder::NodeOut> data_out;
  data_out.emplace_back(contents_n, 0);
  data_out.emplace_back(read_file_n, 0);
  Assert(g, equal_n, data_out);

  return g;
}

static Graph* TransferDeadTensor() {
  Graph* g = new Graph(OpRegistry::Global());
  const int32 slice_size = 1024;
  const int64 timeout_ms = 5000;
  std::string recv_dir = "/tmp/FileSliceTransferTestRecv";
  std::string filename = "/tmp/FileSliceTransferTestSend/send_dead_tensor";

  // val
  Tensor val_t(DT_STRING, TensorShape({}));
  val_t.scalar<tstring>()() = filename;
  Node* val_n = test::graph::Constant(g, val_t);

  Tensor pred_t(DT_BOOL, TensorShape({}));
  pred_t.scalar<bool>()() = true;
  Node* pred_n = test::graph::Constant(g, pred_t);

  Node* switch_n = test::graph::Switch(g, val_n, pred_n);
  FileSliceSend(g, switch_n, "dead_tensor", "/cpu:0", 1, "/cpu:0", slice_size);
  FileSliceRecv(g, "dead_tensor", "/cpu:0", 1, "/cpu:0", recv_dir, slice_size,
                timeout_ms);

  return g;
}

static Graph* FileSliceSendTransferDeadTensorToSliceRecv() {
  Graph* g = new Graph(OpRegistry::Global());
  const int32 slice_size = 1024;
  const int64 timeout_ms = 5000;
  std::string recv_dir = "/tmp/FileSliceTransferTestRecv";
  std::string filename = "/tmp/FileSliceTransferTestSend/send_dead_tensor";

  // val
  Tensor val_t(DT_STRING, TensorShape({}));
  val_t.scalar<tstring>()() = filename;
  Node* val_n = test::graph::Constant(g, val_t);

  Tensor pred_t(DT_BOOL, TensorShape({}));
  pred_t.scalar<bool>()() = true;
  Node* pred_n = test::graph::Constant(g, pred_t);

  Node* switch_n = test::graph::Switch(g, val_n, pred_n);
  FileSliceSend(g, switch_n, "dead_tensor", "/cpu:0", 1, "/cpu:0", slice_size);
  SliceRecv(g, "dead_tensor", "/cpu:0", 1, "/cpu:0", slice_size, timeout_ms);

  return g;
}

static Graph* SliceSendTransferDeadTensorToFileSliceRecv() {
  Graph* g = new Graph(OpRegistry::Global());
  const int32 slice_size = 1024;
  const int64 timeout_ms = 5000;
  std::string recv_dir = "/tmp/FileSliceTransferTestRecv";
  std::string contents = \
    "The quick brown fox jumps over the lazy dog."; // 44 chars.

  // val
  Tensor val_t(DT_STRING, TensorShape({}));
  val_t.scalar<tstring>()() = contents;
  Node* val_n = test::graph::Constant(g, val_t);

  Tensor pred_t(DT_BOOL, TensorShape({}));
  pred_t.scalar<bool>()() = true;
  Node* pred_n = test::graph::Constant(g, pred_t);

  Node* switch_n = test::graph::Switch(g, val_n, pred_n);
  SliceSend(g, switch_n, "dead_tensor", "/cpu:0", 1, "/cpu:0", slice_size);
  FileSliceRecv(g, "dead_tensor", "/cpu:0", 1, "/cpu:0", recv_dir, slice_size,
                timeout_ms);

  return g;
}

static Graph* TransferSmallFile() {
  return TransferFile("small_file", 1024);
}

static Graph* TransferBigFile() {
  return TransferFile("big_file", 16);
}

static Graph* FileSliceSendTransferSmallFileToSliceRecv() {
  return FileSliceSendTransferFileToSliceRecv("small_file", 1024);
}

static Graph* FileSliceSendTransferBigFileToSliceRecv() {
  return FileSliceSendTransferFileToSliceRecv("big_file", 16);
}

static Graph* SliceSendTransferSmallFileToFileSliceRecv() {
  return SliceSendTransferFileToFileSliceRecv("small_file", 1024);
}

static Graph* SliceSendTransferBigFileToFileSliceRecv() {
  return SliceSendTransferFileToFileSliceRecv("big_file", 16);
}

//------------------------------------------------------------------------------
// Test Function.

static void BM_TransferSmallFile(int iters) {
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters));
  test::Benchmark("cpu", TransferSmallFile(), nullptr, nullptr,
                  new DummyRendezvous).Run(iters);
}

static void BM_TransferBigFile(int iters) {
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters));
  test::Benchmark("cpu", TransferBigFile(), nullptr, nullptr,
                  new DummyRendezvous).Run(iters);
}

static void BM_FileSliceSendTransferSmallFileToSliceRecv(int iters) {
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters));
  test::Benchmark("cpu", FileSliceSendTransferSmallFileToSliceRecv(), nullptr,
                  nullptr, new DummyRendezvous).Run(iters);
}

static void BM_FileSliceSendTransferBigFileToSliceRecv(int iters) {
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters));
  test::Benchmark("cpu", FileSliceSendTransferBigFileToSliceRecv(), nullptr,
                  nullptr, new DummyRendezvous).Run(iters);
}

static void BM_SliceSendTransferSmallFileToFileSliceRecv(int iters) {
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters));
  test::Benchmark("cpu", SliceSendTransferSmallFileToFileSliceRecv(), nullptr,
                  nullptr, new DummyRendezvous).Run(iters);
}

static void BM_SliceSendTransferBigFileToFileSliceRecv(int iters) {
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters));
  test::Benchmark("cpu", SliceSendTransferBigFileToFileSliceRecv(), nullptr,
                  nullptr, new DummyRendezvous).Run(iters);
}

static void BM_TransferDeadTensor(int iters) {
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters));
  test::Benchmark("cpu", TransferDeadTensor(), nullptr, nullptr,
                  new DummyRendezvous).Run(iters);
}

static void BM_FileSliceSendTransferDeadTensorToSliceRecv(int iters) {
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters));
  test::Benchmark("cpu", FileSliceSendTransferDeadTensorToSliceRecv(), nullptr,
                  nullptr, new DummyRendezvous).Run(iters);
}

static void BM_SliceSendTransferDeadTensorToFileSliceRecv(int iters) {
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters));
  test::Benchmark("cpu", SliceSendTransferDeadTensorToFileSliceRecv(), nullptr,
                  nullptr, new DummyRendezvous).Run(iters);
}

BENCHMARK(BM_TransferSmallFile);
BENCHMARK(BM_TransferBigFile);
BENCHMARK(BM_FileSliceSendTransferSmallFileToSliceRecv);
BENCHMARK(BM_FileSliceSendTransferBigFileToSliceRecv);
BENCHMARK(BM_SliceSendTransferSmallFileToFileSliceRecv);
BENCHMARK(BM_SliceSendTransferBigFileToFileSliceRecv);
BENCHMARK(BM_TransferDeadTensor);
BENCHMARK(BM_FileSliceSendTransferDeadTensorToSliceRecv);
BENCHMARK(BM_SliceSendTransferDeadTensorToFileSliceRecv);

} // End of anonymous namespace

} // End of namespace tensorflow
