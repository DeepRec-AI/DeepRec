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

  Status FlowControlSend(const StringPiece& tag, const ParsedKey& key,
                         const Args& args, const Tensor& val,
                         const bool is_dead) {
    return Send(key, args, val, is_dead);
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

  void FlowControlRecvAsync(const StringPiece& tag, const ParsedKey& parsed_key,
                            const Args& args, DoneCallback done) {
    RecvAsync(parsed_key, args, done);
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

Node* SliceRecv(Graph* g, const string& tensor, const string& type,
                const string& sender, const uint64 sender_incarnation,
                const string& receiver, const int32 slice_size,
                const int64 timeout_ms) {
  Node* ret;
  DataType dtype;
  CHECK(DataTypeFromString(type, &dtype));
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "_SliceRecv")
              .Attr("tensor_type", dtype)
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

Node* Equal(Graph* g, Node* x, Node* y) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Equal")
              .Input(x)
              .Input(y)
              .Finalize(g, &ret));
  return ret;
}

Node* ReduceAll(Graph* g, Node* input, Node* axes) {
  return test::graph::Reduce(g, "All", input, axes);
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

static Graph* TransferStringTensor() {
  Graph* g = new Graph(OpRegistry::Global());
  const int32 slice_size = 1024;
  const int64 timeout_ms = 5000;
  std::string str = "The quick brown fox jumps over the lazy dog."; // 44 chars.

  Tensor input_t(DT_STRING, TensorShape({2, 4}));
  input_t.flat<tstring>().setConstant(str); // total bytes: 44*8=352 bytes.
  Node* input_n = test::graph::Constant(g, input_t);
  SliceSend(g, input_n, "T", "/cpu:0", 1, "/cpu:0", slice_size);
  Node* recv_n = \
    SliceRecv(g, "T", "string", "/cpu:0", 1, "/cpu:0", slice_size, timeout_ms);

  Node* equal_n = Equal(g, input_n, recv_n);

  Tensor axes_t(DT_INT32, TensorShape({input_t.dims()}));
  auto axes_flat = axes_t.flat<int32>();
  for (int i = 0; i < input_t.dims(); i++) {
    axes_flat(i) = i;
  }
  Node* reduce_all_n = ReduceAll(g, equal_n, test::graph::Constant(g, axes_t));

  std::vector<NodeBuilder::NodeOut> data_out;
  data_out.emplace_back(input_n, 0);
  data_out.emplace_back(recv_n, 0);
  Assert(g, reduce_all_n, data_out);

  return g;
}

static Graph* TransferBasicTypeTensor() {
  Graph* g = new Graph(OpRegistry::Global());
  const int32 slice_size = 1024;
  const int64 timeout_ms = 5000;

  Tensor input_t(DT_FLOAT, TensorShape({2, 8}));
  input_t.flat<float>().setConstant(2); // total bytes = 4*2*8=64 bytes.
  Node* input_n = test::graph::Constant(g, input_t);
  SliceSend(g, input_n, "T", "/cpu:0", 1, "/cpu:0", slice_size);
  Node* recv_n = \
    SliceRecv(g, "T", "float32", "/cpu:0", 1, "/cpu:0", slice_size, timeout_ms);

  Node* equal_n = Equal(g, input_n, recv_n);

  Tensor axes_t(DT_INT32, TensorShape({input_t.dims()}));
  auto axes_flat = axes_t.flat<int32>();
  for (int i = 0; i < input_t.dims(); i++) {
    axes_flat(i) = i;
  }
  Node* reduce_all_n = ReduceAll(g, equal_n, test::graph::Constant(g, axes_t));

  std::vector<NodeBuilder::NodeOut> data_out;
  data_out.emplace_back(input_n, 0);
  data_out.emplace_back(recv_n, 0);
  Assert(g, reduce_all_n, data_out);

  return g;
}

static Graph* TransferBigStringTensor() {
  Graph* g = new Graph(OpRegistry::Global());
  const int32 slice_size = 16;
  const int64 timeout_ms = 5000;
  std::string str = "The quick brown fox jumps over the lazy dog."; // 44 chars.

  Tensor input_t(DT_STRING, TensorShape({2, 4}));
  input_t.flat<tstring>().setConstant(str);
  input_t.flat<tstring>()(0) = "short str";
  Node* input_n = \
    test::graph::Constant(g, input_t); // total bytes: 44*7+9=317 bytes.
  SliceSend(g, input_n, "T", "/cpu:0", 1, "/cpu:0", slice_size);
  Node* recv_n = \
    SliceRecv(g, "T", "string", "/cpu:0", 1, "/cpu:0", slice_size, timeout_ms);

  Node* equal_n = Equal(g, input_n, recv_n);

  Tensor axes_t(DT_INT32, TensorShape({input_t.dims()}));
  auto axes_flat = axes_t.flat<int32>();
  for (int i = 0; i < input_t.dims(); i++) {
    axes_flat(i) = i;
  }
  Node* reduce_all_n = ReduceAll(g, equal_n, test::graph::Constant(g, axes_t));

  std::vector<NodeBuilder::NodeOut> data_out;
  data_out.emplace_back(input_n, 0);
  data_out.emplace_back(recv_n, 0);
  Assert(g, reduce_all_n, data_out);

  return g;
}

static Graph* TransferBigBasicTypeTensor() {
  Graph* g = new Graph(OpRegistry::Global());
  const int32 slice_size = 16;
  const int64 timeout_ms = 5000;

  Tensor input_t(DT_FLOAT, TensorShape({2, 8}));
  input_t.flat<float>().setConstant(2); // total bytes: 4*2*8=64
  Node* input_n = test::graph::Constant(g, input_t);
  SliceSend(g, input_n, "T", "/cpu:0", 1, "/cpu:0", slice_size);
  Node* recv_n = \
    SliceRecv(g, "T", "float32", "/cpu:0", 1, "/cpu:0", slice_size, timeout_ms);

  Node* equal_n = Equal(g, input_n, recv_n);

  Tensor axes_t(DT_INT32, TensorShape({input_t.dims()}));
  auto axes_flat = axes_t.flat<int32>();
  for (int i = 0; i < input_t.dims(); i++) {
    axes_flat(i) = i;
  }
  Node* reduce_all_n = ReduceAll(g, equal_n, test::graph::Constant(g, axes_t));

  std::vector<NodeBuilder::NodeOut> data_out;
  data_out.emplace_back(input_n, 0);
  data_out.emplace_back(recv_n, 0);
  Assert(g, reduce_all_n, data_out);

  return g;
}

static Graph* TransferDeadTensor() {
  Graph* g = new Graph(OpRegistry::Global());
  const int32 slice_size = 1024;
  const int64 timeout_ms = 5000;

  // val
  Tensor val_t(DT_FLOAT, TensorShape({}));
  val_t.scalar<float>()() = 2;
  Node* val_n = test::graph::Constant(g, val_t);

  Tensor pred_t(DT_BOOL, TensorShape({}));
  pred_t.scalar<bool>()() = true;
  Node* pred_n = test::graph::Constant(g, pred_t);

  Node* switch_n = test::graph::Switch(g, val_n, pred_n);
  SliceSend(g, switch_n, "T", "/cpu:0", 1, "/cpu:0", slice_size);
  SliceRecv(g, "T", "float32", "/cpu:0", 1, "/cpu:0", slice_size, timeout_ms);

  return g;
}

static void BM_TransferStringTensor(int iters) {
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters));
  test::Benchmark("cpu", TransferStringTensor(), nullptr, nullptr,
                  new DummyRendezvous).Run(iters);
}

static void BM_TransferBasicTypeTensor(int iters) {
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters));
  test::Benchmark("cpu", TransferBasicTypeTensor(), nullptr, nullptr,
                  new DummyRendezvous).Run(iters);
}

static void BM_TransferBigStringTensor(int iters) {
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters));
  test::Benchmark("cpu", TransferBigStringTensor(), nullptr, nullptr,
                  new DummyRendezvous).Run(iters);
}

static void BM_TransferBigBasicTypeTensor(int iters) {
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters));
  test::Benchmark("cpu", TransferBigBasicTypeTensor(), nullptr, nullptr,
                  new DummyRendezvous).Run(iters);
}

static void BM_TransferDeadTensor(int iters) {
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters));
  test::Benchmark("cpu", TransferDeadTensor(), nullptr, nullptr,
                  new DummyRendezvous).Run(iters);
}

BENCHMARK(BM_TransferStringTensor);
BENCHMARK(BM_TransferBasicTypeTensor);
BENCHMARK(BM_TransferBigStringTensor);
BENCHMARK(BM_TransferBigBasicTypeTensor);
BENCHMARK(BM_TransferDeadTensor);

} // End of anonymous namespace

} // End of namespace tensorflow
