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

#ifdef INTEL_MKL

#include "dnnl.hpp"
#include "absl/algorithm/container.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* Matmul(const string& kind, int m, int k, int n, bool transpose_a, bool transpose_b) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "MatMul" : "_MklMatMul";

  Tensor in0(type, transpose_a ? TensorShape({k, m}) : TensorShape({m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, transpose_b ? TensorShape({n, k}) : TensorShape({k, n}));
  in1.flat<T>().setRandom();

  Node* input_in0 = test::graph::Constant(g, in0);
  Node* input_in1 = test::graph::Constant(g, in1);

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(input_in0)
                    .Input(input_in1)
                    .Attr("transpose_a", transpose_a)
                    .Attr("transpose_b", transpose_b);

  isDefault ? nodeBuilder : nodeBuilder.Attr("_kernel", "MklNameChangeOp");

  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_Matmul_Base(kind, M, K, N, TA, TB, T, DEVICE, NTH)                              \
  static void BM_Matmul##_##kind##_##M##_##K##_##N##_##TA##_##TB##_##T##_##DEVICE##_##NTH( \
      int iters) {                                                                         \
    testing::UseRealTime();                                                                \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2);                    \
    SessionOptions opts;                                                                   \
    opts.config.set_intra_op_parallelism_threads(NTH);                                     \
    test::Benchmark(#DEVICE, Matmul<T>(#kind, M, K, N, TA, TB), &opts).Run(iters);         \
  }                                                                                        \
  BENCHMARK(BM_Matmul##_##kind##_##M##_##K##_##N##_##TA##_##TB##_##T##_##DEVICE##_##NTH);  \

#define BM_Matmul_kind(M, K, N, TA, TB, T, DEVICE, NTH)     \
  BM_Matmul_Base(Default, M, K, N, TA, TB, T, DEVICE, NTH); \
  BM_Matmul_Base(Mkl, M, K, N, TA, TB, T, DEVICE, NTH);     \

#define BM_Matmul_NTH(M, K, N, TA, TB, T, DEVICE) \
  BM_Matmul_kind(M, K, N, TA, TB, T, DEVICE, 1);  \
  BM_Matmul_kind(M, K, N, TA, TB, T, DEVICE, 4);  \
  BM_Matmul_kind(M, K, N, TA, TB, T, DEVICE, 8);  \

#define BM_Matmul(M, K, N, TA, TB)               \
  BM_Matmul_NTH(M, K, N, TA, TB, float, cpu);    \
  BM_Matmul_NTH(M, K, N, TA, TB, bfloat16, cpu); \

/*
// Batch size of 1 included for inference.
// Typical fully connected layers
BM_Matmul(1, 512, 512, false, false);
BM_Matmul(8, 512, 512, false, false);
BM_Matmul(16, 512, 512, false, false);
BM_Matmul(128, 512, 512, false, false);

BM_Matmul(1, 1024, 1024, false, false);
BM_Matmul(8, 1024, 1024, false, false);
BM_Matmul(16, 1024, 1024, false, false);
BM_Matmul(128, 1024, 1024, false, false);
BM_Matmul(4096, 4096, 4096, false, false);

// Backward for fully connected layers
BM_Matmul(1, 1024, 1024, false, true);
BM_Matmul(8, 1024, 1024, false, true);
BM_Matmul(16, 1024, 1024, false, true);
BM_Matmul(128, 1024, 1024, false, true);

// Forward softmax with large output size
BM_Matmul(1, 200, 10000, false, false);
BM_Matmul(8, 200, 10000, false, false);
BM_Matmul(20, 200, 10000, false, false);
BM_Matmul(20, 200, 20000, false, false);

// Backward softmax with large output size
BM_Matmul(1, 10000, 200, false, true);
BM_Matmul(1, 10000, 200, false, false);
BM_Matmul(8, 10000, 200, false, true);
BM_Matmul(20, 10000, 200, false, true);
BM_Matmul(20, 20000, 200, false, true);

// Test some matrix-vector multiplies.
BM_Matmul(50, 50, 1, false, false);
BM_Matmul(50, 50, 1, true, false);
BM_Matmul(50, 50, 1, false, true);
BM_Matmul(50, 50, 1, true, true);
BM_Matmul(500, 500, 1, false, false);
BM_Matmul(500, 500, 1, true, false);
BM_Matmul(500, 500, 1, false, true);
BM_Matmul(500, 500, 1, true, true);
BM_Matmul(2000, 2000, 1, false, false);
BM_Matmul(2000, 2000, 1, true, false);
BM_Matmul(2000, 2000, 1, false, true);
BM_Matmul(2000, 2000, 1, true, true);

// Test some vector-matrix multiplies.
BM_Matmul(1, 50, 50, false, false);
BM_Matmul(1, 50, 50, true, false);
BM_Matmul(1, 50, 50, false, true);
BM_Matmul(1, 50, 50, true, true);
BM_Matmul(1, 500, 500, false, false);
BM_Matmul(1, 500, 500, true, false);
BM_Matmul(1, 500, 500, false, true);
BM_Matmul(1, 500, 500, true, true);
BM_Matmul(1, 2000, 2000, false, false);
BM_Matmul(1, 2000, 2000, true, false);
BM_Matmul(1, 2000, 2000, false, true);
BM_Matmul(1, 2000, 2000, true, true);

// Test some rank-one products.
BM_Matmul(50, 1, 50, false, false);
BM_Matmul(50, 1, 50, true, false);
BM_Matmul(50, 1, 50, false, true);
BM_Matmul(50, 1, 50, true, true);
BM_Matmul(500, 1, 500, false, false);
BM_Matmul(500, 1, 500, true, false);
BM_Matmul(500, 1, 500, false, true);
BM_Matmul(500, 1, 500, true, true);
BM_Matmul(2000, 1, 2000, false, false);
BM_Matmul(2000, 1, 2000, true, false);
BM_Matmul(2000, 1, 2000, false, true);
BM_Matmul(2000, 1, 2000, true, true);
*/

// Vector * Vector
BM_Matmul(1, 50, 1, false, false);
BM_Matmul(1, 2000, 1, false, false);

BM_Matmul(50, 1, 50, false, false);
BM_Matmul(2000, 1, 2000, false, false);

// Vector * Matrix
BM_Matmul(1, 50, 50, false, false);
BM_Matmul(1, 2000, 2000, false, false);

BM_Matmul(50, 50, 1, false, false);
BM_Matmul(2000, 2000, 1, false, false);

// Matrix * Matrix
BM_Matmul(32, 32, 32, false, false);
BM_Matmul(51200, 64, 64, false, false);
BM_Matmul(8, 512, 512, false, false);
BM_Matmul(128, 512, 512, false, false);
BM_Matmul(16, 1024, 1024, false, false);
BM_Matmul(256, 1024, 1024, false, false);
BM_Matmul(4096, 4096, 4096, false, false);

BM_Matmul(2560, 64, 1, false, false);
BM_Matmul(2560, 448, 1, false, false);
BM_Matmul(2560, 2304, 64, false, false);
BM_Matmul(2560, 1040, 1536, false, false);
BM_Matmul(2560, 14435, 2304, false, false);

/*
BM_Matmul(14435, 2560, 2304, true, false);
BM_Matmul(2560, 2304, 14435, false, true);

BM_Matmul(64, 2560, 1, true, false);
BM_Matmul(2560, 1, 64, false, true);

BM_Matmul(448, 2560, 1, true, false);
BM_Matmul(2560, 1, 448, false, true);

BM_Matmul(2304, 2560, 64, true, false);
BM_Matmul(2560, 64, 2304, false, true);

BM_Matmul(1040, 2560, 1536, true, false);
BM_Matmul(2560, 1536, 1040, false, true);
*/

template <typename T>
static Graph* FusedMatMul(const string& kind, int m, int k, int n,
                          bool transpose_a, bool transpose_b, const string& activation = "") {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  std::vector<string> fused_ops{"BiasAdd"};

  if(activation != "" && activation != "null"){
    fused_ops.push_back(activation);
  }

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "_FusedMatMul" : "_MklFusedMatMul";

  int num_args = 1;
  Tensor in0(type, transpose_a ? TensorShape({k, m}) : TensorShape({m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, transpose_b ? TensorShape({n, k}) : TensorShape({k, n}));
  in1.flat<T>().setRandom();
  Tensor bias(type, TensorShape({transpose_b ? k : n}));
  bias.flat<T>().setRandom();

  Node* input_in0 = test::graph::Constant(g, in0);
  Node* input_in1 = test::graph::Constant(g, in1);
  Node* input_bias = test::graph::Constant(g, bias, absl::StrCat("arg", 1));

  Node* not_mkl_shape =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

  std::vector<NodeBuilder::NodeOut> args;
  std::vector<NodeBuilder::NodeOut> args_not_mkl;
  args.push_back(input_bias);
  args_not_mkl.push_back(not_mkl_shape);

  auto nodeBuilder = NodeBuilder(g->NewName("fused_matmul"), op_name)
                    .Input(input_in0)
                    .Input(input_in1)
                    .Input(args)
                    .Attr("T", type)
                    .Attr("num_args", num_args)
                    .Attr("fused_ops", fused_ops)
                    .Attr("transpose_a", transpose_a)
                    .Attr("transpose_b", transpose_b);

  isDefault ? nodeBuilder : nodeBuilder.Attr("_kernel", "MklLayoutDependentOp")
                                       .Input(not_mkl_shape)
                                       .Input(not_mkl_shape)
                                       .Input(args_not_mkl);

  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_FusedMatMul_Base(kind, ACT, M, K, N, TA, TB, T, DEVICE, NTH)                                 \
  static void BM_FusedMatMul##_##kind##_##ACT##_##M##_##K##_##N##_##TA##_##TB##_##T##_##DEVICE##_##NTH( \
      int iters) {                                                                                      \
    testing::UseRealTime();                                                                             \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2);                                 \
    SessionOptions opts;                                                                                \
    opts.config.set_intra_op_parallelism_threads(NTH);                                                  \
    test::Benchmark(#DEVICE, FusedMatMul<T>(#kind, M, K, N, TA, TB, #ACT), &opts).Run(iters);           \
  }                                                                                                     \
  BENCHMARK(BM_FusedMatMul##_##kind##_##ACT##_##M##_##K##_##N##_##TA##_##TB##_##T##_##DEVICE##_##NTH);  \

#define BM_FusedMatMul_kind(ACT, M, K, N, TA, TB, T, DEVICE, NTH)     \
  BM_FusedMatMul_Base(Default, ACT, M, K, N, TA, TB, T, DEVICE, NTH); \
  BM_FusedMatMul_Base(Mkl, ACT, M, K, N, TA, TB, T, DEVICE, NTH);     \

#define BM_FusedMatMul_NTH(ACT, M, K, N, TA, TB, T, DEVICE) \
  BM_FusedMatMul_kind(ACT, M, K, N, TA, TB, T, DEVICE, 1);  \
  BM_FusedMatMul_kind(ACT, M, K, N, TA, TB, T, DEVICE, 4);  \
  BM_FusedMatMul_kind(ACT, M, K, N, TA, TB, T, DEVICE, 8);  \

#define BM_FusedMatMul_ACT(M, K, N, TA, TB, T, DEVICE)  \
  BM_FusedMatMul_NTH(null, M, K, N, TA, TB, T, DEVICE); \
  BM_FusedMatMul_NTH(Relu, M, K, N, TA, TB, T, DEVICE); \

#define BM_FusedMatMul(M, K, N, TA, TB)               \
  BM_FusedMatMul_ACT(M, K, N, TA, TB, float, cpu);    \
  BM_FusedMatMul_ACT(M, K, N, TA, TB, bfloat16, cpu); \

// Vector * Vector
BM_FusedMatMul(1, 50, 1, false, false);
BM_FusedMatMul(1, 2000, 1, false, false);

BM_FusedMatMul(50, 1, 50, false, false);
BM_FusedMatMul(2000, 1, 2000, false, false);

// Vector * Matrix
BM_FusedMatMul(1, 50, 50, false, false);
BM_FusedMatMul(1, 2000, 2000, false, false);

BM_FusedMatMul(50, 50, 1, false, false);
BM_FusedMatMul(2000, 2000, 1, false, false);

// Matrix * Matrix
BM_FusedMatMul(32, 32, 32, false, false);
BM_FusedMatMul(51200, 64, 64, false, false);
BM_FusedMatMul(8, 512, 512, false, false);
BM_FusedMatMul(128, 512, 512, false, false);
BM_FusedMatMul(16, 1024, 1024, false, false);
BM_FusedMatMul(256, 1024, 1024, false, false);
BM_FusedMatMul(4096, 4096, 4096, false, false);

BM_FusedMatMul(2560, 64, 1, false, false);
BM_FusedMatMul(2560, 448, 1, false, false);
BM_FusedMatMul(2560, 2304, 64, false, false);
BM_FusedMatMul(2560, 1040, 1536, false, false);
BM_FusedMatMul(2560, 14435, 2304, false, false);

}  // end namespace tensorflow

#endif  // INTEL_MKL
