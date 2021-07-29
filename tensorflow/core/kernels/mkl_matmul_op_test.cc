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

#include "mkldnn.hpp"
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

#define BM_Matmul_NTH(kind, M, K, N, TA, TB, T, DEVICE) \
  BM_Matmul_Base(kind, M, K, N, TA, TB, T, DEVICE, 1);  \
  BM_Matmul_Base(kind, M, K, N, TA, TB, T, DEVICE, 4);  \
  BM_Matmul_Base(kind, M, K, N, TA, TB, T, DEVICE, 8);  \

#define BM_Matmul_kind(M, K, N, TA, TB, T, DEVICE)    \
  BM_Matmul_NTH(Default, M, K, N, TA, TB, T, DEVICE); \
  BM_Matmul_NTH(Mkl, M, K, N, TA, TB, T, DEVICE);     \

#define BM_Matmul(M, K, N, TA, TB)                \
  BM_Matmul_kind(M, K, N, TA, TB, float, cpu);    \
  BM_Matmul_kind(M, K, N, TA, TB, bfloat16, cpu); \

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

}  // end namespace tensorflow

#endif  // INTEL_MKL
