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
static Graph* LRN(const string& kind, int DR, const TensorShape& in_shape, 
		  float BIAS = 1.0f, float ALPHA = 0.1f,float BETA = 2.0f) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "LRN" : "_MklLRN";

  Tensor in0(type, in_shape);
  in0.flat<T>().setRandom();

  Node* input_in0 = test::graph::Constant(g, in0);

  Node* not_mkl_shape =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(input_in0)
                    .Attr("depth_radius", DR)
                    .Attr("bias", BIAS)
                    .Attr("alpha", ALPHA)
                    .Attr("beta", BETA);

  isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
                                       .Attr("_kernel", "MklLayoutDependentOp");

  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_LRN_Base(kind, DR, in_shape, name, T, DEVICE, NTH)                   \
  static void BM_LRN##_##kind##_##DR##name##_##T##_##DEVICE##_##NTH(            \
      int iters) {                                                              \
    int64 num_elements = in_shape.num_elements();                               \
    testing::UseRealTime();                                                     \
    testing::ItemsProcessed(static_cast<int64>(iters) * num_elements * DR * 4); \
    SessionOptions opts;                                                        \
    opts.config.set_intra_op_parallelism_threads(NTH);                          \
    test::Benchmark(#DEVICE, LRN<T>(#kind, DR, in_shape), &opts).Run(iters);    \
  }                                                                             \
  BENCHMARK(BM_LRN##_##kind##_##DR##name##_##T##_##DEVICE##_##NTH);             \

#define BM_LRN_kind(DR, in_shape, name, T, DEVICE, NTH)     \
  BM_LRN_Base(Default, DR, in_shape, name, T, DEVICE, NTH); \
  BM_LRN_Base(Mkl, DR, in_shape, name, T, DEVICE, NTH);     \

#define BM_LRN_NTH(DR, in_shape, name, T, DEVICE) \
  BM_LRN_kind(DR, in_shape, name, T, DEVICE, 1);  \
  BM_LRN_kind(DR, in_shape, name, T, DEVICE, 4);  \
  BM_LRN_kind(DR, in_shape, name, T, DEVICE, 8);  \

#define BM_LRN_DT(DR, in_shape, name)         \
  BM_LRN_NTH(DR, in_shape, name, float, cpu); \

#define BM_LRN(name, DR, ...)                      \
  BM_LRN_DT(DR, TensorShape({__VA_ARGS__}), name); \

BM_LRN(_128x12x12x64, 4, 128, 12, 12, 64);
BM_LRN(_128x56x56x64, 2, 128, 56, 56, 64);
BM_LRN(_128x27x27x192, 2, 128, 27, 27, 192);

template <typename T>
static Graph* LRNGrad(const string& kind, int DR, const TensorShape& in_shape,
		      float BIAS = 1.0f, float ALPHA = 0.1f, float BETA = 2.0f) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "LRNGrad" : "_MklLRNGrad";

  Tensor inGrad(type, in_shape);
  inGrad.flat<T>().setRandom();

  Tensor in0(type, in_shape);
  in0.flat<T>().setRandom();

  Tensor out(DT_FLOAT, in_shape);

  Node* input_inGrad = test::graph::Constant(g, inGrad);
  Node* input_in0 = test::graph::Constant(g, in0);
  Node* output = test::graph::Constant(g, out);

  Node* not_mkl_shape =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(input_inGrad)
                    .Input(input_in0)
                    .Input(output)
                    .Attr("depth_radius", DR)
                    .Attr("bias", BIAS)
                    .Attr("alpha", ALPHA)
                    .Attr("beta", BETA);

  isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
                                       .Input(not_mkl_shape)
                                       .Input(not_mkl_shape)
                                       .Input(not_mkl_shape)
                                       .Input(not_mkl_shape)
                                       .Attr("_kernel", "MklLayoutDependentOp");

  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_LRNGrad_Base(kind, DR, in_shape, name, T, DEVICE, NTH)                \
  static void BM_LRNGrad##_##kind##_##DR##name##_##T##_##DEVICE##_##NTH(         \
      int iters) {                                                               \
    int64 num_elements = in_shape.num_elements();                                \
    testing::UseRealTime();                                                      \
    testing::ItemsProcessed(static_cast<int64>(iters) * num_elements * DR * 4);  \
    SessionOptions opts;                                                         \
    opts.config.set_intra_op_parallelism_threads(NTH);                           \
    test::Benchmark(#DEVICE, LRNGrad<T>(#kind, DR, in_shape), &opts).Run(iters); \
  }                                                                              \
  BENCHMARK(BM_LRNGrad##_##kind##_##DR##name##_##T##_##DEVICE##_##NTH);          \

#define BM_LRNGrad_kind(DR, in_shape, name, T, DEVICE, NTH)     \
  BM_LRNGrad_Base(Default, DR, in_shape, name, T, DEVICE, NTH); \
  BM_LRNGrad_Base(Mkl, DR, in_shape, name, T, DEVICE, NTH);     \

#define BM_LRNGrad_NTH(DR, in_shape, name, T, DEVICE) \
  BM_LRNGrad_kind(DR, in_shape, name, T, DEVICE, 1);  \
  BM_LRNGrad_kind(DR, in_shape, name, T, DEVICE, 4);  \
  BM_LRNGrad_kind(DR, in_shape, name, T, DEVICE, 8);  \

#define BM_LRNGrad_DT(DR, in_shape, name)         \
  BM_LRNGrad_NTH(DR, in_shape, name, float, cpu); \

#define BM_LRNGrad(name, DR, ...)                      \
  BM_LRNGrad_DT(DR, TensorShape({__VA_ARGS__}), name); \

BM_LRNGrad(_128x12x12x64, 4, 128, 12, 12, 64);
BM_LRNGrad(_128x56x56x64, 2, 128, 56, 56, 64);
BM_LRNGrad(_128x27x27x192, 2, 128, 27, 27, 192);

}  // end namespace tensorflow

#endif  // INTEL_MKL
