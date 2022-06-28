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

#include <vector>
#include "dnnl.hpp"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

template <typename T>
static Graph* Identity(const string& kind, const TensorShape& in_shape) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "Identity" : "_MklIdentity";

  Tensor input(type, in_shape);
  input.flat<T>().setRandom();

  Node* not_mkl_shape =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(test::graph::Constant(g, input))
                    .Attr("T", type);

  isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
                                       .Attr("_kernel", "MklLayoutDependentOp");
  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));
  return g;
}

#define BM_Identity_Base(kind, T, name, in_shape, DEVICE, NTH)                \
  static void BM_Identity##_##kind##_##T##name##_##DEVICE##_##NTH(            \
      int iters) {                                                            \
    int64 num_elements = in_shape.num_elements();  	                          \
    testing::UseRealTime();                                                   \
    testing::ItemsProcessed(static_cast<int64>(iters) * num_elements);        \
    SessionOptions opts;                                                      \
    opts.config.set_intra_op_parallelism_threads(NTH);                        \
    test::Benchmark(#DEVICE, Identity<T>(#kind, in_shape), &opts).Run(iters); \
  }                                                                           \
  BENCHMARK(BM_Identity##_##kind##_##T##name##_##DEVICE##_##NTH);             \

#define BM_Identity_kind(T, name, in_shape, DEVICE, NTH)     \
  BM_Identity_Base(Default, T, name, in_shape, DEVICE, NTH); \
  BM_Identity_Base(Mkl, T, name, in_shape, DEVICE, NTH);     \

#define BM_Identity_NTH(T, name, in_shape, DEVICE) \
  BM_Identity_kind(T, name, in_shape, DEVICE, 1);  \
  BM_Identity_kind(T, name, in_shape, DEVICE, 4);  \
  BM_Identity_kind(T, name, in_shape, DEVICE, 8);  \

#define BM_Identity_DT(name, in_shape)            \
  BM_Identity_NTH(float, name, in_shape, cpu);    \
  BM_Identity_NTH(bfloat16, name, in_shape, cpu); \

#define BM_IdentityND(name, ...)                    \
  BM_Identity_DT(name, TensorShape({__VA_ARGS__})); \

BM_IdentityND(_2D_1024x1024, 1024, 1024);
BM_IdentityND(_3D_1024x1024x1024, 1024, 1024, 1024);
BM_IdentityND(_4D_1024x1024x1024x1024, 1024, 1024, 1024, 1024);

}  // namespace tensorflow

#endif  // INTEL_MKL
