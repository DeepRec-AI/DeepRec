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
static Graph* AddN(const string& kind, int num_inputs, const TensorShape& shape) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "AddN" : "_MklAddN";
  std::vector<NodeBuilder::NodeOut> inputs;
  std::vector<NodeBuilder::NodeOut> inputs_not_mkl;
  inputs.reserve(num_inputs);
  inputs_not_mkl.reserve(num_inputs);

  for (int i = 0; i < num_inputs; ++i) {
    Tensor in(type, shape);
    in.flat<T>().setRandom();
    inputs.push_back(test::graph::Constant(g, in));
    inputs_not_mkl.push_back(test::graph::Constant(g, GetMklMetaTensor(), "not_mkl"));
  }

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(inputs)
                    .Attr("T", type);
  isDefault ? nodeBuilder : nodeBuilder.Input(inputs_not_mkl)
                                       .Attr("_kernel", "MklLayoutDependentOp");
  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));
  return g;
}

#define BM_AddN_Base(kind, T, name, nums, tf_shape, DEVICE, NTH)                          \
  static void BM_AddN##_##kind##_##T##_##name##_##DEVICE##_##NTH(                         \
      int iters) {                                                                        \
    int64 num_elements = tf_shape.num_elements();                                         \
    testing::UseRealTime();                                                               \
    testing::BytesProcessed(static_cast<int64>(iters) * nums * num_elements * sizeof(T)); \
    SessionOptions opts;                                                                  \
    opts.config.set_intra_op_parallelism_threads(NTH);                                    \
    test::Benchmark(#DEVICE, AddN<T>(#kind, nums, tf_shape), &opts).Run(iters);           \
  }                                                                                       \
  BENCHMARK(BM_AddN##_##kind##_##T##_##name##_##DEVICE##_##NTH);                          \

#define BM_AddN_kind(T, name, nums, tf_shape, DEVICE, NTH)     \
  BM_AddN_Base(Default, T, name, nums, tf_shape, DEVICE, NTH); \
  BM_AddN_Base(Mkl, T, name, nums, tf_shape, DEVICE, NTH);     \

#define BM_AddN_NTH(T, name, nums, tf_shape, DEVICE) \
  BM_AddN_kind(T, name, nums, tf_shape, DEVICE, 1);  \
  BM_AddN_kind(T, name, nums, tf_shape, DEVICE, 4);  \
  BM_AddN_kind(T, name, nums, tf_shape, DEVICE, 8);  \

#define BM_AddN_DT(name, nums, tf_shape)            \
  BM_AddN_NTH(float, name, nums, tf_shape, cpu);    \
  BM_AddN_NTH(bfloat16, name, nums, tf_shape, cpu); \

#define BM_AddN_2D(num_inputs, A, B)                                            \
  BM_AddN_DT(num_inputs##_##2D##_##A##x##B, num_inputs, TensorShape({A, B})); \

#define BM_AddN_3D(num_inputs, A, B, C)                                            \
  BM_AddN_DT(num_inputs##_##3D##_##A##x##B##x##C, num_inputs, TensorShape({A, B, C})); \

// dims = 2
BM_AddN_2D(4, 128, 128);
BM_AddN_2D(4, 128, 512);
BM_AddN_2D(4, 128, 2048);
BM_AddN_2D(4, 128, 8192);
BM_AddN_2D(4, 512, 128);
BM_AddN_2D(4, 2048, 128);
BM_AddN_2D(4, 8192, 128);
BM_AddN_2D(4, 512, 512);
BM_AddN_2D(4, 2048, 2048);
BM_AddN_2D(4, 8192, 8192);

BM_AddN_2D(16, 128, 128);
BM_AddN_2D(16, 128, 512);
BM_AddN_2D(16, 128, 2048);
BM_AddN_2D(16, 128, 8192);
BM_AddN_2D(16, 512, 128);
BM_AddN_2D(16, 2048, 128);
BM_AddN_2D(16, 8192, 128);
BM_AddN_2D(16, 512, 512);
BM_AddN_2D(16, 2048, 2048);
BM_AddN_2D(16, 8192, 8192);

BM_AddN_2D(64, 128, 128);
BM_AddN_2D(64, 128, 512);
BM_AddN_2D(64, 128, 2048);
BM_AddN_2D(64, 128, 8192);
BM_AddN_2D(64, 512, 128);
BM_AddN_2D(64, 2048, 128);
BM_AddN_2D(64, 8192, 128);
BM_AddN_2D(64, 512, 512);
BM_AddN_2D(64, 2048, 2048);
BM_AddN_2D(64, 8192, 8192);

// dims = 3
BM_AddN_3D(4, 128, 128, 128);
BM_AddN_3D(4, 128, 128, 1024);
BM_AddN_3D(4, 128, 128, 8192);
BM_AddN_3D(4, 128, 1024, 128);
BM_AddN_3D(4, 128, 8192, 128);
BM_AddN_3D(4, 1024, 128, 128);
BM_AddN_3D(4, 8192, 128, 128);
BM_AddN_3D(4, 128, 1024, 1024);
BM_AddN_3D(4, 128, 8192, 8192);
BM_AddN_3D(4, 1024, 128, 1024);
BM_AddN_3D(4, 8192, 128, 8192);
BM_AddN_3D(4, 1024, 1024, 128);
BM_AddN_3D(4, 8192, 8192, 128);
BM_AddN_3D(4, 1024, 1024, 1024);
BM_AddN_3D(4, 8192, 8192, 8192);

BM_AddN_3D(16, 128, 128, 128);
BM_AddN_3D(16, 128, 128, 1024);
BM_AddN_3D(16, 128, 128, 8192);
BM_AddN_3D(16, 128, 1024, 128);
BM_AddN_3D(16, 128, 8192, 128);
BM_AddN_3D(16, 1024, 128, 128);
BM_AddN_3D(16, 8192, 128, 128);
BM_AddN_3D(16, 128, 1024, 1024);
BM_AddN_3D(16, 128, 8192, 8192);
BM_AddN_3D(16, 1024, 128, 1024);
BM_AddN_3D(16, 8192, 128, 8192);
BM_AddN_3D(16, 1024, 1024, 128);
BM_AddN_3D(16, 8192, 8192, 128);
BM_AddN_3D(16, 1024, 1024, 1024);
BM_AddN_3D(16, 8192, 8192, 8192);

BM_AddN_3D(64, 128, 128, 128);
BM_AddN_3D(64, 128, 128, 1024);
BM_AddN_3D(64, 128, 128, 8192);
BM_AddN_3D(64, 128, 1024, 128);
BM_AddN_3D(64, 128, 8192, 128);
BM_AddN_3D(64, 1024, 128, 128);
BM_AddN_3D(64, 8192, 128, 128);
BM_AddN_3D(64, 128, 1024, 1024);
BM_AddN_3D(64, 128, 8192, 8192);
BM_AddN_3D(64, 1024, 128, 1024);
BM_AddN_3D(64, 8192, 128, 8192);
BM_AddN_3D(64, 1024, 1024, 128);
BM_AddN_3D(64, 8192, 8192, 128);
BM_AddN_3D(64, 1024, 1024, 1024);
BM_AddN_3D(64, 8192, 8192, 8192);

}  // namespace tensorflow

#endif  // INTEL_MKL
