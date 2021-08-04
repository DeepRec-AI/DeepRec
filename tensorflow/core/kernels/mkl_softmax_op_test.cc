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

#include <functional>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

template <typename T>
static Graph* Softmax(const string& kind, const TensorShape& in_shape) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "Softmax" : "_MklSoftmax";

  // Create inputs
  Tensor input1(type, in_shape);
  input1.flat<T>().setRandom();

  Node* input_in0 = test::graph::Constant(g, input1);

  Node* not_mkl_shape =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");
  // Create NodeDef
  auto nodeBuilder = NodeBuilder(g->NewName("softmax"), op_name)
                  .Input(input_in0);

  isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
                                       .Attr("_kernel", "MklLayoutDependentOp");

  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_Softmax_Base(T, kind, name, in_shape, DEVICE, NTH)                \
  static void BM_Softmax_##T##_##kind##name##_##NTH(int iters) {             \
    int64 num_elements = in_shape.num_elements();                            \
    testing::UseRealTime();                                                  \
    testing::ItemsProcessed(static_cast<int64>(iters) * num_elements);       \
    SessionOptions opts;                                                     \
    opts.config.set_intra_op_parallelism_threads(NTH);                       \
    test::Benchmark(#DEVICE, Softmax<T>(#kind, in_shape), &opts).Run(iters); \
  }                                                                          \
  BENCHMARK(BM_Softmax_##T##_##kind##name##_##NTH);                          \

#define BM_Softmax_Kind(T, name, in_shape, DEVICE, NTH)     \
  BM_Softmax_Base(T, Default, name, in_shape, DEVICE, NTH); \
  BM_Softmax_Base(T, MKL, name, in_shape, DEVICE, NTH);     \

#define BM_Softmax_NTH(T, name, in_shape, DEVICE) \
  BM_Softmax_Kind(T, name, in_shape, DEVICE, 1);  \
  BM_Softmax_Kind(T, name, in_shape, DEVICE, 4);  \
  BM_Softmax_Kind(T, name, in_shape, DEVICE, 8);  \

#define BM_Softmax_DT(name, in_shape)             \
  BM_Softmax_NTH(float, name, in_shape, cpu);    \
  BM_Softmax_NTH(bfloat16, name, in_shape, cpu); \

#define BM_SoftmaxND(name, ...)                    \
  BM_Softmax_DT(name, TensorShape({__VA_ARGS__})); \

// dims == 2
BM_SoftmaxND(_2D_32x32, 32, 32);
BM_SoftmaxND(_2D_32x1024, 32, 1024);
BM_SoftmaxND(_2D_32x4096, 32, 4096);
BM_SoftmaxND(_2D_1024x32, 1024, 32);
BM_SoftmaxND(_2D_4096x32, 4096, 32);
BM_SoftmaxND(_2D_1024x1024, 1024, 1024);
BM_SoftmaxND(_2D_4096x4096, 4096, 4096);

// dims == 3
BM_SoftmaxND(_3D_32x32x32, 32, 32, 32);
BM_SoftmaxND(_3D_32x32x1024, 32, 32, 1024);
BM_SoftmaxND(_3D_32x32x4096, 32, 32, 4096);
BM_SoftmaxND(_3D_32x1024x32, 32, 1024, 32);
BM_SoftmaxND(_3D_32x4096x32, 32, 4096, 32);
BM_SoftmaxND(_3D_32x1024x1024, 32, 1024, 1024);
BM_SoftmaxND(_3D_32x4096x4096, 32, 4096, 4096);
BM_SoftmaxND(_3D_1024x1024x1024, 1024, 1024, 1024);

}  // namespace tensorflow

#endif  // INTEL_MKL
