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
static Graph* Reshape(const string& kind, const TensorShape& in_shape, const Tensor& shape_tensor) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "Reshape" : "_MklReshape";

  Tensor input(type, in_shape);
  input.flat<T>().setRandom();

  Node* not_mkl_shape =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(test::graph::Constant(g, input))
                    .Input(test::graph::Constant(g, shape_tensor))
                    .Attr("T", type);

  isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
                                       .Input(not_mkl_shape)
                                       .Attr("_kernel", "MklLayoutDependentOp");
  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));
  return g;
}

// Constructs a tensor of "shape" with values "vals".
template <typename T>
Tensor AsTensor(gtl::ArraySlice<T> vals, const TensorShape& shape) {
  Tensor ret;
  CHECK(ret.CopyFrom(AsTensor(vals), shape));
  return ret;
}

#define S_TENSOR(size, ...) test::AsTensor<int32>({__VA_ARGS__}, {size})

#define BM_Reshape_Base(kind, T, name, in_shape, shape_tensor, DEVICE, NTH)                \
  static void BM_Reshape##_##kind##_##T##name##_##DEVICE##_##NTH(                          \
      int iters) {                                                                         \
    int64 num_elements = in_shape.num_elements();  	                                   \
    testing::UseRealTime();                                                                \
    testing::ItemsProcessed(static_cast<int64>(iters) * num_elements);                     \
    SessionOptions opts;                                                                   \
    opts.config.set_intra_op_parallelism_threads(NTH);                                     \
    test::Benchmark(#DEVICE, Reshape<T>(#kind, in_shape, shape_tensor), &opts).Run(iters); \
  }                                                                                        \
  BENCHMARK(BM_Reshape##_##kind##_##T##name##_##DEVICE##_##NTH);                           \

#define BM_Reshape_kind(T, name, in_shape, shape_tensor, DEVICE, NTH)     \
  BM_Reshape_Base(Default, T, name, in_shape, shape_tensor, DEVICE, NTH); \
  BM_Reshape_Base(Mkl, T, name, in_shape, shape_tensor, DEVICE, NTH);     \

#define BM_Reshape_NTH(T, name, in_shape, shape_tensor, DEVICE) \
  BM_Reshape_kind(T, name, in_shape, shape_tensor, DEVICE, 1);  \
  BM_Reshape_kind(T, name, in_shape, shape_tensor, DEVICE, 4);  \
  BM_Reshape_kind(T, name, in_shape, shape_tensor, DEVICE, 8);  \

#define BM_Reshape_DT(name, in_shape, shape_tensor)             \
  BM_Reshape_NTH(float, name, in_shape, shape_tensor, cpu);    \
  BM_Reshape_NTH(bfloat16, name, in_shape, shape_tensor, cpu); \

#define BM_Reshape2D(name, A, B, size, ...)                                   \
  BM_Reshape_DT(_2D##name, TensorShape({A, B}), S_TENSOR(size, __VA_ARGS__)); \

#define BM_Reshape3D(name, A, B, C, size, ...)                                   \
  BM_Reshape_DT(_3D##name, TensorShape({A, B, C}), S_TENSOR(size, __VA_ARGS__)); \

#define BM_Reshape4D(name, A, B, C, D, size, ...)                                   \
  BM_Reshape_DT(_4D##name, TensorShape({A, B, C, D}), S_TENSOR(size, __VA_ARGS__)); \

BM_Reshape2D(_1024x1024_To_256x4096, 1024, 1024, 2, 256, 4096);
BM_Reshape2D(_1024x1024_To_16x256x256, 1024, 1024, 3, 16, 256, 256);

BM_Reshape3D(_128x128x256_To_256x128x128, 128, 128, 256, 3, 256, 128, 128);

BM_Reshape4D(_128x128x16x16_To_256x128x128, 128, 128, 16, 16, 3, 256, 128, 128);

}  // namespace tensorflow

#endif  // INTEL_MKL
