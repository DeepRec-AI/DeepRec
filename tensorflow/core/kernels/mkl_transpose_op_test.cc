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

namespace tensorflow {

template <typename T>
static Graph* Transpose(const string& kind, const TensorShape& in_shape, const Tensor& perm_tensor) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "Transpose" : "_MklTranspose";

  // Create inputs
  Tensor input1(type, in_shape);
  input1.flat<T>().setRandom();

  Node* input_in0 = test::graph::Constant(g, input1);
  Node* input_in1 = test::graph::Constant(g, perm_tensor);

  // Create NodeDef
  auto nodeBuilder = NodeBuilder(g->NewName("transpose"), op_name)
                  .Input(input_in0)
                  .Input(input_in1);

  isDefault ? nodeBuilder : nodeBuilder.Attr("_kernel", "MklNameChangeOp");

  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define S_TENSOR(size, ...) test::AsTensor<int32>({__VA_ARGS__}, {size})

#define BM_Transpose_Base(T, kind, name, in_shape, perm_tensor, DEVICE, NTH)                \
  static void BM_Transpose_##T##_##kind##name##_##NTH(int iters) {                          \
    int64 num_elements = in_shape.num_elements();                                           \
    testing::UseRealTime();                                                                 \
    testing::BytesProcessed(static_cast<int64>(iters) * num_elements * sizeof(T));          \
    SessionOptions opts;                                                                    \
    opts.config.set_intra_op_parallelism_threads(NTH);                                      \
    test::Benchmark(#DEVICE, Transpose<T>(#kind, in_shape, perm_tensor), &opts).Run(iters); \
  }                                                                                         \
  BENCHMARK(BM_Transpose_##T##_##kind##name##_##NTH);                                       \

#define BM_Transpose_Kind(T, name, in_shape, perm_tensor, DEVICE, NTH)     \
  BM_Transpose_Base(T, Default, name, in_shape, perm_tensor, DEVICE, NTH); \
  BM_Transpose_Base(T, MKL, name, in_shape, perm_tensor, DEVICE, NTH);     \

#define BM_Transpose_NTH(T, name, in_shape, perm_tensor, DEVICE) \
  BM_Transpose_Kind(T, name, in_shape, perm_tensor, DEVICE, 1);  \
  BM_Transpose_Kind(T, name, in_shape, perm_tensor, DEVICE, 4);  \
  BM_Transpose_Kind(T, name, in_shape, perm_tensor, DEVICE, 8);  \

#define BM_Transpose_DT(name, in_shape, perm_tensor)            \
  BM_Transpose_NTH(float, name, in_shape, perm_tensor, cpu);    \
  BM_Transpose_NTH(bfloat16, name, in_shape, perm_tensor, cpu); \

#define BM_Transpose2D(name, A, B, size, ...)                                   \
  BM_Transpose_DT(_2D##name, TensorShape({A, B}), S_TENSOR(size, __VA_ARGS__)); \

#define BM_Transpose3D(name, A, B, C, size, ...)                                   \
  BM_Transpose_DT(_3D##name, TensorShape({A, B, C}), S_TENSOR(size, __VA_ARGS__)); \

#define BM_Transpose4D(name, A, B, C, D, size, ...)                                   \
  BM_Transpose_DT(_4D##name, TensorShape({A, B, C, D}), S_TENSOR(size, __VA_ARGS__)); \

BM_Transpose2D(_128x512, 128, 512, 2, 1, 0);
BM_Transpose2D(_128x1024, 128, 1024, 2, 1, 0);
BM_Transpose2D(_128x2048, 128, 2048, 2, 1, 0);
BM_Transpose2D(_128x4096, 128, 4096, 2, 1, 0);

BM_Transpose2D(_512x128, 512, 128, 2, 1, 0);
BM_Transpose2D(_1024x128, 1024, 128, 2, 1, 0);
BM_Transpose2D(_2048x128, 2048, 128, 2, 1, 0);
BM_Transpose2D(_4096x128, 4096, 128, 2, 1, 0);

BM_Transpose2D(_128x128, 128, 128, 2, 1, 0);
BM_Transpose2D(_512x512, 512, 512, 2, 1, 0);
BM_Transpose2D(_1024x1024, 1024, 1024, 2 , 1, 0);
BM_Transpose2D(_2048x2048, 2048, 2048, 2 , 1, 0);
BM_Transpose2D(_4096x4096, 4096, 4096, 2 , 1, 0);

BM_Transpose3D(_128x128x128, 128, 128, 128, 3, 0, 2, 1);
BM_Transpose3D(_256x256x256, 256, 256, 256, 3, 0, 2, 1);
BM_Transpose3D(_512x512x512, 512, 512, 512, 3, 0, 2, 1);

BM_Transpose4D(_128x128x128x128, 128, 128, 128, 128, 4, 3, 1, 2, 0);

}  // namespace tensorflow

#endif  // INTEL_MKL
