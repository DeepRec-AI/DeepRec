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
static Graph* Slice2D(const string& kind, DataType type, int size) {
  Graph* g = new Graph(OpRegistry::Global());

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "Slice" : "_MklSlice";

  int kDim = 100;
  int kMaxSize = 15000;
  CHECK_LT(size, kMaxSize);

  Tensor input(type, TensorShape({2 * kDim, kMaxSize}));
  input.flat<T>().setRandom();

  Tensor begin(DT_INT32, TensorShape({2}));
  begin.flat<int32>()(0) = 10;
  begin.flat<int32>()(1) = 10;

  Tensor sizes(DT_INT32, TensorShape({2}));
  sizes.flat<int32>()(0) = kDim;
  sizes.flat<int32>()(1) = size;

  Node* not_mkl_shape =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(test::graph::Constant(g, input))
                    .Input(test::graph::Constant(g, begin))
                    .Input(test::graph::Constant(g, sizes))
                    .Attr("T", type);
  isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
	                               .Input(not_mkl_shape)
	                               .Input(not_mkl_shape)
				       .Attr("_kernel", "MklLayoutDependentOp");
  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_Slice2D_Base(kind, size, T, TFTYPE, DEVICE, NTH)                        \
  static void BM_Slice2D##_##kind##_##size##_##T##_##TFTYPE##_##DEVICE##_##NTH(    \
      int iters) {                                                                 \
    testing::UseRealTime();                                                        \
    testing::BytesProcessed(static_cast<int64>(iters) * 100 * size * sizeof(T));   \
    SessionOptions opts;                                                           \
    opts.config.set_intra_op_parallelism_threads(NTH);                             \
    test::Benchmark(#DEVICE, Slice2D<T>(#kind, TFTYPE, size), &opts).Run(iters);   \
  }                                                                                \
  BENCHMARK(BM_Slice2D##_##kind##_##size##_##T##_##TFTYPE##_##DEVICE##_##NTH);     \

#define BM_Slice2D_kind(size, T, TFTYPE, DEVICE, NTH)     \
  BM_Slice2D_Base(Default, size, T, TFTYPE, DEVICE, NTH); \
  BM_Slice2D_Base(Mkl, size, T, TFTYPE, DEVICE, NTH);     \

#define BM_Slice2D_NTH(size, T, TFTYPE, DEVICE) \
  BM_Slice2D_kind(size, T, TFTYPE, DEVICE, 1);  \
  BM_Slice2D_kind(size, T, TFTYPE, DEVICE, 4);  \
  BM_Slice2D_kind(size, T, TFTYPE, DEVICE, 8);  \

#define BM_Slice2D_DT(size)                         \
  BM_Slice2D_NTH(size, float, DT_FLOAT, cpu);       \
  BM_Slice2D_NTH(size, bfloat16, DT_BFLOAT16, cpu); \

#define BM_Slice2D(size) \
  BM_Slice2D_DT(size)    \

BM_Slice2D(100);
BM_Slice2D(1000);
BM_Slice2D(10000);

}  // namespace tensorflow

#endif  // INTEL_MKL
