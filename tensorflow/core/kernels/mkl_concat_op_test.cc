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
static Graph* Concat(const string& kind, int num_inputs,
		const TensorShape& in_shape, int concat_dims) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "Concat" : "_MklConcat";

  Tensor concat_dim(DT_INT32, TensorShape({}));
  concat_dim.scalar<int32>()() = concat_dims;

  Node* not_mkl_shape =
      test::graph::Constant(g, GetMklMetaTensor(), "not_mkl");

  std::vector<NodeBuilder::NodeOut> inputs;
  std::vector<NodeBuilder::NodeOut> inputs_not_mkl;
  inputs.reserve(num_inputs);
  inputs_not_mkl.reserve(num_inputs);

  for (int i = 0; i < num_inputs; ++i) {
    Tensor in(type, in_shape);
    in.flat<T>().setRandom();
    inputs.push_back(test::graph::Constant(g, in));
    inputs_not_mkl.push_back(test::graph::Constant(g, GetMklMetaTensor(), "not_mkl"));
  }

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Input(test::graph::Constant(g, concat_dim))
                    .Input(inputs)
                    .Attr("N", num_inputs)
                    .Attr("T", type);

  isDefault ? nodeBuilder : nodeBuilder.Input(not_mkl_shape)
	                               .Input(inputs_not_mkl)
				       .Attr("_kernel", "MklLayoutDependentOp");
  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define S_TENSOR(...) test::AsTensor<int32>({__VA_ARGS__})

#define BM_Concat_Base(kind, name, NI, in_shape, CD, T, DEVICE, NTH)                    \
  static void BM_Concat##_##kind##_##NI##name##_##T##_##CD##_##DEVICE##_##NTH(          \
      int iters) {                                                                      \
    int64 num_elements = in_shape.num_elements();                                       \
    testing::UseRealTime();                                                             \
    testing::ItemsProcessed(static_cast<int64>(iters) * num_elements * NI * sizeof(T)); \
    SessionOptions opts;                                                                \
    opts.config.set_intra_op_parallelism_threads(NTH);                                  \
    test::Benchmark(#DEVICE, Concat<T>(#kind, NI, in_shape, CD), &opts).Run(iters);     \
  }                                                                                     \
  BENCHMARK(BM_Concat##_##kind##_##NI##name##_##T##_##CD##_##DEVICE##_##NTH);           \

#define BM_Concat_kind(name, NI, in_shape, CD, T, DEVICE, NTH)     \
  BM_Concat_Base(Default, name, NI, in_shape, CD, T, DEVICE, NTH); \
  BM_Concat_Base(Mkl, name, NI, in_shape, CD, T, DEVICE, NTH);     \

#define BM_Concat_NTH(name, NI, in_shape, CD, T, DEVICE) \
  BM_Concat_kind(name, NI, in_shape, CD, T, DEVICE, 1);  \
  BM_Concat_kind(name, NI, in_shape, CD, T, DEVICE, 4);  \
  BM_Concat_kind(name, NI, in_shape, CD, T, DEVICE, 8);  \

#define BM_Concat_DT(name, NI, in_shape, CD)             \
  BM_Concat_NTH(name, NI, in_shape, CD, float, cpu);    \
  BM_Concat_NTH(name, NI, in_shape, CD, bfloat16, cpu); \

#define BM_ConcatND(name, NI, ...)                       \
  BM_Concat_DT(name, NI, TensorShape({__VA_ARGS__}), 0); \

// dims == 2
BM_ConcatND(_2Dx2x32x32, 2, 32, 32)
BM_ConcatND(_2Dx2x32x256, 2, 32, 256)
BM_ConcatND(_2Dx2x32x2048, 2, 32, 2048)
BM_ConcatND(_2Dx2x256x32, 2, 256, 32)
BM_ConcatND(_2Dx2x2048x32, 2, 2048, 32)
BM_ConcatND(_2Dx2x256x256, 2, 256, 256)
BM_ConcatND(_2Dx2x2048x2048, 2, 2048, 2048)

BM_ConcatND(_2Dx8x32x32, 8, 32, 32)
BM_ConcatND(_2Dx8x32x256, 8, 32, 256)
BM_ConcatND(_2Dx8x32x2048, 8, 32, 2048)
BM_ConcatND(_2Dx8x256x32, 8, 256, 32)
BM_ConcatND(_2Dx8x2048x32, 8, 2048, 32)
BM_ConcatND(_2Dx8x256x256, 8, 256, 256)
BM_ConcatND(_2Dx8x2048x2048, 8, 2048, 2048)

BM_ConcatND(_2Dx32x32x32, 32, 32, 32)
BM_ConcatND(_2Dx32x32x256, 32, 32, 256)
BM_ConcatND(_2Dx32x32x2048, 32, 32, 2048)
BM_ConcatND(_2Dx32x256x32, 32, 256, 32)
BM_ConcatND(_2Dx32x2048x32, 32, 2048, 32)
BM_ConcatND(_2Dx32x256x256, 32, 256, 256)
BM_ConcatND(_2Dx32x2048x2048, 32, 2048, 2048)

// dims == 3
BM_ConcatND(_3Dx2x32x32x32, 2, 32, 32, 32)
BM_ConcatND(_3Dx2x32x32x256, 2, 32, 32, 256)
BM_ConcatND(_3Dx2x32x32x2048, 2, 32, 32, 2048)
BM_ConcatND(_3Dx2x32x256x32, 2, 32, 256, 32)
BM_ConcatND(_3Dx2x32x2048x32, 2, 32, 2048, 32)
BM_ConcatND(_3Dx2x32x256x256, 2, 32, 256, 256)
BM_ConcatND(_3Dx2x32x2048x2048, 2, 32, 2048, 2048)
BM_ConcatND(_3Dx2x256x32x32, 2, 256, 32, 32)
BM_ConcatND(_3Dx2x256x32x2048, 2, 256, 32, 2048)
BM_ConcatND(_3Dx2x256x2048x32, 2, 256, 2048, 32)
BM_ConcatND(_3Dx2x256x256x256, 2, 256, 256, 256)

BM_ConcatND(_3Dx8x32x32x32, 8, 32, 32, 32)
BM_ConcatND(_3Dx8x32x32x256, 8, 32, 32, 256)
BM_ConcatND(_3Dx8x32x32x2048, 8, 32, 32, 2048)
BM_ConcatND(_3Dx8x32x256x32, 8, 32, 256, 32)
BM_ConcatND(_3Dx8x32x2048x32, 8, 32, 2048, 32)
BM_ConcatND(_3Dx8x32x256x256, 8, 32, 256, 256)
BM_ConcatND(_3Dx8x32x2048x2048, 8, 32, 2048, 2048)
BM_ConcatND(_3Dx8x256x32x32, 8, 256, 32, 32)
BM_ConcatND(_3Dx8x256x32x2048, 8, 256, 32, 2048)
BM_ConcatND(_3Dx8x256x2048x32, 8, 256, 2048, 32)
BM_ConcatND(_3Dx8x256x256x256, 8, 256, 256, 256)

BM_ConcatND(_3Dx32x32x32x32, 32, 32, 32, 32)
BM_ConcatND(_3Dx32x32x32x256, 32, 32, 32, 256)
BM_ConcatND(_3Dx32x32x32x2048, 32, 32, 32, 2048)
BM_ConcatND(_3Dx32x32x256x32, 32, 32, 256, 32)
BM_ConcatND(_3Dx32x32x2048x32, 32, 32, 2048, 32)
BM_ConcatND(_3Dx32x32x256x256, 32, 32, 256, 256)
BM_ConcatND(_3Dx32x32x2048x2048, 32, 32, 2048, 2048)
BM_ConcatND(_3Dx32x256x32x32, 32, 256, 32, 32)
BM_ConcatND(_3Dx32x256x32x2048, 32, 256, 32, 2048)
BM_ConcatND(_3Dx32x256x2048x32, 32, 256, 2048, 32)
BM_ConcatND(_3Dx32x256x256x256, 32, 256, 256, 256)

// dims == 4
BM_ConcatND(_4Dx2x32x32x32x32, 2, 32, 32, 32, 32)
BM_ConcatND(_4Dx2x256x256x256x256, 2, 256, 256, 256, 256)

BM_ConcatND(_4Dx8x32x32x32x32, 8, 32, 32, 32, 32)
BM_ConcatND(_4Dx8x256x256x256x256, 8, 256, 256, 256, 256)

BM_ConcatND(_4Dx32x32x32x32x32, 32, 32, 32, 32, 32)
BM_ConcatND(_4Dx32x256x256x256x256, 32, 256, 256, 256, 256)

}  // namespace tensorflow

#endif  // INTEL_MKL
