/* Copyright 2023 The DeepRec Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

#define BM_FUNC(OPERATOR)                                                    \
  static Graph* BM_##OPERATOR(int input_x_size, int input_y_size) {          \
    Graph* graph = new Graph(OpRegistry::Global());                          \
                                                                             \
    Tensor data(DT_FLOAT, TensorShape({input_x_size, input_y_size}));        \
    data.flat<float>().setConstant(1.0f);                                    \
                                                                             \
    Tensor segment_ids(DT_INT64, TensorShape({input_x_size}));               \
    auto segment_ids_vec = segment_ids.vec<int64>();                         \
    for (int i = 0; i < input_x_size; i++)                                   \
      segment_ids_vec(i) = i / 3;                                            \
                                                                             \
    Node* ret = nullptr;                                                     \
    TF_CHECK_OK(                                                             \
        NodeBuilder(graph->NewName("n"), #OPERATOR)                          \
            .Input(test::graph::Constant(graph, data, "data"))               \
            .Input(test::graph::Constant(graph, segment_ids, "segment_ids")) \
            .Attr("T", DT_FLOAT)                                             \
            .Attr("Tindices", DT_INT64)                                      \
            .Finalize(graph, &ret));                                         \
    return graph;                                                            \
  }

BM_FUNC(SegmentSum)
BM_FUNC(SegmentMean)
BM_FUNC(SegmentProd)
BM_FUNC(SegmentMin)
BM_FUNC(SegmentMax)

#define BM_FUNC_DEV(FUNC, DEVICE, X_SIZE, Y_SIZE)			\
  static void BM_##FUNC##_##DEVICE##_##X_SIZE##_##Y_SIZE(int iters) {	\
    test::Benchmark(#DEVICE, BM_##FUNC(X_SIZE, Y_SIZE)).Run(iters);	\
  }									\
  BENCHMARK(BM_##FUNC##_##DEVICE##_##X_SIZE##_##Y_SIZE)

#define BM_FUNC_DEV_DEVICE(FUNC, DEVICE) \
  BM_FUNC_DEV(FUNC, DEVICE, 4096, 16);   \
  BM_FUNC_DEV(FUNC, DEVICE, 8192, 16);   \
  BM_FUNC_DEV(FUNC, DEVICE, 16384, 16);


/*----------------------------- SegmentSum Begin ------------------------------*/
// CPU
BM_FUNC_DEV_DEVICE(SegmentSum, cpu)

// GPU
#ifdef GOOGLE_CUDA
BM_FUNC_DEV_DEVICE(SegmentSum, gpu)
#endif
/*----------------------------- SegmentSum End --------------------------------*/

/*---------------------------- SegmentMean Begin ------------------------------*/
// CPU
BM_FUNC_DEV_DEVICE(SegmentMean, cpu)

// GPU
#ifdef GOOGLE_CUDA
BM_FUNC_DEV_DEVICE(SegmentMean, gpu)
#endif
/*---------------------------- SegmentMean End --------------------------------*/

/*---------------------------- SegmentProd Begin ------------------------------*/
// CPU
BM_FUNC_DEV_DEVICE(SegmentProd, cpu)

// GPU
#ifdef GOOGLE_CUDA
BM_FUNC_DEV_DEVICE(SegmentProd, gpu)
#endif
/*---------------------------- SegmentProd End --------------------------------*/

/*----------------------------- SegmentMin Begin ------------------------------*/
// CPU
BM_FUNC_DEV_DEVICE(SegmentMin, cpu)

// GPU
#ifdef GOOGLE_CUDA
BM_FUNC_DEV_DEVICE(SegmentMin, gpu)
#endif
/*----------------------------- SegmentMin End --------------------------------*/

/*----------------------------- SegmentMax Begin ------------------------------*/
// CPU
BM_FUNC_DEV_DEVICE(SegmentMax, cpu)

// GPU
#ifdef GOOGLE_CUDA
BM_FUNC_DEV_DEVICE(SegmentMax, gpu)
#endif
/*----------------------------- SegmentMax End --------------------------------*/

#undef BM_FUNC_DEV_DEVICE
#undef BM_FUNC_DEV
#undef BM_FUNC
} // End of namespace tensorflow
