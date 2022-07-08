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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

static Graph* BM_SparseReshape(int indices_num, int input_dims, int output_dims) {
    Graph* g = new Graph(OpRegistry::Global());
    assert(input_dims > output_dims && "In this benchmark, we assume intput_dims > output_dims");
    int diff = input_dims - output_dims;

    Tensor indices_tensor(DT_INT64, TensorShape({indices_num, input_dims}));
    indices_tensor.flat<int64>().setConstant(2);

    Tensor input_shape_tensor(DT_INT64, TensorShape({input_dims}));
    auto input_shape_tensor_ptr = input_shape_tensor.flat<int64>().data();
    int product = 1;
    for (int i = 0; i < input_dims; i++) {
        input_shape_tensor_ptr[i] = 6;
        if (i <= diff) {
            product *= input_shape_tensor_ptr[i];
        }
    }

    Tensor output_shape_tensor(DT_INT64, TensorShape({output_dims}));
    auto output_shape_tensor_ptr = output_shape_tensor.flat<int64>().data();
    output_shape_tensor_ptr[0] = product;
    for (int i = 1; i < output_dims; i++) {
        output_shape_tensor_ptr[i] = 6;
    }

    Node *ret;
    TF_CHECK_OK(NodeBuilder(g->NewName("n"), "SparseReshape")
                        .Input(test::graph::Constant(g, indices_tensor))
                        .Input(test::graph::Constant(g, input_shape_tensor))
                        .Input(test::graph::Constant(g, output_shape_tensor))
                        .Finalize(g, &ret));
    return g;
}

#define BM_SparseReshapeDev(DEVICE, NUM, OLD_RANK, NEW_RANK) \
    static void BM_SparseReshape_##DEVICE##_##NUM##_##OLD_RANK##_##NEW_RANK(int iters) { \
        test::Benchmark(#DEVICE, BM_SparseReshape(NUM, OLD_RANK, NEW_RANK)).Run(iters); \
    } \
    BENCHMARK(BM_SparseReshape_##DEVICE##_##NUM##_##OLD_RANK##_##NEW_RANK)

BM_SparseReshapeDev(cpu, 16, 8, 4);
BM_SparseReshapeDev(cpu, 32, 8, 4);
BM_SparseReshapeDev(cpu, 64, 8, 4);
BM_SparseReshapeDev(cpu, 128, 8, 4);
BM_SparseReshapeDev(cpu, 256, 8, 4);
BM_SparseReshapeDev(cpu, 512, 8, 4);
BM_SparseReshapeDev(cpu, 1024, 8, 4);
BM_SparseReshapeDev(cpu, 2048, 8, 4);
BM_SparseReshapeDev(cpu, 4096, 8, 4);
BM_SparseReshapeDev(cpu, 8192, 8, 4);
BM_SparseReshapeDev(cpu, 16384, 8, 4);

} // End of namespace tensorflow