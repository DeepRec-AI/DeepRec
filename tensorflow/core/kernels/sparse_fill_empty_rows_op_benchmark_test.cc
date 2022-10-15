#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
static Graph *BM_SparseFillEmptyRows(int indices_num, int input_dims) {
  Graph *graph = new Graph(OpRegistry::Global());

  Tensor indices(DT_INT64, TensorShape({indices_num, input_dims}));
  indices.flat<int64>().setConstant(0);
  auto indices_matrix = indices.matrix<int64>();
  for (int i = 0; i < indices_num; i++)
    indices_matrix(i, 0) = i;
  
  Tensor dense_shape(DT_INT64, TensorShape({2}));
  auto dense_shape_vec = dense_shape.vec<int64>();
  dense_shape_vec(0) = indices_num*2;
  dense_shape_vec(1) = input_dims;

  Tensor values(DT_DOUBLE, TensorShape({indices_num}));
  values.flat<double>().setConstant(2);

  Tensor default_value((double)-1.0);

  Node *ret = nullptr;
  TF_CHECK_OK(NodeBuilder(graph->NewName("n"), "SparseFillEmptyRows")
	      .Input(test::graph::HostConstant(graph, indices, "indices"))
	      .Input(test::graph::Constant(graph, values, "values"))
	      .Input(test::graph::HostConstant(graph, dense_shape, "dense_shape"))
	      .Input(test::graph::Constant(graph, default_value, "default_value"))
	      .Attr("T", DT_DOUBLE)
	      .Finalize(graph, &ret));

  return graph;
}

#define BM_SparseFillEmptyRowsDev(DEVICE, NUM, RANK)                        \
  static void BM_SparseFillEmptyRows_##DEVICE##_##NUM##_##RANK(int iters) { \
    test::Benchmark(#DEVICE, BM_SparseFillEmptyRows(NUM, RANK)).Run(iters); \
  }                                                                         \
  BENCHMARK(BM_SparseFillEmptyRows_##DEVICE##_##NUM##_##RANK)

BM_SparseFillEmptyRowsDev(cpu, 256, 16);
BM_SparseFillEmptyRowsDev(cpu, 512, 16);
BM_SparseFillEmptyRowsDev(cpu, 1024, 16);
BM_SparseFillEmptyRowsDev(cpu, 2048, 16);
BM_SparseFillEmptyRowsDev(cpu, 4096, 16);
BM_SparseFillEmptyRowsDev(cpu, 8192, 16);
BM_SparseFillEmptyRowsDev(cpu, 16384, 16);

#ifdef GOOGLE_CUDA
BM_SparseFillEmptyRowsDev(gpu, 256, 16);
BM_SparseFillEmptyRowsDev(gpu, 512, 16);
BM_SparseFillEmptyRowsDev(gpu, 1024, 16);
BM_SparseFillEmptyRowsDev(gpu, 2048, 16);
BM_SparseFillEmptyRowsDev(gpu, 4096, 16);
BM_SparseFillEmptyRowsDev(gpu, 8192, 16);
BM_SparseFillEmptyRowsDev(gpu, 16384, 16);

#endif // end of GOOGLE_CUDA
    
} // end of namespace tensorflow
