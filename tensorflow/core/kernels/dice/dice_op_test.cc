#if defined(__GNUC__) && (__GNUC__ > 6) && (__AVX512F__)

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {

enum class Device { CPU, GPU };

class DiceOpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, DataType dtype, int axis,
                          float epsilon) {
    TF_EXPECT_OK(NodeDefBuilder("dice", "Dice")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(DiceOpTest, Dice_Test) {
  const int rows = 7;
  const int cols = 255;

  MakeOpAndSetDevice(Device::CPU, DT_FLOAT, 0, 1e-12);

  AddInput<float>(TensorShape({rows, cols}),
                  [](int i) -> float { return 2.0; });
  AddInput<float>(TensorShape({cols}), [](int i) -> float { return 2.0; });
  AddInput<float>(TensorShape({cols}), [](int i) -> float { return 1.0; });
  AddInput<float>(TensorShape({cols}), [](int i) -> float { return 0.4; });

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_output(allocator(), DT_FLOAT, TensorShape({rows, cols}));
    float output_array[rows * cols];
    for (int i = 0; i < rows * cols; i++) {
      output_array[i] = 1.4;
    }
    test::FillValues<float>(&expected_output, output_array);
    test::ExpectTensorNear<float>(expected_output, *GetOutput(0), 1e-5);
  }
}

//----------------------------------------------------------------------------//
// Performance benchmarks                                                     //
//----------------------------------------------------------------------------//
static Graph* Dice(int rows, int cols) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType dtype = DT_FLOAT;

  Tensor in(dtype, TensorShape({rows, cols}));
  in.flat<float>().setRandom();
  Tensor mean(dtype, TensorShape({cols}));
  mean.flat<float>().setRandom();
  Tensor rvar(dtype, TensorShape({cols}));
  rvar.flat<float>().setRandom();
  Tensor gamma(dtype, TensorShape({cols}));
  gamma.flat<float>().setRandom();

  Node* input_in = test::graph::Constant(g, in);
  Node* input_mean = test::graph::Constant(g, mean);
  Node* input_rvar = test::graph::Constant(g, rvar);
  Node* input_gamma = test::graph::Constant(g, gamma);
  auto nodeBuilder = NodeBuilder(g->NewName("dice"), "Dice")
                         .Input(input_in)
                         .Input(input_mean)
                         .Input(input_rvar)
                         .Input(input_gamma);
  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_DICE(ROWS, COLS, NTH)                                          \
  static void BM_DICE##_##ROWS##_##COLS##_##NTH##_CPU(int iters) {        \
    testing::UseRealTime();                                               \
    testing::ItemsProcessed(static_cast<int64>(iters) * ROWS * COLS * 3); \
    SessionOptions opts;                                                  \
    opts.config.set_intra_op_parallelism_threads(NTH);                    \
    test::Benchmark("cpu", Dice(ROWS, COLS), &opts).Run(iters);           \
  }                                                                       \
  BENCHMARK(BM_DICE##_##ROWS##_##COLS##_##NTH##_CPU);

#define BM_DICE_NTH(ROWS, COLS) \
  BM_DICE(ROWS, COLS, 1);       \
  BM_DICE(ROWS, COLS, 4);       \
  BM_DICE(ROWS, COLS, 8);

// BM_DICE_NTH(40, 600);
// BM_DICE_NTH(40, 400);
// BM_DICE_NTH(40, 300);
// BM_DICE_NTH(40, 200);
// BM_DICE_NTH(40, 100);
// BM_DICE_NTH(100, 600);
// BM_DICE_NTH(100, 400);
// BM_DICE_NTH(100, 300);
// BM_DICE_NTH(100, 200);
// BM_DICE_NTH(100, 100);
// BM_DICE_NTH(200, 600);
// BM_DICE_NTH(200, 400);
// BM_DICE_NTH(200, 300);
// BM_DICE_NTH(200, 200);
// BM_DICE_NTH(200, 100);
// BM_DICE_NTH(400, 600);
// BM_DICE_NTH(400, 400);
// BM_DICE_NTH(400, 300);
// BM_DICE_NTH(400, 200);
// BM_DICE_NTH(400, 100);
BM_DICE_NTH(500, 600);
// BM_DICE_NTH(500, 400);
// BM_DICE_NTH(500, 300);
// BM_DICE_NTH(500, 200);
// BM_DICE_NTH(500, 100);

}  // namespace
}  // namespace tensorflow

#endif  // AVX512F