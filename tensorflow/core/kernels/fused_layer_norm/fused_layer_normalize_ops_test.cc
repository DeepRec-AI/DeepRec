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
#include "tensorflow/cc/ops/standard_ops.h"

namespace tensorflow {
namespace {

enum class Device { CPU, GPU };

class FusedLayerNormalizeOpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, DataType dtype, int axis, float epsilon) {
    TF_EXPECT_OK(NodeDefBuilder("fused_layer_normalize", "FusedLayerNorm")
                     .Attr("T", dtype)
                     .Attr("epsilon", epsilon)
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(FusedLayerNormalizeOpTest, 2Dims_Float) {
  const int rows = 7;
  const int cols = 255;

  MakeOpAndSetDevice(Device::CPU, DT_FLOAT, 0, 1e-12);

  float input_array[1785];
  for (int i = 0; i < sizeof(input_array) / sizeof(float); i++) {
    input_array[i] = 1.0;
  }
  for (int i = 0; i < rows; i++) {
    input_array[i * cols] = 2.0;
  }
  AddInputFromArray<float>(TensorShape({rows, cols}), input_array);
  AddInput<float>(TensorShape({cols}), [](int i) -> float { return 2.0; });
  AddInput<float>(TensorShape({cols}), [](int i) -> float { return 1.0; });

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_output(allocator(), DT_FLOAT,
                                TensorShape({rows, cols}));
    Tensor mean(allocator(), DT_FLOAT, TensorShape({rows}));
    Tensor rvariance(allocator(), DT_FLOAT, TensorShape({rows}));
    float output_array[1785];
    float rvar_value = 16.000125885009766f;
    float mean_value = 256.0f / 255.0f; 
    //1.00392162799835205f;
    for (int i = 0; i < sizeof(output_array) / sizeof(float); i++) {
      output_array[i] = 0.87450695037841797;
    }
    for (int i = 0; i < rows; i++) {
      output_array[i * cols] = 2.0f * sqrtf(254.0f) + 1.0f; 
      // 32.874755859375;
    }

    float mean_array[rows];
    for (int i = 0; i < sizeof(mean_array) / sizeof(float); i++) {
      mean_array[i] = mean_value;
    }

    float rvariance_array[rows];
    
    for (int i = 0; i < sizeof(rvariance_array) / sizeof(float); i++) {
      rvariance_array[i] = rvar_value;
    }
    test::FillValues<float>(&expected_output, output_array);
    test::FillValues<float>(&mean, mean_array);
    test::FillValues<float>(&rvariance, rvariance_array);
    test::ExpectTensorNear<float>(expected_output, *GetOutput(0), 1e-5);
    test::ExpectTensorNear<float>(mean, *GetOutput(1), 1e-5);
    test::ExpectTensorNear<float>(rvariance, *GetOutput(2), 1e-5);
  }
}

class FusedLayerNormalizeGradOpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, DataType dtype, int axis, float epsilon) {
    TF_EXPECT_OK(NodeDefBuilder("fused_layer_normalize_grad", "FusedLayerNormGrad")
                     .Attr("T", dtype)
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(FusedLayerNormalizeGradOpTest, 2Dims_Float) {
  const int rows = 7;
  const int cols = 255;

  MakeOpAndSetDevice(Device::CPU, DT_FLOAT, 0, 1e-12);

  AddInput<float>(TensorShape({rows, cols}), [](int i) -> float { return 1.0f; }); // y_grad
  AddInput<float>(TensorShape({rows, cols}), [](int i) -> float { return (i % cols) ? 1.0f : 2.0f; }); // x
  AddInput<float>(TensorShape({rows}), [](int i) -> float { return 256.0f / 255.0f; }); // mean
  AddInput<float>(TensorShape({rows}), [](int i) -> float { return 16.00012302493275484; }); // rvariance
  AddInput<float>(TensorShape({cols}), [](int i) -> float { return 2.0f; }); // gamma

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_output(allocator(), DT_FLOAT,
                                TensorShape({rows, cols}));
    Tensor gamma_grad(allocator(), DT_FLOAT, TensorShape({cols}));
    Tensor beta_grad(allocator(), DT_FLOAT, TensorShape({cols}));
    float x_grad[1785];
    for (int i = 0; i < sizeof(x_grad) / sizeof(float); i++) {
      x_grad[i] = 0.0f;
    }
    for (int i = 0; i < rows; i++){
      x_grad[i * cols] = 0.00048447030712850392f;
    }

    float gamma_grads[cols];
    for (int i = 0; i < sizeof(gamma_grads) / sizeof(float); i++) {
      gamma_grads[i] = -0.4392257034778595;
    }
    gamma_grads[0] = 111.56163787841797;

    float beta_grads[cols];
    for (int i = 0; i < sizeof(beta_grads) / sizeof(float); i++) {
      beta_grads[i] = 7.0f;
    }
    test::FillValues<float>(&expected_output, x_grad);
    test::FillValues<float>(&gamma_grad, gamma_grads);
    test::FillValues<float>(&beta_grad, beta_grads);
    test::ExpectTensorNear<float>(expected_output, *GetOutput(0), 1e-5);
    test::ExpectTensorNear<float>(gamma_grad, *GetOutput(1), 1e-5);
    test::ExpectTensorNear<float>(beta_grad, *GetOutput(2), 1e-5);
  }
}


//----------------------------------------------------------------------------//
// Performance benchmarks                                                     //
//----------------------------------------------------------------------------//
static Graph* FusedLayerNormalize(int rows, int cols) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType dtype = DT_FLOAT;

  Tensor in(dtype, TensorShape({rows, cols}));
  in.flat<float>().setRandom();
  Tensor gamma(dtype, TensorShape({cols}));
  gamma.flat<float>().setRandom();
  Tensor beta(dtype, TensorShape({cols}));
  beta.flat<float>().setRandom();

  Node* input_in = test::graph::Constant(g, in);
  Node* input_gamma = test::graph::Constant(g, gamma);
  Node* input_beta = test::graph::Constant(g, beta);
  auto nodeBuilder = NodeBuilder(g->NewName("n"), "FusedLayerNorm")
                    .Input(input_in)
                    .Input(input_gamma)
                    .Input(input_beta)
                    .Attr("T", dtype)
                    .Attr("epsilon", 1e-12);
  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_FusedLayerNorm(ROWS, COLS, NTH)                                   \
  static void BM_FusedLayerNorm##_##ROWS##_##COLS##_##NTH##_CPU(             \
  int iters) {                                                               \
  testing::UseRealTime();                                                    \
  testing::ItemsProcessed(static_cast<int64>(iters) * ROWS * COLS * 3);      \
  SessionOptions opts;                                                       \
  opts.config.set_intra_op_parallelism_threads(NTH);                         \
  test::Benchmark("cpu", FusedLayerNormalize(ROWS, COLS), &opts).Run(iters); \
  }                                                                          \
  BENCHMARK(BM_FusedLayerNorm##_##ROWS##_##COLS##_##NTH##_CPU);              \

#define BM_FusedLayerNorm_NTH(ROWS, COLS) \
  BM_FusedLayerNorm(ROWS, COLS, 1);       \
  BM_FusedLayerNorm(ROWS, COLS, 4);       \
  BM_FusedLayerNorm(ROWS, COLS, 8);       \

BM_FusedLayerNorm_NTH(1024, 63);
BM_FusedLayerNorm_NTH(1024, 255);
BM_FusedLayerNorm_NTH(1024, 511);
BM_FusedLayerNorm_NTH(1024, 1023);
BM_FusedLayerNorm_NTH(1024, 1024);
BM_FusedLayerNorm_NTH(1024, 2048);
BM_FusedLayerNorm_NTH(1024, 4096);

}

static Graph* FusedLayerNormalizeGrad(int rows, int cols) {
  Graph* g = new Graph(OpRegistry::Global());
  DataType dtype = DT_FLOAT;

  Tensor y_grad(dtype, TensorShape({rows, cols}));
  y_grad.flat<float>().setRandom();
  Tensor x(dtype, TensorShape({rows, cols}));
  x.flat<float>().setRandom();
  Tensor mean(dtype, TensorShape({rows}));
  mean.flat<float>().setRandom();
  Tensor rvarance(dtype, TensorShape({rows}));
  rvarance.flat<float>().setRandom();
  Tensor gamma(dtype, TensorShape({cols}));
  gamma.flat<float>().setRandom();

  Node* input_y_grad = test::graph::Constant(g, y_grad);
  Node* input_x = test::graph::Constant(g, x);
  Node* input_mean = test::graph::Constant(g, mean);
  Node* input_rvarance = test::graph::Constant(g, rvarance);
  Node* input_gamma = test::graph::Constant(g, gamma);
  auto nodeBuilder = NodeBuilder(g->NewName("n"), "FusedLayerNormGrad")
                    .Input(input_y_grad)
                    .Input(input_x)
                    .Input(input_mean)
                    .Input(input_rvarance)
                    .Input(input_gamma)
                    .Attr("T", dtype);
  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));

  return g;
}

#define BM_FusedLayerNormGrad(ROWS, COLS, NTH)                                    \
  static void BM_FusedLayerNormGrad##_##ROWS##_##COLS##_##NTH##_CPU(              \
  int iters) {                                                                    \
  testing::UseRealTime();                                                         \
  testing::ItemsProcessed(static_cast<int64>(iters) * ROWS * COLS * 3);           \
  SessionOptions opts;                                                            \
  opts.config.set_intra_op_parallelism_threads(NTH);                              \
  test::Benchmark("cpu", FusedLayerNormalizeGrad(ROWS, COLS), &opts).Run(iters);  \
  }                                                                               \
  BENCHMARK(BM_FusedLayerNormGrad##_##ROWS##_##COLS##_##NTH##_CPU);               \

#define BM_FusedLayerNormGrad_NTH(ROWS, COLS) \
  BM_FusedLayerNormGrad(ROWS, COLS, 1);       \
  BM_FusedLayerNormGrad(ROWS, COLS, 4);       \
  BM_FusedLayerNormGrad(ROWS, COLS, 8);       \

BM_FusedLayerNormGrad_NTH(1024, 63);
BM_FusedLayerNormGrad_NTH(1024, 255);
BM_FusedLayerNormGrad_NTH(1024, 511);
BM_FusedLayerNormGrad_NTH(1024, 1023);
BM_FusedLayerNormGrad_NTH(1024, 1024);
BM_FusedLayerNormGrad_NTH(1024, 2048);
BM_FusedLayerNormGrad_NTH(1024, 4096);

}

