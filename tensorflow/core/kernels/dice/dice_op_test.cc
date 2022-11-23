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

}  // namespace
}  // namespace tensorflow