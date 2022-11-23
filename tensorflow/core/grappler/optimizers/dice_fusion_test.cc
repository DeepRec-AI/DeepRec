#include "tensorflow/core/grappler/optimizers/dice_fusion.h"

#include "absl/algorithm/container.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace tensorflow {
namespace grappler {

class DiceFusionTest : public GrapplerTest {};

TEST_F(DiceFusionTest, DiceFusionOp) {
  using test::function::NDef;

  DiceFusion optimizer;

  const Tensor input = test::AsTensor<float>({1.1, 0.9, 1.2, 0.8}, {2, 2});
  const Tensor sub_1_in = test::AsTensor<float>({1.2, 0.8});
  const Tensor sub_2_in = test::AsTensor<float>({1.0});
  const Tensor mul_1_in = test::AsTensor<float>({1.2, 0.8});
  const Tensor mul_3_in = test::AsTensor<float>({1.2, 0.8});

  const auto scalar = PartialTensorShape({-1, 2});

  GrapplerItem item;
  item.graph = test::function::GDef({

      NDef("input", "Const", {}, {{"T", DT_FLOAT}, {"value", input}}),
      NDef("sub_1_in", "Const", {}, {{"T", DT_FLOAT}, {"value", sub_1_in}}),
      NDef("sub_2_in", "Const", {}, {{"T", DT_FLOAT}, {"value", sub_2_in}}),
      NDef("mul_1_in", "Const", {}, {{"T", DT_FLOAT}, {"value", mul_1_in}}),
      NDef("mul_3_in", "Const", {}, {{"T", DT_FLOAT}, {"value", mul_3_in}}),

      NDef("sub_1", "Sub", {"input", "sub_1_in"},
           {{"T", DT_FLOAT}, {"_output_shapes", scalar}}),
      NDef("mul_1", "Mul", {"sub_1", "mul_1_in"},
           {{"T", DT_FLOAT}, {"_output_shapes", scalar}}),
      NDef("sigmoid", "Sigmoid", {"mul_1"},
           {{"T", DT_FLOAT}, {"_output_shapes", scalar}}),
      NDef("sub_2", "Sub", {"sigmoid", "sub_2_in"},
           {{"T", DT_FLOAT}, {"_output_shapes", scalar}}),
      NDef("mul_2", "Mul", {"sub_2", "input"},
           {{"T", DT_FLOAT}, {"_output_shapes", scalar}}),
      NDef("mul_3", "Mul", {"mul_2", "mul_3_in"},
           {{"T", DT_FLOAT}, {"_output_shapes", scalar}}),
      NDef("mul_4", "Mul", {"sigmoid", "input"},
           {{"T", DT_FLOAT}, {"_output_shapes", scalar}}),
      NDef("add", "Add", {"mul_3", "mul_4"},
           {{"T", DT_FLOAT}, {"_output_shapes", scalar}}),
  });

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(/*cluster=*/nullptr, item, &output));

  GraphDef expected = test::function::GDef({
      NDef("input", "Const", {}, {{"T", DT_FLOAT}, {"value", input}}),
      NDef("sub_1_in", "Const", {}, {{"T", DT_FLOAT}, {"value", sub_1_in}}),
      NDef("mul_1_in", "Const", {}, {{"T", DT_FLOAT}, {"value", mul_1_in}}),
      NDef("mul_3_in", "Const", {}, {{"T", DT_FLOAT}, {"value", mul_3_in}}),

      // fusing node.
      NDef("add", "Dice",
           {"input", "sub_1_in", "mul_1_in", "mul_3_in"},
           {{"T", DT_FLOAT}}),
  });

  CompareGraphs(expected, output);
}

}  // namespace grappler
}  // namespace tensorflow