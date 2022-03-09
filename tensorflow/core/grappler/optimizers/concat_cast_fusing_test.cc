/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/concat_cast_fusing.h"

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace grappler {
namespace {

class ConcatCastFusingTest : public GrapplerTest { };

TEST_F(ConcatCastFusingTest, SimpleFusing) {
    // Build a simple graph with a few trivially prunable ops.
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    Output a = ops::Const(s.WithOpName("a"), 1.0f, {1});
    Output b = ops::Const(s.WithOpName("b"), 2.0f, {1});
    auto ax = ops::Const(s.WithOpName("axis"), 0);
    Output c = ops::Concat(s.WithOpName("c").WithDevice("/CPU:0"), {a, b}, ax);
    Output d = ops::Cast(s.WithOpName("d"), c, DataType::DT_FLOAT);

    GrapplerItem item;
    item.fetch.push_back("d");
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    ConcatCastFusing optimizer(/*cpu_device=*/nullptr);
    GraphDef output;
    Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
    TF_EXPECT_OK(status);

  //EXPECT_EQ(1, output.node_size());
//
  //const NodeDef& node_d = output.node(0);
  //EXPECT_EQ("d", node_d.name());
  //EXPECT_EQ("Const", node_d.op());
//
    std::vector<string> fetch = {"d"};
    auto tensors_expected = EvaluateNodes(item.graph, fetch);
    auto tensors = EvaluateNodes(output, fetch);
  //EXPECT_EQ(1, tensors_expected.size());
  //EXPECT_EQ(1, tensors.size());
    //std::cout << tensors[0] << std::endl;
    test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
