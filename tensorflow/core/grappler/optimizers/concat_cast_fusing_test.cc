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
namespace ConcatCastFusingTestDefs {
    typedef std::tuple<
        std::vector<DataType>,          // src_type & dst_type
        long long int,                  // num_input_concat
        std::vector<long long int>,     // sizes
        long long int                   // axis
    > ConcatCastFusingTestParams;
    std::vector<std::vector<DataType>> dataTypes {
        {DataType::DT_FLOAT, DataType::DT_INT32}//,
        //{DataType::DT_BFLOAT16}
    };
    std::vector<long long int> numInputs = {2};//, 4};
    std::vector<long long int> AXIS_2D = {0};//, 1};
    std::vector<long long int> AXIS_3D = {0, 1, 2};
    std::vector<long long int> AXIS_4D = {0, 1, 2, 3};
    std::vector<std::vector<long long int>> SIZES_2D = {{1, 1}};//, {32, 21}, {64, 64}};
    std::vector<std::vector<long long int>> SIZES_3D = {{32, 16, 1}, {128, 128, 128}, {1, 1, 1}};
    std::vector<std::vector<long long int>> SIZES_4D = {{32, 32, 32, 32}, {16, 1, 1, 1}, {31, 63, 15, 7}};
} // namespace ConcatCastFusingTestDefs

using namespace ConcatCastFusingTestDefs;
class ConcatCastFusingTest :
    public ::testing::WithParamInterface<ConcatCastFusingTestDefs::ConcatCastFusingTestParams>,
    public GrapplerTest {
    public:
    static std::string getTestCaseName(::testing::TestParamInfo<ConcatCastFusingTestParams> obj) {
        DataType src_type;
        DataType dst_type;
        std::vector<DataType> dt;
        long long int num_inputs;
        std::vector<long long int> input_size;
        long long int ax;
        std::tie(dt, num_inputs, input_size, ax) = obj.param;
        src_type = dt[0];
        dst_type = dt[1];
        std::ostringstream result;
        result << "ConcatCastFusing_" << "_SrcType_";
        switch(src_type) {
            case DataType::DT_FLOAT:
                result << "FLOAT";
                break;
            case DataType::DT_BFLOAT16:
                result << "BFLOAT16";
                break;
            case DataType::DT_INT32:
                result << "INT32";
                break;
            default:
                result << "UNRECOGNISED_TYPE";
        }
        result << "_DstType_";
        switch(dst_type) {
            case DataType::DT_FLOAT:
                result << "FLOAT";
                break;
            case DataType::DT_BFLOAT16:
                result << "BFLOAT16";
                break;
            case DataType::DT_INT32:
                result << "INT32";
                break;
            default:
                result << "UNRECOGNISED_TYPE";
        }

        result << "_NumInputs_" << num_inputs;

        result << "_InputSizes";
        for (auto &x : input_size) {
            result << "_" << x;
        }

        result << "_Axis_" << ax;
        return result.str();
    }

    void SetUp() {
        std::vector<DataType> dt;
        std::tie(dt, num_inputs, input_size, ax) = this->GetParam();
        src_type = dt[0];
        dst_type = dt[1];

        inputs = {};
        for (uint i = 0; i < num_inputs; ++i){
            Tensor input = Tensor(src_type, TensorShape(tensorflow::gtl::ArraySlice<long long int>(input_size.data(), input_size.size())));
            switch(src_type) {
                case DT_FLOAT:
                    input.flat<float>() = input.flat<float>().template setRandom<Eigen::internal::NormalRandomGenerator<float>>(); // input
                    break;
                case DT_BFLOAT16:
                    input.flat<Eigen::bfloat16>() = input.flat<Eigen::bfloat16>().template setRandom<Eigen::internal::UniformRandomGenerator<Eigen::bfloat16>>(); // input
		            input.flat<Eigen::bfloat16>() = input.flat<Eigen::bfloat16>() - input.flat<Eigen::bfloat16>().constant((Eigen::bfloat16)0.5);
		            input.flat<Eigen::bfloat16>() = input.flat<Eigen::bfloat16>() * input.flat<Eigen::bfloat16>().constant((Eigen::bfloat16)200.0);
                    break;
                default:
                    GTEST_FAIL() << "Unexpected DataType";
            }
            inputs.push_back(input);
        }
        axis = Tensor((int32)ax);
    }

    protected:
    // Test definition (straight from Params, filled in SetUp)
    DataType src_type;
    DataType dst_type;
    long long int num_inputs;
    std::vector<long long int> input_size;
    long long int ax;
    // Test input Tensors (filled in SetUp)
    std::vector<Tensor> inputs;
    Tensor axis;
    // Test output Tensors (filled in Run method)
    Tensor values;
    Tensor default_values;
};

class ConcatCastFusingTestSimpleFusing : public ConcatCastFusingTest {
    public:
    void RunAndValidate() {
        tensorflow::Scope s = tensorflow::Scope::NewRootScope();

        std::vector<Input> in_values;
        for (int i = 0; i < num_inputs; ++i) {
            std::cout << "Shape " << i << ": " << inputs[i].shape() << std::endl;
            const string input_name = absl::StrCat("input_", i);
            auto tmp = ops::Const(s.WithOpName(input_name), Input::Initializer(inputs[i]));
            in_values.push_back(tmp);
        }
        auto a = ops::Const(s.WithOpName("axis"), axis);

        Output c = ops::Concat(s.WithOpName("concat").WithDevice("/CPU:0"), absl::Span<const Input>(in_values), a);
        Output d = ops::Cast(s.WithOpName("cast"), c, dst_type);

        GrapplerItem item;
        item.fetch.push_back("cast");
        TF_CHECK_OK(s.ToGraphDef(&item.graph));

        ConcatCastFusing optimizer(/*cpu_device=*/nullptr);
        GraphDef output;
        Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
        TF_EXPECT_OK(status);

        auto expected_graph_size = num_inputs + 1 /* axis */ + 1 /* FusedConcatCast */;
        EXPECT_EQ(expected_graph_size, output.node_size());
        for (int i = 0; i < output.node_size(); ++i) {
            const NodeDef& node = output.node(i);
            const string& name = node.name();
            if (name == "cast") {
                EXPECT_EQ("_FusedConcatCast", node.op());
                for (int i = 0; i < num_inputs; ++i) {
                    EXPECT_EQ(absl::StrCat("input_", i), node.input(i));
                }
                EXPECT_EQ("axis", node.input(num_inputs));
            }
        }

        std::vector<string> fetch = {"cast"};
        auto tensors_expected = EvaluateNodes(item.graph, fetch);
        auto tensors = EvaluateNodes(output, fetch);
        EXPECT_EQ(1, tensors_expected.size());
        EXPECT_EQ(1, tensors.size());
        test::ExpectTensorEqual<int>(tensors_expected[0], tensors[0]);
    }
};

TEST_P(ConcatCastFusingTestSimpleFusing, CompareWithRefs) {
    SetUp();
    RunAndValidate();
};

// TEST_F(ConcatCastFusingTest, SimpleFusingNotLast) {
//     tensorflow::Scope s = tensorflow::Scope::NewRootScope();

//     Output a = ops::Const(s.WithOpName("a"), 1.0f, {1});
//     Output b = ops::Const(s.WithOpName("b"), 2.0f, {1});
//     auto ax = ops::Const(s.WithOpName("axis"), 0);
//     Output c = ops::Concat(s.WithOpName("c").WithDevice("/CPU:0"), {a, b}, ax);
//     Output d = ops::Cast(s.WithOpName("d"), c, DataType::DT_INT32);
//     Output e = ops::Const(s.WithOpName("e"), {2, 2}, {1});
//     Output f = ops::Add(s.WithOpName("f"), d, e);

//     GrapplerItem item;
//     item.fetch.push_back("f");
//     TF_CHECK_OK(s.ToGraphDef(&item.graph));

//     ConcatCastFusing optimizer(/*cpu_device=*/nullptr);
//     GraphDef output;
//     Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
//     TF_EXPECT_OK(status);

//     const NodeDef& node_d = output.node(0);
//     EXPECT_EQ("f", node_d.name());
//     EXPECT_EQ("Add", node_d.op());

//     std::vector<string> fetch = {"f"};
//     auto tensors_expected = EvaluateNodes(item.graph, fetch);
//     auto tensors = EvaluateNodes(output, fetch);
//     EXPECT_EQ(1, tensors_expected.size());
//     EXPECT_EQ(1, tensors.size());
//     test::ExpectTensorEqual<int>(tensors_expected[0], tensors[0]);
// }

INSTANTIATE_TEST_CASE_P(Concat2D, ConcatCastFusingTestSimpleFusing,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(numInputs),
        ::testing::ValuesIn(SIZES_2D),
        ::testing::ValuesIn(AXIS_2D)),
    ConcatCastFusingTestSimpleFusing::getTestCaseName);

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
