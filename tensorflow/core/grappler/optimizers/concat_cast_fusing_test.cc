/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/public/session.h"

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
        {DataType::DT_FLOAT, DataType::DT_INT32},
        //{DataType::DT_INT32, DataType::DT_FLOAT},
        //{DataType::DT_BFLOAT16, DataType::DT_FLOAT},
        //{DataType::DT_FLOAT, DataType::DT_BFLOAT16},
        //{DataType::DT_BFLOAT16, DataType::DT_INT32},
        //{DataType::DT_INT32, DataType::DT_BFLOAT16}
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
        result << "ConcatCastFusing_SrcType_";
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

    void SetUp(string input_type) {
        std::vector<DataType> dt;
        std::tie(dt, num_inputs, input_size, ax) = this->GetParam();
        src_type = dt[0];
        dst_type = dt[1];

        std::vector<string> input_names;
        GraphDef ref_graph;
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
                case DT_INT32:
                    input.flat<int32_t>() = input.flat<int32_t>().template setRandom<Eigen::internal::NormalRandomGenerator<int32_t>>(); // input
                    break;
                default:
                    GTEST_FAIL() << "Unexpected DataType";
            }
            inputs.push_back(input);
            const string input_name = absl::StrCat("input_", i);
            input_names.push_back(input_name);
            AddNode(input_name, input_type, {}, {}, &ref_graph);
        }
        axis = Tensor((int32)ax);
        AddNode("axis", "Const", {}, {}, &ref_graph);
        input_names.push_back("axis");
        AddNode("cast", "_FusedConcatCast", input_names, {}, &ref_graph);
        want = ref_graph;
    }

    protected:
    void Validate(std::vector<Tensor> tensors, std::vector<Tensor> tensors_expected) {
        EXPECT_EQ(1, tensors_expected.size());
        EXPECT_EQ(1, tensors.size());
        EXPECT_EQ(dst_type, tensors_expected[0].dtype());
        EXPECT_EQ(dst_type, tensors[0].dtype());
        switch(dst_type) {
            case DT_FLOAT:
                test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
                break;
            case DT_BFLOAT16:
                test::ExpectTensorEqual<Eigen::bfloat16>(tensors_expected[0], tensors[0]);
                break;
            case DT_INT32:
                test::ExpectTensorEqual<int32_t>(tensors_expected[0], tensors[0]);
                break;
            default:
                GTEST_FAIL() << "Unexpected DataType";
        }
    }

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

    GraphDef want;
};

class ConcatCastFusingTestSimpleFusing : public ConcatCastFusingTest {
    public:
    void RunAndValidate() {
        tensorflow::Scope s = tensorflow::Scope::NewRootScope();

        std::vector<Input> in_values;
        for (int i = 0; i < num_inputs; ++i) {
            const string input_name = absl::StrCat("input_", i);
            auto tmp = ops::Const(s.WithOpName(input_name), Input::Initializer(inputs[i]));
            in_values.push_back(tmp);
        }
        auto a = ops::Const(s.WithOpName("axis"), axis);

        Output c = ops::Concat(s.WithOpName("concat"), absl::Span<const Input>(in_values), a);
        Output d = ops::Cast(s.WithOpName("cast"), c, dst_type);

        GrapplerItem item;
        item.fetch.push_back("cast");
        TF_CHECK_OK(s.ToGraphDef(&item.graph));

        ConcatCastFusing optimizer;
        GraphDef output;
        Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
        TF_EXPECT_OK(status);

        CompareGraphs(want, output);

        std::vector<string> fetch = {"cast"};
        auto tensors_expected = EvaluateNodes(item.graph, fetch);
        auto tensors = EvaluateNodes(output, fetch);
        Validate(tensors, tensors_expected);
    }
};

class ConcatCastFusingTestMultithreaded : public ConcatCastFusingTest {
    public:
    void RunAndValidate() {
        tensorflow::Scope s = tensorflow::Scope::NewRootScope();

        std::vector<Output> in_values;
        TensorShape shape(input_size);
        for (int i = 0; i < num_inputs; ++i) {
            const string input_name = absl::StrCat("input_", i);
            auto tmp = ops::Placeholder(s.WithOpName(input_name), src_type, ops::Placeholder::Shape(shape));
            in_values.push_back(tmp);
        }
        auto a = ops::Const(s.WithOpName("axis"), axis);

        Output c = ops::Concat(s.WithOpName("concat"), in_values, a);
        Output d = ops::Cast(s.WithOpName("cast"), c, dst_type);

        ClientSession::FeedType feed_list;
        for (int i = 0; i < num_inputs; i++) {
            feed_list.insert({in_values[i], inputs[i]});
        }

        GrapplerItem item;
        item.fetch.push_back("cast");
        TF_CHECK_OK(s.ToGraphDef(&item.graph));
        ConcatCastFusing optimizer;
        GraphDef output;
        Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
        TF_EXPECT_OK(status);

        CompareGraphs(want, output);

        tensorflow::SessionOptions session_options_;
        session_options_.config.set_intra_op_parallelism_threads(8);
        session_options_.config.set_inter_op_parallelism_threads(1);

        tensorflow::RewriterConfig* cfg = session_options_.config.mutable_graph_options()->mutable_rewrite_options();
        cfg->set_constant_folding(tensorflow::RewriterConfig::OFF);
        cfg->set_layout_optimizer(tensorflow::RewriterConfig::OFF);
        cfg->set_remapping(tensorflow::RewriterConfig::OFF);
        tensorflow::ClientSession session(s, session_options_);
        std::vector<Tensor> tensors;
        RunOptions run_options;
        TF_ASSERT_OK(session.Run(feed_list, {d}, &tensors));

        // Create expected graph
        std::vector<Input> in_values_expected;
        for (int i = 0; i < num_inputs; ++i) {
            const string input_name = absl::StrCat("expected_input_", i);
            auto tmp = ops::Const(s.WithOpName(input_name), Input::Initializer(inputs[i]));
            in_values_expected.push_back(tmp);
        }
        auto a_expected = ops::Const(s.WithOpName("expected_axis"), axis);

        Output c_expected = ops::Concat(s.WithOpName("expected_concat"), absl::Span<const Input>(in_values_expected), a_expected);
        Output d_expected = ops::Cast(s.WithOpName("expected_cast"), c_expected, dst_type);

        GrapplerItem item_expected;
        item_expected.fetch.push_back("expected_cast");
        TF_CHECK_OK(s.ToGraphDef(&item_expected.graph));
        std::vector<string> fetch = {"expected_cast"};
        auto tensors_expected = EvaluateNodes(item_expected.graph, fetch);

        Validate(tensors, tensors_expected);
    }
};

TEST_P(ConcatCastFusingTestSimpleFusing, CompareWithRefs) {
    SetUp("Const");
    RunAndValidate();
};

TEST_P(ConcatCastFusingTestMultithreaded, CompareWithRefs) {
    SetUp("Placeholder");
    RunAndValidate();
};

// INSTANTIATE_TEST_CASE_P(Concat2D, ConcatCastFusingTestSimpleFusing,
//     ::testing::Combine(
//         ::testing::ValuesIn(dataTypes),
//         ::testing::ValuesIn(numInputs),
//         ::testing::ValuesIn(SIZES_2D),
//         ::testing::ValuesIn(AXIS_2D)),
//     ConcatCastFusingTestSimpleFusing::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Concat2D, ConcatCastFusingTestMultithreaded,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(numInputs),
        ::testing::ValuesIn(SIZES_2D),
        ::testing::ValuesIn(AXIS_2D)),
    ConcatCastFusingTestSimpleFusing::getTestCaseName);

// INSTANTIATE_TEST_CASE_P(Concat3D, ConcatCastFusingTestSimpleFusing,
//     ::testing::Combine(
//         ::testing::ValuesIn(dataTypes),
//         ::testing::ValuesIn(numInputs),
//         ::testing::ValuesIn(SIZES_3D),
//         ::testing::ValuesIn(AXIS_3D)),
//     ConcatCastFusingTestSimpleFusing::getTestCaseName);

// INSTANTIATE_TEST_CASE_P(Concat4D, ConcatCastFusingTestSimpleFusing,
//     ::testing::Combine(
//         ::testing::ValuesIn(dataTypes),
//         ::testing::ValuesIn(numInputs),
//         ::testing::ValuesIn(SIZES_4D),
//         ::testing::ValuesIn(AXIS_4D)),
//     ConcatCastFusingTestSimpleFusing::getTestCaseName);

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
