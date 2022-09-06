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
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {
namespace {
//----------------------------------------------------------------------------//
// Concat+Cast Functional Tests are below.                                    //
//----------------------------------------------------------------------------//
namespace ConcatCastFusingTestDefs {
    typedef std::tuple<
        std::vector<DataType>,          // src_type & dst_type
        long long int,                  // num_input_concat
        std::vector<long long int>,     // sizes
        long long int                   // axis
    > ConcatCastFusingTestParams;
    std::vector<std::vector<DataType>> dataTypes {
        {DataType::DT_BFLOAT16, DataType::DT_FLOAT},
        {DataType::DT_FLOAT, DataType::DT_BFLOAT16}
    };
    std::vector<long long int> numInputs = {2, 4};
    std::vector<long long int> AXIS_2D = {0, 1, -1};
    std::vector<long long int> AXIS_3D = {-1, 0, 1, 2};
    std::vector<long long int> AXIS_4D = {0, 1, 2, 3};
    std::vector<std::vector<long long int>> SIZES_2D = {{1, 1}, {32, 21}, {64, 64}};
    std::vector<std::vector<long long int>> MULTITHREADED_SIZES_2D = {{10000, 4}};
    std::vector<std::vector<long long int>> SIZES_3D = {{32, 16, 1}, {128, 128, 128}, {1, 1, 1}};
    std::vector<std::vector<long long int>> MULTITHREADED_SIZES_3D = {{7, 10000, 4}};
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

        if (ax < 0)
            result << "_Axis_negative_" << abs(ax);
        else
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
        AddNode("cast", "FusedConcatCast", input_names, {}, &ref_graph);
        want = ref_graph;

        for (auto& node : *want.mutable_node()) {
            node.set_device("/cpu:0");
        }
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

        for (auto& node : *item.graph.mutable_node()) {
            node.set_device("/cpu:0");
        }

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
        for (auto& node : *item.graph.mutable_node()) {
            node.set_device("/cpu:0");
        }
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

        for (auto& node : *item_expected.graph.mutable_node()) {
            node.set_device("/cpu:0");
        }

        tensorflow::SessionOptions expected_session_options_;
        tensorflow::RewriterConfig* expected_cfg = expected_session_options_.config.mutable_graph_options()->mutable_rewrite_options();
        cfg->set_disable_meta_optimizer(tensorflow::RewriterConfig::OFF);
        std::unique_ptr<Session> expected_session(NewSession(expected_session_options_));
        TF_CHECK_OK(expected_session->Create(item_expected.graph));
        std::vector<Tensor> tensors_expected;
        RunOptions run_options;
        TF_CHECK_OK(expected_session->Run(run_options, {}, fetch, fetch, &tensors_expected, nullptr));
        TF_CHECK_OK(expected_session->Close());

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

INSTANTIATE_TEST_CASE_P(Concat2D, ConcatCastFusingTestSimpleFusing,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(numInputs),
        ::testing::ValuesIn(SIZES_2D),
        ::testing::ValuesIn(AXIS_2D)),
    ConcatCastFusingTestSimpleFusing::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Concat3D, ConcatCastFusingTestSimpleFusing,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(numInputs),
        ::testing::ValuesIn(SIZES_3D),
        ::testing::ValuesIn(AXIS_3D)),
    ConcatCastFusingTestSimpleFusing::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Concat4D, ConcatCastFusingTestSimpleFusing,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(numInputs),
        ::testing::ValuesIn(SIZES_4D),
        ::testing::ValuesIn(AXIS_4D)),
    ConcatCastFusingTestSimpleFusing::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Concat2D, ConcatCastFusingTestMultithreaded,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(numInputs),
        ::testing::ValuesIn(MULTITHREADED_SIZES_2D),
        ::testing::ValuesIn(AXIS_2D)),
    ConcatCastFusingTestSimpleFusing::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Concat3D, ConcatCastFusingTestMultithreaded,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(numInputs),
        ::testing::ValuesIn(MULTITHREADED_SIZES_3D),
        ::testing::ValuesIn(AXIS_3D)),
    ConcatCastFusingTestSimpleFusing::getTestCaseName);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//
template <typename SrcT, typename DstT>
static Graph* ConcatCastFusion(bool if_fused, int num_inputs,
                               int axis, std::vector<long long int> input_shape) {
    Graph* g = new Graph(OpRegistry::Global());
    DataType source_dt = DataTypeToEnum<SrcT>::v();
    DataType dst_dt = DataTypeToEnum<DstT>::v();
    Tensor concat_dim(DT_INT32, TensorShape({}));
    concat_dim.scalar<int32>()() = axis;
    std::vector<NodeBuilder::NodeOut> inputs;
    inputs.reserve(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
        Tensor in(source_dt, TensorShape(tensorflow::gtl::ArraySlice<long long int>(input_shape.data(), input_shape.size())));
        in.flat<SrcT>().setRandom();
        inputs.push_back(test::graph::Constant(g, in));
    }

    if (if_fused) {
        Node* concat_cast;
        TF_CHECK_OK(NodeBuilder(g->NewName("concatcast"), "FusedConcatCast")
                      .Input(inputs)
                      .Input(test::graph::Constant(g, concat_dim))
                      .Attr("N", num_inputs)
                      .Attr("SrcT", source_dt)
                      .Attr("DstT", dst_dt)
                      .Attr("Truncate", false)
                      .Finalize(g, &concat_cast));

        return g;
    } else {
        Node* concat;
        TF_CHECK_OK(NodeBuilder(g->NewName("concat"), "Concat")
                      .Input(test::graph::Constant(g, concat_dim))
                      .Input(inputs)
                      .Attr("N", num_inputs)
                      .Attr("T", source_dt)
                      .Finalize(g, &concat));

        Node* cast;
        TF_CHECK_OK(NodeBuilder(g->NewName("cast"), "Cast")
                      .Input(concat)
                      .Attr("SrcT", source_dt)
                      .Attr("DstT", dst_dt)
                      .Attr("Truncate", false)
                      .Finalize(g, &cast));

        return g;
    }
}

using fp32 = float;
using bfloat16 = Eigen::bfloat16;

#define BM_NAME(name, IF_FUSED, SRCT, DSTT, NUM_INPUTS, axis_name, input_shape_name)    \
    name##_##IF_FUSED##_##SRCT##_##DSTT##_##NUM_INPUTS##_##axis_name##_##input_shape_name

#define BM_ConcatCastFusionBench(IF_FUSED, NUM_INPUTS, axis_name, AXIS, SRCT, DSTT, input_shape_name, INPUT_SHAPE, LABEL)                                                                                                                 \
    static void BM_NAME(BM_ConcatCastFusion, IF_FUSED, SRCT, DSTT, NUM_INPUTS, axis_name, input_shape_name)(int iters) {          \
      testing::StopTiming();                                                                                                      \
      std::string base_label = IF_FUSED ? "fused_concat_cast" : "nonfused_concat_cast";                                           \
      size_t input_shape_size = INPUT_SHAPE.size();                                                                               \
      int all_elems = 1;                                                                                                          \
      for (int i = 0; i < input_shape_size; ++i) {                                                                                \
          all_elems *= INPUT_SHAPE[i];                                                                                            \
      }                                                                                                                           \
      testing::SetLabel(base_label + "_" + LABEL);                                                                                \
      testing::BytesProcessed(static_cast<int64_t>(iters) * all_elems * NUM_INPUTS * sizeof(DSTT));                               \
      testing::StartTiming();                                                                                                     \
      test::Benchmark("cpu", ConcatCastFusion<SRCT, DSTT>(IF_FUSED, NUM_INPUTS, AXIS, INPUT_SHAPE))                               \
          .Run(iters);                                                                                                            \
    }                                                                                                                             \
    BENCHMARK(BM_NAME(BM_ConcatCastFusion, IF_FUSED, SRCT, DSTT, NUM_INPUTS, axis_name, input_shape_name));

#define DLRM_BENCHMARK(IF_FUSED)                                                                                   \
    /* DLRM concat + cast operations */                                                                            \
    BM_ConcatCastFusionBench(IF_FUSED, /*num_inputs*/ 2, /*axis_name*/ 1, /*axis*/ 1, /*Source T*/ fp32,           \
                             /*Dst T*/ bfloat16, /*Input shape name*/ 512x16,                                      \
                             /*Input shape*/ (std::vector<long long int>{512, 16}),                                \
                             "2_1_fp32_bfloat16_512x16");                                                          \
                                                                                                                   \
    BM_ConcatCastFusionBench(IF_FUSED, /*num_inputs*/ 13, /*axis name*/ neg_1, /*axis*/ (-1), /*Source T*/ fp32,   \
                             /*Dst T*/ bfloat16, /*Input shape name*/ 512x1,                                       \
                             /*Input shape*/ (std::vector<long long int>{512, 1}),                                 \
                             "13_-1_fp32_bfloat16_512x1");                                                         \
                                                                                                                   \
    BM_ConcatCastFusionBench(IF_FUSED, /*num_inputs*/ 13, /*axis_name*/ neg_1, /*axis*/ (-1), /*Source T*/ fp32,   \
                             /*Dst T*/ bfloat16, /*Input shape name*/ 128x1,                                       \
                             /*Input shape*/ (std::vector<long long int>{128, 1}),                                 \
                             "13_-1_fp32_bfloat16_128x1");

#define OTHER_BENCHMARK(IF_FUSED)                                                                                  \
    BM_ConcatCastFusionBench(IF_FUSED, /*num_inputs*/ 16, /*axis_name*/ 2, /*axis*/ 2, /*Source T*/ fp32,          \
                             /*Dst T*/ bfloat16, /*Input shape name*/ 512x32x16,                                   \
                             /*Input shape*/ (std::vector<long long int>{512, 32, 16}),                            \
                             "16_2_fp32_bfloat16_512x32x16");                                                      \
                                                                                                                   \
    BM_ConcatCastFusionBench(IF_FUSED, /*num_inputs*/ 64, /*axis_name*/ 1, /*axis*/ 1, /*Source T*/ bfloat16,      \
                             /*Dst T*/ fp32, /*Input shape name*/ 400x60,                                          \
                             /*Input shape*/ (std::vector<long long int>{400, 60}),                                \
                             "64_1_bfloat16_fp32_400x60");


DLRM_BENCHMARK(/*IFFUSED=*/true)
DLRM_BENCHMARK(/*IFFUSED=*/false)

OTHER_BENCHMARK(/*IFFUSED=*/true)
OTHER_BENCHMARK(/*IFFUSED=*/false)

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
