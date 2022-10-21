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

#include "tensorflow/core/grappler/optimizers/split_concat_fuse.h"

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
// Functional tests are below.                                                //
//----------------------------------------------------------------------------//

namespace SplitConcatFuseTestDefs{
    typedef std::tuple<
        DataType,
        std::vector<long long int>, // Input tensor size
        std::vector<long long int>,
        long long int             // num_of_splits
    > SplitConcatFuseTestParams;
    std::vector<DataType> dataTypes{
        DataType::DT_FLOAT
    };
    std::vector<std::vector<long long int>> SCRIPT_SIZES = {{1024, 50, 256}};
    std::vector<std::vector<long long int>> SCRIPT_SPLIT_CONCAT = {{2, 0}};
    std::vector<long long int> SCRIPT_NUM_OF_SPLITS = {8};

    std::vector<std::vector<long long int>> SIZES_2D = {{216, 192}, {6, 36}, {48, 144}, {216, 36}};
    std::vector<std::vector<long long int>> SPLIT_DIM_CONCAT_AXIS_2D = {{0, 1}, {1, 0}, {1, 1}, {0, 0},
                                                                        {-1, -1}, {-1, 0}, {0, -1}};
    std::vector<long long int> NUM_OF_SPLITS_2D = {2, 3};

    std::vector<std::vector<long long int>> SIZES_3D = {{36, 6, 48}, {48, 48, 48}, {18, 216, 6}};
    std::vector<std::vector<long long int>> SPLIT_DIM_CONCAT_AXIS_3D = {{0, 0}, {0, 1}, {1, 0}, {1, 1},
                                                                        {0, 2}, {2, 0}, {2, 2}, {2, 1}, {1, 2}, 
                                                                        {-2, 0}, {-1, 0}};
    std::vector<long long int> NUM_OF_SPLITS_3D = {2, 3};

    std::vector<std::vector<long long int>> SIZES_4D = {{18, 6, 192, 48}, {36, 36, 36, 36}, {6, 18, 6, 36}};
    std::vector<std::vector<long long int>> SPLIT_DIM_CONCAT_AXIS_4D = {{0, 0}, {0, 1}, {1, 0}, {1, 1},
                                                                        {0, 2}, {2, 0}, {2, 2}, {2, 1}, {1, 2},
                                                                        {0, 3}, {3, 0}, {1, 3}, {3, 1}, {2, 3}, {3, 2}, {3, 3},
                                                                        {-3, 0}, {-1, 3}};
    std::vector<long long int> NUM_OF_SPLITS_4D = {2, 3};
} // namespace SplitConcatFuseTestDefs

using namespace SplitConcatFuseTestDefs;
class SplitConcatFuseTest :
public ::testing::WithParamInterface<SplitConcatFuseTestDefs::SplitConcatFuseTestParams>,
public GrapplerTest {
    public:
    static std::string getTestCaseName(::testing::TestParamInfo<SplitConcatFuseTestParams> obj){
        DataType dtype;
        std::vector<long long int> input_size;
        std::vector<long long int> split_dim_concat_axis;
        long long int num_split;
        std::tie(dtype, input_size, split_dim_concat_axis, num_split) = obj.param;

        std::ostringstream result;
        result << "SplitConcatFuse_DataType_";
        switch(dtype) {
            case DataType::DT_FLOAT:
                result << "FLOAT";
                break;
            default:
                result << "UNRECOGNISED_TYPE";
        }
        result << "_InputSize";
        for (auto &x : input_size){
            result << "_" << x;
        }
        if(split_dim_concat_axis[0] < 0){
            result << "_SplitDim_negative_" << abs(split_dim_concat_axis[0]); 
        } else{
            result << "_SplitDim_" << split_dim_concat_axis[0];
        }
        if(split_dim_concat_axis[1] < 0){
            result << "_ConcatAxis_negative_" << abs(split_dim_concat_axis[1]);
        } else{
            result << "_ConcatAxis_" << split_dim_concat_axis[1];
        }
        result << "_NumSplit_" << num_split;
        return result.str();
    }

    void SetUp(){
        std::tie(dtype, input_size, split_dim_concat_axis, num_of_splits) = this->GetParam();
        std::vector<string> input_names;
        GraphDef ref_graph;
        input = Tensor(dtype, TensorShape(tensorflow::gtl::ArraySlice<long long int>(input_size.data(), input_size.size())));
        switch(dtype){
            case DataType::DT_FLOAT:
                input.flat<float>() = input.flat<float>().template setRandom<Eigen::internal::NormalRandomGenerator<float>>(); // input
                break;
            default:
                GTEST_FAIL() << "Unexpected DataType";
        }
        split_dim_tensor = Tensor((int32)split_dim_concat_axis[0]);
        concat_axis_tensor = Tensor((int32)split_dim_concat_axis[1]);
    }

    protected:
    void Validate(std::vector<Tensor> tensor, std::vector<Tensor> tensor_expected){
        EXPECT_EQ(dtype, tensor_expected[0].dtype());
        EXPECT_EQ(dtype, tensor[0].dtype());
        test::ExpectTensorEqual<float>(tensor_expected[0], tensor[0]);
    }

    // Test definition (straight from Params, filled in Setup)
    DataType dtype;
    std::vector<long long int> input_size;
    std::vector<long long int> split_dim_concat_axis;
    long long int num_of_splits;
    
    Tensor input;
    Tensor split_dim_tensor;
    Tensor concat_axis_tensor;

    GraphDef want;
};

class SplitConcatFuseTestSimpleFusing : public SplitConcatFuseTest{
    public:
    void RunAndValidate(){
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();

        auto value = 
            ops::Const(root.WithOpName("value"), input);
        auto concat_axis_op = 
            ops::Const(root.WithOpName("axis"), concat_axis_tensor);
        auto split_dim_op = 
            ops::Const(root.WithOpName("split_dim"), split_dim_tensor);
        auto s = 
            ops::Split(root.WithOpName("split"), split_dim_op, value, num_of_splits);
        
        if(num_of_splits == 2){
            auto concat_out =
                ops::Concat(root.WithOpName("concat"), {s[0], s[1]}, concat_axis_op);
        } else if( num_of_splits == 3){
            auto concat_out =
                ops::Concat(root.WithOpName("concat"), {s[0], s[1], s[2]}, concat_axis_op);
        } else if( num_of_splits == 4){
            auto concat_out =
                ops::Concat(root.WithOpName("concat"), {s[0], s[1], s[2], s[3]}, concat_axis_op);
        } else if( num_of_splits == 8){
            auto concat_out = 
                ops::Concat(root.WithOpName("concat"), {s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]}, concat_axis_op);
        } else{
            GTEST_FAIL() << "This num of splits is not coded in tests, try between 2 and 4.";
        }

        GrapplerItem item;
        item.fetch.push_back("concat");
        TF_CHECK_OK(root.ToGraphDef(&item.graph));

        SplitConcatFuse optimizer(nullptr);
        GraphDef output;
        Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
        TF_EXPECT_OK(status);

        std::vector<string> fetch = {"concat"};
        auto tensors_expected = EvaluateNodes(item.graph, fetch);
        auto tensors = EvaluateNodes(output, fetch);
        Validate(tensors, tensors_expected);
    }
};

TEST_P(SplitConcatFuseTestSimpleFusing, CompareWithRefs){
    SetUp();
    RunAndValidate();
}

INSTANTIATE_TEST_CASE_P(SCRIPTplit, SplitConcatFuseTestSimpleFusing,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SCRIPT_SIZES),
        ::testing::ValuesIn(SCRIPT_SPLIT_CONCAT),
        ::testing::ValuesIn(SCRIPT_NUM_OF_SPLITS)),
    SplitConcatFuseTestSimpleFusing::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Split2D, SplitConcatFuseTestSimpleFusing,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_2D),
        ::testing::ValuesIn(SPLIT_DIM_CONCAT_AXIS_2D),
        ::testing::ValuesIn(NUM_OF_SPLITS_2D)),
    SplitConcatFuseTestSimpleFusing::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Split3D, SplitConcatFuseTestSimpleFusing,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_3D),
        ::testing::ValuesIn(SPLIT_DIM_CONCAT_AXIS_3D),
        ::testing::ValuesIn(NUM_OF_SPLITS_3D)),
    SplitConcatFuseTestSimpleFusing::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Split4D, SplitConcatFuseTestSimpleFusing,
    ::testing::Combine(
        ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(SIZES_4D),
        ::testing::ValuesIn(SPLIT_DIM_CONCAT_AXIS_4D),
        ::testing::ValuesIn(NUM_OF_SPLITS_4D)),
    SplitConcatFuseTestSimpleFusing::getTestCaseName);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* SplitConcatFuse(bool fused, std::vector<long long int> input_shape, int split_d, int concat_a, int num_split){
    Graph* g = new Graph(OpRegistry::Global());
    DataType dt = DataTypeToEnum<T>::v();
    Tensor split_dim(DT_INT32, TensorShape({}));
    split_dim.scalar<int32>()() = split_d;
    Tensor concat_axis(DT_INT32, TensorShape({}));
    concat_axis.scalar<int32>()() = concat_a;
    Tensor input(dt, TensorShape(tensorflow::gtl::ArraySlice<long long int>(input_shape.data(), input_shape.size())));
    input.flat<T>().setRandom();

    if(fused){
        Node* split_concat;
        TF_CHECK_OK(NodeBuilder(g->NewName("splitconcat"), "_FusedSplitConcat")
            .Input(test::graph::Constant(g, split_dim))
            .Input(test::graph::Constant(g, input))
            .Input(test::graph::Constant(g, concat_axis))
            .Attr("num_split", num_split)
            .Attr("T", dt)
            .Attr("N", num_split)
            .Attr("Tidx", DT_INT32)
            .Finalize(g, &split_concat));

        return g;
    } else{
        Node* split;
        TF_CHECK_OK(NodeBuilder(g->NewName("split"), "Split")
            .Input(test::graph::Constant(g, split_dim))
            .Input(test::graph::Constant(g, input))
            .Attr("num_split", num_split)
            .Attr("T", dt)
            .Finalize(g, &split));
        
        
        std::vector<NodeBuilder::NodeOut> out_list;
        for (int i = 0; i < num_split; ++i){
            Output buf(split, i);
            out_list.push_back(buf.node());
        }

        Node* concat;
        TF_CHECK_OK(NodeBuilder(g->NewName("concat"), "Concat")
            .Input(test::graph::Constant(g, concat_axis))
            .Input(out_list)
            .Attr("N", num_split)
            .Attr("T", dt)
            .Finalize(g, &concat));
        
        return g;
    }
}

using fp32 = float;

#define BM_NAME(name, FUSED, T, split_name, concat_name, NUM_SPLIT, input_shape_name)      \
    name##_##FUSED##_##T##_##split_name##_##concat_name##_##NUM_SPLIT##_##input_shape_name

#define BM_SplitConcatFuseBenchmark(FUSED, T, INPUT_SHAPE, split_name, SPLIT_DIM, concat_name, CONCAT_AXIS, NUM_SPLIT, input_shape_name, LABEL)         \
    static void BM_NAME(BM_SplitConcatFuse, FUSED, T, split_name, concat_name, NUM_SPLIT, input_shape_name)(int iters){                                 \
        testing::StopTiming();                                                                                                                          \
        std::string base_label = FUSED ? "fused_split_concat" : "not_fused_split_concat";                                                               \
        size_t input_shape_size = INPUT_SHAPE.size();                                                                                                   \
        int all_elements = 1;                                                                                                                           \
        for (int i = 0; i < input_shape_size; ++i){                                                                                                     \
            all_elements *= INPUT_SHAPE[i];                                                                                                             \
        }                                                                                                                                               \
        testing::SetLabel(base_label + "_" + LABEL);                                                                                                    \
        testing::BytesProcessed(static_cast<int64>(iters) * all_elements * sizeof(T));                                                                  \
        testing::StartTiming();                                                                                                                         \
        test::Benchmark("cpu", SplitConcatFuse<T>(FUSED, INPUT_SHAPE, SPLIT_DIM, CONCAT_AXIS, NUM_SPLIT))                                               \
            .Run(iters);                                                                                                                                \
    }                                                                                                                                                   \
    BENCHMARK(BM_NAME(BM_SplitConcatFuse, FUSED, T, split_name, concat_name, NUM_SPLIT, input_shape_name));

#define DLRM_BENCHMARK(FUSED)                                                                                                                                                           \
    BM_SplitConcatFuseBenchmark(FUSED, fp32, (std::vector<long long int>{1024, 50, 256}), 2, 2, 0, 0, 8, 1024x50x256, "float_1024x50x356_split_2_concat_0_numsplit_8");                 \
    BM_SplitConcatFuseBenchmark(FUSED, fp32, (std::vector<long long int>{1024, 50, 256}), neg_1, -1, 0, 0, 8, 1024x50x256, "float_1024x50x356_split_-1_concat_0_numsplit_8");            

#define EQUAL_BENCHMARK(FUSED)                                                                                                                                                          \
    /* split_dim == concat_axis */                                                                                                                                                      \
    BM_SplitConcatFuseBenchmark(FUSED, fp32, (std::vector<long long int>{1024, 50, 256}), 0, 0, 0, 0, 8, 1024x50x256, "float_1024x50x356_split_0_concat_0_numsplit_8");                 \
    BM_SplitConcatFuseBenchmark(FUSED, fp32, (std::vector<long long int>{128, 512, 2, 16}), 3, 3, neg_1, -1, 2, 128x512x2z16, "float_128x512x2x16_split_3_concat_-1_numsplit_2");       \
    BM_SplitConcatFuseBenchmark(FUSED, fp32, (std::vector<long long int>{192, 64, 96}), neg_1, -1, 2, 2, 3, 192x64x96, "float_192x64x96_split_-1_concat_2_numsplit_3");

#define SPLIT_MAJOR_BENCHMARK(FUSED)                                                                                                                                                    \
    /* split_dim > concat_axis */                                                                                                                                                       \
    BM_SplitConcatFuseBenchmark(FUSED, fp32, (std::vector<long long int>{1024, 50, 256}), 2, 2, 1, 1, 2, 1024x50x256, "float_1024x50x256_split_2_concat_1_numsplit_2");                 \
    BM_SplitConcatFuseBenchmark(FUSED, fp32, (std::vector<long long int>{1024, 512}), 1, 1, 0, 0, 2, 1024x512, "float_1024x512_split_1_concat_0_numsplit_2");                           \
    BM_SplitConcatFuseBenchmark(FUSED, fp32, (std::vector<long long int>{384, 192, 64, 6}), 3, 3, neg_3, -3, 3, 384x192x64x6, "float_384x192x64x6_split_3_concat_-3_numsplit_3");

#define CONCAT_MAJOR_BENCHMARK(FUSED)                                                                                                                                                   \
    /* concat_axis > split_dim */                                                                                                                                                       \
    BM_SplitConcatFuseBenchmark(FUSED, fp32, (std::vector<long long int>{1024, 50, 256}), 0, 0, 2, 2, 8, 1024x50x256, "float_1024x50x356_split_0_concat_2_numsplit_8");                 \
    BM_SplitConcatFuseBenchmark(FUSED, fp32, (std::vector<long long int>{1024, 50, 256, 16}), neg_3, -3, 3, 3, 2, 1024x50z256x16, "float_1024x50x256x16_split_-3_concat_3_numsplit_2"); \
    BM_SplitConcatFuseBenchmark(FUSED, fp32, (std::vector<long long int>{384, 192, 6}), 0, 0, 1, 1, 3, 384x192x6, "float_384x192x6_split_0_concat_1_numsplit_1");       

DLRM_BENCHMARK(true)
DLRM_BENCHMARK(false)

EQUAL_BENCHMARK(true)
EQUAL_BENCHMARK(false)

SPLIT_MAJOR_BENCHMARK(true)
SPLIT_MAJOR_BENCHMARK(false)

CONCAT_MAJOR_BENCHMARK(true)
CONCAT_MAJOR_BENCHMARK(false)

} // namespace
} // namespace grappler
} // namespace tensorflow