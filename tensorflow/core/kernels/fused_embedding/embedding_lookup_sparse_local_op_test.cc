/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
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

enum TestCase { Sqrtn, Mean, Sum, SqrtnAndMaxNorm200, MeanAndMaxNorm100 };

template <TestCase test_case>
void get_node_attr_from_test_case(string& combiner_str, float& max_norm) {
  if (test_case == Sqrtn) {
    combiner_str = "sqrtn";
    max_norm = -1.0f;
  } else if (test_case == Mean) {
    combiner_str = "mean";
    max_norm = -1.0f;
  } else if (test_case == Sum) {
    combiner_str = "sum";
    max_norm = -1.0f;
  } else if (test_case == SqrtnAndMaxNorm200) {
    combiner_str = "sqrtn";
    max_norm = 200.0f;
  } else if (test_case == MeanAndMaxNorm100) {
    combiner_str = "mean";
    max_norm = 100.0f;
  }
}

template <TestCase test_case>
void fill_emb_vector_expected(Tensor* expected);

template <>
void fill_emb_vector_expected<Sqrtn>(Tensor* expected) {
  test::FillValues<float>(
      expected, {22.627416610717773, 24.0416316986084,   25.45584487915039,
                 26.870058059692383, 28.284271240234375, 29.698484420776367,
                 31.112699508666992, 32.526912689208984, 73.90083312988281,
                 75.63288879394531,  77.36493682861328,  79.09698486328125,
                 80.82904052734375,  82.56108856201172,  84.29314422607422,
                 86.02519226074219,  124.70765686035156, 126.43971252441406,
                 128.17176818847656, 129.90380859375,    131.6358642578125,
                 133.367919921875,   135.09996032714844, 136.83201599121094,
                 107.48023223876953, 108.89444732666016, 110.30866241455078,
                 111.72286987304688, 113.1370849609375,  114.55130004882812,
                 115.96551513671875, 117.37973022460938});
}

template <>
void fill_emb_vector_expected<Mean>(Tensor* expected) {
  test::FillValues<float>(
      expected, {16.00000000000000, 17.00000000000000, 18.00000000000000,
                 19.00000000000000, 20.00000000000000, 21.00000000000000,
                 22.00000000000000, 23.00000000000000, 42.66666793823242,
                 43.66666793823242, 44.66666793823242, 45.66666793823242,
                 46.66666793823242, 47.66666793823242, 48.66666793823242,
                 49.66666793823242, 72.00000000000000, 73.00000000000000,
                 74.00000000000000, 75.00000000000000, 76.00000000000000,
                 77.00000000000000, 78.00000000000000, 79.00000000000000,
                 76.00000000000000, 77.00000000000000, 78.00000000000000,
                 79.00000000000000, 80.00000000000000, 81.00000000000000,
                 82.00000000000000, 83.00000000000000});
}

template <>
void fill_emb_vector_expected<Sum>(Tensor* expected) {
  test::FillValues<float>(
      expected, {32.0,  34.0,  36.0,  38.0,  40.0,  42.0,  44.0,  46.0,
                 128.0, 131.0, 134.0, 137.0, 140.0, 143.0, 146.0, 149.0,
                 216.0, 219.0, 222.0, 225.0, 228.0, 231.0, 234.0, 237.0,
                 152.0, 154.0, 156.0, 158.0, 160.0, 162.0, 164.0, 166.0});
}

template <>
void fill_emb_vector_expected<SqrtnAndMaxNorm200>(Tensor* expected) {
  test::FillValues<float>(
      expected,
      {22.62741661, 24.04163170, 25.45584488,  26.87005806,  28.28427124,
       29.69848442, 31.11269951, 32.52691269,  73.90083313,  75.63288879,
       77.36493683, 79.09698486, 80.82904053,  82.56108856,  84.29314423,
       86.02519226, 92.61308289, 94.01081848,  95.40855408,  96.80628204,
       98.20401764, 99.60175323, 100.99948120, 102.39721680, 71.20205688,
       72.31395721, 73.42584991, 74.53774261,  75.64963531,  76.76153564,
       77.87342834, 78.98532867});
}

class FusedEmbeddingLocalSparseLookUpOpTest : public OpsTestBase {
 protected:
  template <typename T, TestCase test_case>
  void Run(Device device) {
    if (device == Device::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }
    DataType dtype = DataTypeToEnum<T>::value;
    std::string combiner_str;
    float max_norm;

    get_node_attr_from_test_case<test_case>(combiner_str, max_norm);

    TF_EXPECT_OK(NodeDefBuilder("fused_embedding_local_sparse_look_up",
                                "FusedEmbeddingLocalSparseLookUp")
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(dtype))
                     .Attr("T", dtype)
                     .Attr("combiner", combiner_str)
                     .Attr("max_norm", max_norm)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());

    const int nnz = 10;
    const int batch_size = 4;
    const int emb_vector_dim = 8;
    const int entries = 8;
    const int bucket_size = 16;

    Tensor sp_values(DT_INT64, {nnz});
    Tensor sp_indices(DT_INT64, {nnz, 2});
    Tensor sp_dense_shape(DT_INT64, {2});
    Tensor emb_variable(dtype, {bucket_size, emb_vector_dim});

    test::FillValues<int64>(&sp_values, {3, 1, 4, 5, 7, 3, 12, 12, 15, 4});
    test::FillValues<int64>(&sp_indices, {0, 1, 0, 5, 1, 2, 1, 1, 1, 7,
                                          2, 1, 2, 4, 2, 7, 3, 0, 3, 6});
    test::FillValues<int64>(&sp_dense_shape, {batch_size, entries});
    test::FillValues<T>(
        &emb_variable,
        {0.0,   1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   7.0,   8.0,   9.0,
         10.0,  11.0,  12.0,  13.0,  14.0,  15.0,  16.0,  17.0,  18.0,  19.0,
         20.0,  21.0,  22.0,  23.0,  24.0,  25.0,  26.0,  27.0,  28.0,  29.0,
         30.0,  31.0,  32.0,  33.0,  34.0,  35.0,  36.0,  37.0,  38.0,  39.0,
         40.0,  41.0,  42.0,  43.0,  44.0,  45.0,  46.0,  47.0,  48.0,  49.0,
         50.0,  51.0,  52.0,  53.0,  54.0,  55.0,  56.0,  57.0,  58.0,  59.0,
         60.0,  61.0,  62.0,  63.0,  64.0,  65.0,  66.0,  67.0,  68.0,  69.0,
         70.0,  71.0,  72.0,  73.0,  74.0,  75.0,  76.0,  77.0,  78.0,  79.0,
         80.0,  81.0,  82.0,  83.0,  84.0,  85.0,  86.0,  87.0,  88.0,  89.0,
         90.0,  91.0,  92.0,  93.0,  94.0,  95.0,  96.0,  97.0,  98.0,  99.0,
         100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
         110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
         120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0});

    AddInputFromArray<int64>(sp_values.shape(), sp_values.flat<int64>());
    AddInputFromArray<int64>(sp_indices.shape(), sp_indices.flat<int64>());
    AddInputFromArray<int64>(sp_dense_shape.shape(),
                             sp_dense_shape.flat<int64>());
    AddInputFromArray<T>(emb_variable.shape(), emb_variable.flat<T>());

    TF_ASSERT_OK(RunOpKernel());

    Tensor emb_vector_expected(dtype, {batch_size, emb_vector_dim});
    Tensor sp_values_offset_expected(DT_INT32, {batch_size});
    fill_emb_vector_expected<test_case>(&emb_vector_expected);
    test::FillValues<int32>(&sp_values_offset_expected, {0, 2, 5, 8});

    const Tensor& emb_vector = *GetOutput(0);
    const Tensor& values_offset = *GetOutput(1);
    TF_EXPECT_OK(device_->Sync());

    test::ExpectTensorNear<T>(emb_vector_expected, emb_vector, 1e-4);
    test::ExpectTensorEqual<int32>(sp_values_offset_expected, values_offset);
  }
};

template <TestCase test_case>
void fill_grad_expected(Tensor* expected);

template <>
void fill_grad_expected<Sqrtn>(Tensor* expected) {
  test::FillValues<float>(
      expected, {0.000000000000000,  0.7071067690849304, 1.4142135381698608,
                 2.1213204860687256, 2.8284270763397217, 3.535533905029297,
                 4.242640972137451,  4.949747562408447,  0.000000000000000,
                 0.7071067690849304, 1.4142135381698608, 2.1213204860687256,
                 2.8284270763397217, 3.535533905029297,  4.242640972137451,
                 4.949747562408447,  4.618802070617676,  5.196152687072754,
                 5.773502826690674,  6.350852966308594,  6.928203582763672,
                 7.505553722381592,  8.082903861999512,  8.66025447845459,
                 4.618802070617676,  5.196152687072754,  5.773502826690674,
                 6.350852966308594,  6.928203582763672,  7.505553722381592,
                 8.082903861999512,  8.66025447845459,   4.618802070617676,
                 5.196152687072754,  5.773502826690674,  6.350852966308594,
                 6.928203582763672,  7.505553722381592,  8.082903861999512,
                 8.66025447845459,   9.237604141235352,  9.81495475769043,
                 10.392305374145508, 10.96965503692627,  11.547005653381348,
                 12.124356269836426, 12.701705932617188, 13.279056549072266,
                 9.237604141235352,  9.81495475769043,   10.392305374145508,
                 10.96965503692627,  11.547005653381348, 12.124356269836426,
                 12.701705932617188, 13.279056549072266, 9.237604141235352,
                 9.81495475769043,   10.392305374145508, 10.96965503692627,
                 11.547005653381348, 12.124356269836426, 12.701705932617188,
                 13.279056549072266, 16.970563888549805, 17.677669525146484,
                 18.384777069091797, 19.091882705688477, 19.79899024963379,
                 20.5060977935791,   21.21320343017578,  21.920310974121094,
                 16.970563888549805, 17.677669525146484, 18.384777069091797,
                 19.091882705688477, 19.79899024963379,  20.5060977935791,
                 21.21320343017578,  21.920310974121094});
}

template <>
void fill_grad_expected<Mean>(Tensor* expected) {
  test::FillValues<float>(
      expected, {0.000000000000000,  0.500000000000000,  1.000000000000000,
                 1.500000000000000,  2.000000000000000,  2.500000000000000,
                 3.000000000000000,  3.500000000000000,  0.000000000000000,
                 0.500000000000000,  1.000000000000000,  1.500000000000000,
                 2.000000000000000,  2.500000000000000,  3.000000000000000,
                 3.500000000000000,  2.6666667461395264, 3.000000000000000,
                 3.3333332538604736, 3.6666667461395264, 4.000000000000000,
                 4.333333492279053,  4.666666507720947,  5.000000000000000,
                 2.6666667461395264, 3.000000000000000,  3.3333332538604736,
                 3.6666667461395264, 4.000000000000000,  4.333333492279053,
                 4.666666507720947,  5.000000000000000,  2.6666667461395264,
                 3.000000000000000,  3.3333332538604736, 3.6666667461395264,
                 4.000000000000000,  4.333333492279053,  4.666666507720947,
                 5.000000000000000,  5.333333492279053,  5.666666507720947,
                 6.000000000000000,  6.333333492279053,  6.666666507720947,
                 7.000000000000000,  7.333333492279053,  7.666666507720947,
                 5.333333492279053,  5.666666507720947,  6.000000000000000,
                 6.333333492279053,  6.666666507720947,  7.000000000000000,
                 7.333333492279053,  7.666666507720947,  5.333333492279053,
                 5.666666507720947,  6.000000000000000,  6.333333492279053,
                 6.666666507720947,  7.000000000000000,  7.333333492279053,
                 7.666666507720947,  12.000000000000000, 12.500000000000000,
                 13.000000000000000, 13.500000000000000, 14.000000000000000,
                 14.500000000000000, 15.000000000000000, 15.500000000000000,
                 12.000000000000000, 12.500000000000000, 13.000000000000000,
                 13.500000000000000, 14.000000000000000, 14.500000000000000,
                 15.000000000000000, 15.500000000000000});
}

template <>
void fill_grad_expected<Sum>(Tensor* expected) {
  test::FillValues<float>(
      expected,
      {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  0.0,  1.0,  2.0,  3.0,
       4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
       8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 8.0,  9.0,  10.0, 11.0,
       12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
       16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 16.0, 17.0, 18.0, 19.0,
       20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
       24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0});
}

template <>
void fill_grad_expected<MeanAndMaxNorm100>(Tensor* expected) {
  test::FillValues<float>(
      expected,
      {0.00000000,  0.50000000,  1.00000000,  1.50000000,  2.00000000, 2.50000000,  3.00000000, 3.50000000,
       0.00000000,  0.50000000, 1.00000000,  1.50000000,  2.00000000,  2.50000000,  3.00000000, 3.50000000,
       2.65028572,  2.98157120,  3.31285667,  3.64414287, 3.97542834,  4.30671406,  4.63799953,  4.96928549,
       2.16437674, 2.43492365,  2.70547056,  2.97601795,  3.24656487,  3.51711202, 3.78765893,  4.05820608,
       1.58337951,  1.78130186,  1.97922409, 2.17714667,  2.37506914,  2.57299161,  2.77091384,  2.96883631,
       5.33333349,  5.66666651,  6.00000000,  6.33333349,  6.66666651, 7.00000000,  7.33333349,  7.66666651,
       1.89459133,  2.01300311, 2.13141513,  2.24982715,  2.36823893,  2.48665094,  2.60506320,  2.72347474,
       1.89459133,  2.01300311,  2.13141513,  2.24982715, 2.36823893,  2.48665094,  2.60506320,  2.72347474,
       3.43474555, 3.57786012,  3.72097445,  3.86408877,  4.00720310,  4.15031767, 4.29343224,  4.43654633,
       11.92628479, 12.42321396, 12.92014217, 13.41707039, 13.91399956, 14.41092777, 14.90785599, 15.40478516});
}

class FusedEmbeddingLocalSparseLookUpGradOpTest : public OpsTestBase {
 protected:
  template <typename T, TestCase test_case>
  void Run(Device device) {
    if (device == Device::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }
    DataType dtype = DataTypeToEnum<T>::value;
    std::string combiner_str;
    float max_norm;
    get_node_attr_from_test_case<test_case>(combiner_str, max_norm);

    TF_EXPECT_OK(NodeDefBuilder("fused_embedding_local_sparse_look_up_grad",
                                "FusedEmbeddingLocalSparseLookUpGrad")
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT32))
                     .Attr("T", dtype)
                     .Attr("combiner", combiner_str)
                     .Attr("max_norm", max_norm)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());

    const int nnz = 10;
    const int batch_size = 4;
    const int emb_vector_dim = 8;
    const int bucket_size = 16;

    Tensor top_grad(dtype, {batch_size, emb_vector_dim});
    Tensor emb_variable(dtype, {bucket_size, emb_vector_dim});
    Tensor sp_values(DT_INT64, {nnz});
    Tensor sp_values_offset(DT_INT32, {batch_size});

    test::FillValues<T>(
        &top_grad,
        {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0,
         11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
         22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0});
    test::FillValues<T>(
        &emb_variable,
        {0.0,   1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   7.0,   8.0,   9.0,
         10.0,  11.0,  12.0,  13.0,  14.0,  15.0,  16.0,  17.0,  18.0,  19.0,
         20.0,  21.0,  22.0,  23.0,  24.0,  25.0,  26.0,  27.0,  28.0,  29.0,
         30.0,  31.0,  32.0,  33.0,  34.0,  35.0,  36.0,  37.0,  38.0,  39.0,
         40.0,  41.0,  42.0,  43.0,  44.0,  45.0,  46.0,  47.0,  48.0,  49.0,
         50.0,  51.0,  52.0,  53.0,  54.0,  55.0,  56.0,  57.0,  58.0,  59.0,
         60.0,  61.0,  62.0,  63.0,  64.0,  65.0,  66.0,  67.0,  68.0,  69.0,
         70.0,  71.0,  72.0,  73.0,  74.0,  75.0,  76.0,  77.0,  78.0,  79.0,
         80.0,  81.0,  82.0,  83.0,  84.0,  85.0,  86.0,  87.0,  88.0,  89.0,
         90.0,  91.0,  92.0,  93.0,  94.0,  95.0,  96.0,  97.0,  98.0,  99.0,
         100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
         110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
         120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0});
    test::FillValues<int64>(&sp_values, {3, 1, 4, 5, 7, 3, 12, 12, 15, 4});
    test::FillValues<int32>(&sp_values_offset, {0, 2, 5, 8});

    AddInputFromArray<T>(top_grad.shape(), top_grad.flat<T>());
    AddInputFromArray<T>(emb_variable.shape(), emb_variable.flat<T>());
    AddInputFromArray<int64>(sp_values.shape(), sp_values.flat<int64>());
    AddInputFromArray<int32>(sp_values_offset.shape(),
                             sp_values_offset.flat<int32>());

    TF_ASSERT_OK(RunOpKernel());

    Tensor grad_expected(dtype, {nnz, emb_vector_dim});
    fill_grad_expected<test_case>(&grad_expected);

    const Tensor& grad = *GetOutput(0);
    TF_EXPECT_OK(device_->Sync());

    test::ExpectTensorNear<T>(grad_expected, grad, 1e-4);
  }
};

TEST_F(FusedEmbeddingLocalSparseLookUpOpTest, LocalFloatSumCpu) {

  TF_EXPECT_OK(NodeDefBuilder("FusedSafeEmbeddingLookupSparseLocal",
                              "FusedSafeEmbeddingLookupSparseLocal")
                    .Input(FakeInput(DT_FLOAT))
                    .Input(FakeInput(DT_INT64))
                    .Input(FakeInput(DT_INT64))
                    .Input(FakeInput(DT_INT64))
                    .Input(FakeInput(DT_INT64))
                    .Attr("T", DT_FLOAT)
                    .Attr("combiner", "sum")
                    .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());

  const int nnz = 10;
  const int batch_size = 4;
  const int emb_vector_dim = 8;
  const int entries = 8;
  const int bucket_size = 16;

  Tensor sp_values(DT_INT64, {nnz});
  Tensor sp_weight(DT_INT64, {nnz});
  Tensor sp_indices(DT_INT64, {nnz, 2});
  Tensor sp_dense_shape(DT_INT64, {2});
  Tensor emb_variable(DT_FLOAT, {bucket_size, emb_vector_dim});

  // [3, 1, 4, 5, 7, 3, 12, 12, 15, 4]
  test::FillValues<int64>(&sp_values, {3, 1, 4, 5, 7, 3, 12, 12, 15, 4});
  test::FillValues<int64>(&sp_weight, {3, 1, 4, 5, 7, 3, 12, 12, 15, 4});
  // [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
  test::FillValues<int64>(&sp_indices, {0, 1, 0, 5, 1, 2, 1, 1, 1, 7,
                                        2, 1, 2, 4, 2, 7, 3, 0, 3, 6});
  test::FillValues<int64>(&sp_dense_shape, {batch_size, entries});
  test::FillValues<float>(
      &emb_variable,
      {0.0,   1.0,    2.0,   3.0,   4.0,   5.0,   6.0,   7.0,
       8.0,   9.0,   10.0,  11.0,  12.0,  13.0,  14.0,  15.0,
       16.0,  17.0,  18.0,  19.0,  20.0,  21.0,  22.0,  23.0,
       24.0,  25.0,  26.0,  27.0,  28.0,  29.0,  30.0,  31.0,
       32.0,  33.0,  34.0,  35.0,  36.0,  37.0,  38.0,  39.0,
       40.0,  41.0,  42.0,  43.0,  44.0,  45.0,  46.0,  47.0,
       48.0,  49.0,  50.0,  51.0,  52.0,  53.0,  54.0,  55.0,
       56.0,  57.0,  58.0,  59.0,  60.0,  61.0,  62.0,  63.0,
       64.0,  65.0,  66.0,  67.0,  68.0,  69.0,  70.0,  71.0,
       72.0,  73.0,  74.0,  75.0,  76.0,  77.0,  78.0,  79.0,
       80.0,  81.0,  82.0,  83.0,  84.0,  85.0,  86.0,  87.0,
       88.0,  89.0,  90.0,  91.0,  92.0,  93.0,  94.0,  95.0,
       96.0,  97.0,  98.0,  99.0, 100.0, 101.0, 102.0, 103.0,
       104.0,105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
       112.0,113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
       120.0,121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0});

  AddInputFromArray<float>(emb_variable.shape(), emb_variable.flat<float>());
  AddInputFromArray<int64>(sp_values.shape(), sp_values.flat<int64>());
  AddInputFromArray<int64>(sp_dense_shape.shape(),
                            sp_dense_shape.flat<int64>());
  AddInputFromArray<int64>(sp_indices.shape(), sp_indices.flat<int64>());

  TF_ASSERT_OK(RunOpKernel());

  Tensor emb_vector_expected(DT_FLOAT, {batch_size, emb_vector_dim});
  // Tensor sp_values_offset_expected(DT_INT32, {batch_size});
  fill_emb_vector_expected<Sum>(&emb_vector_expected);
  // test::FillValues<int32>(&sp_values_offset_expected, {0, 2, 5, 8});

  const Tensor& emb_vector = *GetOutput(0);
  // const Tensor& values_offset = *GetOutput(1);
  // TF_EXPECT_OK(device_->Sync());

  float *output = (float *)emb_vector.tensor_data().data();
  float *output_ex = (float *)emb_vector_expected.tensor_data().data();

  test::ExpectTensorNear<float>(emb_vector_expected, emb_vector, 1e-2);
  // test::ExpectTensorEqual<int32>(sp_values_offset_expected, values_offset);
}

TEST_F(FusedEmbeddingLocalSparseLookUpOpTest, LocalGradFloatSumCpu) {

  TF_EXPECT_OK(NodeDefBuilder("FusedSafeEmbeddingLookupSparseLocalGrad",
                              "FusedSafeEmbeddingLookupSparseLocalGrad")
                    .Input(FakeInput(DT_FLOAT)) // gradients
                    .Input(FakeInput(DT_INT64)) // input hash value
                    .Input(FakeInput(DT_INT64)) // dense_shape
                    .Input(FakeInput(DT_INT64)) // indices
                    .Attr("T", DT_FLOAT)
                    .Attr("Tinput", DT_INT64)
                    .Attr("Tindices", DT_INT64)
                    .Attr("Tdense_shape", DT_INT64)
                    .Attr("combiner", "sum")
                    .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());

  const int nnz = 32;
  const int batch_size = 32;
  const int emb_vector_dim = 4;
  const int entries = 1;
  const int bucket_size = 16;

  Tensor sp_values(DT_INT64, {nnz});
  Tensor sp_indices(DT_INT64, {nnz, 2});
  Tensor sp_dense_shape(DT_INT64, {2});
  Tensor grad_variable(DT_FLOAT, {batch_size, emb_vector_dim});

  test::FillValues<float>(
      &grad_variable,
      {-0.00363823911, 0.0138593055, 0.00232614437, 0.00241222954,
       -0.000268990319, -0.00410466315, 0.00478722388, -0.000196215493,
       -0.0044340631, -0.00725936424, -0.00691315765, -0.00612797868,
       -0.00678675482, -0.00246100035, 0.00216219737, -0.00346030248,
       0.00100048154, -0.00852716807, 0.00803291425, -0.000800206966,
       -3.03583856e-05, 0.00524863973, -0.0163001865, -0.0109826243,
       0.0830041766, 0.153927863, -0.0508279465, -0.00474824524,
       7.8225421e-05, -0.000293536956, 0.00610643439, -0.00019871055,
       -0.000780000235, -0.00221115421, 0.00387162319, 0.00222597015,
       -0.0102384416, -0.00801581, -0.0017716008, 0.00598057127,
       -0.00808391348, -0.00166459556, 0.00106997311, -0.00185864791,
       0.00491535058, -0.00633693347, 0.0212651137, 0.00704831816,
       -0.00338345463, -0.00668374076, -0.0000871402444, -0.000196078254,
       0.00254824688, -0.00249796058, -0.0034719836, -0.003478111,
       6.03029093e-06, -0.00211180653, 0.000114592229, -0.00240143575,
       -0.00592383416, -0.00984606426, 0.00129341101, 0.00100650277,
       0.000906444562, -0.00139640097, -0.000192714069, 0.00277191238,
       -0.000245573436, -0.00680374401, 0.00356984767, -0.00120577728,
       -0.000766036392, -0.00487764599, 0.000532136182, -0.00413817167,
       -0.0302855149, -0.0406391025, 0.0006130244, 0.0183675159,
       -0.00247384049, -0.00609699031, 0.00127684267, -0.00235637,
       0.00715987338, 0.00783564895, -0.00139878597, -0.0048744888,
       0.00356917572, -0.0164020304, 0.0179400034, 0.000975746894,
       -0.00529623777, -0.00490315, 0.00691250199, 0.00286021968,
       -0.00426661829, -0.00417789398, -0.00597105641, -0.00605484238,
       0.00197085389, -0.00757023226, 0.00458694575, 0.00153650146,
       -0.00345475, -0.00823391136, 0.000807857723, 0.0121598523,
       -0.00745406374, -0.0135948248, 0.004774753, -0.00390140619,
       -0.00208005216, -0.00362896058, 0.00558064319, -0.000532045437,
       -0.00854093302, 0.00566324079, -0.00435794424, 0.00403016619,
       0.000468764076, 0.000297251798, -0.00617758604, -0.00338481856,
       0.00280403625, -0.00649327, -0.000154057736, -0.000479023496});
  test::FillValues<int64>(&sp_values, {9, 2, 9, 2, 2, 9, 2, 2, 2, 2, 2, 2, 9, 2, 2, 2, 9, 2, 2, 2, 2, 9, 2, 9, 2, 2, 2, 9, 2, 9, 2, 2});
  test::FillValues<int64>(&sp_indices, {0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31, 0});
  test::FillValues<int64>(&sp_dense_shape, {batch_size, entries});

  AddInputFromArray<float>(grad_variable.shape(), grad_variable.flat<float>());
  AddInputFromArray<int64>(sp_values.shape(), sp_values.flat<int64>());
  AddInputFromArray<int64>(sp_indices.shape(), sp_indices.flat<int64>());
  AddInputFromArray<int64>(sp_dense_shape.shape(), sp_dense_shape.flat<int64>());

  TF_ASSERT_OK(RunOpKernel());

  Tensor output1_tensor_expected(DT_FLOAT, {2, emb_vector_dim});
  Tensor output2_tensor_expected(DT_INT64, {2});

  test::FillValues<float>(&output1_tensor_expected,
      {-0.0247110315, -0.00123064546, -0.0152365314, -0.0140080471,
       0.0247110203, 0.00123063289, 0.0152365509, 0.0140080536});

  test::FillValues<int64>(&output2_tensor_expected, {9, 2});
  float *output1_ex = (float *)output1_tensor_expected.tensor_data().data();
  int64 *output2_ex = (int64 *)output2_tensor_expected.tensor_data().data();

  const Tensor& output1_tensor = *GetOutput(0);
  const Tensor& output2_tensor = *GetOutput(1);

  float *output1 = (float *)output1_tensor.tensor_data().data();
  int64 *output2 = (int64 *)output2_tensor.tensor_data().data();

  printf("out = %.11f , expect = %.11f\n", output1[5], output1_ex[5]);
  printf("out = %.11f , expect = %.11f\n", output1[7], output1_ex[7]);
  test::ExpectTensorNear<float>(output1_tensor_expected, output1_tensor, 1e-8);
  test::ExpectTensorEqual<int64>(output2_tensor_expected, output2_tensor);
}

TEST_F(FusedEmbeddingLocalSparseLookUpOpTest, LocalGradFloatMeanCpu) {

  TF_EXPECT_OK(NodeDefBuilder("FusedSafeEmbeddingLookupSparseLocalGrad",
                              "FusedSafeEmbeddingLookupSparseLocalGrad")
                    .Input(FakeInput(DT_FLOAT)) // gradients
                    .Input(FakeInput(DT_INT64)) // input hash value
                    .Input(FakeInput(DT_INT64)) // dense_shape
                    .Input(FakeInput(DT_INT64)) // indices
                    .Attr("T", DT_FLOAT)
                    .Attr("Tinput", DT_INT64)
                    .Attr("Tindices", DT_INT64)
                    .Attr("Tdense_shape", DT_INT64)
                    .Attr("combiner", "mean")
                    .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());

  const int nnz = 9;
  const int batch_size = 5;
  const int emb_vector_dim = 4;
  const int entries = 8;
  const int bucket_size = 16;

  Tensor sp_values(DT_INT64, {nnz});
  Tensor sp_indices(DT_INT64, {nnz, 2});
  Tensor sp_dense_shape(DT_INT64, {2});
  Tensor grad_variable(DT_FLOAT, {batch_size, emb_vector_dim});

  test::FillValues<float>(
      &grad_variable,
      {0.0103125420, 0.018807490, -0.0106398590, -0.029409127,
       0.0054132286, 0.013920069, -0.0190976150, -0.023196392,
       0.0100601720, 0.015330995, -0.0055795530, -0.024889620,
       0.0108455080, 0.018832123, -0.0095151365, -0.029357582,
       0.0100478110, 0.018798435, -0.0112019650, -0.029439624});
  test::FillValues<int64>(&sp_values, {1, 1, 0, 4, 1, 1, 1, 0, 1});
  test::FillValues<int64>(&sp_indices, {0, 1, 0, 3, 0, 6, 1, 3, 1, 6,
                                        3, 3, 3, 4, 4, 1, 4, 7});
  test::FillValues<int64>(&sp_dense_shape, {batch_size, entries});

  AddInputFromArray<float>(grad_variable.shape(), grad_variable.flat<float>());
  AddInputFromArray<int64>(sp_values.shape(), sp_values.flat<int64>());
  AddInputFromArray<int64>(sp_indices.shape(), sp_indices.flat<int64>());
  AddInputFromArray<int64>(sp_dense_shape.shape(), sp_dense_shape.flat<int64>());

  TF_ASSERT_OK(RunOpKernel());

  Tensor output1_tensor_expected(DT_FLOAT, {3, emb_vector_dim});
  Tensor output2_tensor_expected(DT_INT64, {3});
  test::FillValues<float>(&output1_tensor_expected,
      {0.0254510570, 0.0477297000, -0.0317581670, -0.075281680,
       0.0084614195, 0.0156683810, -0.0091476020, -0.024522856,
       0.0027066143, 0.0069600344, -0.0095488075, -0.011598196});
  test::FillValues<int64>(&output2_tensor_expected, {1, 0, 4});
  float *output1_ex = (float *)output1_tensor_expected.tensor_data().data();
  int64 *output2_ex = (int64 *)output2_tensor_expected.tensor_data().data();

  const Tensor& output1_tensor = *GetOutput(0);
  const Tensor& output2_tensor = *GetOutput(1);

  float *output1 = (float *)output1_tensor.tensor_data().data();
  int64 *output2 = (int64 *)output2_tensor.tensor_data().data();

  // printf("out = %f , expect = %f\n", output1[0], output1_ex[0]);
  // printf("out = %f , expect = %f\n", output1[1], output1_ex[1]);
  // printf("out = %f , expect = %f\n", output1[2], output1_ex[2]);
  // printf("out = %f , expect = %f\n", output1[3], output1_ex[3]);

  // printf("out = %d , expect = %d\n", output2[0], output2_ex[0]);
  // printf("out = %d , expect = %d\n", output2[1], output2_ex[1]);
  // printf("out = %d , expect = %d\n", output2[2], output2_ex[2]);

  test::ExpectTensorNear<float>(output1_tensor_expected, output1_tensor, 1e-8);
  test::ExpectTensorEqual<int64>(output2_tensor_expected, output2_tensor);
}

TEST_F(FusedEmbeddingLocalSparseLookUpOpTest, FloatSumCpu) {

  TF_EXPECT_OK(NodeDefBuilder("FusedSafeEmbeddingLookupSparse",
                              "FusedSafeEmbeddingLookupSparse")
                    .Input(FakeInput(DT_FLOAT))
                    .Input(FakeInput(DT_INT64))
                    .Input(FakeInput(DT_INT64))
                    .Input(FakeInput(DT_INT64))
                    .Input(FakeInput(DT_INT64))
                    .Attr("T", DT_FLOAT)
                    .Attr("combiner", "sum")
                    .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());

  const int nnz = 9;
  const int batch_size = 5;
  const int emb_vector_dim = 4;
  const int entries = 8;
  const int gathered_weight_size = 3;

  Tensor sp_values(DT_INT64, {nnz});
  Tensor sp_weight(DT_INT64, {nnz});
  Tensor sp_indices(DT_INT64, {nnz, 2});
  Tensor sp_dense_shape(DT_INT64, {2});
  Tensor emb_variable(DT_FLOAT, {gathered_weight_size, emb_vector_dim});

  // [1 1 0 4 1 1 1 0 1] -> [1 0 4], [0 0 1 2 0 0 0 1 0]
  test::FillValues<int64>(&sp_values, {0, 0, 1, 2, 0, 0, 0, 1, 0});
  test::FillValues<int64>(&sp_weight, {0, 0, 1, 2, 0, 0, 0, 1, 0});
  // [0 0 0 1 1 3 3 4 4]
  test::FillValues<int64>(&sp_indices, {0, 1, 0, 3, 0, 6, 1, 3, 1, 6,
                                        3, 3, 3, 4, 4, 1, 4, 7});
  test::FillValues<int64>(&sp_dense_shape, {batch_size, entries});
  test::FillValues<float>(
      &emb_variable,
      {-0.023765106, -0.248630840,  0.275294270, 0.228118000,
       -0.147108670, -0.298352200, -0.067187610, 0.274558250,
        0.491792620, -0.094891705,  0.064489834, 0.058840238});
  
  AddInputFromArray<float>(emb_variable.shape(), emb_variable.flat<float>());
  AddInputFromArray<int64>(sp_values.shape(), sp_values.flat<int64>());
  AddInputFromArray<int64>(sp_dense_shape.shape(),
                            sp_dense_shape.flat<int64>());
  AddInputFromArray<int64>(sp_indices.shape(), sp_indices.flat<int64>());

  TF_ASSERT_OK(RunOpKernel());

  Tensor emb_vector_expected(DT_FLOAT, {batch_size, emb_vector_dim});

  test::FillValues<float>(&emb_vector_expected,
      {-0.19463888, -0.79561390, 0.48340094, 0.73079425,
        0.46802750, -0.34352255, 0.33978412, 0.28695825,
        0.00000000,  0.00000000, 0.00000000, 0.00000000,
       -0.04753021, -0.49726167, 0.55058855, 0.45623600,
       -0.17087378, -0.54698306, 0.20810667, 0.50267625});

  const Tensor& emb_vector = *GetOutput(0);

  float *output = (float *)emb_vector.tensor_data().data();
  float *output_ex = (float *)emb_vector_expected.tensor_data().data();

  test::ExpectTensorNear<float>(emb_vector_expected, emb_vector, 1e-8);
}

TEST_F(FusedEmbeddingLocalSparseLookUpOpTest, FloatMeanCpu) {

  TF_EXPECT_OK(NodeDefBuilder("FusedSafeEmbeddingLookupSparse",
                              "FusedSafeEmbeddingLookupSparse")
                    .Input(FakeInput(DT_FLOAT))
                    .Input(FakeInput(DT_INT64))
                    .Input(FakeInput(DT_INT64))
                    .Input(FakeInput(DT_INT64))
                    .Input(FakeInput(DT_INT64))
                    .Attr("T", DT_FLOAT)
                    .Attr("combiner", "mean")
                    .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());

  const int nnz = 9;
  const int batch_size = 5;
  const int emb_vector_dim = 4;
  const int entries = 8;
  const int gathered_weight_size = 3;

  Tensor sp_values(DT_INT64, {nnz});
  Tensor sp_weight(DT_INT64, {nnz});
  Tensor sp_indices(DT_INT64, {nnz, 2});
  Tensor sp_dense_shape(DT_INT64, {2});
  Tensor emb_variable(DT_FLOAT, {gathered_weight_size, emb_vector_dim});

  // [1 1 0 4 1 1 1 0 1] -> [1 0 4], [0 0 1 2 0 0 0 1 0]
  test::FillValues<int64>(&sp_values, {0, 0, 1, 2, 0, 0, 0, 1, 0});
  test::FillValues<int64>(&sp_weight, {0, 0, 1, 2, 0, 0, 0, 1, 0});
  // [0 0 0 1 1 3 3 4 4]
  test::FillValues<int64>(&sp_indices, {0, 1, 0, 3, 0, 6, 1, 3, 1, 6,
                                        3, 3, 3, 4, 4, 1, 4, 7});
  test::FillValues<int64>(&sp_dense_shape, {batch_size, entries});
  test::FillValues<float>(
      &emb_variable,
      {-0.02299355, -0.247596220,  0.27484232, 0.226618130,
       -0.14686598, -0.297978460, -0.06733219, 0.273977040,
        0.49191360, -0.094738655,  0.06426916, 0.058573183});

  AddInputFromArray<float>(emb_variable.shape(), emb_variable.flat<float>());
  AddInputFromArray<int64>(sp_values.shape(), sp_values.flat<int64>());
  AddInputFromArray<int64>(sp_dense_shape.shape(),
                            sp_dense_shape.flat<int64>());
  AddInputFromArray<int64>(sp_indices.shape(), sp_indices.flat<int64>());

  TF_ASSERT_OK(RunOpKernel());

  Tensor emb_vector_expected(DT_FLOAT, {batch_size, emb_vector_dim});
  test::FillValues<float>(&emb_vector_expected,
      {-0.064284360, -0.26439032, 0.160784140, 0.24240442,
        0.234460030, -0.17116743, 0.169555740, 0.14259565,
        0.000000000,  0.00000000, 0.000000000, 0.00000000,
       -0.022993550, -0.24759622, 0.274842320, 0.22661813,
       -0.084929764, -0.27278733, 0.103755064, 0.25029758});

  const Tensor& emb_vector = *GetOutput(0);

  float *output = (float *)emb_vector.tensor_data().data();
  float *output_ex = (float *)emb_vector_expected.tensor_data().data();

  test::ExpectTensorNear<float>(emb_vector_expected, emb_vector, 1e-7);
}

TEST_F(FusedEmbeddingLocalSparseLookUpOpTest, GradFloatSumCpu) {

  TF_EXPECT_OK(NodeDefBuilder("FusedSafeEmbeddingLookupSparseGrad",
                              "FusedSafeEmbeddingLookupSparseGrad")
                    .Input(FakeInput(DT_FLOAT)) // gradients
                    .Input(FakeInput(DT_INT64)) // unique_id
                    .Input(FakeInput(DT_INT64)) // unique_indices
                    .Input(FakeInput(DT_INT64)) // dense_shape
                    .Input(FakeInput(DT_INT64)) // indices
                    .Attr("T", DT_FLOAT)
                    .Attr("Tinput", DT_INT64)
                    .Attr("Tindices", DT_INT64)
                    .Attr("Tdense_shape", DT_INT64)
                    .Attr("combiner", "sum")
                    .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());

  const int unique_size = 3;
  const int nnz = 9;
  const int batch_size = 5;
  const int emb_vector_dim = 4;
  const int entries = 8;

  Tensor unique_id(DT_INT64, {unique_size});
  Tensor unique_indices(DT_INT64, {nnz});
  Tensor sp_indices(DT_INT64, {nnz, 2});
  Tensor sp_dense_shape(DT_INT64, {2});
  Tensor grad_variable(DT_FLOAT, {batch_size, emb_vector_dim});

  test::FillValues<float>(
      &grad_variable,
      {0.0076283700764179229736328125, 0.0121669657528400421142578125, -0.0049919090233743190765380859, -0.0190300568938255310058593750,
       0.0065145129337906837463378906, 0.0117923058569431304931640625, -0.0164990965276956558227539062, -0.0200323350727558135986328125,
       0.0100607946515083312988281250, 0.0153625328093767166137695312, -0.0056031607091426849365234375, -0.0249206330627202987670898438,
       0.0099571626633405685424804688, 0.0154269225895404815673828125, -0.0055019007995724678039550781, -0.0239365808665752410888671875,
       0.0084272380918264389038085938, 0.0152924191206693649291992188, -0.0086676068603992462158203125, -0.0239860229194164276123046875});
  test::FillValues<int64>(&unique_id, {1, 0, 4});
  test::FillValues<int64>(&unique_indices, {0, 0, 1, 2, 0, 0, 0, 1, 0});
  test::FillValues<int64>(&sp_indices, {0, 1, 0, 3, 0, 6, 1, 3, 1, 6,
                                        3, 3, 3, 4, 4, 1, 4, 7});
  test::FillValues<int64>(&sp_dense_shape, {batch_size, entries});

  AddInputFromArray<float>(grad_variable.shape(), grad_variable.flat<float>());
  AddInputFromArray<int64>(unique_id.shape(), unique_id.flat<int64>());
  AddInputFromArray<int64>(unique_indices.shape(), unique_indices.flat<int64>());
  AddInputFromArray<int64>(sp_indices.shape(), sp_indices.flat<int64>());
  AddInputFromArray<int64>(sp_dense_shape.shape(), sp_dense_shape.flat<int64>());

  TF_ASSERT_OK(RunOpKernel());

  Tensor output1_tensor_expected(DT_FLOAT, {unique_size, emb_vector_dim});
  Tensor output2_tensor_expected(DT_INT64, {unique_size});
  test::FillValues<float>(&output1_tensor_expected,
      {0.0501128211617469787597656250, 0.0822724997997283935546875000, -0.0461543202400207519531250000, -0.1299516409635543823242187500,
       0.0160556081682443618774414062, 0.0274593848735094070434570312, -0.0136595163494348526000976562, -0.0430160798132419586181640625,
       0.0065145129337906837463378906, 0.0117923058569431304931640625, -0.0164990965276956558227539062, -0.0200323369354009628295898438});
  test::FillValues<int64>(&output2_tensor_expected, {1, 0, 4});
  float *output1_ex = (float *)output1_tensor_expected.tensor_data().data();
  int64 *output2_ex = (int64 *)output2_tensor_expected.tensor_data().data();

  const Tensor& output1_tensor = *GetOutput(0);
  const Tensor& output2_tensor = *GetOutput(1);

  float *output1 = (float *)output1_tensor.tensor_data().data();
  int64 *output2 = (int64 *)output2_tensor.tensor_data().data();

  printf("out = %.28f , expect = %.28f\n", output1[11], output1_ex[11]);

  test::ExpectTensorNear<float>(output1_tensor_expected, output1_tensor, 1e-8);
  test::ExpectTensorEqual<int64>(output2_tensor_expected, output2_tensor);
}

TEST_F(FusedEmbeddingLocalSparseLookUpOpTest, GradFloatMeanCpu) {

  TF_EXPECT_OK(NodeDefBuilder("FusedSafeEmbeddingLookupSparseGrad",
                              "FusedSafeEmbeddingLookupSparseGrad")
                    .Input(FakeInput(DT_FLOAT)) // gradients
                    .Input(FakeInput(DT_INT64)) // unique_id
                    .Input(FakeInput(DT_INT64)) // unique_indices
                    .Input(FakeInput(DT_INT64)) // dense_shape
                    .Input(FakeInput(DT_INT64)) // indices
                    .Attr("T", DT_FLOAT)
                    .Attr("Tinput", DT_INT64)
                    .Attr("Tindices", DT_INT64)
                    .Attr("Tdense_shape", DT_INT64)
                    .Attr("combiner", "mean")
                    .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());

  const int unique_size = 3;
  const int nnz = 9;
  const int batch_size = 5;
  const int emb_vector_dim = 4;
  const int entries = 8;

  Tensor unique_id(DT_INT64, {unique_size});
  Tensor unique_indices(DT_INT64, {nnz});
  Tensor sp_indices(DT_INT64, {nnz, 2});
  Tensor sp_dense_shape(DT_INT64, {2});
  Tensor grad_variable(DT_FLOAT, {batch_size, emb_vector_dim});

  test::FillValues<float>(
      &grad_variable,
      {0.0103125420, 0.018807490, -0.0106398590, -0.029409127,
       0.0054132286, 0.013920069, -0.0190976150, -0.023196392,
       0.0100601720, 0.015330995, -0.0055795530, -0.024889620,
       0.0108455080, 0.018832123, -0.0095151365, -0.029357582,
       0.0100478110, 0.018798435, -0.0112019650, -0.029439624});
  test::FillValues<int64>(&unique_id, {1, 0, 4});
  test::FillValues<int64>(&unique_indices, {0, 0, 1, 2, 0, 0, 0, 1, 0});
  test::FillValues<int64>(&sp_indices, {0, 1, 0, 3, 0, 6, 1, 3, 1, 6,
                                        3, 3, 3, 4, 4, 1, 4, 7});
  test::FillValues<int64>(&sp_dense_shape, {batch_size, entries});

  AddInputFromArray<float>(grad_variable.shape(), grad_variable.flat<float>());
  AddInputFromArray<int64>(unique_id.shape(), unique_id.flat<int64>());
  AddInputFromArray<int64>(unique_indices.shape(), unique_indices.flat<int64>());
  AddInputFromArray<int64>(sp_indices.shape(), sp_indices.flat<int64>());
  AddInputFromArray<int64>(sp_dense_shape.shape(), sp_dense_shape.flat<int64>());

  TF_ASSERT_OK(RunOpKernel());

  Tensor output1_tensor_expected(DT_FLOAT, {unique_size, emb_vector_dim});
  Tensor output2_tensor_expected(DT_INT64, {unique_size});
  test::FillValues<float>(&output1_tensor_expected,
      {0.0254510570, 0.0477297000, -0.0317581670, -0.075281680,
       0.0084614195, 0.0156683810, -0.0091476020, -0.024522856,
       0.0027066143, 0.0069600344, -0.0095488075, -0.011598196});
  test::FillValues<int64>(&output2_tensor_expected, {1, 0, 4});
  float *output1_ex = (float *)output1_tensor_expected.tensor_data().data();
  int64 *output2_ex = (int64 *)output2_tensor_expected.tensor_data().data();

  const Tensor& output1_tensor = *GetOutput(0);
  const Tensor& output2_tensor = *GetOutput(1);

  float *output1 = (float *)output1_tensor.tensor_data().data();
  int64 *output2 = (int64 *)output2_tensor.tensor_data().data();

  test::ExpectTensorNear<float>(output1_tensor_expected, output1_tensor, 1e-8);
  test::ExpectTensorEqual<int64>(output2_tensor_expected, output2_tensor);
}

}  // namespace
}  // namespace tensorflow