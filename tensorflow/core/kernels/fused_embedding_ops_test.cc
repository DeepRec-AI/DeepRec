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
enum Combiner { Sqrtn, Mean, Sum };

template <typename T, Combiner combiner>
void fill_emb_vector_expected(Tensor* expected);

template <>
void fill_emb_vector_expected<float, Sqrtn>(Tensor* expected) {
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
void fill_emb_vector_expected<float, Mean>(Tensor* expected) {
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
void fill_emb_vector_expected<float, Sum>(Tensor* expected) {
  test::FillValues<float>(
      expected, {32.0,  34.0,  36.0,  38.0,  40.0,  42.0,  44.0,  46.0,
                 128.0, 131.0, 134.0, 137.0, 140.0, 143.0, 146.0, 149.0,
                 216.0, 219.0, 222.0, 225.0, 228.0, 231.0, 234.0, 237.0,
                 152.0, 154.0, 156.0, 158.0, 160.0, 162.0, 164.0, 166.0});
}

class FusedEmbeddingSparseLookUpOpTest : public OpsTestBase {
 protected:
  template <typename T, Combiner combiner>
  void Run(Device device) {
    if (device == Device::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }
    DataType dtype = DataTypeToEnum<T>::value;
    std::string combiner_str;
    if (combiner == Sqrtn) {
      combiner_str = "sqrtn";
    } else if (combiner == Mean) {
      combiner_str = "mean";
    } else {
      combiner_str = "sum";
    }

    TF_EXPECT_OK(NodeDefBuilder("fused_embedding_sparse_look_up",
                                "FusedEmbeddingSparseLookUp")
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(dtype))
                     .Attr("T", dtype)
                     .Attr("combiner", combiner_str)
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
    fill_emb_vector_expected<T, combiner>(&emb_vector_expected);
    test::FillValues<int32>(&sp_values_offset_expected, {0, 2, 5, 8});

    const Tensor& emb_vector = *GetOutput(0);
    const Tensor& values_offset = *GetOutput(1);
    TF_EXPECT_OK(device_->Sync());

    test::ExpectTensorNear<T>(emb_vector_expected, emb_vector, 1e-5);
    test::ExpectTensorEqual<int32>(sp_values_offset_expected, values_offset);
  }
};

template <typename T, Combiner combiner>
void fill_grad_expected(Tensor* expected);

template <>
void fill_grad_expected<float, Sqrtn>(Tensor* expected) {
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
void fill_grad_expected<float, Mean>(Tensor* expected) {
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
void fill_grad_expected<float, Sum>(Tensor* expected) {
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

class FusedEmbeddingSparseLookUpGradOpTest : public OpsTestBase {
 protected:
  template <typename T, Combiner combiner>
  void Run(Device device) {
    if (device == Device::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }
    DataType dtype = DataTypeToEnum<T>::value;
    std::string combiner_str;
    if (combiner == Sqrtn) {
      combiner_str = "sqrtn";
    } else if (combiner == Mean) {
      combiner_str = "mean";
    } else {
      combiner_str = "sum";
    }

    TF_EXPECT_OK(NodeDefBuilder("fused_embedding_sparse_look_up_grad",
                                "FusedEmbeddingSparseLookUpGrad")
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT32))
                     .Attr("T", dtype)
                     .Attr("combiner", combiner_str)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());

    const int nnz = 10;
    const int batch_size = 4;
    const int emb_vector_dim = 8;

    Tensor top_grad(dtype, {batch_size, emb_vector_dim});
    Tensor sp_values(DT_INT64, {nnz});
    Tensor sp_values_offset(DT_INT32, {batch_size});

    test::FillValues<T>(
        &top_grad,
        {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0,
         11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
         22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0});
    test::FillValues<int64>(&sp_values, {3, 1, 4, 5, 7, 3, 12, 12, 15, 4});
    test::FillValues<int32>(&sp_values_offset, {0, 2, 5, 8});

    AddInputFromArray<T>(top_grad.shape(), top_grad.flat<T>());
    AddInputFromArray<int64>(sp_values.shape(), sp_values.flat<int64>());
    AddInputFromArray<int32>(sp_values_offset.shape(),
                             sp_values_offset.flat<int32>());

    TF_ASSERT_OK(RunOpKernel());

    Tensor grad_expected(dtype, {nnz, emb_vector_dim});
    fill_grad_expected<T, combiner>(&grad_expected);

    const Tensor& grad = *GetOutput(0);
    TF_EXPECT_OK(device_->Sync());

    test::ExpectTensorNear<T>(grad_expected, grad, 1e-5);
  }
};

#ifdef GOOGLE_CUDA
TEST_F(FusedEmbeddingSparseLookUpOpTest, EmbeddingSparseLookUpFloatSqrtnGpu) {
  Run<float, Sqrtn>(Device::GPU);
}

TEST_F(FusedEmbeddingSparseLookUpOpTest, EmbeddingSparseLookUpFloatMeanGpu) {
  Run<float, Mean>(Device::GPU);
}

TEST_F(FusedEmbeddingSparseLookUpOpTest, EmbeddingSparseLookUpFloatSumGpu) {
  Run<float, Sum>(Device::GPU);
}

TEST_F(FusedEmbeddingSparseLookUpGradOpTest,
       EmbeddingSparseLookUpGradFloatGpu) {
  Run<float, Sqrtn>(Device::GPU);
}

TEST_F(FusedEmbeddingSparseLookUpGradOpTest,
       EmbeddingSparseLookUpGradFloatMeanGpu) {
  Run<float, Mean>(Device::GPU);
}

TEST_F(FusedEmbeddingSparseLookUpGradOpTest,
       EmbeddingSparseLookUpGradFloatSumGpu) {
  Run<float, Sum>(Device::GPU);
}

#endif

}  // namespace
}  // namespace tensorflow