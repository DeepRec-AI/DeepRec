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
enum Combiner { Sqrt };

template <typename T, Combiner combiner>
void fill_emb_vector_expected(Tensor* expected);

template <>
void fill_emb_vector_expected<float, Sqrt>(Tensor* expected) {
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

class FusedEmbeddingOpTest : public OpsTestBase {
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
    if(combiner == Sqrt) {
      combiner_str = "sqrt";
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
    Tensor emb_variable(DT_FLOAT, {bucket_size, emb_vector_dim});

    test::FillValues<int64>(&sp_values, {3, 1, 4, 5, 7, 3, 12, 12, 15, 4});
    test::FillValues<int64>(&sp_indices, {0, 1, 0, 5, 1, 2, 1, 1, 1, 7,
                                          2, 1, 2, 4, 2, 7, 3, 0, 3, 6});
    test::FillValues<int64>(&sp_dense_shape, {batch_size, entries});
    test::FillValues<T>(
        &emb_variable,
        {
            0.0,   1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   7.0,   8.0,
            9.0,   10.0,  11.0,  12.0,  13.0,  14.0,  15.0,  16.0,  17.0,
            18.0,  19.0,  20.0,  21.0,  22.0,  23.0,  24.0,  25.0,  26.0,
            27.0,  28.0,  29.0,  30.0,  31.0,  32.0,  33.0,  34.0,  35.0,
            36.0,  37.0,  38.0,  39.0,  40.0,  41.0,  42.0,  43.0,  44.0,
            45.0,  46.0,  47.0,  48.0,  49.0,  50.0,  51.0,  52.0,  53.0,
            54.0,  55.0,  56.0,  57.0,  58.0,  59.0,  60.0,  61.0,  62.0,
            63.0,  64.0,  65.0,  66.0,  67.0,  68.0,  69.0,  70.0,  71.0,
            72.0,  73.0,  74.0,  75.0,  76.0,  77.0,  78.0,  79.0,  80.0,
            81.0,  82.0,  83.0,  84.0,  85.0,  86.0,  87.0,  88.0,  89.0,
            90.0,  91.0,  92.0,  93.0,  94.0,  95.0,  96.0,  97.0,  98.0,
            99.0,  100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0,
            108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0,
            117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0,
            126.0, 127.0,
        });

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

#ifdef GOOGLE_CUDA
TEST_F(FusedEmbeddingOpTest, FusedEmbeddingFloatGpu) {
  Run<float, Sqrt>(Device::GPU);
}
#endif

}  // namespace
}  // namespace tensorflow