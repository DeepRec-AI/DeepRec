#include <sys/resource.h>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

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

class GroupEmbeddingForWardOpTest : public OpsTestBase {
 protected:
  template <typename TKey, typename TValue, TestCase test_case>
  void Run() {
    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));

    DataType k_dtype = DataTypeToEnum<TKey>::value;
    DataType v_dtype = DataTypeToEnum<TValue>::value;
    std::string combiner_str;
    float max_norm;

    const int nnz = 10;
    const int batch_size = 4;
    const int emb_vector_dim = 8;
    const int entries = 8;
    const int bucket_size = 16;
    const int num_lookups = 2;
    get_node_attr_from_test_case<test_case>(combiner_str, max_norm);

    TF_EXPECT_OK(NodeDefBuilder("multi_embedding_sparse_look_up",
                                "MultiEmbeddingSparseLookUp")
                     .Input(FakeInput(num_lookups, v_dtype))
                     .Input(FakeInput(num_lookups, k_dtype))
                     .Input(FakeInput(num_lookups, DT_INT64))
                     .Input(FakeInput(num_lookups, DT_INT64))
                     .Attr("dtype", v_dtype)
                     .Attr("Tkeys", k_dtype)
                     .Attr("combiner", combiner_str)
                     .Attr("max_norm", max_norm)
                     .Attr("dimension", emb_vector_dim)
                     .Attr("num_lookups", num_lookups)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());

    for (int i = 0; i < num_lookups; ++i) {
      Tensor emb_variable(v_dtype, {bucket_size, emb_vector_dim});
      test::FillValues<TValue>(
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

      AddInputFromArray<TValue>(emb_variable.shape(),
                                emb_variable.flat<TValue>());
    }

    for (int i = 0; i < num_lookups; ++i) {
      Tensor sp_values(k_dtype, {nnz});
      test::FillValues<TKey>(&sp_values, {3, 1, 4, 5, 7, 3, 12, 12, 15, 4});
      AddInputFromArray<TKey>(sp_values.shape(), sp_values.flat<TKey>());
    }

    for (int i = 0; i < num_lookups; ++i) {
      Tensor sp_indices(DT_INT64, {nnz, 2});
      test::FillValues<int64>(&sp_indices, {0, 1, 0, 5, 1, 2, 1, 1, 1, 7,
                                            2, 1, 2, 4, 2, 7, 3, 0, 3, 6});
      AddInputFromArray<int64>(sp_indices.shape(), sp_indices.flat<int64>());
    }

    for (int i = 0; i < num_lookups; ++i) {
      Tensor sp_dense_shape(DT_INT64, {2});
      test::FillValues<int64>(&sp_dense_shape, {batch_size, entries});
      AddInputFromArray<int64>(sp_dense_shape.shape(),
                               sp_dense_shape.flat<int64>());
    }

    TF_ASSERT_OK(RunOpKernel());

    Tensor emb_vector_expected(v_dtype, {batch_size, emb_vector_dim});
    Tensor sp_values_offset_expected(DT_INT32, {batch_size});
    fill_emb_vector_expected<test_case>(&emb_vector_expected);
    test::FillValues<int32>(&sp_values_offset_expected, {0, 2, 5, 8});

    TF_EXPECT_OK(device_->Sync());

    for (int i = 0; i < num_lookups; ++i) {
      const Tensor& emb_vector = *GetOutput(i);
      const Tensor& values_offset = *GetOutput(num_lookups + i);

      test::ExpectTensorNear<TValue>(emb_vector_expected, emb_vector, 1e-4);
      test::ExpectTensorEqual<int32>(sp_values_offset_expected, values_offset);
    }
  }
};

#ifdef GOOGLE_CUDA
TEST_F(GroupEmbeddingForWardOpTest, EmbeddingLocalSparseLookUpFloatSqrtnGpu) {
  Run<int64, float, Sqrtn>();
}

TEST_F(GroupEmbeddingForWardOpTest, EmbeddingLocalSparseLookUpFloatMeanGpu) {
  Run<int64, float, Mean>();
}

TEST_F(GroupEmbeddingForWardOpTest, EmbeddingLocalSparseLookUpFloatSumGpu) {
  Run<int64, float, Sum>();
}

TEST_F(GroupEmbeddingForWardOpTest,
       EmbeddingLocalSparseLookUpFloatSqrtnAndMaxNorm200Gpu) {
  Run<int64, float, SqrtnAndMaxNorm200>();
}
#endif  // GOOGLE_CUDA

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
      {0.00000000,  0.50000000,  1.00000000,  1.50000000,  2.00000000,
       2.50000000,  3.00000000,  3.50000000,  0.00000000,  0.50000000,
       1.00000000,  1.50000000,  2.00000000,  2.50000000,  3.00000000,
       3.50000000,  2.65028572,  2.98157120,  3.31285667,  3.64414287,
       3.97542834,  4.30671406,  4.63799953,  4.96928549,  2.16437674,
       2.43492365,  2.70547056,  2.97601795,  3.24656487,  3.51711202,
       3.78765893,  4.05820608,  1.58337951,  1.78130186,  1.97922409,
       2.17714667,  2.37506914,  2.57299161,  2.77091384,  2.96883631,
       5.33333349,  5.66666651,  6.00000000,  6.33333349,  6.66666651,
       7.00000000,  7.33333349,  7.66666651,  1.89459133,  2.01300311,
       2.13141513,  2.24982715,  2.36823893,  2.48665094,  2.60506320,
       2.72347474,  1.89459133,  2.01300311,  2.13141513,  2.24982715,
       2.36823893,  2.48665094,  2.60506320,  2.72347474,  3.43474555,
       3.57786012,  3.72097445,  3.86408877,  4.00720310,  4.15031767,
       4.29343224,  4.43654633,  11.92628479, 12.42321396, 12.92014217,
       13.41707039, 13.91399956, 14.41092777, 14.90785599, 15.40478516});
}

class GroupEmbeddingBackWardOpTest : public OpsTestBase {
 protected:
  template <typename TKey, typename TValue, TestCase test_case>
  void Run() {
    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));

    DataType k_dtype = DataTypeToEnum<TKey>::value;
    DataType v_dtype = DataTypeToEnum<TValue>::value;
    std::string combiner_str;
    float max_norm;

    const int nnz = 10;
    const int batch_size = 4;
    const int emb_vector_dim = 8;
    const int entries = 8;
    const int bucket_size = 16;
    const int num_lookups = 2;
    get_node_attr_from_test_case<test_case>(combiner_str, max_norm);

    TF_EXPECT_OK(NodeDefBuilder("multi_embedding_sparse_look_up_grad",
                                "MultiEmbeddingSparseLookUpGrad")
                     .Input(FakeInput(num_lookups, DT_FLOAT))
                     .Input(FakeInput(num_lookups, v_dtype))
                     .Input(FakeInput(num_lookups, k_dtype))
                     .Input(FakeInput(num_lookups, DT_INT32))
                     .Attr("dtype", v_dtype)
                     .Attr("Tkeys", k_dtype)
                     .Attr("combiner", combiner_str)
                     .Attr("max_norm", max_norm)
                     .Attr("dimension", emb_vector_dim)
                     .Attr("num_lookups", num_lookups)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());

    for (int i = 0; i < num_lookups; ++i) {
      Tensor top_grad(DT_FLOAT, {batch_size, emb_vector_dim});
      test::FillValues<float>(
          &top_grad,
          {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0,
           11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
           22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0});

      AddInputFromArray<float>(top_grad.shape(), top_grad.flat<float>());
    }

    for (int i = 0; i < num_lookups; ++i) {
      Tensor emb_variable(v_dtype, {bucket_size, emb_vector_dim});
      test::FillValues<TValue>(
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
      AddInputFromArray<TValue>(emb_variable.shape(),
                                emb_variable.flat<TValue>());
    }

    for (int i = 0; i < num_lookups; ++i) {
      Tensor sp_values(k_dtype, {nnz});
      test::FillValues<TKey>(&sp_values, {3, 1, 4, 5, 7, 3, 12, 12, 15, 4});
      AddInputFromArray<TKey>(sp_values.shape(), sp_values.flat<TKey>());
    }

    for (int i = 0; i < num_lookups; ++i) {
      Tensor sp_values_offset(DT_INT32, {batch_size});
      test::FillValues<int32>(&sp_values_offset, {0, 2, 5, 8});
      AddInputFromArray<int32>(sp_values_offset.shape(),
                               sp_values_offset.flat<int32>());
    }

    TF_ASSERT_OK(RunOpKernel());

    Tensor grad_expected(v_dtype, {nnz, emb_vector_dim});
    fill_grad_expected<test_case>(&grad_expected);

    TF_EXPECT_OK(device_->Sync());

    for (int i = 0; i < num_lookups; ++i) {
      const Tensor& grad = *GetOutput(i);
      test::ExpectTensorNear<TValue>(grad_expected, grad, 1e-4);
    }
  }
};

#ifdef GOOGLE_CUDA
TEST_F(GroupEmbeddingBackWardOpTest, EmbeddingLocalSparseLookUpGradFloatGpu) {
  Run<int64, float, Sqrtn>();
}

TEST_F(GroupEmbeddingBackWardOpTest,
       EmbeddingLocalSparseLookUpGradFloatMeanGpu) {
  Run<int64, float, Mean>();
}

TEST_F(GroupEmbeddingBackWardOpTest,
       EmbeddingLocalSparseLookUpGradFloatSumGpu) {
  Run<int64, float, Sum>();
}

TEST_F(GroupEmbeddingBackWardOpTest,
       EmbeddingLocalSparseLookUpGradFloatMeanAndMaxNorm100Gpu) {
  Run<int64, float, MeanAndMaxNorm100>();
}

#endif  // GOOGLE_CUDA

}  // namespace tensorflow