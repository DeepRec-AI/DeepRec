#include <sys/resource.h>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/resource_mgr.h"
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

enum DEVICE { CPU, GPU };

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
void fill_var_vector_expected(Tensor* expected);

template <>
void fill_var_vector_expected<Sqrtn>(Tensor* expected) {
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
void fill_var_vector_expected<Mean>(Tensor* expected) {
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
void fill_var_vector_expected<Sum>(Tensor* expected) {
  test::FillValues<float>(
      expected, {32.0,  34.0,  36.0,  38.0,  40.0,  42.0,  44.0,  46.0,
                 128.0, 131.0, 134.0, 137.0, 140.0, 143.0, 146.0, 149.0,
                 216.0, 219.0, 222.0, 225.0, 228.0, 231.0, 234.0, 237.0,
                 152.0, 154.0, 156.0, 158.0, 160.0, 162.0, 164.0, 166.0});
}

template <>
void fill_var_vector_expected<SqrtnAndMaxNorm200>(Tensor* expected) {
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

class GroupVariableForWardOpTest : public OpsTestBase {
 protected:
  template <typename TKey, typename TValue, TestCase test_case>
  void Run(DEVICE device) {
    if (device == DEVICE::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }

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

    TF_EXPECT_OK(NodeDefBuilder("group_variable_lookup", "GroupVariableLookup")
                     .Input(FakeInput(num_lookups, v_dtype))   // ev
                     .Input(FakeInput(num_lookups, k_dtype))   // sp_values
                     .Input(FakeInput(num_lookups, DT_INT64))  // sp_indices
                     .Input(FakeInput(num_lookups, v_dtype))   // sp_weights
                     .Input(FakeInput(DT_INT32))               // dense_shape
                     .Input(FakeInput(v_dtype))                // default_value
                     .Attr("dtype", v_dtype)
                     .Attr("Tkeys", k_dtype)
                     .Attr("combiner", combiner_str)
                     .Attr("max_norm", max_norm)
                     .Attr("dimension", emb_vector_dim)
                     .Attr("num_lookups", num_lookups)
                     .Attr("ignore_weights", true)
                     .Attr("is_use_default_value_tensor", false)
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
      Tensor sp_indices(DT_INT64, {nnz});
      test::FillValues<int64>(&sp_indices, {0, 0, 1, 1, 1, 2, 2, 2, 3, 3});
      AddInputFromArray<int64>(sp_indices.shape(), sp_indices.flat<int64>());
    }

    for (int i = 0; i < num_lookups; ++i) {
      Tensor sp_weights(v_dtype, {nnz});
      test::FillValues<TValue>(&sp_weights, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                             1.0f, 1.0f, 1.0f, 1.0f});
      AddInputFromArray<TValue>(sp_weights.shape(), sp_weights.flat<TValue>());
    }

    Tensor* batch_size_tensor =
        AddInput(DataTypeToEnum<int32>::v(), TensorShape({}));
    auto batch_size_data = batch_size_tensor->flat<int>().data();
    batch_size_data[0] = batch_size;

    Tensor* default_v_tensor =
        AddInput(DataTypeToEnum<TValue>::v(), TensorShape({}));
    auto default_v = default_v_tensor->flat<float>().data();
    default_v[0] = 1.0f;

    TF_ASSERT_OK(RunOpKernel());

    Tensor emb_vector_expected(v_dtype, {batch_size, emb_vector_dim});
    Tensor unique_values_expected(DT_INT64, {7});
    Tensor unique_idx_expected(DT_INT32, {nnz});
    Tensor batch_size_expected(DT_INT32, {batch_size});

    fill_var_vector_expected<test_case>(&emb_vector_expected);

    if (device == DEVICE::GPU) {
      test::FillValues<int32>(&batch_size_expected, {0, 2, 5, 8});
    } else {
      test::FillValues<int64>(&unique_values_expected, {3, 1, 4, 5, 7, 12, 15});
      test::FillValues<int32>(&unique_idx_expected,
                              {0, 1, 2, 3, 4, 0, 5, 5, 6, 2});
      test::FillValues<int32>(&batch_size_expected, {2, 5, 8, 10});
    }
    TF_EXPECT_OK(device_->Sync());

    for (int i = 0; i < num_lookups; ++i) {
      const Tensor& emb_vector = *GetOutput(i);
      const Tensor& unique_values = *GetOutput(num_lookups + i);
      const Tensor& unique_idx_output = *GetOutput(2 * num_lookups + i);
      const Tensor& batch_size_output = *GetOutput(3 * num_lookups + i);
      test::ExpectTensorNear<TValue>(emb_vector_expected, emb_vector, 1e-4);
      if (device == DEVICE::CPU) {
        test::ExpectTensorEqual<int64>(unique_values_expected, unique_values);
        test::ExpectTensorEqual<int32>(unique_idx_expected, unique_idx_output);
      }
      test::ExpectTensorEqual<int32>(batch_size_expected, batch_size_output);
    }
  }
};

#ifdef GOOGLE_CUDA
TEST_F(GroupVariableForWardOpTest, EmbeddingLocalSparseLookUpFloatSqrtnGpu) {
  Run<int64, float, Sqrtn>(DEVICE::GPU);
}

TEST_F(GroupVariableForWardOpTest, EmbeddingLocalSparseLookUpFloatMeanGpu) {
  Run<int64, float, Mean>(DEVICE::GPU);
}

TEST_F(GroupVariableForWardOpTest, EmbeddingLocalSparseLookUpFloatSumGpu) {
  Run<int64, float, Sum>(DEVICE::GPU);
}

// TEST_F(GroupVariableForWardOpTest,
//        EmbeddingLocalSparseLookUpFloatSqrtnAndMaxNorm200Gpu) {
//   Run<int64, float, SqrtnAndMaxNorm200>(DEVICE::GPU);
// }
#endif  // GOOGLE_CUDA

TEST_F(GroupVariableForWardOpTest, EmbeddingLocalSparseLookUpFloatSqrtnCpu) {
  Run<int64, float, Sqrtn>(DEVICE::CPU);
}

TEST_F(GroupVariableForWardOpTest, EmbeddingLocalSparseLookUpFloatMeanCpu) {
  Run<int64, float, Mean>(DEVICE::CPU);
}

TEST_F(GroupVariableForWardOpTest, EmbeddingLocalSparseLookUpFloatSumCpu) {
  Run<int64, float, Sum>(DEVICE::CPU);
}

// TEST_F(GroupVariableForWardOpTest,
//        EmbeddingLocalSparseLookUpFloatSqrtnAndMaxNorm200Cpu) {
//   Run<int64, float, SqrtnAndMaxNorm200>(DEVICE::CPU);
// }

template <DEVICE device, TestCase test_case>
void fill_var_grad_expected(Tensor* expected);

template <>
void fill_var_grad_expected<DEVICE::CPU, Sqrtn>(Tensor* expected) {
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
                 16.970563888549805, 17.677669525146484, 18.384777069091797,
                 19.091882705688477, 19.79899024963379,  20.5060977935791,
                 21.21320343017578,  21.920310974121094});
}

template <>
void fill_var_grad_expected<DEVICE::CPU, Mean>(Tensor* expected) {
  test::FillValues<float>(
      expected, {0.000000000000000,  0.500000000000000,  1.000000000000000,
                 1.500000000000000,  2.000000000000000,  2.500000000000000,
                 3.000000000000000,  3.500000000000000,  0.000000000000000,
                 0.500000000000000,  1.000000000000000,  1.500000000000000,
                 2.000000000000000,  2.500000000000000,  3.000000000000000,
                 3.500000000000000,

                 2.6666667461395264, 3.000000000000000,  3.3333332538604736,
                 3.6666667461395264, 4.000000000000000,  4.333333492279053,
                 4.666666507720947,  5.000000000000000,  2.6666667461395264,
                 3.000000000000000,  3.3333332538604736, 3.6666667461395264,
                 4.000000000000000,  4.333333492279053,  4.666666507720947,
                 5.000000000000000,  2.6666667461395264, 3.000000000000000,
                 3.3333332538604736, 3.6666667461395264, 4.000000000000000,
                 4.333333492279053,  4.666666507720947,  5.000000000000000,
                 5.333333492279053,  5.666666507720947,  6.000000000000000,
                 6.333333492279053,  6.666666507720947,  7.000000000000000,
                 7.333333492279053,  7.666666507720947,  12.000000000000000,
                 12.500000000000000, 13.000000000000000, 13.500000000000000,
                 14.000000000000000, 14.500000000000000, 15.000000000000000,
                 15.500000000000000});
}

template <>
void fill_var_grad_expected<DEVICE::CPU, Sum>(Tensor* expected) {
  test::FillValues<float>(
      expected,
      {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  0.0,  1.0,  2.0,  3.0,
       4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
       8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 8.0,  9.0,  10.0, 11.0,
       12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
       24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0});
}

template <>
void fill_var_grad_expected<DEVICE::GPU, Sqrtn>(Tensor* expected) {
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
void fill_var_grad_expected<DEVICE::GPU, Mean>(Tensor* expected) {
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
void fill_var_grad_expected<DEVICE::GPU, Sum>(Tensor* expected) {
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

// template <>
// void fill_var_grad_expected<MeanAndMaxNorm100>(Tensor* expected) {
//   test::FillValues<float>(
//       expected,
//       {0.00000000,  0.50000000,  1.00000000,  1.50000000,  2.00000000,
//        2.50000000,  3.00000000,  3.50000000,  0.00000000,  0.50000000,
//        1.00000000,  1.50000000,  2.00000000,  2.50000000,  3.00000000,
//        3.50000000,  2.65028572,  2.98157120,  3.31285667,  3.64414287,
//        3.97542834,  4.30671406,  4.63799953,  4.96928549,  2.16437674,
//        2.43492365,  2.70547056,  2.97601795,  3.24656487,  3.51711202,
//        3.78765893,  4.05820608,  1.58337951,  1.78130186,  1.97922409,
//        2.17714667,  2.37506914,  2.57299161,  2.77091384,  2.96883631,
//        5.33333349,  5.66666651,  6.00000000,  6.33333349,  6.66666651,
//        7.00000000,  7.33333349,  7.66666651,  1.89459133,  2.01300311,
//        2.13141513,  2.24982715,  2.36823893,  2.48665094,  2.60506320,
//        2.72347474,  1.89459133,  2.01300311,  2.13141513,  2.24982715,
//        2.36823893,  2.48665094,  2.60506320,  2.72347474,  3.43474555,
//        3.57786012,  3.72097445,  3.86408877,  4.00720310,  4.15031767,
//        4.29343224,  4.43654633,  11.92628479, 12.42321396, 12.92014217,
//        13.41707039, 13.91399956, 14.41092777, 14.90785599, 15.40478516});
// }

class GroupVariableBackWardOpTest : public OpsTestBase {
 protected:
  template <typename TKey, typename TValue, TestCase test_case>
  void Run(DEVICE device) {
    if (device == DEVICE::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }

    DataType k_dtype = DataTypeToEnum<TKey>::value;
    DataType v_dtype = DataTypeToEnum<TValue>::value;
    std::string combiner_str;
    float max_norm;

    const int nnz = 7;
    const int nums = 10;
    const int batch_size = 4;
    const int emb_vector_dim = 8;
    const int entries = 8;
    const int bucket_size = 16;
    const int num_lookups = 2;
    get_node_attr_from_test_case<test_case>(combiner_str, max_norm);

    TF_EXPECT_OK(
        NodeDefBuilder("group_variable_lookup_grad", "GroupVariableLookupGrad")
            .Input(FakeInput(num_lookups, DT_FLOAT))  // grads
            .Input(FakeInput(num_lookups, v_dtype))   // variable
            .Input(FakeInput(num_lookups, k_dtype))   // unique_key
            .Input(FakeInput(num_lookups, DT_INT64))  // unique_idx
            .Input(FakeInput(num_lookups, DT_INT32))  // batch_nums
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
    if (device == DEVICE::GPU) {
      for (int i = 0; i < num_lookups; ++i) {
        Tensor sp_values(k_dtype, {nums});
        test::FillValues<TKey>(&sp_values, {3, 1, 4, 5, 7, 3, 12, 12, 15, 4});
        AddInputFromArray<TKey>(sp_values.shape(), sp_values.flat<TKey>());
      }

      for (int i = 0; i < num_lookups; ++i) {
        Tensor sp_values_offset(DT_INT64, {nnz});
        test::FillValues<int64>(&sp_values_offset, {0, 0, 1, 1, 1, 2, 3});
        AddInputFromArray<int64>(sp_values_offset.shape(),
                                 sp_values_offset.flat<int64>());
      }

      for (int i = 0; i < num_lookups; ++i) {
        Tensor sp_values_offset(DT_INT32, {batch_size});
        test::FillValues<int32>(&sp_values_offset, {0, 2, 5, 8});
        AddInputFromArray<int32>(sp_values_offset.shape(),
                                 sp_values_offset.flat<int>());
      }
      TF_ASSERT_OK(RunOpKernel());

      Tensor grad_expected(v_dtype, {nums, emb_vector_dim});
      fill_var_grad_expected<DEVICE::GPU, test_case>(&grad_expected);

      TF_EXPECT_OK(device_->Sync());

      for (int i = 0; i < num_lookups; ++i) {
        const Tensor& grad = *GetOutput(i);
        test::ExpectTensorNear<TValue>(grad_expected, grad, 1e-4);
      }
    } else {
      for (int i = 0; i < num_lookups; ++i) {
        Tensor sp_values(k_dtype, {nnz});
        test::FillValues<TKey>(&sp_values, {3, 1, 4, 5, 7, 12, 15});
        AddInputFromArray<TKey>(sp_values.shape(), sp_values.flat<TKey>());
      }

      for (int i = 0; i < num_lookups; ++i) {
        Tensor sp_values_offset(DT_INT64, {nnz});
        test::FillValues<int64>(&sp_values_offset, {0, 0, 1, 1, 1, 2, 3});
        AddInputFromArray<int64>(sp_values_offset.shape(),
                                 sp_values_offset.flat<int64>());
      }

      for (int i = 0; i < num_lookups; ++i) {
        Tensor sp_values_offset(DT_INT32, {batch_size});
        test::FillValues<int32>(&sp_values_offset, {2, 5, 8, 10});
        AddInputFromArray<int32>(sp_values_offset.shape(),
                                 sp_values_offset.flat<int>());
      }
      TF_ASSERT_OK(RunOpKernel());

      Tensor grad_expected(v_dtype, {nnz, emb_vector_dim});
      fill_var_grad_expected<DEVICE::CPU, test_case>(&grad_expected);

      TF_EXPECT_OK(device_->Sync());

      for (int i = 0; i < num_lookups; ++i) {
        const Tensor& grad = *GetOutput(i);
        test::ExpectTensorNear<TValue>(grad_expected, grad, 1e-4);
      }
    }
  }
};

#ifdef GOOGLE_CUDA
TEST_F(GroupVariableBackWardOpTest, EmbeddingLocalSparseLookUpGradFloatGpu) {
  Run<int64, float, Sqrtn>(DEVICE::GPU);
}

TEST_F(GroupVariableBackWardOpTest,
       EmbeddingLocalSparseLookUpGradFloatMeanGpu) {
  Run<int64, float, Mean>(DEVICE::GPU);
}

TEST_F(GroupVariableBackWardOpTest, EmbeddingLocalSparseLookUpGradFloatSumGpu) {
  Run<int64, float, Sum>(DEVICE::GPU);
}

// TEST_F(GroupVariableBackWardOpTest,
//        EmbeddingLocalSparseLookUpGradFloatMeanAndMaxNorm100Gpu) {
//   Run<int64, float, MeanAndMaxNorm100>(DEVICE::GPU);
// }
#endif  // GOOGLE_CUDA

TEST_F(GroupVariableBackWardOpTest,
       EmbeddingLocalSparseLookUpGradFloatSqrtCpu) {
  Run<int64, float, Sqrtn>(DEVICE::CPU);
}

TEST_F(GroupVariableBackWardOpTest,
       EmbeddingLocalSparseLookUpGradFloatMeanCpu) {
  Run<int64, float, Mean>(DEVICE::CPU);
}

TEST_F(GroupVariableBackWardOpTest, EmbeddingLocalSparseLookUpGradFloatSumCpu) {
  Run<int64, float, Sum>(DEVICE::CPU);
}

// TEST_F(GroupVariableBackWardOpTest,
//        EmbeddingLocalSparseLookUpGradFloatMeanAndMaxNorm100Cpu) {
//   Run<int64, float, MeanAndMaxNorm100>(DEVICE::CPU);
// }

template <TestCase test_case>
void fill_ev_vector_expected(Tensor* expected);

template <>
void fill_ev_vector_expected<Sqrtn>(Tensor* expected) {
  test::FillValues<float>(
      expected,
      {1.41421, 1.41421, 1.41421, 1.41421, 1.41421, 1.41421, 1.41421, 1.41421,
       1.73205, 1.73205, 1.73205, 1.73205, 1.73205, 1.73205, 1.73205, 1.73205,
       1.73205, 1.73205, 1.73205, 1.73205, 1.73205, 1.73205, 1.73205, 1.73205,
       1.41421, 1.41421, 1.41421, 1.41421, 1.41421, 1.41421, 1.41421, 1.41421});
}

template <>
void fill_ev_vector_expected<Mean>(Tensor* expected) {
  test::FillValues<float>(
      expected, {
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                });
}

template <>
void fill_ev_vector_expected<Sum>(Tensor* expected) {
  test::FillValues<float>(
      expected, {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
                 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0});
}

template <>
void fill_ev_vector_expected<SqrtnAndMaxNorm200>(Tensor* expected) {
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

class GroupEmbeddingVariableForWardOpTest : public OpsTestBase {
 protected:
  template <typename TKey, typename TValue, TestCase test_case>
  void Run(DEVICE device) {
    if (device == DEVICE::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }

    DataType k_dtype = DataTypeToEnum<TKey>::value;
    DataType v_dtype = DataTypeToEnum<TValue>::value;
    // TensorShapeProto tshape_proto;
    // tshape_proto.add_dim()->set_size(8);
    // TF_EXPECT_OK(NodeDefBuilder("kv_var_handle", "KvVarHandleOp")
    //                  .Attr("dtype", v_dtype)
    //                  .Attr("Tkeys", k_dtype)
    //                  .Attr("shape", tshape_proto)
    //                  .Attr("container", "EV")
    //                  .Attr("shared_name", "EV")
    //                  .Finalize(node_def()));
    // TF_EXPECT_OK(InitOp());
    // TF_ASSERT_OK(RunOpKernel());
    // const Tensor& ev_resource = *GetOutput(0);
    // ResourceHandle ev_handle = ev_resource.flat<ResourceHandle>()(0);

    // TF_EXPECT_OK(NodeDefBuilder("initialize_kv_variable",
    //                             "InitializeKvVariableOp")
    //                  .Input(FakeInput(DT_RESOURCE))  // ev
    //                  .Input(FakeInput(DT_RESOURCE))  // ev
    //                  .Input(FakeInput(v_dtype))      // sp_values
    //                  .Input(FakeInput(k_dtype))      // sp_indices
    //                  .Attr("dtype", v_dtype)
    //                  .Attr("Tkeys", k_dtype)
    //                  .Attr("slot_num", 0)
    //                  .Attr("shape", tshape_proto)
    //                  .Attr("initial_num_buckets", 131072)  // 2^17
    //                  .Attr("max_load_factor", 0.8)
    //                  .Attr("steps_to_live", 0)
    //                  .Attr("emb_index", 0)
    //                  .Attr("block_num", 1)
    //                  .Attr("slot_index", 0)
    //                  .Attr("ht_partition_num", 1000)
    //                  .Attr("filter_freq", 0)
    //                  .Attr("max_freq", 999999)
    //                  .Attr("max_element_size", 0)
    //                  .Attr("counter_type", k_dtype)
    //                  .Attr("false_positive_probability", -1.0)
    //                  .Attr("l2_weight_threshold", -1.0)
    //                  .Attr("layout", "")
    //                  .Attr("storage_type", 0)
    //                  .Attr("default_value_dim", 8)
    //                  .Attr("default_value_no_permission", 0.0)
    //                  .Attr("record_freq", false)
    //                  .Attr("record_version", false)
    //                  .Finalize(node_def()));
    // TF_EXPECT_OK(InitOp());

    // AddInputFromArray<ResourceHandle>(TensorShape({}), {ev_handle});
    // AddInputFromArray<ResourceHandle>(TensorShape({}), {ev_handle});

    // Tensor default_values(v_dtype, {8});
    // test::FillValues<TValue>(&default_values,
    //                          {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // AddInputFromArray<TValue>(default_values.shape(),
    //                           default_values.flat<TValue>());
    // Tensor empty_key(k_dtype, {1});
    // test::FillValues<TKey>(&empty_key, {-1});
    // AddInputFromArray<TKey>(empty_key.shape(), empty_key.flat<TKey>());
    // TF_ASSERT_OK(RunOpKernel());

    // Clear Resource
    // inputs_.clear();
    // gtl::STLDeleteElements(&tensors_);
    // gtl::STLDeleteElements(&managed_outputs_);

    std::string combiner_str;
    float max_norm;

    const int nnz = 10;
    const int batch_size = 4;
    const int emb_vector_dim = 8;
    const int num_lookups = 2;
    std::vector<TKey> sp_values_vec{3, 1, 4, 5, 7, 3, 12, 12, 15, 4};
    get_node_attr_from_test_case<test_case>(combiner_str, max_norm);

    TF_EXPECT_OK(NodeDefBuilder("group_embedding_variable_lookup",
                                "GroupEmbeddingVarLookup")
                     .Input(FakeInput(num_lookups, DT_RESOURCE))  // ev
                     .Input(FakeInput(num_lookups, k_dtype))      // sp_values
                     .Input(FakeInput(num_lookups, DT_INT64))     // sp_indices
                     .Input(FakeInput(num_lookups, v_dtype))      // sp_weights
                     .Input(FakeInput(DT_INT32))                  // dense_shape
                     .Input(FakeInput(v_dtype))  // default_value
                     .Attr("dtype", v_dtype)
                     .Attr("Tkeys", k_dtype)
                     .Attr("combiner", combiner_str)
                     .Attr("max_norm", max_norm)
                     .Attr("dimension", emb_vector_dim)
                     .Attr("num_lookups", num_lookups)
                     .Attr("ignore_weights", true)
                     .Attr("is_use_default_value_tensor", false)
                     .Attr("is_inference", false)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());

    for (int i = 0; i < num_lookups; ++i) {
      EmbeddingVar<TKey, TValue>* embedding_var = nullptr;
      Allocator* gpu_allocator = device_->GetAllocator(AllocatorAttributes());
      auto embedding_config =
          EmbeddingConfig(0, 0, 1, 1, "", 0, 0, 99999, 14.0);
      auto storage = embedding::StorageFactory::Create<TKey, TValue>(
          embedding::StorageConfig(embedding::StorageType::DRAM, "",
                                   {1024, 1024, 1024, 1024}, "normal",
                                   embedding_config),
          gpu_allocator,
          "EV" + std::to_string(i));
      embedding_var = new EmbeddingVar<TKey, TValue>(
          "EV" + std::to_string(i), storage, embedding_config,
          gpu_allocator);
      Tensor value(DT_FLOAT, TensorShape({emb_vector_dim}));
      test::FillValues<TValue>(&value,
                               std::vector<TValue>(emb_vector_dim, 1.0));
      embedding_var->Init(value, 1);

      for (int64 j = 0; j < nnz; ++j) {
        ValuePtr<TValue>* value_ptr = nullptr;
        Status s =
            embedding_var->LookupOrCreateKey(sp_values_vec[j], &value_ptr);
        typename TTypes<TValue>::Flat vflat = embedding_var->flat(value_ptr);
      }
      AddResourceInput<EmbeddingVar<TKey, TValue>>("", "EV" + std::to_string(i),
                                                   embedding_var);
    }

    for (int i = 0; i < num_lookups; ++i) {
      Tensor sp_values(k_dtype, {nnz});
      test::FillValues<TKey>(&sp_values, sp_values_vec);
      AddInputFromArray<TKey>(sp_values.shape(), sp_values.flat<TKey>());
    }

    for (int i = 0; i < num_lookups; ++i) {
      Tensor sp_indices(DT_INT64, {nnz});
      test::FillValues<int64>(&sp_indices, {0, 0, 1, 1, 1, 2, 2, 2, 3, 3});
      AddInputFromArray<int64>(sp_indices.shape(), sp_indices.flat<int64>());
    }

    for (int i = 0; i < num_lookups; ++i) {
      Tensor sp_weights(v_dtype, {nnz});
      test::FillValues<TValue>(&sp_weights, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                             1.0f, 1.0f, 1.0f, 1.0f});
      AddInputFromArray<TValue>(sp_weights.shape(), sp_weights.flat<TValue>());
    }

    Tensor* batch_size_tensor =
        AddInput(DataTypeToEnum<int32>::v(), TensorShape({}));
    auto batch_size_data = batch_size_tensor->flat<int>().data();
    batch_size_data[0] = batch_size;

    TF_ASSERT_OK(RunOpKernel());

    Tensor emb_vector_expected(v_dtype, {batch_size, emb_vector_dim});
    Tensor sp_values_offset_expected(DT_INT64, {7});
    Tensor unique_idx_expected(DT_INT32, {nnz});
    Tensor batch_size_expected(DT_INT32, {batch_size});
    fill_ev_vector_expected<test_case>(&emb_vector_expected);

    if (device == DEVICE::GPU) {
      test::FillValues<int32>(&batch_size_expected, {0, 2, 5, 8});
    } else {
      test::FillValues<int64>(&sp_values_offset_expected,
                              {3, 1, 4, 5, 7, 12, 15});
      test::FillValues<int32>(&unique_idx_expected,
                              {0, 1, 2, 3, 4, 0, 5, 5, 6, 2});
      test::FillValues<int32>(&batch_size_expected, {2, 5, 8, 10});
    }
    TF_EXPECT_OK(device_->Sync());

    for (int i = 0; i < num_lookups; ++i) {
      const Tensor& emb_vector = *GetOutput(i);
      const Tensor& values_offset = *GetOutput(num_lookups + i);
      const Tensor& unique_idx_output = *GetOutput(2 * num_lookups + i);
      const Tensor& batch_size_output = *GetOutput(3 * num_lookups + i);
      test::ExpectTensorNear<TValue>(emb_vector_expected, emb_vector, 1e-4);
      // Currently GPU do not have Unique logic.
      if (device == DEVICE::CPU) {
        test::ExpectTensorEqual<int64>(sp_values_offset_expected,
                                       values_offset);
        test::ExpectTensorEqual<int32>(unique_idx_expected, unique_idx_output);
      }
      test::ExpectTensorEqual<int32>(batch_size_expected, batch_size_output);
    }
  }
};

#ifdef GOOGLE_CUDA
// TODO(junqi): Complete GPUEV related test
// TEST_F(GroupEmbeddingVariableForWardOpTest,
//        EmbeddingLocalSparseLookUpFloatSqrtnGpu) {
//   Run<int64, float, Sqrtn>(DEVICE::GPU);
// }

// TEST_F(GroupEmbeddingVariableForWardOpTest,
//        EmbeddingLocalSparseLookUpFloatMeanGpu) {
//   Run<int64, float, Mean>(DEVICE::GPU);
// }

// TEST_F(GroupEmbeddingVariableForWardOpTest,
//        EmbeddingLocalSparseLookUpFloatSumGpu) {
//   Run<int64, float, Sum>(DEVICE::GPU);
// }

// TEST_F(GroupEmbeddingVariableForWardOpTest,
//        EmbeddingLocalSparseLookUpFloatSqrtnAndMaxNorm200Gpu) {
//   Run<int64, float, SqrtnAndMaxNorm200>(DEVICE::GPU);
// }
#endif  // GOOGLE_CUDA

TEST_F(GroupEmbeddingVariableForWardOpTest,
       EmbeddingVarLocalSparseLookUpFloatSqrtnCpu) {
  Run<int64, float, Sqrtn>(DEVICE::CPU);
}

TEST_F(GroupEmbeddingVariableForWardOpTest,
       EmbeddingVarLocalSparseLookUpFloatMeanCpu) {
  Run<int64, float, Mean>(DEVICE::CPU);
}

TEST_F(GroupEmbeddingVariableForWardOpTest,
       EmbeddingVarLocalSparseLookUpFloatSumCpu) {
  Run<int64, float, Sum>(DEVICE::CPU);
}

// TEST_F(GroupEmbeddingForWardOpTest,
//        EmbeddingLocalSparseLookUpFloatSqrtnAndMaxNorm200Cpu) {
//   Run<int64, float, SqrtnAndMaxNorm200>(DEVICE::CPU);
// }

class GroupEmbeddingVariableBackWardOpTest : public OpsTestBase {
 protected:
  template <typename TKey, typename TValue, TestCase test_case>
  void Run(DEVICE device) {
    if (device == DEVICE::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }

    DataType k_dtype = DataTypeToEnum<TKey>::value;
    DataType v_dtype = DataTypeToEnum<TValue>::value;
    std::string combiner_str;
    float max_norm;

    const int nums = 10;
    const int nnz = 7;
    const int batch_size = 4;
    const int emb_vector_dim = 8;
    const int entries = 8;
    const int bucket_size = 16;
    const int num_lookups = 2;
    std::vector<TKey> sp_values_vec{3, 1, 4, 5, 7, 3, 12, 12, 15, 4};
    get_node_attr_from_test_case<test_case>(combiner_str, max_norm);

    TF_EXPECT_OK(NodeDefBuilder("group_embedding_variable_lookup_grad",
                                "GroupEmbeddingVariableLookupGrad")
                     .Input(FakeInput(num_lookups, DT_FLOAT))     // grads
                     .Input(FakeInput(num_lookups, DT_RESOURCE))  // ev
                     .Input(FakeInput(num_lookups, k_dtype))      // unique_key
                     .Input(FakeInput(num_lookups, DT_INT64))     // unique_idx
                     .Input(FakeInput(num_lookups, DT_INT32))     // batch_nums
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
      EmbeddingVar<TKey, TValue>* embedding_var = nullptr;
      Allocator* gpu_allocator = device_->GetAllocator(AllocatorAttributes());
      auto embedding_config =
          EmbeddingConfig(0, 0, 1, 1, "", 0, 0, 99999, 14.0);
      auto storage = embedding::StorageFactory::Create<TKey, TValue>(
          embedding::StorageConfig(embedding::StorageType::DRAM, "",
                                   {1024, 1024, 1024, 1024}, "normal",
                                   embedding_config),
          gpu_allocator,
          "EV" + std::to_string(i));
      embedding_var = new EmbeddingVar<TKey, TValue>(
          "EV" + std::to_string(i), storage, embedding_config,
          gpu_allocator);
      Tensor value(DT_FLOAT, TensorShape({emb_vector_dim}));
      test::FillValues<TValue>(&value,
                               std::vector<TValue>(emb_vector_dim, 1.0));
      embedding_var->Init(value, 1);

      for (int64 j = 0; j < nnz; ++j) {
        ValuePtr<TValue>* value_ptr = nullptr;
        Status s =
            embedding_var->LookupOrCreateKey(sp_values_vec[j], &value_ptr);
        typename TTypes<TValue>::Flat vflat = embedding_var->flat(value_ptr);
      }
      AddResourceInput<EmbeddingVar<TKey, TValue>>("", "EV" + std::to_string(i),
                                                   embedding_var);
    }

    if (device == DEVICE::GPU) {
      for (int i = 0; i < num_lookups; ++i) {
        Tensor sp_values(k_dtype, {nums});
        test::FillValues<TKey>(&sp_values, {3, 1, 4, 5, 7, 3, 12, 12, 15, 4});
        AddInputFromArray<TKey>(sp_values.shape(), sp_values.flat<TKey>());
      }

      for (int i = 0; i < num_lookups; ++i) {
        Tensor sp_values_offset(DT_INT64, {nnz});
        test::FillValues<int64>(&sp_values_offset, {0, 0, 1, 1, 1, 2, 3});
        AddInputFromArray<int64>(sp_values_offset.shape(),
                                 sp_values_offset.flat<int64>());
      }

      for (int i = 0; i < num_lookups; ++i) {
        Tensor sp_values_offset(DT_INT32, {batch_size});
        test::FillValues<int32>(&sp_values_offset, {0, 2, 5, 8});
        AddInputFromArray<int32>(sp_values_offset.shape(),
                                 sp_values_offset.flat<int>());
      }
      TF_ASSERT_OK(RunOpKernel());

      Tensor grad_expected(v_dtype, {nums, emb_vector_dim});
      fill_var_grad_expected<DEVICE::GPU, test_case>(&grad_expected);

      TF_EXPECT_OK(device_->Sync());

      for (int i = 0; i < num_lookups; ++i) {
        const Tensor& grad = *GetOutput(i);
        test::ExpectTensorNear<TValue>(grad_expected, grad, 1e-4);
      }
    } else {
      for (int i = 0; i < num_lookups; ++i) {
        Tensor sp_values(k_dtype, {nnz});
        test::FillValues<TKey>(&sp_values, {3, 1, 4, 5, 7, 12, 15});
        AddInputFromArray<TKey>(sp_values.shape(), sp_values.flat<TKey>());
      }

      for (int i = 0; i < num_lookups; ++i) {
        Tensor sp_values_offset(DT_INT64, {nnz});
        test::FillValues<int64>(&sp_values_offset, {0, 0, 1, 1, 1, 2, 3});
        AddInputFromArray<int64>(sp_values_offset.shape(),
                                 sp_values_offset.flat<int64>());
      }

      for (int i = 0; i < num_lookups; ++i) {
        Tensor sp_values_offset(DT_INT32, {batch_size});
        test::FillValues<int32>(&sp_values_offset, {2, 5, 8, 10});
        AddInputFromArray<int32>(sp_values_offset.shape(),
                                 sp_values_offset.flat<int>());
      }
      TF_ASSERT_OK(RunOpKernel());

      Tensor grad_expected(v_dtype, {nnz, emb_vector_dim});
      fill_var_grad_expected<DEVICE::CPU, test_case>(&grad_expected);

      TF_EXPECT_OK(device_->Sync());

      for (int i = 0; i < num_lookups; ++i) {
        const Tensor& grad = *GetOutput(i);
        test::ExpectTensorNear<TValue>(grad_expected, grad, 1e-4);
      }
    }
  }
};

#ifdef GOOGLE_CUDA
// TODO(junqi): Complete GPUEV related test

// TEST_F(GroupEmbeddingVariableBackWardOpTest,
//        EmbeddingLocalSparseLookUpGradFloatGpu) {
//   Run<int64, float, Sqrtn>(DEVICE::GPU);
// }

// TEST_F(GroupEmbeddingVariableBackWardOpTest,
//        EmbeddingLocalSparseLookUpGradFloatMeanGpu) {
//   Run<int64, float, Mean>(DEVICE::GPU);
// }

// TEST_F(GroupEmbeddingVariableBackWardOpTest,
//        EmbeddingLocalSparseLookUpGradFloatSumGpu) {
//   Run<int64, float, Sum>(DEVICE::GPU);
// }

// TEST_F(GroupEmbeddingVariableBackWardOpTest,
//        EmbeddingLocalSparseLookUpGradFloatMeanAndMaxNorm100Gpu) {
//   Run<int64, float, MeanAndMaxNorm100>(DEVICE::GPU);
// }
#endif  // GOOGLE_CUDA

TEST_F(GroupEmbeddingVariableBackWardOpTest,
       EmbeddingVarLocalSparseLookUpGradFloatSqrtCpu) {
  Run<int64, float, Sqrtn>(DEVICE::CPU);
}

TEST_F(GroupEmbeddingVariableBackWardOpTest,
       EmbeddingVarLocalSparseLookUpGradFloatMeanCpu) {
  Run<int64, float, Mean>(DEVICE::CPU);
}

TEST_F(GroupEmbeddingVariableBackWardOpTest,
       EmbeddingVarLocalSparseLookUpGradFloatSumCpu) {
  Run<int64, float, Sum>(DEVICE::CPU);
}

// TEST_F(GroupEmbeddingVariableBackWardOpTest,
//        EmbeddingLocalSparseLookUpGradFloatMeanAndMaxNorm100Cpu) {
//   Run<int64, float, MeanAndMaxNorm100>(DEVICE::CPU);
// }

}  // namespace tensorflow
