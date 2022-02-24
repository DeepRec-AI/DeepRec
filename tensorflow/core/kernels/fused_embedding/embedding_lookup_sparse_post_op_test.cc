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
#include "tensorflow/cc/ops/standard_ops.h"

namespace tensorflow {
namespace {

enum class Device { CPU, GPU };
class FusedSafeEmbeddingPostLookupOpTest : public OpsTestBase {
 protected:
  void MakeOpAndSetDevice(Device device, int num_partitions, DataType dtype,
                          const std::string& combiner, const float max_norm,
                          const int default_id) {
    if (device == Device::GPU) {
      SetDevice(DEVICE_GPU,
                std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                    "GPU", {}, "/job:a/replica:0/task:0")));
    }

    TF_EXPECT_OK(NodeDefBuilder("fused_safe_embedding_post_look_up",
                                "FusedEmbeddingSparsePostLookUp")
                     .Attr("T", dtype)
                     .Attr("num_partitions", num_partitions)
                     .Attr("partition_axis", 0)
                     .Attr("combiner", combiner)
                     .Attr("max_norm", max_norm)
                     .Attr("default_id", default_id)
                     .Input(FakeInput(num_partitions, dtype))
                     .Input(FakeInput(num_partitions, DT_INT64))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT64))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

// TEST_F(FusedSafeEmbeddingPostLookupOpTest,
//        Partition3_Sqrtn_MaxNorm200_Float) {
//   const int nnz = 10;
//   const int batch_size = 4;
//   const int emb_vector_dim = 8;
//   const int entries = 8;

//   MakeOpAndSetDevice(Device::CPU, 3, DT_FLOAT, "sqrtn", 200.0, -1);

//   // emb_shards
//   AddInputFromArray<float>(
//       TensorShape({6, emb_vector_dim}),
//       {
//           8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 24.0, 25.0,
//           26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 24.0, 25.0, 26.0, 27.0,
//           28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0,
//           38.0, 39.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0,
//           40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
//       });
//   AddInputFromArray<float>(TensorShape({1, emb_vector_dim}),
//                            {56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0});
//   AddInputFromArray<float>(
//       TensorShape({3, emb_vector_dim}),
//       {96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0,
//        96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0,
//        120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0});

//   // partitioned_indices
//   AddInputFromArray<int64>(TensorShape({6, 2}),
//                            {0, 5, 0, 1, 2, 1, 1, 2, 3, 6, 1, 1});
//   AddInputFromArray<int64>(TensorShape({1, 2}), {1, 7});
//   AddInputFromArray<int64>(TensorShape({3, 2}), {2, 4, 2, 7, 3, 0});

//   // sp_dense_shape
//   AddInputFromArray<int64>(TensorShape({2}), {batch_size, entries});

//   // row_empty_and_invalid_flags
//   AddInputFromArray<int>(TensorShape({batch_size + nnz}),
//                          {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

//   TF_ASSERT_OK(RunOpKernel());
//   TF_EXPECT_OK(device_->Sync());

//   {
//     Tensor expected_emb_vectors(allocator(), DT_FLOAT,
//                                 TensorShape({batch_size, emb_vector_dim}));
//     test::FillValues<float>(
//         &expected_emb_vectors,
//         {22.62741661, 24.04163170, 25.45584488,  26.87005806,  28.28427124,
//          29.69848442, 31.11269951, 32.52691269,  73.90083313,  75.63288879,
//          77.36493683, 79.09698486, 80.82904053,  82.56108856,  84.29314423,
//          86.02519226, 92.61308289, 94.01081848,  95.40855408,  96.80628204,
//          98.20401764, 99.60175323, 100.99948120, 102.39721680, 71.20205688,
//          72.31395721, 73.42584991, 74.53774261,  75.64963531,  76.76153564,
//          77.87342834, 78.98532867});
//     test::ExpectTensorNear<float>(expected_emb_vectors, *GetOutput(0), 1e-4);
//   }
//   {
//     Tensor feature_nums_expected(allocator(), DT_INT32,
//                                  TensorShape({batch_size}));
//     test::FillValues<int>(&feature_nums_expected, {2, 3, 3, 2});
//     test::ExpectTensorEqual<int32>(feature_nums_expected, *GetOutput(1));
//   }
// }

TEST_F(FusedSafeEmbeddingPostLookupOpTest,
       Partition3_Sqrtn_Float) {
  const int nnz = 10;
  const int batch_size = 4;
  const int emb_vector_dim = 8;
  const int entries = 8;

  MakeOpAndSetDevice(Device::CPU, 3, DT_FLOAT, "sqrtn", -1.0, -1);

  // emb_shards
  AddInputFromArray<float>(
      TensorShape({6, emb_vector_dim}),
      {
          8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 24.0, 25.0,
          26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 24.0, 25.0, 26.0, 27.0,
          28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0,
          38.0, 39.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0,
          40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
      });
  AddInputFromArray<float>(TensorShape({1, emb_vector_dim}),
                           {56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0});
  AddInputFromArray<float>(
      TensorShape({3, emb_vector_dim}),
      {96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0,
       96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0,
       120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0});

  // partitioned_indices
  AddInputFromArray<int64>(TensorShape({6, 2}),
                           {0, 5, 0, 1, 2, 1, 1, 2, 3, 6, 1, 1});
  AddInputFromArray<int64>(TensorShape({1, 2}), {1, 7});
  AddInputFromArray<int64>(TensorShape({3, 2}), {2, 4, 2, 7, 3, 0});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {batch_size, entries});

  // row_empty_and_invalid_flags
  AddInputFromArray<int>(TensorShape({batch_size + nnz}),
                         {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_emb_vectors(allocator(), DT_FLOAT,
                                TensorShape({batch_size, emb_vector_dim}));
    test::FillValues<float>(
        &expected_emb_vectors,
        {22.62741661, 24.04162979, 25.45584297, 26.87005806,
         28.28427124, 29.69848442, 31.11269760, 32.52691269,
         73.90083313, 75.63288116, 77.36493683, 79.09698486,
         80.82903290, 82.56108856, 84.29313660, 86.02519226,
         124.70765686, 126.43970490, 128.17175293, 129.90380859,
         131.63586426, 133.36790466, 135.09996033, 136.83201599,
         107.48023224, 108.89443970, 110.30865479, 111.72286987,
         113.13708496, 114.55130005, 115.96550751, 117.37972260});
    test::ExpectTensorNear<float>(expected_emb_vectors, *GetOutput(0), 1e-4);
  }
  {
    Tensor feature_nums_expected(allocator(), DT_INT32,
                                 TensorShape({batch_size}));
    test::FillValues<int>(&feature_nums_expected, {2, 3, 3, 2});
    test::ExpectTensorEqual<int32>(feature_nums_expected, *GetOutput(1));
  }
}

TEST_F(FusedSafeEmbeddingPostLookupOpTest, Partition2_Sum_No_Default) {
  const int nnz = 3;
  const int batch_size = 3;
  const int emb_vector_dim = 4;
  const int entries = 8;

  MakeOpAndSetDevice(Device::CPU, 2, DT_FLOAT, "sum", -1.0, -1);

  // emb_shards
  AddInputFromArray<float>(TensorShape({2, emb_vector_dim}),
                           {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0});
  AddInputFromArray<float>(TensorShape({2, emb_vector_dim}),
                           {10.0, 10.0, 10.0, 10.0, 13.0, 13.0, 13.0, 13.0});

  // partitioned_indices
  AddInputFromArray<int64>(TensorShape({2, 2}), {0, 0, 0, 5});
  AddInputFromArray<int64>(TensorShape({2, 2}), {1, 4, 2, 0});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {batch_size, entries});

  // row_empty_and_invalid_flags
  AddInputFromArray<int>(TensorShape({batch_size + nnz}), {0, 0, 1, 1, 1, 1});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_emb_vectors(allocator(), DT_FLOAT,
                                TensorShape({batch_size, emb_vector_dim}));
    test::FillValues<float>(
        &expected_emb_vectors,
        {3.0, 3.0, 3.0, 3.0, 10.0, 10.0, 10.0, 10.0, 13.0, 13.0, 13.0, 13.0});
    test::ExpectTensorNear<float>(expected_emb_vectors, *GetOutput(0), 1e-4);
  }
  {
    Tensor feature_nums_expected(allocator(), DT_INT32,
                                 TensorShape({batch_size}));
    test::FillValues<int>(&feature_nums_expected, {2, 1, 1});
    test::ExpectTensorEqual<int32>(feature_nums_expected, *GetOutput(1));
  }
}

TEST_F(FusedSafeEmbeddingPostLookupOpTest, Partition2_Sum_Default_0) {
  const int nnz = 3;
  const int batch_size = 3;
  const int emb_vector_dim = 4;
  const int entries = 8;

  MakeOpAndSetDevice(Device::CPU, 2, DT_FLOAT, "sum", -1.0, 0);

  // emb_shards
  AddInputFromArray<float>(TensorShape({2, emb_vector_dim}),
                           {1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0});
  AddInputFromArray<float>(TensorShape({2, emb_vector_dim}),
                           {10.0, 10.0, 10.0, 10.0, 13.0, 13.0, 13.0, 13.0});

  // partitioned_indices
  AddInputFromArray<int64>(TensorShape({2, 2}), {0, 0, 0, 5});
  AddInputFromArray<int64>(TensorShape({2, 2}), {1, 4, 2, 0});

  // sp_dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {batch_size, entries});

  // row_empty_and_invalid_flags
  AddInputFromArray<int>(TensorShape({batch_size + nnz}), {0, 0, 1, 1, 1, 1});

  TF_ASSERT_OK(RunOpKernel());
  TF_EXPECT_OK(device_->Sync());

  {
    Tensor expected_emb_vectors(allocator(), DT_FLOAT,
                                TensorShape({batch_size, emb_vector_dim}));
    test::FillValues<float>(
        &expected_emb_vectors,
        {3.0, 3.0, 3.0, 3.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0});
    test::ExpectTensorNear<float>(expected_emb_vectors, *GetOutput(0), 1e-4);
  }
  {
    Tensor feature_nums_expected(allocator(), DT_INT32,
                                 TensorShape({batch_size}));
    test::FillValues<int>(&feature_nums_expected, {2, 1, 1});
    test::ExpectTensorEqual<int32>(feature_nums_expected, *GetOutput(1));
  }
}


//----------------------------------------------------------------------------//
// Performance benchmarks                                                     //
//----------------------------------------------------------------------------//

template <typename T>
void FillValues(Tensor* tensor, gtl::ArraySlice<T> vals) {
  auto flat = tensor->flat<T>();
  CHECK_EQ(flat.size(), vals.size());
  if (flat.size() > 0) {
    std::copy_n(vals.data(), vals.size(), flat.data());
  }
}

template <typename T>
void FillZerosValues(Tensor* tensor) {
  auto flat = tensor->flat<T>();
  for (int i = 0; i < flat.size(); ++i) {
    flat.data()[i] = 0.0;
  }
}

template <typename T>
void FillOnesValues(Tensor* tensor) {
  auto flat = tensor->flat<T>();
  float scale = std::rand()/((RAND_MAX + 1u)/6);
  for (int i = 0; i < flat.size(); ++i) {
    flat.data()[i] = 1.1 * scale;
  }
}

template <typename T>
void FillIndiceValues(Tensor* tensor, const int partitions, const int batch_size, const int entries) {
  auto flat = tensor->flat<T>();
  int k = 0;
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < entries; ++j) {
      flat.data()[k] = i + partitions;
      flat.data()[k+1] = j;
      k += 2;
    }
  }
}

template <typename T>
void PrintValues(Tensor* tensor) {
  auto flat = tensor->flat<T>();
  for (int i = 0; i < flat.size(); ++i) {
    std::cout << flat.data()[i] << ", ";
  }
  std::cout << std::endl;
}

template <typename T>
static Graph* EmbPostOp(const string& kind, int num_partitions, const std::string& combiner,
                      const float max_norm, const int default_id) {
  const int nnz = 3;
  const int batch_size = 512;
  const int emb_vector_dim = 32;
  const int entries = 8;
  const float sparsity = 0.5;
  const int total_inputs = batch_size*entries*sparsity;

  Graph* g = new Graph(OpRegistry::Global());
  DataType type = DataTypeToEnum<T>::v();

  const bool isDefault = (kind == "Default");
  string op_name = isDefault ? "FusedEmbeddingSparsePostLookUpOrigin" : "FusedEmbeddingSparsePostLookUp";

  // emb_shards
  std::vector<NodeBuilder::NodeOut> input_emb_shards;
  input_emb_shards.reserve(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    Tensor emb_shards(type, TensorShape({total_inputs/num_partitions, emb_vector_dim}));
    FillOnesValues<T>(&emb_shards);
    input_emb_shards.push_back(test::graph::Constant(g, emb_shards));
    // PrintValues<T>(&emb_shards);
  }

  // partitioned_indices
  std::vector<NodeBuilder::NodeOut> partitioned_indices;
  partitioned_indices.reserve(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    Tensor sub_partitioned_indice(DT_INT64, TensorShape({total_inputs/num_partitions, 2}));
    FillIndiceValues<int64>(&sub_partitioned_indice, i, batch_size/num_partitions, entries*sparsity);
    partitioned_indices.push_back(test::graph::Constant(g, sub_partitioned_indice));
    // PrintValues<int64>(&sub_partitioned_indice);
  }

  // sp_dense_shape
  Tensor sp_dense_shape(DT_INT64, TensorShape({2}));
  FillValues<int64>(&sp_dense_shape, {batch_size, entries});

  // row_empty_and_invalid_flags
  Tensor row_empty_and_invalid_flags(DT_INT32, TensorShape({batch_size + nnz}));
  FillZerosValues<int>(&row_empty_and_invalid_flags);

  auto nodeBuilder = NodeBuilder(g->NewName("n"), op_name)
                    .Attr("T", type)
                    .Attr("num_partitions", num_partitions)
                    .Attr("partition_axis", 0)
                    .Attr("combiner", combiner)
                    .Attr("max_norm", max_norm)
                    .Attr("default_id", default_id)
                    .Input(input_emb_shards)
                    .Input(partitioned_indices)
                    .Input(test::graph::Constant(g, sp_dense_shape))
                    .Input(test::graph::Constant(g, row_empty_and_invalid_flags))
                    .Input(partitioned_indices);
  TF_CHECK_OK(nodeBuilder.Finalize(g, nullptr));
  return g;
}

#define BM_EMB_POST_OP(kind, NP, C, T, DEVICE, NTH)                                    \
  static void BM_EMB_POST_OP##_##kind##_##NP##_##C##_##T##_##DEVICE##_##NTH(           \
      int iters) {                                                                     \
    testing::UseRealTime();                                                            \
    SessionOptions opts;                                                               \
    opts.config.set_intra_op_parallelism_threads(NTH);                                 \
    test::Benchmark(#DEVICE, EmbPostOp<T>(#kind, NP, #C, -1.0, -1), &opts).Run(iters); \
  }                                                                                    \
  BENCHMARK(BM_EMB_POST_OP##_##kind##_##NP##_##C##_##T##_##DEVICE##_##NTH);            \

#define BM_EMB_POST_OP_kind(NP, C, NTH)            \
  BM_EMB_POST_OP(OPT, NP, C, float, CPU, NTH);     \

#define BM_EMB_POST_OP_NTH(NP, C) \
  BM_EMB_POST_OP_kind(NP, C, 1);  \
  BM_EMB_POST_OP_kind(NP, C, 4);  \
  BM_EMB_POST_OP_kind(NP, C, 8);  \

BM_EMB_POST_OP_NTH(2, sum);

}  // namespace
}  // namespace tensorflow
