#include <stdio.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("PruneInvalidAndFillEmptyRows")
    .Attr("fill_empty_row: bool = false")
    .Attr("prune_invalid_id: bool = false")
    .Attr("default_id: int = -1")
    .Input("sp_values: int64")
    .Input("sp_indices: int64")
    .Input("sp_dense_shape: int64")
    .Output("sp_values_out: int64")
    .Output("sp_indices_out: int64")
    .Output("row_empty_and_invalid_flags: int32")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle unused;
      std::vector<ShapeHandle> unused_list;

      ctx->input("sp_values", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      ctx->input("sp_indices", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 2, &unused));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(unused, 1), 2, &unused_dim));

      ctx->input("sp_dense_shape", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      unused_list.clear();
      unused_list.resize(1);
      unused_list[0] = ctx->MakeShape({ctx->UnknownDim()});
      ctx->set_output("sp_values_out", unused_list);
      ctx->set_output("row_empty_flags", unused_list);
      unused_list[0] = ctx->MakeShape({ctx->UnknownDim(), 2});
      ctx->set_output("sp_indices_out", unused_list);

      return Status::OK();
    });

REGISTER_OP("UniqueWithMultiOutput")
    .Input("input: T")
    .Output("unique_keys: T")
    .Output("unique_idxs: out_idx")
    .Output("unique_counts: out_idx")
    .Output("idx_of_input_to_unique: out_idx")
    .Output("unique_offsets: out_idx")
    .Attr("T: type")
    .Attr("out_idx: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle unused;
      std::vector<ShapeHandle> unused_list;

      ctx->input("input", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      unused_list.clear();
      unused_list.resize(1);
      unused_list[0] = ctx->MakeShape({ctx->UnknownDim()});

      ctx->set_output("unique_keys", unused_list);
      ctx->set_output("unique_idxs", unused_list);
      ctx->set_output("unique_counts", unused_list);
      ctx->set_output("idx_of_input_to_unique", unused_list);
      ctx->set_output("unique_offsets", unused_list);

      return Status::OK();
    });

REGISTER_OP("PartitionWithPermutation")
    .Attr("num_partitions: int >= 1 = 1")
    .Attr("partition_axis: int >= 0 = 0")  // for now only support = 0,
                                           // will consider support = 1
                                           // if necessary
    .Attr("partition_strategy : {'div'}")
    .Input("input: int64")
    .Input("partition_shapes: num_partitions * int64")
    .Output("partitioned_values: num_partitions * int64")
    .Output("partition_permutations: num_partitions * int64")
    .Attr("T: type")
    .Attr("out_idx: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle unused;
      std::vector<ShapeHandle> unused_list;
      DimensionHandle unused_dim;

      int num_partitions;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_partitions", &num_partitions));
      int partition_axis;
      TF_RETURN_IF_ERROR(ctx->GetAttr("partition_axis", &partition_axis));

      ctx->input("input", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      ctx->input("partition_shapes", &unused_list);
      for (int i = 0; i < num_partitions; i++) {
        TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[i], 1, &unused));
        TF_RETURN_IF_ERROR(
            ctx->WithValue(ctx->NumElements(unused), 2, &unused_dim));
      }

      unused_list.clear();
      unused_list.resize(num_partitions);
      for (int i = 0; i < num_partitions; i++) {
        unused_list[i] = ctx->MakeShape({ctx->UnknownDim()});
      }
      ctx->set_output("partitioned", unused_list);
      ctx->set_output("partition_permutations", unused_list);

      return Status::OK();
    });

REGISTER_OP("FusedEmbeddingSparsePostLookUp")
    .Attr("T : {float32}")
    .Attr("num_partitions: int >= 1 = 1")
    .Attr("fill_empty_row: bool = false")
    .Attr("default_id: int = -1")
    .Attr("partition_axis: int >= 0 = 0")  // for now only support = 0,
                                           // will consider support = 1
                                           // if necessary
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("max_norm: float = -1.0")
    .Input("emb_shards: num_partitions * T")
    .Input("partition_permutations: num_partitions * int64")
    .Input("sp_dense_shape: int64")
    .Input("indices_before_unique: int64")
    .Input("row_empty_and_invalid_flags: int32")
    .Input("unique_counts: int64")
    .Input("idx_of_input_to_unique: int64")
    .Input("unique_offsets: int64")
    .Output("emb_vectors: T")
    .Output("feature_nums: int32")
    .SetShapeFn([](InferenceContext* ctx) {
      int num_partitions;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_partitions", &num_partitions));

      std::vector<ShapeHandle> unused_list;
      ShapeHandle unused;
      DimensionHandle unused_dim;

      // emb_shards
      ctx->input("emb_shards", &unused_list);
      ShapeHandle first_emb_shard_shape;
      TF_RETURN_IF_ERROR(
          ctx->WithRank(unused_list[0], 2, &first_emb_shard_shape));
      DimensionHandle emb_vec_size_dim = ctx->Dim(first_emb_shard_shape, 1);
      int64 emb_vec_size = ctx->Value(emb_vec_size_dim);

      for (int i = 0; i < num_partitions; i++) {
        TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[i], 2, &unused));
        TF_RETURN_IF_ERROR(
            ctx->WithValue(ctx->Dim(unused, 1), emb_vec_size, &unused_dim));
      }

      // partition_permutations
      ctx->input("partition_permutations", &unused_list);
      for (int i = 0; i < num_partitions; i++) {
        TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[i], 1, &unused));
      }

      // sp_dense_shape
      ctx->input("sp_dense_shape", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // indices_before_unique
      ctx->input("indices_before_unique", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 2, &unused));

      // row_empty_and_invalid_flags
      ctx->input("row_empty_and_invalid_flags", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // unique_counts
      ctx->input("unique_counts", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // idx_of_input_to_unique
      ctx->input("idx_of_input_to_unique", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // unique_offsets
      ctx->input("unique_offsets", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // emb_vectors
      unused_list.clear();
      unused_list.resize(1);
      unused_list[0] = ctx->MakeShape({ctx->UnknownDim(), emb_vec_size_dim});
      ctx->set_output("emb_vectors", unused_list);

      // feature_nums
      unused_list[0] = ctx->MakeShape({ctx->UnknownDim()});
      ctx->set_output("feature_nums", unused_list);
      return Status::OK();
    });

//     .Doc(R"doc(
// A fused embedding op, usually using for partitioned and distriuted embedding
// variables. FusedEmbeddingSparse`LookUp, FusedEmbeddingSparsePostLookUp
// should be used together. There should be several Gather ops before this op.
// The Gather ops gather embedding shards from embedding variable and this op
// glue them together, then apply combiner and max_morm according to embedding
// indices.
//     )doc");

REGISTER_OP("FusedEmbeddingSparsePostLookUpGrad")
    .Attr("T : {float32}")
    .Attr("num_partitions: int >= 1 = 1")
    .Attr("fill_empty_row: bool = false")
    .Attr("partition_axis: int >= 0 = 0")  // for now only support = 0,
                                           // will consider support = 1
                                           // if necessary
    .Attr("default_id: int = -1")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("max_norm: float = -1.0")
    .Input("top_grad: T")
    .Input("emb_shards: num_partitions * T")
    .Input("partition_permutations: num_partitions * int64")
    .Input("feature_nums: int32")
    .Input("indices_before_unique: int64")
    .Input("unique_counts: int64")
    .Input("idx_of_input_to_unique: int64")
    .Input("unique_offsets: int64")
    .Input("row_empty_and_invalid_flags: int32")
    .Output("grad_shards: num_partitions * T")
    .SetShapeFn([](InferenceContext* ctx) {
      int num_partitions;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_partitions", &num_partitions));

      std::vector<ShapeHandle> unused_list;
      ShapeHandle unused;
      DimensionHandle unused_dim;

      ctx->input("top_grad", &unused_list);
      // top_grad
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 2, &unused));
      DimensionHandle emb_vec_size_dim = ctx->Dim(unused, 1);

      // emb_shards
      ctx->input("emb_shards", &unused_list);
      for (int i = 0; i < num_partitions; i++) {
        TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[i], 2, &unused));
      }
      // partition_permutations
      ctx->input("partition_permutations", &unused_list);
      for (int i = 0; i < num_partitions; i++) {
        TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[i], 1, &unused));
      }

      // feature_nums
      ctx->input("feature_nums", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // indices_before_unique
      ctx->input("indices_before_unique", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 2, &unused));

      // unique_counts
      ctx->input("unique_counts", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // idx_of_input_to_unique
      ctx->input("idx_of_input_to_unique", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // unique_offsets
      ctx->input("unique_offsets", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // row_empty_and_invalid_flags
      ctx->input("row_empty_and_invalid_flags", &unused_list);
      TF_RETURN_IF_ERROR(ctx->WithRank(unused_list[0], 1, &unused));

      // grad_shards
      unused_list.clear();
      unused_list.resize(1);
      unused_list[0] = ctx->MakeShape({ctx->UnknownDim(), emb_vec_size_dim});
      ctx->set_output("grad_shards", unused_list);

      return Status::OK();
    });

//     .Doc(R"doc(
// Calculate gradient of FusedEmbeddingSparsePostLookUp
//     )doc");

}  // namespace tensorflow
