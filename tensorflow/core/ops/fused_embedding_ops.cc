#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("FusedEmbeddingLocalSparseLookUp")
    .Attr("T: {float32}")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("max_norm: float = -1.0")
    .Input("sp_values: int64")
    .Input("sp_indices: int64")
    .Input("sp_dense_shape: int64")
    .Input("emb_variable: T")
    .Output("emb_vectors: T")
    .Output("sp_values_offset: int32")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle temp;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 1, &temp));
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(1), 2, &temp));
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(2), 1, &temp));
      ShapeHandle emb_var_shape;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(3), 2, &emb_var_shape));

      DimensionHandle emb_vec_size_dim = ctx->Dim(emb_var_shape, 1);
      DimensionHandle batch_dim = ctx->UnknownDim();

      ShapeHandle output_shape = ctx->MakeShape({batch_dim, emb_vec_size_dim});
      ctx->set_output(0, output_shape);

      return Status::OK();
    })
    .Doc(R"doc()doc");

REGISTER_OP("FusedEmbeddingLocalSparseLookUpGrad")
    .Attr("T: {float32}")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("max_norm: float = -1.0")
    .Input("top_grad: T")
    .Input("emb_variable: T")
    .Input("sp_values: int64")
    .Input("sp_values_offset: int32")
    .Output("grad_emb_weight_sp_values: T")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle top_grad_shape;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 2, &top_grad_shape));
      DimensionHandle emb_vec_size_dim = ctx->Dim(top_grad_shape, 1);
      ctx->set_output(0, ctx->MakeShape({ctx->UnknownDim(), emb_vec_size_dim}));
      return Status::OK();
    })
    .Doc(R"doc()doc");

REGISTER_OP("FusedEmbeddingDistributedSparsePreLookUp")
    .Attr("T: {int32, int64}")
    .Attr("num_partitions: T >= 1 = 1")
    .Attr("partition_axis: T <= 0 = 0")  // for now only support = 0,
                                         // will consider support = 1
                                         // if necessary
    .Input("partition_shapes: num_partitions * T")
    .Input("sp_values: int64")
    .Input("sp_indices: int64")
    .Output("partitioned_values: num_partitions * int64")
    .Output("partitioned_indices: num_partitions * int64")
    .SetShapeFn([](InferenceContext* ctx) {
      int64 num_partitions;
      TF_RETURN_IF_ERROR(ctx->GetAttr("num_partitions", &num_partitions));
      int64 partition_axis;
      TF_RETURN_IF_ERROR(ctx->GetAttr("partition_axis", &partition_axis));

      ShapeHandle unused;
      TF_RETURN_IF_ERROR(
          ctx->WithRank(c->input(int(num_partitions)), 1, &unused));
      TF_RETURN_IF_ERROR(
          ctx->WithRank(c->input(int(num_partitions) + 1), 2, &unused));
      DimensionHandle unused_dim;
      TF_RETURN_IF_ERROR(c->WithValue(ctx->Dim(unused, 1), 2, &unused_dim));

      for (int i = 0; i < int(num_partitions); i++) {
        ShapeHandle partition_shape;
        TF_RETURN_IF_ERROR(c->WithRank(ctx->input(i), 1, &partition_shape));
        TF_RETURN_IF_ERROR(
            c->WithValue(ctx->NumElements(partition_shape), 2, &unused_dim));

        ShapeHandle values_result_shape, indices_result_shape;
        if (int(partition_axis) == 0) {
          values_result_shape ctx->MakeShape({ctx->UnknownDim()});
          indices_result_shape = ctx->MakeShape({ctx->UnknownDim(), 2});
        } else {
          return errors::InvalidArgument("partition_axis > 0 not implemented!");
        }
        c->set_output(i, values_result_shape);
        c->set_output(i + int(num_partitions), indices_result_shape);
      }
      return Status::OK();
    })
    .Doc(R"doc()doc");

REGISTER_OP("FusedEmbeddingDistributedSparsePostLookUp")
    .Attr("num_partitions: T >= 1 = 1")
    .Attr("partition_axis: T <= 0 = 0")  // for now only support = 0,
                                         // will consider support = 1
                                         // if necessary
    .Attr("T : {float32}")
    .Attr("combiner: {'sqrtn', 'mean', 'sum'}")
    .Attr("max_norm: float = -1.0")
    .Input("emb_shards: num_partitions * T")
    .Input("partitioned_indices: num_partitions * int64")
    .Input("sp_dense_shape: int64")
    .Output("emb_vectors: T")
    .Output("sp_values_offset: int32")

}  // namespace tensorflow