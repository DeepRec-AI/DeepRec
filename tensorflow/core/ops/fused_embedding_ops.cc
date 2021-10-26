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
    .Output("emb_vector: T")
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

}  // namespace tensorflow