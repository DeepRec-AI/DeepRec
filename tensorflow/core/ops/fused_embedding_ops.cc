#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("FusedEmbeddingSparseLookUp")
    .Attr("T: {float32}")
    .Attr("combiner: string")
    .Input("sp_values: int64")
    .Input("sp_indices: int64")
    .Input("sp_dense_shape: int64")
    .Input("emb_variable: T")
    .Output("emb_vector: T")
    .Output("values_offset: int64")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle emb_var_shape;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(2), 2, &emb_var_shape));

      DimensionHandle emb_vec_size_dim = ctx->Dim(emb_var_shape, 1);
      DimensionHandle batch_dim = ctx->UnknownDim();

      ShapeHandle output_shape = ctx->MakeShape({batch_dim, emb_vec_size_dim});
      ctx->set_output(0, output_shape);

      return Status::OK();
    })
    .Doc(R"doc()doc");

REGISTER_OP("FusedEmbeddingSparseLookUpGrad")
    .Attr("T: {float32}")
    .Attr("combiner: string")
    .Input("top_grad: T")
    .Input("sp_values: int64")
    .Input("sp_values_offset: int64")
    .Output("grad_sp_values: T")
    .Output("grad_sp_indices: int64")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->MakeShape({ctx->UnknownDim()}));
      ctx->set_output(0, ctx->MakeShape({ctx->UnknownDim(), 2}));

      return Status::OK();
    })
    .Doc(R"doc()doc");

}  // namespace tensorflow