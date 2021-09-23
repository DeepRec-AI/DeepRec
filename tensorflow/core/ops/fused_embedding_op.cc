#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("SparseFusedEmbedding")
    .Attr("value_dtype: {int64}")
    .Attr("T: {float32}")
    .Attr("combiner: string")
    .Input("values: value_dtype")
    .Input("indices: int64")
    .Input("dense_shape: int64")
    .Input("emb_variable: T")
    .Output("emb_vector: T")
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

}  // namespace tensorflow