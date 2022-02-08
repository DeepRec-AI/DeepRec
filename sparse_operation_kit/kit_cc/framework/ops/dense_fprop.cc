/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using namespace tensorflow::shape_inference;

REGISTER_OP("PluginDenseFprop")
    .Input("emb_var_handle: resource")
    .Input("emb_handle: variant")
    .Input("values: value_dtype")
    .Input("global_replica_id: int32")
    .Output("emb_vector: dtype")
    .Attr("training: bool")
    .Attr("value_dtype: {int64}")
    .Attr("dtype: type")
    .Attr("unique_op_name: string")
    .Attr("dynamic_input: bool = false")
    .SetShapeFn([](InferenceContext* ctx) {
      std::vector<ShapeAndType> handle_shape_and_type;
      TF_RETURN_IF_ERROR(
          shape_inference::ValidateVariableResourceHandle(ctx, &handle_shape_and_type));

      ShapeHandle variable_shape;
      TF_RETURN_IF_ERROR(ctx->WithRank(handle_shape_and_type[0].shape, 2, &variable_shape));

      ShapeHandle emb_vec_size_shape;
      TF_RETURN_IF_ERROR(
          ctx->Subshape(variable_shape, /*start=*/1, /*end=*/2, &emb_vec_size_shape));

      bool dynamic_input = false;
      TF_RETURN_IF_ERROR(ctx->GetAttr("dynamic_input", &dynamic_input));
      if (dynamic_input) {
        // when dynamic_input is true, then values must be 1-D tensor.
        ShapeHandle values_shape;
        TF_RETURN_IF_ERROR(ctx->WithRankAtMost(ctx->input(2), 1, &values_shape));
      }

      // output_shape = [input[2].shape, embedding_vec_size]
      ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(ctx->Concatenate(ctx->input(2), emb_vec_size_shape, &output_shape));
      ctx->set_output(0, output_shape);

      return Status::OK();
    })
    .Doc(R"doc(
        This op can be used for all kinds of embedding forward propagation,
        which requires the unique_op_name to identify different op instance.
        For example:
            vec0 = plugin_dense_fprop(emb_handle0, values, unique_op_name='1')
            vec1 = plugin_dense_fprop(emb_handle1, values, unique_op_name='2')

            where different unique_op_name are set for different embedding layer instance.

        If dynamic_input is true, the input to this op must 1-D tensor, and its output
        tensor will be a 2-D tensor, whose shape is [input.shape, embedding_vec_size].
    )doc");