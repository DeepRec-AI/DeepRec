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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/types.h"

using namespace tensorflow;
using namespace tensorflow::shape_inference;

#if TF_VERSION_MAJOR == 2

REGISTER_OP("CreateVar")
    .Input("var_name: string")
    .Input("initial_value: dtype")
    .Input("local_replica_id: int32")
    .Output("var_handle: resource")
    .Output("handle: resource")
    .Output("unique_var_name: string")
    .Attr("trainable: bool = true")
    .Attr("shape: shape")
    .Attr("use_hashtable: bool = true")
    .Attr("dtype: {float, string, resource}")
    .SetShapeFn([](InferenceContext* ctx) {
        // TODO: this function is not called under Eager mode

        TensorShape shape;
        TF_RETURN_IF_ERROR(ctx->GetAttr("shape", &shape));
        DataType dtype;
        TF_RETURN_IF_ERROR(ctx->GetAttr("dtype", &dtype));
        if (2 != shape.dims()) return errors::Aborted("shape must be [vocabulary_size_per_gpu, embedding_vector_size].");
        if (!shape.IsFullyDefined()) return errors::Aborted("shape must be fully defined.");

        if (DT_FLOAT == dtype) {
            ShapeHandle initial_value_shape = ctx->input(0);
            int rank = ctx->Rank(initial_value_shape);
            if (2 != rank) return errors::Aborted("initial_value must be 2 ranks, which is [None, embedding_vector_size].");
            DimensionHandle dim = ctx->Dim(initial_value_shape, 1);
            if (!ctx->ValueKnown(dim)) return errors::Aborted("The second rank of initial_value must not be None.");

            if (shape.dim_size(1) != ctx->Value(dim)) return errors::Aborted("The second dim of initial_value must be equal"
                " to that of shape.");
        }

        ShapeHandle output_shape;
        TF_RETURN_IF_ERROR(ctx->MakeShapeFromTensorShape(shape, &output_shape));
        ctx->set_output(0, output_shape);
        ctx->set_output(1, output_shape);

        ctx->set_output_handle_shapes_and_types(0, std::vector<ShapeAndType>{{output_shape, DT_FLOAT}});
        ctx->set_output_handle_shapes_and_types(1, std::vector<ShapeAndType>{{output_shape, DT_FLOAT}});

        return Status::OK();
    })
    .Doc(R"doc(
        This op is used create variables used by a embedding layer on all GPUs in single worker.
        shape specify the variable's shape created on ONE GPU.
    )doc");

#else 

REGISTER_OP("CreateVar")
    .Attr("var_name: string")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Output("emb_var_handle: resource")
    .Output("tf_var_handle: resource")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Scalar());
        c->set_output(1, c->Scalar());

        DataType t;
        TF_RETURN_IF_ERROR(c->GetAttr("dtype", &t));
        PartialTensorShape p;
        TF_RETURN_IF_ERROR(c->GetAttr("shape", &p));
        ShapeHandle s;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(p, &s));
        c->set_output_handle_shapes_and_types(0, std::vector<ShapeAndType>{{s, t}});
        c->set_output_handle_shapes_and_types(1, std::vector<ShapeAndType>{{s, t}});

        return Status::OK();
    });

#endif