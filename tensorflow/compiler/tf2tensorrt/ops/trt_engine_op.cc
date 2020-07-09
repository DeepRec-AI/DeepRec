/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

// NOTE: when making changes please follow
// https://www.tensorflow.org/guide/extend/op#backwards_compatibility to not
// break backward compatibility.
//
// TODO(laigd): consider making this op stateful. The only problem is it uses TF
// function which has to be stateless, but we can use function library as the
// key to cache the instantiated functions for different executor subgraphs.
REGISTER_OP("TRTEngineOp")
    .Attr("serialized_segment: string")
    .Attr("segment_func: func = {}")
    .Attr("InT: list({int8,float16,float32,int32})")
    .Attr("OutT: list({int8,float16,float32,int32})")
    .Attr("max_cached_engines_count: int = 1")
    .Attr("workspace_size_bytes: int")
    .Attr("precision_mode: {'FP32', 'FP16', 'INT8'}")
    .Attr("calibration_data: string = ''")
    .Attr("use_calibration: bool = true")
    .Input("in_tensor: InT")
    .Output("out_tensor: OutT")
    // TODO(jie): TF requires concrete output shape for concrete input shapes.
    // This is tricky for batch dimension, since we cannot ensure which input
    // would carry the correct batch dimension (for the current stage of the
    // implementation, we do require all input tensor to carry the same batch
    // size, but this could change in the future). Hence we disable shape
    // inference function as a workaround.
    // .SetShapeFn(shape_inference::UnknownShape)
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // ::tensorflow::shape_inference::ShapeHandle input;
      // TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      // c->set_output(0, input);

      // ::tensorflow::shape_inference::ShapeHandle output_shapes;
      std::vector<tensorflow::TensorShape> output_shapes;
      // std::vector<tensorflow::TensorShapeProto> output_shapes;
      // std::unique_ptr<tensorflow::TensorShapeProto[]> proto output_shapes;
      // c->GetAttr("output_shapes", &output_shapes);
      TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));

      for(int i=0; i<output_shapes.size(); i++) {
        // ShapeHandle shape;
        ::tensorflow::shape_inference::ShapeHandle shape;

        // ShapeHandle handle;
        // TensorShapeProto proto;
        shape_inference::ShapeHandle output_shape_handle;
        // TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(
        //     output_shapes[i], &output_shape_handle));
        // ShapeHandle handle;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromTensorShape(
              output_shapes.at(i), &output_shape_handle));

        // TF_RETURN_IF_ERROR(
        //   c->MakeShapeFromShapeProto(output_shapes[i], &shape));
        c->set_output(i, output_shape_handle);
      }

      return Status::OK();
    })
    // Deprecated attributes.
    .Attr("segment_funcdef_name: string = ''")
    .Attr("cached_engine_batches: list(int) >= 0 = []")
    .Attr("fixed_input_size: bool = true")
    .Attr("input_shapes: list(shape) = []")
    .Attr("output_shapes: list(shape) = []")
    .Attr("static_engine: bool = true");
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
