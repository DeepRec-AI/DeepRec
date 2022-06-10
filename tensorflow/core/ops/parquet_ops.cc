// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("ParquetTabularDatasetV1")
    .Output("handle: variant")
    .Input("filename: string")
    .Input("batch_size: int64")
    .Attr("field_names: list(string) >= 1")
    .Attr("field_dtypes: list(type) >= 1")
    .Attr("field_ragged_ranks: list(int) >= 1")
    .Attr("partition_count: int = 1")
    .Attr("partition_index: int = 0")
    .Attr("drop_remainder: bool = false")
    .SetIsStateful()  // NOTE: Source dataset ops must be marked stateful to
                      // inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

}  // namespace tensorflow
