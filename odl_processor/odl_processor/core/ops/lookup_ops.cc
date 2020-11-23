/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace processor {

// Init storage
REGISTER_OP("KvInit")
    .Input("version: string")
    .Attr("feature_names: list(string) >= 0")
    .SetShapeFn(shape_inference::UnknownShape);

// EV
REGISTER_OP("KvLookup")
    .Input("version: string")
    .Input("indices: Tkeys")
    .Input("default_value: dtype")
    .Output("output: dtype")
    .Attr("var_name: string")
    .Attr("dim_len: int")
    .Attr("dtype: type")
    .Attr("Tkeys: {int64}")
    .SetShapeFn(shape_inference::UnknownShape);

// EV
REGISTER_OP("KvImport")
    .Input("version: string")
    .Input("prefix: string")
    .Input("tensor_name: string")
    .Attr("var_name: string")
    .Attr("dim_len: int")
    .Attr("Tkeys: {int64}")
    .Attr("dtype: type")
    .SetShapeFn(shape_inference::UnknownShape);

} // namespace processor
} // namespace tensorflow
