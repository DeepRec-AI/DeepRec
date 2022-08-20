/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_QUANTIZE_EMBEDDING_VARIABLE_H_
#define TENSORFLOW_C_QUANTIZE_EMBEDDING_VARIABLE_H_

#include <string>
#include <vector>

#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace checkpoint {

Status QuantizeEmbeddingVariable(const string& input_prefix,
                                 const string& output_prefix,
                                 const std::vector<string>& names,
                                 const std::vector<string>& quant_names,
                                 const std::vector<string>& scale_names,
                                 TF_DataType data_type);

}  // namespace checkpoint
}  // namespace tensorflow

#endif  // TENSORFLOW_C_QUANTIZE_EMBEDDING_VARIABLE_H_
