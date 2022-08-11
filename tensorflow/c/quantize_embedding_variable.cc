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

#include "tensorflow/c/quantize_embedding_variable.h"

// #include <unordered_set>
// #include <utility>

// #include "tensorflow/core/lib/core/status.h"
// #include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace checkpoint {

Status QuantizeEmbeddingVariable(const string& input_prefix,
                                 const string& output_prefix,
                                 const std::vector<string>& names,
                                 const std::vector<string>& quant_names,
                                 const std::vector<string>& scale_names,
                                 TF_DataType data_type) {
  BundleReader reader(Env::Default(), input_prefix);
  BundleWriter writer(Env::Default(), output_prefix);
  std::set<string> ev_suffix = {
      "-freqs",         "-freqs_filtered",          "-keys",
      "-keys_filtered", "-partition_filter_offset", "-partition_offset",
      "-versions",      "-versions_filtered"};

  Status status;
  DataType dtype;
  TensorShape shape;
  std::set<string> updated_names;
  for (int idx = 0; idx < names.size(); ++idx) {
    string value_name = names[idx] + "-values";
    status = reader.LookupDtypeAndShape(value_name, &dtype, &shape);
    if (!status.ok()) errors::InvalidArgument("Invalid name:", value_name);
    Tensor in_tensor(dtype, shape);
    status = reader.Lookup(value_name, &in_tensor);
    auto in_data = in_tensor.flat<float>();

    if (data_type == TF_DataType::TF_BFLOAT16) {
      Tensor out_tensor(DataTypeToEnum<bfloat16>::v(), shape);
      auto out_data = out_tensor.flat<bfloat16>();
      for (int i = 0; i < out_tensor.NumElements(); ++i) {
        out_data(i) = static_cast<bfloat16>(in_data(i));
      }
      writer.Add(quant_names[idx] + "-values", out_tensor);
    } else if (data_type == TF_DataType::TF_INT8) {
      Tensor out_tensor(DataTypeToEnum<int8_t>::v(), shape);
      auto out_data = out_tensor.flat<int8_t>();
      int embedding_dim = shape.dim_size(shape.dims() - 1);
      Tensor scale_tensor(DT_FLOAT, TensorShape({embedding_dim}));
      auto scale_data = scale_tensor.flat<float>();
      for (int i = 0; i < embedding_dim; ++i) {
        int size = in_tensor.NumElements() / embedding_dim;
        float max_value = std::numeric_limits<float>::min();
        for (int j = 0; j < size; ++j) {
          max_value = std::max(max_value, std::abs(in_data(i * size + j)));
        }
        scale_data(i) = max_value / 127.0;
        for (int j = 0; j < size; ++j) {
          float int8_value = in_data(i * size + j) / scale_data(i);
          out_data(i * size + j) = static_cast<int8_t>(round(int8_value));
        }
      }
      writer.Add(scale_names[idx], scale_tensor);
      writer.Add(quant_names[idx] + "-values", out_tensor);
    }
    updated_names.insert(value_name);

    for (auto it = ev_suffix.cbegin(); it != ev_suffix.cend(); it++) {
      string tensor_name = names[idx] + *it;
      status = reader.LookupDtypeAndShape(tensor_name, &dtype, &shape);
      if (status.ok()) {
        Tensor tensor(dtype, shape);
        status = reader.Lookup(tensor_name, &tensor);
        if (status.ok()) {
          writer.Add(quant_names[idx] + *it, tensor);
          updated_names.insert(tensor_name);
        }
      }
    }
  }

  std::vector<std::string> tensor_names;
  reader.Seek(kHeaderEntryKey);
  reader.Next();
  for (; reader.Valid(); reader.Next()) {
    tensor_names.emplace_back(reader.key());
  }
  for (auto& tensor_name : tensor_names) {
    if (updated_names.count(tensor_name)) continue;
    status = reader.LookupDtypeAndShape(tensor_name, &dtype, &shape);
    if (status.ok()) {
      Tensor tensor(dtype, shape);
      status = reader.Lookup(tensor_name, &tensor);
      writer.Add(tensor_name, tensor);
    }
  }

  writer.Finish();

  return Status::OK();
}

}  // namespace checkpoint
}  // namespace tensorflow
