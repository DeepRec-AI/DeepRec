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
#include "tensorflow/core/framework/bfloat16.h"

#ifdef INTEL_MKL
#include <omp.h>
#include "dnnl.hpp"
#endif  // INTEL_MKL

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

#if INTEL_MKL
#pragma omp parallel for num_threads(omp_get_num_procs())
#endif  // INTEL_MKL
  for (int idx = 0; idx < names.size(); ++idx) {
    Status status;
    DataType dtype;
    TensorShape shape;
    string value_name = names[idx] + "-values";
    status = reader.LookupDtypeAndShape(value_name, &dtype, &shape);
    if (!status.ok()) {
      errors::InvalidArgument("Invalid variable name:", value_name);
    }
    Tensor in_tensor(dtype, shape);
    status = reader.Lookup(value_name, &in_tensor);
    auto in_data = in_tensor.flat<float>();

    if (data_type == TF_DataType::TF_BFLOAT16) {
      Tensor out_tensor(DataTypeToEnum<bfloat16>::v(), shape);
      auto out_data = out_tensor.flat<bfloat16>();
      int64 data_size = out_tensor.NumElements();
#if INTEL_MKL
      dnnl::cvt_float_to_bfloat16((dnnl_bfloat16_t*)out_data.data(),
                                  (const float*)in_data.data(), data_size);
#else
      FloatToBFloat16(in_data.data(), out_data.data(), data_size);
#endif  // INTEL_MKL
      writer.Add(quant_names[idx] + "-values", out_tensor);
    } else if (data_type == TF_DataType::TF_HALF) {
      Tensor out_tensor(DataTypeToEnum<Eigen::half>::v(), shape);
      auto out_data = out_tensor.flat<Eigen::half>();
      for (size_t i = 0; i < out_tensor.NumElements(); ++i) {
        out_data(i) = static_cast<Eigen::half>(in_data(i));
      }
      writer.Add(quant_names[idx] + "-values", out_tensor);
    } else if (data_type == TF_DataType::TF_INT8) {
      Tensor out_tensor(DataTypeToEnum<int8_t>::v(), shape);
      auto out_data = out_tensor.flat<int8_t>();
      int embed_dim = shape.dim_size(shape.dims() - 1);
      Tensor scale_tensor(DT_FLOAT, TensorShape({embed_dim}));
      auto scale_data = scale_tensor.flat<float>();
      for (int i = 0; i < embed_dim; ++i) {
        int64 voc_size = in_tensor.NumElements() / embed_dim;
        float max_val = 0;
        for (size_t j = 0; j < voc_size; ++j) {
          max_val = std::max(max_val, std::abs(in_data(j * embed_dim + i)));
        }
        scale_data(i) = max_val / 127.0;
        for (size_t j = 0; j < voc_size; ++j) {
          float int8_value = in_data(j * embed_dim + i) / scale_data(i);
          out_data(j * embed_dim + i) = static_cast<int8_t>(round(int8_value));
        }
      }
      writer.Add(scale_names[idx], scale_tensor);
      writer.Add(quant_names[idx] + "-values", out_tensor);
    } else {
      errors::InvalidArgument("Unsupported data type:", data_type);
    }

    for (auto it = ev_suffix.cbegin(); it != ev_suffix.cend(); ++it) {
      string tensor_name = names[idx] + *it;
      status = reader.LookupDtypeAndShape(tensor_name, &dtype, &shape);
      if (status.ok()) {
        Tensor tensor(dtype, shape);
        status = reader.Lookup(tensor_name, &tensor);
        if (status.ok()) {
          writer.Add(quant_names[idx] + *it, tensor);
        }
      }
    }
  }

  std::set<string> updated_names;
  for (int idx = 0; idx < names.size(); ++idx) {
    updated_names.insert(names[idx] + "-values");
    for (auto it = ev_suffix.cbegin(); it != ev_suffix.cend(); ++it) {
      updated_names.insert(names[idx] + *it);
    }
  }

  std::vector<std::string> tensor_names;
  reader.Seek(kHeaderEntryKey);
  reader.Next();
  for (; reader.Valid(); reader.Next()) {
    tensor_names.emplace_back(reader.key());
  }
  for (auto& tensor_name : tensor_names) {
    Status status;
    DataType dtype;
    TensorShape shape;
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
