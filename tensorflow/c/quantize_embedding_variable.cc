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

void WriteRestVariables(BundleReader& reader, BundleWriter& writer,
                        const std::vector<string>& ignored_names) {
  std::set<string> excluded_names(ignored_names.cbegin(), ignored_names.cend());
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
    if (excluded_names.count(tensor_name)) continue;
    status = reader.LookupDtypeAndShape(tensor_name, &dtype, &shape);
    if (status.ok()) {
      Tensor tensor(dtype, shape);
      status = reader.Lookup(tensor_name, &tensor);
      writer.Add(tensor_name, tensor);
    }
  }
}

void WriteRestVariables(BundleReader& reader, BundleWriter& writer,
                        const std::vector<string>& ignored_names,
                        const std::set<string>& ev_suffix) {
  std::vector<string> ev_names;
  for (int idx = 0; idx < ignored_names.size(); ++idx) {
    for (auto it = ev_suffix.cbegin(); it != ev_suffix.cend(); ++it) {
      ev_names.push_back(ignored_names[idx] + *it);
    }
  }
  WriteRestVariables(reader, writer, ev_names);
}

void ConvertToBF16Value(const Tensor& in_tensor, const string name,
                        BundleWriter& writer) {
  auto in_data = in_tensor.flat<float>();
  Tensor out_tensor(DataTypeToEnum<bfloat16>::v(), in_tensor.shape());
  auto out_data = out_tensor.flat<bfloat16>();
  int64 data_size = out_tensor.NumElements();
#if TEL_MKL
  dnnl::cvt_float_to_bfloat16((dnnl_bfloat16_t*)out_data.data(),
                              (const float*)in_data.data(), data_size);
#else
  FloatToBFloat16(in_data.data(), out_data.data(), data_size);
#endif  // INTEL_MKL
  writer.Add(name, out_tensor);
}

void ConvertToHalfValue(const Tensor& in_tensor, const string name,
                        BundleWriter& writer) {
  auto in_data = in_tensor.flat<float>();
  Tensor out_tensor(DataTypeToEnum<Eigen::half>::v(), in_tensor.shape());
  auto out_data = out_tensor.flat<Eigen::half>();
#if INTEL_MKL
#pragma omp parallel for num_threads(omp_get_num_procs())
#endif  // INTEL_MKL
  for (size_t i = 0; i < out_tensor.NumElements(); ++i) {
    out_data(i) = static_cast<Eigen::half>(in_data(i));
  }
  writer.Add(name, out_tensor);
}

void ConvertToInt8Value(const Tensor& in_tensor, const string name,
                        const string scale_name, BundleWriter& writer) {
  auto in_data = in_tensor.flat<float>();
  TensorShape shape = in_tensor.shape();
  Tensor out_tensor(DataTypeToEnum<int8_t>::v(), shape);
  auto out_data = out_tensor.flat<int8_t>();
  int embed_dim = shape.dim_size(shape.dims() - 1);
  Tensor scale_tensor(DT_FLOAT, TensorShape({embed_dim}));
  auto scale_data = scale_tensor.flat<float>();
  std::vector<float> max_val(embed_dim, 0.0);
#if INTEL_MKL
#pragma omp parallel for num_threads(omp_get_num_procs())
#endif  // INTEL_MKL
  for (size_t i = 0; i < out_tensor.NumElements(); ++i) {
    int embed_i = i % embed_dim;
    max_val[embed_i] = std::max(max_val[embed_i], std::abs(in_data(i)));
  }
  for (size_t i = 0; i < embed_dim; ++i) {
    scale_data(i) = max_val[i] / 127.0;
  }
#if INTEL_MKL
#pragma omp parallel for num_threads(omp_get_num_procs())
#endif  // INTEL_MKL
  for (size_t i = 0; i < out_tensor.NumElements(); ++i) {
    int embed_i = i % embed_dim;
    out_data(i) = static_cast<int8_t>(round(in_data(i) / scale_data(embed_i)));
  }
  writer.Add(scale_name, scale_tensor);
  writer.Add(name, out_tensor);
}

Status QuantizeEmbeddingVariable(const string& input_prefix,
                                 const string& output_prefix,
                                 const std::vector<string>& names,
                                 const std::vector<string>& quant_names,
                                 const std::vector<string>& scale_names,
                                 const TF_DataType data_type,
                                 const bool is_ev) {
  BundleReader reader(Env::Default(), input_prefix);
  BundleWriter writer(Env::Default(), output_prefix);
  const std::set<string> ev_suffix = {
      "-freqs",         "-freqs_filtered",          "-keys",
      "-keys_filtered", "-partition_filter_offset", "-partition_offset",
      "-versions",      "-versions_filtered",       "-values"};

  for (int idx = 0; idx < names.size(); ++idx) {
    Status status;
    DataType dtype;
    TensorShape shape;
    string suffix = is_ev ? "-values" : "";
    string value_name = names[idx] + suffix;
    status = reader.LookupDtypeAndShape(value_name, &dtype, &shape);
    if (!status.ok()) {
      errors::InvalidArgument("Invalid variable name:", value_name);
    }
    Tensor in_tensor(dtype, shape);
    status = reader.Lookup(value_name, &in_tensor);
    auto in_data = in_tensor.flat<float>();

    string quant_name = quant_names[idx] + suffix;
    if (data_type == TF_DataType::TF_BFLOAT16) {
      ConvertToBF16Value(in_tensor, quant_name, writer);
    } else if (data_type == TF_DataType::TF_HALF) {
      ConvertToHalfValue(in_tensor, quant_name, writer);
    } else if (data_type == TF_DataType::TF_INT8) {
      ConvertToInt8Value(in_tensor, quant_name, scale_names[idx], writer);
    } else {
      errors::InvalidArgument("Unsupported data type:", data_type);
    }
    if (is_ev) {
      for (auto it = ev_suffix.cbegin(); it != ev_suffix.cend(); ++it) {
        if (*it == "-values") continue;
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
  }

  if (is_ev) {
    WriteRestVariables(reader, writer, names, ev_suffix);
  } else {
    WriteRestVariables(reader, writer, names);
  }
  writer.Finish();
  return Status::OK();
}

Status RemoveVariable(const string& input_prefix, const string& output_prefix,
                      const std::vector<string>& names) {
  BundleReader reader(Env::Default(), input_prefix);
  BundleWriter writer(Env::Default(), output_prefix);
  WriteRestVariables(reader, writer, names);
  writer.Finish();
  return Status::OK();
}

}  // namespace checkpoint
}  // namespace tensorflow
