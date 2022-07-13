/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

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
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "pybind11/complex.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/core/kernels/data/arrow_util.h"

namespace tensorflow {
namespace data {

namespace {

std::string make_buildinfo() {
  std::string message = "deeprec buildinfo";
  return message;
}

std::string buildinfo() {
  static std::string kBuildInfo = make_buildinfo();
  return kBuildInfo;
}

typedef std::tuple<std::string, std::string, int> parquet_file_field_t;

std::vector<parquet_file_field_t> parquet_file_get_fields(
    const std::string& filename) {
  std::vector<std::string> field_names;
  std::vector<std::string> field_dtypes;
  std::vector<int> field_ragged_ranks;
  auto s = ArrowUtil::GetParquetDataFrameFields(
      &field_names, &field_dtypes, &field_ragged_ranks, filename);
  std::vector<parquet_file_field_t> fields;
  if (!s.ok()) {
    std::cerr << "parquet_file_get_fields failed: " << s.message() << std::endl;
    return fields;
  }
  for (size_t i = 0; i < field_names.size(); ++i) {
    fields.emplace_back(field_names[i], field_dtypes[i], field_ragged_ranks[i]);
  }
  return fields;
}

}  // namespace

PYBIND11_MODULE(_parquet_pybind, m) {
  m.def("buildinfo", &buildinfo, "Get building information.");
  m.def("parquet_file_get_fields", &parquet_file_get_fields,
        "Get fields of a parquet file.");
}

}  // namespace data
}  // namespace tensorflow
