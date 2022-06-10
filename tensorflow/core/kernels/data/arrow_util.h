/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_ARROW_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_DATA_ARROW_UTIL_H_

#include <deque>
#include <string>

#include "arrow/dataset/api.h"
#include "arrow/record_batch.h"
#include "parquet/arrow/reader.h"
#include "parquet/properties.h"
#include "arrow/filesystem/localfs.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

#define TF_RETURN_IF_ARROW_ERROR(...)              \
  do {                                             \
    const ::arrow::Status _status = (__VA_ARGS__); \
    if (TF_PREDICT_FALSE(!_status.ok()))           \
      return errors::Internal(_status.ToString()); \
  } while (0)

#define TF_CHECKED_ARROW_ASSIGN(lhs, rexpr)                 \
  do {                                                      \
    auto&& _result = (rexpr);                               \
    if (TF_PREDICT_FALSE(!_result.ok()))                    \
      return errors::Internal(_result.status().ToString()); \
    lhs = std::move(_result).ValueUnsafe();                 \
  } while (0)

namespace tensorflow {
namespace data {
namespace ArrowUtil {

int UpdateArrowCpuThreadPoolCapacityFromEnv();

int GetArrowFileBufferSizeFromEnv();

::arrow::Status OpenArrowFile(
    std::shared_ptr<::arrow::io::RandomAccessFile>* file,
    const std::string& filename);

::arrow::Status OpenParquetReader(
    std::unique_ptr<::parquet::arrow::FileReader>* reader,
    const std::shared_ptr<::arrow::io::RandomAccessFile>& file);

::arrow::Status GetParquetDataFrameFields(
    std::vector<std::string>* field_names,
    std::vector<std::string>* field_dtypes,
    std::vector<int>* field_ragged_ranks, const std::string& filename);

template <typename TYPE>
struct DataTypeToArrowEnum {
  static constexpr ::arrow::Type::type value = ::arrow::Type::NA;
};

template <::arrow::Type::type VALUE>
struct ArrowEnumToDataType {
  typedef uint8 Type;
};

#define MATCH_TYPE_AND_ARROW_ENUM(TYPE, ENUM)          \
  template <>                                          \
  struct DataTypeToArrowEnum<TYPE> {                   \
    static constexpr ::arrow::Type::type value = ENUM; \
  };                                                   \
  template <>                                          \
  struct ArrowEnumToDataType<ENUM> {                   \
    typedef TYPE Type;                                 \
  }

MATCH_TYPE_AND_ARROW_ENUM(int8, ::arrow::Type::INT8);
MATCH_TYPE_AND_ARROW_ENUM(uint8, ::arrow::Type::UINT8);
MATCH_TYPE_AND_ARROW_ENUM(int32, ::arrow::Type::INT32);
MATCH_TYPE_AND_ARROW_ENUM(uint32, ::arrow::Type::UINT32);
MATCH_TYPE_AND_ARROW_ENUM(int64, ::arrow::Type::INT64);
MATCH_TYPE_AND_ARROW_ENUM(uint64, ::arrow::Type::UINT64);
MATCH_TYPE_AND_ARROW_ENUM(Eigen::half, ::arrow::Type::HALF_FLOAT);
MATCH_TYPE_AND_ARROW_ENUM(float, ::arrow::Type::FLOAT);
MATCH_TYPE_AND_ARROW_ENUM(double, ::arrow::Type::DOUBLE);
MATCH_TYPE_AND_ARROW_ENUM(string, ::arrow::Type::STRING);

Status MakeDataTypeAndRaggedRankFromArrowDataType(
    const std::shared_ptr<::arrow::DataType>& arrow_dtype, DataType* dtype,
    int32* ragged_rank);

Status MakeTensorsFromArrowArray(
    DataType type, int32 ragged_rank,
    const std::shared_ptr<::arrow::Array>& arrow_array,
    std::vector<Tensor>* output_tensors);

}  // namespace ArrowUtil
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_ARROW_UTIL_H_
