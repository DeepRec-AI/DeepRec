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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_PARQUET_BATCH_READER_H_
#define TENSORFLOW_CORE_KERNELS_DATA_PARQUET_BATCH_READER_H_

#include <memory>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace data {

class ParquetBatchReader {
 public:
  ParquetBatchReader(const string& filename, const int64 batch_size,
                     const std::vector<string>& field_names,
                     const DataTypeVector& field_dtypes,
                     const std::vector<int32>& field_ragged_ranks,
                     const int64 partition_count, const int64 partition_index,
                     const bool drop_remainder);

  Status Open();

  Status Read(std::vector<Tensor>* output_tensors);

  virtual ~ParquetBatchReader();

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_PARQUET_BATCH_READER_H_
