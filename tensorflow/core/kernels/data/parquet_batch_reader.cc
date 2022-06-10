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
#include "tensorflow/core/kernels/data/parquet_batch_reader.h"

#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "tensorflow/core/kernels/data/arrow_util.h"

namespace tensorflow {
namespace data {

class ParquetBatchReader::Impl {
 public:
  Impl(const string& filename, const int64 batch_size,
       const std::vector<string>& field_names,
       const DataTypeVector& field_dtypes,
       const std::vector<int32>& field_ragged_ranks,
       const int64 partition_count, const int64 partition_index,
       const bool drop_remainder)
      : filename_(filename),
        batch_size_(batch_size),
        field_names_(field_names),
        field_dtypes_(field_dtypes),
        field_ragged_ranks_(field_ragged_ranks),
        partition_count_(partition_count),
        partition_index_(partition_index),
        drop_remainder_(drop_remainder) {}

  Status Open() {
    if (TF_PREDICT_TRUE(batch_reader_)) {
      return Status::OK();
    }
    if (TF_PREDICT_FALSE(partition_index_ >= partition_count_)) {
      return errors::InvalidArgument("Partition index ", partition_index_,
                                     " must be smaller than partition count ",
                                     partition_count_);
    }
    if (TF_PREDICT_FALSE(partition_index_ < 0)) {
      return errors::InvalidArgument("Partition index ", partition_index_,
                                     "must be greater than 0");
    }

    std::shared_ptr<::arrow::io::RandomAccessFile> file;
    TF_RETURN_IF_ARROW_ERROR(ArrowUtil::OpenArrowFile(&file, filename_));
    TF_RETURN_IF_ARROW_ERROR(ArrowUtil::OpenParquetReader(&reader_, file));

    int num_row_groups = reader_->num_row_groups();
    for (int g = partition_index_; g < num_row_groups; g += partition_count_) {
      row_group_indices_.push_back(g);
    }
    std::shared_ptr<::arrow::Schema> schema;
    TF_RETURN_IF_ARROW_ERROR(reader_->GetSchema(&schema));
    if (TF_PREDICT_FALSE(!schema->HasDistinctFieldNames())) {
      return errors::InvalidArgument(filename_,
                                     " must has distinct column names");
    }
    for (size_t i = 0; i < field_names_.size(); ++i) {
      auto& cname = field_names_[i];
      int column_index = schema->GetFieldIndex(cname);
      if (TF_PREDICT_FALSE(column_index < 0)) {
        return errors::NotFound("No column called `", cname, "` found in ",
                                filename_);
      }
      column_indices_.push_back(column_index);
      const auto& expected_dtype = field_dtypes_[i];
      const auto& expected_ragged_rank = field_ragged_ranks_[i];
      DataType actual_dtype;
      int32 actual_ragged_rank = 0;
      TF_RETURN_IF_ERROR(ArrowUtil::MakeDataTypeAndRaggedRankFromArrowDataType(
          schema->field(column_index)->type(), &actual_dtype,
          &actual_ragged_rank));
      if (TF_PREDICT_FALSE(actual_dtype != expected_dtype)) {
        return errors::InvalidArgument(
            "Field ", cname, " in ", filename_, " has unexpected data type ",
            DataTypeString(actual_dtype), ", which should be ",
            DataTypeString(expected_dtype));
      }
      if (TF_PREDICT_FALSE(actual_ragged_rank != expected_ragged_rank)) {
        return errors::InvalidArgument(
            "Field ", cname, " in ", filename_, " has unexpected ragged rank ",
            actual_ragged_rank, ", which should be ", expected_ragged_rank);
      }
    }
    reader_->set_batch_size(batch_size_);

    TF_RETURN_IF_ARROW_ERROR(reader_->GetRecordBatchReader(
        row_group_indices_, column_indices_, &batch_reader_));
    return Status::OK();
  }

  Status Read(std::vector<Tensor>* output_tensors) {
    // Read next batch from parquet file.
    std::shared_ptr<::arrow::RecordBatch> batch;
    TF_RETURN_IF_ARROW_ERROR(batch_reader_->ReadNext(&batch));
    if (TF_PREDICT_FALSE(!batch)) {
      return errors::OutOfRange("Reached end of parquet file ", filename_);
    }
    if (TF_PREDICT_FALSE(drop_remainder_ && batch->num_rows() < batch_size_)) {
      return errors::OutOfRange("Reached end of parquet file ", filename_,
                                " after dropping reminder batch");
    }

    // Populate tensors from record batch.
    auto arrays = batch->columns();
    for (size_t i = 0; i < arrays.size(); ++i) {
      TF_RETURN_IF_ERROR(ArrowUtil::MakeTensorsFromArrowArray(
          field_dtypes_[i], field_ragged_ranks_[i], arrays[i], output_tensors));
    }

    return Status::OK();
  }

 private:
  const string filename_;
  const int64 batch_size_;
  std::vector<string> field_names_;
  DataTypeVector field_dtypes_;
  std::vector<int32> field_ragged_ranks_;
  int64 partition_count_;
  int64 partition_index_;
  bool drop_remainder_;
  std::unique_ptr<::parquet::arrow::FileReader> reader_;
  std::unique_ptr<::arrow::RecordBatchReader> batch_reader_;
  std::vector<int> row_group_indices_;
  std::vector<int> column_indices_;
};

ParquetBatchReader::ParquetBatchReader(
    const string& filename, const int64 batch_size,
    const std::vector<string>& field_names, const DataTypeVector& field_dtypes,
    const std::vector<int32>& field_ragged_ranks, const int64 partition_count,
    const int64 partition_index, const bool drop_remainder)
    : pimpl_(new ParquetBatchReader::Impl(
          filename, batch_size, field_names, field_dtypes, field_ragged_ranks,
          partition_count, partition_index, drop_remainder)) {}

Status ParquetBatchReader::Open() { return pimpl_->Open(); }

Status ParquetBatchReader::Read(std::vector<Tensor>* output_tensors) {
  return pimpl_->Read(output_tensors);
}

ParquetBatchReader::~ParquetBatchReader() {}

}  // namespace data
}  // namespace tensorflow
