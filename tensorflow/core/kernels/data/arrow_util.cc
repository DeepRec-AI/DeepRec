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
#include "tensorflow/core/kernels/data/arrow_util.h"

#include <cstdlib>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include "arrow/array.h"
#include "arrow/util/thread_pool.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/kernels/data/eigen.h"

namespace tensorflow {
namespace data {
namespace ArrowUtil {

namespace {

int EnvGetInt(const std::string& env_var, int default_val) {
  const char* env_var_val = getenv(env_var.c_str());
  if (env_var_val == nullptr) {
    return default_val;
  }
  std::string env_var_val_str(env_var_val);
  std::istringstream ss(env_var_val_str);
  int result;
  if (!(ss >> result)) {
    result = default_val;
  }
  return result;
}

int SetArrowCpuThreadPoolCapacityFromEnv() {
  int arrow_threads = EnvGetInt("ARROW_NUM_THREADS", 0);
  if (arrow_threads > 0) {  // Set from environment variable
    auto s = ::arrow::SetCpuThreadPoolCapacity(arrow_threads);
    if (ARROW_PREDICT_FALSE(!s.ok())) {
      return 0;
    }
  }
  return arrow_threads;
}

::arrow::Status MakeNumpyDtypeAndRaggedRankFromArrowDataType(
    std::string* numpy_dtype, int* ragged_rank,
    const std::shared_ptr<::arrow::DataType>& arrow_dtype) {
  if (arrow_dtype->id() == ::arrow::Type::LIST) {
    ++(*ragged_rank);
    return MakeNumpyDtypeAndRaggedRankFromArrowDataType(
        numpy_dtype, ragged_rank, arrow_dtype->field(0)->type());
  }

  switch (arrow_dtype->id()) {
    case ::arrow::Type::INT8:
    case ::arrow::Type::UINT8:
    case ::arrow::Type::INT32:
    case ::arrow::Type::INT64:
    case ::arrow::Type::UINT64:
      *numpy_dtype = arrow_dtype->name();
      break;
    case ::arrow::Type::HALF_FLOAT:
      *numpy_dtype = "float16";
      break;
    case ::arrow::Type::FLOAT:
      *numpy_dtype = "float32";
      break;
    case ::arrow::Type::DOUBLE:
      *numpy_dtype = "float64";
      break;
    case ::arrow::Type::STRING:
      *numpy_dtype = "O";
      break;
    default:
      return ::arrow::Status::Invalid(
          "Arrow data type ", arrow_dtype->ToString(), " not supported.");
  }
  return ::arrow::Status::OK();
}

#if DEEPREC_ARROW_ZEROCOPY
class ArrowPrimitiveTensorBuffer : public TensorBuffer {
 public:
  ArrowPrimitiveTensorBuffer() = delete;

  explicit ArrowPrimitiveTensorBuffer(
      const std::shared_ptr<arrow::Buffer>& arrow_buffer)
      : TensorBuffer(const_cast<uint8_t*>(arrow_buffer->data())),
        arrow_buffer_(arrow_buffer) {}

  size_t size() const override { return arrow_buffer_->size(); }

  TensorBuffer* root_buffer() override { return this; }

  void FillAllocationDescription(AllocationDescription* proto) const override {
    proto->set_requested_bytes(size());
    proto->set_allocator_name(::tensorflow::cpu_allocator()->Name());
  }

  bool OwnsMemory() const override { return false; }

 private:
  std::shared_ptr<::arrow::Buffer> arrow_buffer_;
};
#endif

::arrow::Status MakeTensorFromArrowBuffer(
    DataType dtype, const std::shared_ptr<::arrow::Buffer>& arrow_buffer,
    Tensor* tensor) {
  const TensorShape shape = {arrow_buffer->size() / DataTypeSize(dtype)};

#if DEEPREC_ARROW_ZEROCOPY
  // NOTE: Alignment is 64 in Arrow 4.x, same to EIGEN_MAX_ALIGN_BYTES. See:
  // https://github.com/apache/arrow/blob/apache-arrow-4.0.1/cpp/src/arrow/memory_pool.cc#L97
  if (TF_PREDICT_FALSE(!CHECK_EIGEN_ALIGN(arrow_buffer->data()))) {
    *tensor = Tensor(dtype, shape);
    std::memcpy(const_cast<char*>(tensor->tensor_data().data()),
                arrow_buffer->data(), arrow_buffer->size());
    return ::arrow::Status::OK();
  }

  ArrowPrimitiveTensorBuffer* tensor_buffer =
      new ArrowPrimitiveTensorBuffer(arrow_buffer);
  core::ScopedUnref unref(tensor_buffer);
  *tensor = Tensor(dtype, shape, tensor_buffer);
  return ::arrow::Status::OK();
#else
  *tensor = Tensor(dtype, shape);
  std::memcpy(const_cast<char*>(tensor->tensor_data().data()),
              arrow_buffer->data(), arrow_buffer->size());
  return ::arrow::Status::OK();
#endif
}

::arrow::Status MakeStringTensorFromArrowArray(
    const ::arrow::StringArray& array, Tensor* tensor) {
  if (array.null_count() != 0) {
    return ::arrow::Status::Invalid("Null elements not supported");
  }

  const auto num_strings = array.length();

  *tensor = Tensor(DT_STRING, TensorShape({num_strings}));
  auto tensor_vec = tensor->vec<std::string>();

  for (auto i = 0; i < num_strings; ++i) {
    int string_size;
    auto string_data = array.GetValue(i, &string_size);
    tensor_vec(i).assign(reinterpret_cast<const char*>(string_data),
                         string_size);
  }
  return ::arrow::Status::OK();
}

// Primitive Arrow arrays have validity and value buffers.
#define RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(ARRAY_CLASS)                    \
  ::arrow::Status Visit(const ARRAY_CLASS& array) override {                  \
    if (TF_PREDICT_FALSE(ragged_rank_ != 0)) {                                \
      return ::arrow::Status::Invalid("Inconsistent ragged rank");            \
    }                                                                         \
    Tensor tensor;                                                            \
    auto st =                                                                 \
        MakeTensorFromArrowBuffer(dtype_, array.data()->buffers[1], &tensor); \
    if (!st.ok()) {                                                           \
      return st;                                                              \
    }                                                                         \
    ragged_tensor_.push_front(std::move(tensor));                             \
    return ::arrow::Status::OK();                                             \
  }

#define RAGGED_TENSOR_BUILDER_STRING_VISIT(ARRAY_CLASS)            \
  ::arrow::Status Visit(const ARRAY_CLASS& array) override {       \
    if (TF_PREDICT_FALSE(ragged_rank_ != 0)) {                     \
      return ::arrow::Status::Invalid("Inconsistent ragged rank"); \
    }                                                              \
    Tensor tensor;                                                 \
    auto st = MakeStringTensorFromArrowArray(array, &tensor);      \
    if (!st.ok()) {                                                \
      return st;                                                   \
    }                                                              \
    ragged_tensor_.push_front(std::move(tensor));                  \
    return ::arrow::Status::OK();                                  \
  }

class RaggedTensorBuilder : public ::arrow::ArrayVisitor {
 public:
  RaggedTensorBuilder(DataType dtype, int32 ragged_rank)
      : dtype_(dtype), ragged_rank_(ragged_rank) {}

  ::arrow::Status Build(const std::shared_ptr<::arrow::Array>& array,
                        std::vector<Tensor>* output_tensors) {
    auto st = array->Accept(this);
    if (!st.ok()) {
      return st;
    }
    output_tensors->insert(output_tensors->end(), ragged_tensor_.begin(),
                           ragged_tensor_.end());
    return ::arrow::Status::OK();
  }

  ::arrow::Status Visit(const ::arrow::ListArray& array) override {
    --ragged_rank_;
    Tensor tensor;
    auto st =
        MakeTensorFromArrowBuffer(DT_INT32, array.value_offsets(), &tensor);
    if (!st.ok()) {
      return st;
    }
    ragged_tensor_.push_front(std::move(tensor));
    return array.values()->Accept(this);
  }

  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::Int8Array);
  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::UInt8Array);
  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::Int32Array);
  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::UInt32Array);
  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::Int64Array);
  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::UInt64Array);
  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::HalfFloatArray);
  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::FloatArray);
  RAGGED_TENSOR_BUILDER_PRIMITIVE_VISIT(::arrow::DoubleArray);

  RAGGED_TENSOR_BUILDER_STRING_VISIT(::arrow::StringArray);

 private:
  const DataType dtype_;
  int32 ragged_rank_;
  std::deque<Tensor> ragged_tensor_;
};

}  // namespace

#define CASE_ARROW_ENUM_SET_DTYPE(PTR, ENUM)                       \
  case ENUM: {                                                     \
    *PTR = DataTypeToEnum<ArrowEnumToDataType<ENUM>::Type>::value; \
    return Status::OK();                                           \
  }

Status MakeDataTypeAndRaggedRankFromArrowDataType(
    const std::shared_ptr<::arrow::DataType>& arrow_dtype, DataType* dtype,
    int32* ragged_rank) {
  if (arrow_dtype->id() == ::arrow::Type::LIST) {
    ++(*ragged_rank);
    return MakeDataTypeAndRaggedRankFromArrowDataType(
        arrow_dtype->field(0)->type(), dtype, ragged_rank);
  }

  switch (arrow_dtype->id()) {
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::INT8);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::UINT8);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::INT32);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::UINT32);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::INT64);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::UINT64);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::HALF_FLOAT);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::FLOAT);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::DOUBLE);
    CASE_ARROW_ENUM_SET_DTYPE(dtype, ::arrow::Type::STRING);
    default:
      return errors::Unimplemented("Arrow data type ", arrow_dtype->ToString(),
                                   " not supported.");
  }
  return Status::OK();
}

Status MakeTensorsFromArrowArray(
    DataType dtype, int32 ragged_rank,
    const std::shared_ptr<::arrow::Array>& arrow_array,
    std::vector<Tensor>* output_tensors) {
  if (TF_PREDICT_FALSE(arrow_array->null_count() != 0)) {
    return errors::Internal("Arrow array with null values not supported");
  }

  if (TF_PREDICT_FALSE(arrow_array->data()->offset != 0)) {
    return errors::Internal("Arrow array has zero non-offset not supported");
  }

  RaggedTensorBuilder builder(dtype, ragged_rank);
  TF_RETURN_IF_ARROW_ERROR(builder.Build(arrow_array, output_tensors));
  return Status::OK();
}

int UpdateArrowCpuThreadPoolCapacityFromEnv() {
  static int arrow_threads = SetArrowCpuThreadPoolCapacityFromEnv();
  return arrow_threads;
}

int GetArrowFileBufferSizeFromEnv() {
  static int buffer_size = EnvGetInt("ARROW_FILE_BUFFER_SIZE", 4096 * 4);
  return buffer_size;
}

::arrow::Status OpenArrowFile(
    std::shared_ptr<::arrow::io::RandomAccessFile>* file,
    const std::string& filename) {
#if DEEPREC_ARROW_HDFS
  if (filename.rfind("hdfs://", 0) == 0) {
    ::arrow::internal::Uri uri;
    ARROW_RETURN_NOT_OK(uri.Parse(filename));
    ARROW_ASSIGN_OR_RAISE(auto options, ::arrow::fs::HdfsOptions::FromUri(uri));
    std::shared_ptr<::arrow::io::HadoopFileSystem> fs;
    ARROW_RETURN_NOT_OK(::arrow::io::HadoopFileSystem::Connect(
        &options.connection_config, &fs));
    std::shared_ptr<::arrow::io::HdfsReadableFile> hdfs_file;
    ARROW_RETURN_NOT_OK(fs->OpenReadable(uri.path(), &hdfs_file));
    *file = hdfs_file;
    return ::arrow::Status::OK();
  }
#endif
  auto fs = std::make_shared<::arrow::fs::LocalFileSystem>();
  ARROW_ASSIGN_OR_RAISE(*file, fs->OpenInputFile(filename));
  return ::arrow::Status::OK();
}

::arrow::Status OpenParquetReader(
    std::unique_ptr<::parquet::arrow::FileReader>* reader,
    const std::shared_ptr<::arrow::io::RandomAccessFile>& file) {
  auto config = ::parquet::ReaderProperties();
  config.enable_buffered_stream();
  config.set_buffer_size(GetArrowFileBufferSizeFromEnv());
  ARROW_RETURN_NOT_OK(::parquet::arrow::FileReader::Make(
      ::arrow::default_memory_pool(),
      ::parquet::ParquetFileReader::Open(file, config), reader));
  // If ARROW_NUM_THREADS > 0, specified number of threads will be used.
  // If ARROW_NUM_THREADS = 0, no threads will be used.
  // If ARROW_NUM_THREADS < 0, all threads will be used.
  (*reader)->set_use_threads(UpdateArrowCpuThreadPoolCapacityFromEnv() != 0);
  return ::arrow::Status::OK();
}

::arrow::Status GetParquetDataFrameFields(
    std::vector<std::string>* field_names,
    std::vector<std::string>* field_dtypes,
    std::vector<int>* field_ragged_ranks, const std::string& filename) {
  std::shared_ptr<::arrow::io::RandomAccessFile> file;
  ARROW_RETURN_NOT_OK(OpenArrowFile(&file, filename));
  std::unique_ptr<::parquet::arrow::FileReader> reader;
  ARROW_RETURN_NOT_OK(OpenParquetReader(&reader, file));

  std::shared_ptr<::arrow::Schema> schema;
  ARROW_RETURN_NOT_OK(reader->GetSchema(&schema));
  if (ARROW_PREDICT_FALSE(!schema->HasDistinctFieldNames())) {
    return ::arrow::Status::Invalid(filename,
                                    " must has distinct column names");
  }
  for (const auto& field : schema->fields()) {
    field_names->push_back(field->name());
    std::string dtype;
    int ragged_rank = 0;
    ARROW_RETURN_NOT_OK(MakeNumpyDtypeAndRaggedRankFromArrowDataType(
        &dtype, &ragged_rank, field->type()));
    field_dtypes->push_back(dtype);
    field_ragged_ranks->push_back(ragged_rank);
  }
  return ::arrow::Status::OK();
}

}  // namespace ArrowUtil
}  // namespace data
}  // namespace tensorflow
