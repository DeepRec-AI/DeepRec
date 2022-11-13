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

#ifndef TENSORFLOW_CORE_KERNELS_SAVE_RESTORE_TENSOR_H_
#define TENSORFLOW_CORE_KERNELS_SAVE_RESTORE_TENSOR_H_

#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/util/tensor_slice_reader.h"
#include "tensorflow/core/util/tensor_slice_writer.h"
#include "tensorflow/core/framework/hash_table/tensible_variable.h"
#include "tensorflow/core/framework/hash_table/hash_table.h"
#include "tensorflow/core/framework/hash_table/bloom_filter_strategy.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"

namespace tensorflow {

class OpKernelContext;

// Legacy / V1 checkpoint format.

// Save input tensors in *context to a writer built from builder_func().
// context must have the following inputs:
//  0: a single element string tensor that contains the file name.
//  1: names for the remaining tensors
// If save_slices is true:
//  2: shape and slice specifications.
//  rest: tensors to save
void SaveTensors(
    OpKernelContext* context,
    checkpoint::TensorSliceWriter::CreateBuilderFunction builder_func,
    bool save_slices);

// Reads a single tensor from the reader built from open_func() and produces
// it as context->output(restore_index).  "preferred_shard" is the same the
// TensorSliceReader preferred_shard parameter.
//
// context must have the following inputs:
//  0: a single element string tensor that contains the file name.
//  1: string tensor that names the outputs to be restored.
// If restore_slice is true:
//  2: shape and slice specification of the tensors to restore.
//
// restore_index indicates the variable name and slice to lookup
// in context(1) and (2).
void RestoreTensor(OpKernelContext* context,
                   checkpoint::TensorSliceReader::OpenTableFunction open_func,
                   int preferred_shard, bool restore_slice, int restore_index);

// V2 checkpoint format.

// Invokes the V2 checkpoint read path to read tensors.
//
// "context" is only used for allocating outputs.  In particular, the inputs are
// explicitly provided and not accessed via the "input(i)" methods.
// REQUIRES:
//   * "prefix" has 1 element, DT_STRING.
//   * "tensor_names" and "shape_and_slices" shaped {N}, both DT_STRING.
//   * "dtypes" has N elements, the datatypes of the to-restore tensors.
Status RestoreTensorsV2(OpKernelContext* context, const Tensor& prefix,
                        const Tensor& tensor_names,
                        const Tensor& shape_and_slices,
                        gtl::ArraySlice<DataType> dtypes);

Status SaveHashTable(
    BundleWriter* writer, HashTable* table,
    const std::vector<TensibleVariable*>& tensibles,
    const string& table_name, const std::vector<string>& tensibles_name,
    int64 slice_beg, int64 slice_length, int64 slice_size);

void RestoreHashTable(
    std::function<void(std::function<void()>)> runner,
    BundleReader* reader, HashTable* table,
    const std::vector<TensibleVariable*>& tensibles,
    const string& table_name, const std::vector<string>& tensibles_name,
    int64 slice_beg, int64 slice_length, int64 slice_size,
    std::function<void(Status)> done);

void RestoreCoalescedHashTable(
    std::function<void(std::function<void()>)> runner,
    BundleReader* reader, CoalescedHashTable* table,
    const std::vector<TensibleVariable*>& tensibles,
    const std::vector<string>& table_names,
    const std::vector<std::vector<string>>& tensibles_names,
    int64 slice_beg, int64 slice_length, int64 slice_size,
    bool clear, std::function<void(Status)> done);

Status SaveBloomFilter(
    BundleWriter* writer, BloomFilterAdmitStrategy* strategy,
    const string& name, int64 slice_beg, int64 slice_length, int64 slice_size);

Status RestoreBloomFilter(
    BundleReader* reader, BloomFilterAdmitStrategy* strategy,
    const string& name, int64 slice_beg, int64 slice_length, int64 slice_size);

template<class T>
class DumpIterator {
 public:
  virtual ~DumpIterator() {}
  virtual bool HasNext() const = 0;
  virtual T Next() = 0;
};

template<typename T>
Status SaveTensorWithFixedBuffer(const string& tensor_name,
    BundleWriter* writer,
    char* dump_buffer,
    size_t bytes_limit,
    DumpIterator<T>* dump_iter,
    const TensorShape& dump_tensor_shape,
    embedding::Iterator* it = nullptr,
    // -1: save key, x_offset: save embedding(primary or slot offset)
    // -2: save frequency, -3: save version
    int64 value_offset = -1,
    bool use_shape = true) {
  bool dump_happened = false;
  size_t bytes_written = 0;
  int buffer_idx = 0;
  Status st;
  int64 total_bytes_written = 0;
  T* key_dump_buffer = (T*)dump_buffer;
  if (use_shape)
  st = writer->AddTensorHeader(tensor_name, DataTypeToEnum<T>::v(), dump_tensor_shape);
  if (!st.ok())
    return st;

  while (dump_iter->HasNext()) {
    T key = dump_iter->Next();
    if (bytes_written + sizeof(T) > bytes_limit) {
      dump_happened = true;
      writer->AppendSegmentData(dump_buffer, bytes_written);
      bytes_written = 0;
      buffer_idx = 0;
    }
    key_dump_buffer[buffer_idx] = key;
    buffer_idx++;
    bytes_written += sizeof(T);
    total_bytes_written += sizeof(T);
  }
  if (it != nullptr) {
    int64 size = 0; 
    if (value_offset < 0) {
      size = sizeof(T);
    } else {
      size = sizeof(T) * dump_tensor_shape.dim_size(1);
    }
    char val[size];
    for(int i=0; i < size; ++i) {
      val[i] = 0;
    }
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      int64 dim = 0;
      void* start = nullptr;
      if (value_offset < 0) {
        if (value_offset == -1){
          it->Key(val, sizeof(T));
        } else if (value_offset == -2) {
          it->Freq(val, sizeof(T));
        } else {
          it->Version(val, sizeof(T));
        }
        if (bytes_written + sizeof(T) > bytes_limit) {
          dump_happened = true;
          writer->AppendSegmentData(dump_buffer, bytes_written);
          bytes_written = 0;
          buffer_idx = 0;
        }
        key_dump_buffer[buffer_idx] = *((T*)val);
        buffer_idx++;
        bytes_written += sizeof(T);
        total_bytes_written += sizeof(T);

      } else {
        dim = dump_tensor_shape.dim_size(1);
        it->Value(val, dim * sizeof(T), value_offset * sizeof(T));

        for (int j = 0; j < dim; ++j) {
          if (bytes_written + sizeof(T) > bytes_limit) {
            dump_happened = true;
            writer->AppendSegmentData(dump_buffer, bytes_written);
            bytes_written = 0;
            buffer_idx = 0;
          }
          key_dump_buffer[buffer_idx] = *((T*)val + j);
          buffer_idx++;
          bytes_written += sizeof(T);
          total_bytes_written += sizeof(T);
        }
      }
    }
  }
  if (!dump_happened) {
    VLOG(1) << tensor_name << " only one buffer written, size:" << bytes_written;
    writer->AddCompeleteData(dump_buffer, bytes_written);
  } else {
    VLOG(1) << tensor_name << " mutiple buffer written, size:" << total_bytes_written << ", bytes written:" << bytes_written;
    writer->AppendSegmentData(dump_buffer, bytes_written);
    writer->EndSegmentData(total_bytes_written,  bytes_written);
  }
  return Status::OK();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SAVE_RESTORE_TENSOR_H_
