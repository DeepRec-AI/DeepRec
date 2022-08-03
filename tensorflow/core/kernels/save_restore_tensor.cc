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

#include "tensorflow/core/kernels/save_restore_tensor.h"
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/util/tensor_slice_reader.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#include "tensorflow/core/util/tensor_slice_writer.h"
#include "tensorflow/core/framework/hash_table/status_collector.h"

namespace tensorflow {

void SaveTensors(
    OpKernelContext* context,
    checkpoint::TensorSliceWriter::CreateBuilderFunction builder_func,
    bool save_slices) {
  const Tensor& filename_t = context->input(0);
  {
    const int64 size = filename_t.NumElements();
    OP_REQUIRES(
        context, size == 1,
        errors::InvalidArgument(
            "Input 0 (filename) must be a string scalar; got a tensor of ",
            size, "elements"));
  }

  // Path, names, and slices if save_slices is true.
  const int kFixedInputs = save_slices ? 3 : 2;
  const Tensor& tensor_names_t = context->input(1);
  OP_REQUIRES(context,
              FastBoundsCheck(tensor_names_t.NumElements() + kFixedInputs,
                              std::numeric_limits<int>::max()),
              errors::InvalidArgument("Too many inputs to SaveTensors"));
  const int N = static_cast<int>(tensor_names_t.NumElements());
  const tstring* tensor_shapes_and_slices_ptr = nullptr;
  if (save_slices) {
    const Tensor& tensor_shapes_and_slices_t = context->input(2);
    OP_REQUIRES(
        context,
        tensor_shapes_and_slices_t.NumElements() == static_cast<int64>(N),
        errors::InvalidArgument("Expected ", N,
                                " elements for the tensor "
                                "shapes and slices but got ",
                                tensor_shapes_and_slices_t.NumElements()));
    tensor_shapes_and_slices_ptr =
        tensor_shapes_and_slices_t.flat<tstring>().data();
  }
  OP_REQUIRES(context, context->num_inputs() == N + kFixedInputs,
              errors::InvalidArgument("Expected totally ", N + kFixedInputs,
                                      " inputs as input #1 (which is a string "
                                      "tensor of saved names) contains ",
                                      N, " names, but received ",
                                      context->num_inputs(), " inputs"));

  VLOG(1) << "About to save tensors to file " << filename_t.flat<tstring>()(0)
          << "...";
  checkpoint::TensorSliceWriter writer(filename_t.flat<tstring>()(0),
                                       std::move(builder_func));

  Status s;
  auto tensor_names_flat = tensor_names_t.flat<tstring>();

  // Process tensors in sorted name order.  This allows us to avoid seeking
  // during restoration in the common case where we are restoring a full
  // checkpoint.
  std::vector<size_t> sorted_name_idx(tensor_names_flat.size());
  std::iota(sorted_name_idx.begin(), sorted_name_idx.end(), 0);
  std::sort(sorted_name_idx.begin(), sorted_name_idx.end(),
            [&tensor_names_flat](size_t a, size_t b) {
              return tensor_names_flat(a) < tensor_names_flat(b);
            });

  for (const size_t i : sorted_name_idx) {
    const string& name = tensor_names_flat(i);
    const Tensor& input = context->input(i + kFixedInputs);
    TensorShape shape(input.shape());
    TensorSlice slice(input.dims());
    if (save_slices && !tensor_shapes_and_slices_ptr[i].empty()) {
      const tstring& shape_spec = tensor_shapes_and_slices_ptr[i];
      TensorShape slice_shape;
      OP_REQUIRES_OK(context, checkpoint::ParseShapeAndSlice(
                                  shape_spec, &shape, &slice, &slice_shape));
      OP_REQUIRES(context, slice_shape.IsSameSize(input.shape()),
                  errors::InvalidArgument(
                      "Slice in shape_and_slice "
                      "specification does not match the "
                      "shape of the tensor to  save: ",
                      shape_spec, ", tensor: ", input.shape().DebugString()));
    }

#define WRITER_ADD(T)                                           \
  case DataTypeToEnum<T>::value:                                \
    s = writer.Add(name, shape, slice, input.flat<T>().data()); \
    break;

    switch (input.dtype()) {
      TF_CALL_SAVE_RESTORE_TYPES(WRITER_ADD)
      default:
        context->SetStatus(errors::Unimplemented("Saving data type ",
                                                 DataTypeString(input.dtype()),
                                                 " not yet supported"));
        return;
    }
#undef WRITER_ADD
    if (!s.ok()) {
      context->SetStatus(s);
      return;
    }
  }

  s = writer.Finish();
  if (!s.ok()) {
    context->SetStatus(s);
  }
}

void RestoreTensor(OpKernelContext* context,
                   checkpoint::TensorSliceReader::OpenTableFunction open_func,
                   int preferred_shard, bool restore_slice, int restore_index) {
  const Tensor& file_pattern_t = context->input(0);
  {
    const int64 size = file_pattern_t.NumElements();
    OP_REQUIRES(
        context, size == 1,
        errors::InvalidArgument(
            "Input 0 (file_pattern) must be a string scalar; got a tensor of ",
            size, " elements"));
  }
  const string& file_pattern = file_pattern_t.flat<tstring>()(0);

  const Tensor& tensor_name_t = context->input(1);
  {
    const int64_t size = tensor_name_t.NumElements();
    OP_REQUIRES(context, size > restore_index,
                errors::InvalidArgument(
                    "Input 1 (file_pattern) must be a have at least ",
                    restore_index + 1, " elements"));
  }
  const string& tensor_name = tensor_name_t.flat<tstring>()(restore_index);

  // If we cannot find a cached reader we will allocate our own.
  std::unique_ptr<checkpoint::TensorSliceReader> allocated_reader;

  const checkpoint::TensorSliceReader* reader = nullptr;

  if (context->slice_reader_cache()) {
    reader = context->slice_reader_cache()->GetReader(file_pattern, open_func,
                                                      preferred_shard);
  }
  if (!reader) {
    allocated_reader.reset(new checkpoint::TensorSliceReader(
        file_pattern, open_func, preferred_shard));
    reader = allocated_reader.get();
  }
  OP_REQUIRES_OK(context, CHECK_NOTNULL(reader)->status());

  // Get the shape and type from the save file.
  DataType type;
  TensorShape saved_shape;
  OP_REQUIRES(
      context, reader->HasTensor(tensor_name, &saved_shape, &type),
      errors::NotFound("Tensor name \"", tensor_name,
                       "\" not found in checkpoint files ", file_pattern));
  OP_REQUIRES(
      context, type == context->expected_output_dtype(restore_index),
      errors::InvalidArgument("Expected to restore a tensor of type ",
                              DataTypeString(context->expected_output_dtype(0)),
                              ", got a tensor of type ", DataTypeString(type),
                              " instead: tensor_name = ", tensor_name));

  // Shape of the output and slice to load.
  TensorShape output_shape(saved_shape);
  TensorSlice slice_to_load(saved_shape.dims());
  if (restore_slice) {
    const tstring& shape_spec =
        context->input(2).flat<tstring>()(restore_index);
    if (!shape_spec.empty()) {
      TensorShape parsed_shape;
      OP_REQUIRES_OK(context, checkpoint::ParseShapeAndSlice(
                                  shape_spec, &parsed_shape, &slice_to_load,
                                  &output_shape));
      OP_REQUIRES(
          context, parsed_shape.IsSameSize(saved_shape),
          errors::InvalidArgument(
              "Shape in shape_and_slice spec does not match the shape in the "
              "save file: ",
              parsed_shape.DebugString(),
              ", save file shape: ", saved_shape.DebugString()));
    }
  }

  Tensor* t = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(restore_index, output_shape, &t));

  if (output_shape.num_elements() == 0) return;

#define READER_COPY(T)                                                \
  case DataTypeToEnum<T>::value:                                      \
    OP_REQUIRES(context,                                              \
                reader->CopySliceData(tensor_name, slice_to_load,     \
                                      t->flat<T>().data()),           \
                errors::InvalidArgument("Error copying slice data")); \
    break;

  switch (type) {
    TF_CALL_SAVE_RESTORE_TYPES(READER_COPY)
    default:
      context->SetStatus(errors::Unimplemented(
          "Restoring data type ", DataTypeString(type), " not yet supported"));
  }
#undef READER_COPY
}

Status SaveBloomFilter(
    BundleWriter* writer, BloomFilterAdmitStrategy* strategy,
    const string& name, int64 slice_beg, int64 slice_length, int64 slice_size) {
  LOG(INFO) << "Save BloomFilter" << ": name=" << name
      << ", slice_beg=" << slice_beg << ", slice_length=" << slice_length
      << ", slice_size=" << slice_size;
  std::vector<int8> snapshot = strategy->Snapshot();
  TensorSlice slice(1);
  slice.set_start(0, slice_beg);
  slice.set_length(0, slice_length);
  {
    TensorSliceProto* tensor_slice_proto;
    TF_RETURN_IF_ERROR(writer->AddSliceHeader(
        name, TensorShape({0}), DataType::DT_INT8, true,
        &tensor_slice_proto));
    TensorSlice xslice(1);
    xslice.set_start(0, 0);
    xslice.set_length(0, snapshot.size());
    xslice.AsProto(tensor_slice_proto);
    tensor_slice_proto->set_hash_slice_begin(slice_beg);
    tensor_slice_proto->set_hash_slice_length(slice_length);
    tensor_slice_proto->set_hash_slice_size(slice_size);
    std::string xname = checkpoint::EncodeTensorNameSlice(
        name, slice);
    SegmentBundleWriter segment_writer(
        writer, xname, TensorShape({(int64)snapshot.size()}), DataType::DT_INT8);
    TF_RETURN_IF_ERROR(segment_writer.Begin());
    for (auto&& val : snapshot) {
      TF_RETURN_IF_ERROR(segment_writer.WriteData(&val, sizeof(int8)));
    }
    TF_RETURN_IF_ERROR(segment_writer.End());
  }
  LOG(INFO) << "Save BloomFilter Done";
  return Status::OK();
}

Status RestoreBloomFilter(
    BundleReader* reader, BloomFilterAdmitStrategy* strategy,
    const string& name, int64 slice_beg, int64 slice_length,
    int64 slice_size) {
  LOG(INFO) << "Restore BloomFilter" << ": name=" << name
      << ", slice_beg=" << slice_beg << ", slice_length=" << slice_length
      << ", slice_size=" << slice_size;
  std::vector<TensorSliceProto> slices;
  TF_RETURN_IF_ERROR(reader->LookupTensorSliceProtos(
      name, &slices));
  std::sort(slices.begin(), slices.end(),
      [](const TensorSliceProto& lhs, const TensorSliceProto& rhs) -> bool {
        return lhs.hash_slice_begin() < rhs.hash_slice_begin();
  });
  for (auto&& slice : slices) {
    if (slice.hash_slice_begin() >= slice_beg + slice_length ||
        slice_beg >= slice.hash_slice_begin() + slice.hash_slice_length()) {
      continue;
    }
    std::unique_ptr<SegmentBundleReader> bundle_reader;
    std::string xname = checkpoint::EncodeTensorNameSlice(
        name, TensorSlice(slice));
    bundle_reader.reset(new SegmentBundleReader(reader, xname, 0, -1));
    TF_RETURN_IF_ERROR(bundle_reader->Begin());
    int64 size = bundle_reader->shape().dim_size(0);
    std::vector<int8> buffer(size, 0);
    TF_RETURN_IF_ERROR(bundle_reader->Read(buffer.data(), size * sizeof(int8)));
    strategy->Restore(slice.hash_slice_begin(), slice.hash_slice_length(),
                      slice_beg, slice_length, buffer);
  }
  LOG(INFO) << "Restore BloomFilter Done";
  return Status::OK();
}

Status SaveHashTableHelper(
    BundleWriter* writer, const std::vector<std::pair<int64, int64>>& data,
    const std::vector<TensibleVariable*>& tensibles,
    const string& table_name, const std::vector<string>& tensibles_name,
    int64 slice_beg, int64 slice_length, int64 slice_size) {
  if (tensibles.size() != tensibles_name.size()) {
    return errors::InvalidArgument("save tensor name error");
  }
  if (data.size() == 0 && slice_beg != 0) {
    return Status::OK();
  }
  TensorSlice slice(1);
  slice.set_start(0, slice_beg);
  slice.set_length(0, slice_length);
  {
    TensorSliceProto* tensor_slice_proto;
    TF_RETURN_IF_ERROR(writer->AddSliceHeader(
            table_name, TensorShape({0}), DataType::DT_INT64, true,
            &tensor_slice_proto));
    TensorSlice xslice(1);
    xslice.set_start(0, 0);
    xslice.set_length(0, data.size());
    xslice.AsProto(tensor_slice_proto);
    tensor_slice_proto->set_hash_slice_begin(slice_beg);
    tensor_slice_proto->set_hash_slice_length(slice_length);
    tensor_slice_proto->set_hash_slice_size(slice_size);
    string xname = checkpoint::EncodeTensorNameSlice(
        table_name, slice);
    SegmentBundleWriter segment_writer(
        writer, xname,
        TensorShape({(int64)data.size()}), DataType::DT_INT64);
    TF_RETURN_IF_ERROR(segment_writer.Begin());
    for (auto&& item : data) {
      TF_RETURN_IF_ERROR(segment_writer.WriteData(&item.first, sizeof(int64)));
    }
    TF_RETURN_IF_ERROR(segment_writer.End());
  }
  for (size_t i = 0; i < tensibles.size(); i++) {
    TensorShape tsx = tensibles[i]->shape();
    tsx.set_dim(0, 0);
    TensorSliceProto* tensor_slice_proto;
    TF_RETURN_IF_ERROR(writer->AddSliceHeader(
            tensibles_name[i], tsx, tensibles[i]->dtype(), true,
            &tensor_slice_proto));
    TensorSlice xslice(tsx.dims());
    xslice.set_start(0, 0);
    xslice.set_length(0, data.size());
    xslice.AsProto(tensor_slice_proto);
    tensor_slice_proto->set_hash_slice_begin(slice_beg);
    tensor_slice_proto->set_hash_slice_length(slice_length);
    tensor_slice_proto->set_hash_slice_size(slice_size);
    string xname = checkpoint::EncodeTensorNameSlice(
        tensibles_name[i], slice);
    TensorShape ts = tensibles[i]->shape();
    ts.set_dim(0, data.size());
    SegmentBundleWriter segment_writer(
        writer, xname, ts, tensibles[i]->dtype());
    TF_RETURN_IF_ERROR(segment_writer.Begin());
    int64 slice_size = tensibles[i]->SliceSize();
    for (auto&& item : data) {
      TF_RETURN_IF_ERROR(segment_writer.WriteData(
              tensibles[i]->GetSlice<void>(item.second), slice_size));
    }
    TF_RETURN_IF_ERROR(segment_writer.End());
  }
  return Status::OK();
}

Status SaveHashTable(
    BundleWriter* writer, HashTable* table,
    const std::vector<TensibleVariable*>& tensibles,
    const string& table_name, const std::vector<string>& tensibles_name,
    int64 slice_beg, int64 slice_length, int64 slice_size) {
  LOG(INFO) << "Save";
  std::vector<std::pair<int64, int64>> snapshot;
  CoalescedHashTable* coalesced_table =
      dynamic_cast<CoalescedHashTable*>(table);
  if (coalesced_table == nullptr) {
    snapshot = table->Snapshot();
  } else {
    TF_RETURN_IF_ERROR(coalesced_table->ChildSnapshot(table_name, &snapshot));
  }
  TF_RETURN_IF_ERROR(SaveHashTableHelper(
      writer, snapshot, tensibles, table_name, tensibles_name,
      slice_beg, slice_length, slice_size));
  LOG(INFO) << "Save Done";
  return Status::OK();
}

namespace {
struct RestoreHashTableSlice {
  int64 slice_id;
  int64 beg, len;
};
}

Status LookupSliceInfo(
    BundleReader* reader, const string& table_name,
    const std::vector<string>& tensibles_name,
    std::vector<TensorSliceProto>* out_table_slices,
    std::vector<std::vector<TensorSliceProto>>* out_tensible_slices) {
  std::vector<TensorSliceProto> table_slices;
  TF_RETURN_IF_ERROR(reader->LookupTensorSliceProtos(
          table_name, &table_slices));

  std::sort(table_slices.begin(), table_slices.end(),
  [](const TensorSliceProto& lhs, const TensorSliceProto& rhs) -> bool {
    return lhs.hash_slice_begin() < rhs.hash_slice_begin();
  });

  std::vector<std::vector<TensorSliceProto>> tensible_slices(
      tensibles_name.size());
  for (size_t i = 0; i < tensibles_name.size(); i++) {
    Status st = reader->LookupTensorSliceProtos(
        tensibles_name[i], &tensible_slices[i]);
    if (!st.ok())  {
      LOG(ERROR) << "Slice " << tensibles_name[i] << " not found, ignored!";
      tensible_slices[i].clear();
      continue;
    }
    std::sort(tensible_slices[i].begin(), tensible_slices[i].end(),
    [](const TensorSliceProto& lhs, const TensorSliceProto& rhs) -> bool {
      return lhs.hash_slice_begin() < rhs.hash_slice_begin();
    });
  }
  for (auto&& slice : tensible_slices) {
    if (slice.empty()) {
      continue;
    }
    if (slice.size() != table_slices.size()) {
      return errors::FailedPrecondition("Checkpoint's Slice is mismatched");
    }
    for (size_t i = 0; i < table_slices.size(); ++i) {
      if (slice[i].hash_slice_begin() != table_slices[i].hash_slice_begin()) {
        return errors::FailedPrecondition("Checkpoint's Slice is mismatched");
      }
      if (slice[i].hash_slice_length() != table_slices[i].hash_slice_length()) {
        return errors::FailedPrecondition("Checkpoint's Slice is mismatched");
      }
      if (slice[i].extent(0).start() != table_slices[i].extent(0).start()) {
        return errors::FailedPrecondition("Checkpoint's Slice is mismatched");
      }
      if (slice[i].extent(0).length() != table_slices[i].extent(0).length()) {
        return errors::FailedPrecondition("Checkpoint's Slice is mismatched");
      }
    }
  }
  *out_table_slices = std::move(table_slices);
  *out_tensible_slices = std::move(tensible_slices);
  return Status::OK();
}

void BuildRestoreSlice(
    const std::vector<TensorSliceProto>& table_slices,
    int64 slice_beg, int64 slice_length,
    std::vector<RestoreHashTableSlice>* out_slices) {
  int64 kBlock = 1 << 20;
  for (size_t i = 0; i < table_slices.size(); i++) {
    if (table_slices[i].hash_slice_begin() >= slice_beg + slice_length ||
        slice_beg >= table_slices[i].hash_slice_begin() +
            table_slices[i].hash_slice_length()) {
      continue;
    }
    int64 idx = table_slices[i].extent(0).length();
    for (int64 j = 0; j < idx; j += kBlock) {
      int64 len = std::min(idx - j, kBlock);
      out_slices->emplace_back();
      out_slices->back().slice_id = i;
      out_slices->back().beg = j;
      out_slices->back().len = len;
    }
  }
}

Status InsertTable(
    BundleReader* reader, HashTable* table,
    const std::vector<TensibleVariable*>& tensibles,
    const string& table_name, const std::vector<string>& tensibles_name,
    const std::vector<TensorSliceProto>& table_slices,
    const std::vector<std::vector<TensorSliceProto>>& tensible_slices,
    const RestoreHashTableSlice& slice, mutex* mu,
    int64 slice_beg, int64 slice_length, int64 slice_size,
    const std::function<void(int64*,size_t)>& revise_fn = nullptr) {
  std::unique_ptr<SegmentBundleReader> table_bundle_reader;
  std::vector<std::unique_ptr<SegmentBundleReader>> tensible_bundle_readers;
  {
    mutex_lock lock(*mu);
    string xname = checkpoint::EncodeTensorNameSlice(
        table_name, TensorSlice(table_slices[slice.slice_id]));
    table_bundle_reader.reset(new SegmentBundleReader(
            reader, xname, slice.beg, slice.len));
    TF_RETURN_IF_ERROR(table_bundle_reader->Begin());

    for (size_t j = 0; j < tensibles.size(); j++) {
      if (tensible_slices[j].empty()) {
        tensible_bundle_readers.emplace_back(nullptr);
        continue;
      }
      string xname = checkpoint::EncodeTensorNameSlice(
          tensibles_name[j], TensorSlice(tensible_slices[j][slice.slice_id]));
      tensible_bundle_readers.emplace_back(new SegmentBundleReader(
              reader, xname, slice.beg, slice.len));
      TF_RETURN_IF_ERROR(tensible_bundle_readers.back()->Begin());
    }
  }
  int64 id_size = slice.len;
  constexpr int64 kSlice = 1 << 15;
  int64 slice_end = slice_beg + slice_length;
  while (id_size > 0) {
    int64 xid = std::min(id_size, kSlice);
    id_size -= xid;
    int64 key[kSlice];
    TF_RETURN_IF_ERROR(table_bundle_reader->Read(key, xid * sizeof(int64)));
    if (revise_fn) {
      revise_fn(key, xid);
    }
    std::vector<int64> real_key, real_id, real_offset;
    for (int j = 0; j < xid; j++) {
      int64 slice_idx = (uint64_t)(key[j]) % (uint64_t)(slice_size);
      if (slice_beg <= slice_idx && slice_idx < slice_end) {
        real_key.push_back(key[j]);
        real_offset.push_back(j);
      }
    }
    real_id.resize(real_key.size());
    int64 size = table->GetIdsWithoutResize(
        &real_key[0], &real_id[0], real_key.size());
    for (size_t j = 0; j < tensibles.size(); j++) {
      tensibles[j]->ZeroCostResize(size);
    }
    real_offset.push_back(-1);
    int64 idx = 0;
    for (int64 j = 0; j < xid; j++) {
      if (real_offset[idx] == j) {
        for (size_t k = 0; k < tensibles.size(); k++) {
          if (tensible_bundle_readers[k]) {
            TF_RETURN_IF_ERROR(tensible_bundle_readers[k]->Read(
                    tensibles[k]->GetSlice<void>(real_id[idx]),
                    tensibles[k]->SliceSize()));
          }
        }
        idx++;
      } else {
        for (size_t k = 0; k < tensibles.size(); k++) {
          if (tensible_bundle_readers[k]) {
            TF_RETURN_IF_ERROR(tensible_bundle_readers[k]->Skip(
                    tensibles[k]->SliceSize()));
          }
        }
      }
    }
  }
  return Status::OK();
}

void RestoreHashTable(
    std::function<void(std::function<void()>)> runner,
    BundleReader* reader, HashTable* table,
    const std::vector<TensibleVariable*>& tensibles,
    const string& table_name, const std::vector<string>& tensibles_name,
    int64 slice_beg, int64 slice_length, int64 slice_size,
    std::function<void(Status)> done) {
  LOG(INFO) << "Restore";
  table->Clear([=](Status st) {
    if (!st.ok()) {
      done(st);
      return;
    }
    std::vector<TensorSliceProto> table_slices;
    std::vector<std::vector<TensorSliceProto>> tensible_slices(tensibles.size());
    if (tensibles.size() != tensibles_name.size()) {
      done(errors::FailedPrecondition("Tensible Size is not equal to names"));
      return;
    }

    Status res = LookupSliceInfo(
        reader, table_name, tensibles_name, &table_slices, &tensible_slices);
    if (!res.ok()) {
      done(res);
      return;
    }

    std::vector<RestoreHashTableSlice> slices;
    BuildRestoreSlice(table_slices, slice_beg, slice_length, &slices);

    mutex* mu = new mutex;
    auto insert_table = std::bind(
        InsertTable, reader, table, tensibles, table_name,
        tensibles_name, table_slices, tensible_slices, std::placeholders::_1,
        mu, slice_beg, slice_length, slice_size, nullptr);

    auto after_add_table = [=](Status st) {
      delete mu;
      if (!st.ok()) {
        done(st);
        return;
      };
      std::set<TensibleVariable*> restored_variable(
          tensibles.begin(), tensibles.end());
      std::vector<TensibleVariable*> vars = table->Tensibles();
      StatusCollector* stc = new StatusCollector(vars.size(), done);
      for (auto&& tensible : vars) {
        if (restored_variable.find(tensible) != restored_variable.end()) {
          tensible->Pad(table->Size(), stc->AddStatusFunc());
        } else {
          tensible->Resize(table->Size(), stc->AddStatusFunc());
        }
      }
      LOG(INFO) << "Restore Done";
      stc->Start();
    };
    StatusCollector* stc = new StatusCollector(slices.size(), after_add_table);
    for (auto s : slices) {
      runner([=]{
        Status st = insert_table(s);
        stc->AddStatus(st);
      });
    }
    stc->Start();
  });
}

std::vector<string> ChildrenInCheckpoints(
    BundleReader* reader, const std::vector<string>& table_names) {
  std::vector<string> result;
  for (auto&& table_name : table_names) {
    std::vector<TensorSliceProto> slices;
    Status st = reader->LookupTensorSliceProtos(table_name, &slices);
    if (st.ok()) {
      result.push_back(table_name);
    }
  }
  return result;
}

void RestoreCoalescedHashTable(
    std::function<void(std::function<void()>)> runner,
    BundleReader* reader, CoalescedHashTable* table,
    const std::vector<TensibleVariable*>& tensibles,
    const std::vector<string>& table_names,
    const std::vector<std::vector<string>>& tensibles_names,
    int64 slice_beg, int64 slice_length, int64 slice_size,
    bool clear, std::function<void(Status)> done) {
  LOG(INFO) << "Restore CoalescedHashTable";
  auto after_clear = [=](Status st) {
    if (!st.ok()) {
      done(st);
      return;
    }
    if (table_names.size() != tensibles_names.size()) {
      done(errors::FailedPrecondition("Tensible Size is not equal to names"));
      return;
    }
    std::vector<std::function<Status(void)>> insert_table_fns;
    mutex* mu = new mutex;
    for (size_t i = 0; i < table_names.size(); ++i) {
      string table_name = table_names[i];
      std::vector<string> tensibles_name = tensibles_names[i];
      Status res = table->ValidChild(table_name);
      if (!res.ok()) {
        done(res);
        return;
      }
      std::vector<TensorSliceProto> table_slices;
      std::vector<std::vector<TensorSliceProto>> tensible_slices;
      res = LookupSliceInfo(
          reader, table_name, tensibles_name, &table_slices,
          &tensible_slices);
      // ignore non-existed child table
      if (!res.ok()) {
        LOG(ERROR) << res.ToString();
        continue;
      }
      std::vector<RestoreHashTableSlice> slices;
      BuildRestoreSlice(table_slices, slice_beg, slice_length, &slices);
      for (auto&& slice : slices) {
        insert_table_fns.push_back(std::bind(
                InsertTable, reader, table, tensibles, table_name, 
                tensibles_name, table_slices, tensible_slices,
                slice, mu, slice_beg, slice_length, slice_size,
                table->MakeReviserFn(table_name)));
      }
    }
    auto after_add_table = [=](Status st) {
      delete mu;
      if (!st.ok()) {
        done(st);
        return;
      }
      std::set<TensibleVariable*> restored_variable(
          tensibles.begin(), tensibles.end());
      std::vector<TensibleVariable*> vars = table->Tensibles();
      StatusCollector* stc = new StatusCollector(vars.size(), done);
      for (auto&& tensible: vars) {
        if (restored_variable.find(tensible) != restored_variable.end()) {
          tensible->Pad(table->Size(), stc->AddStatusFunc());
        } else {
          tensible->Resize(table->Size(), stc->AddStatusFunc());
        }
      }
      LOG(INFO) << "Restore CoalescedHashTable Done";
      stc->Start();
    };
    StatusCollector* stc = new StatusCollector(
        insert_table_fns.size(), after_add_table);
    for (auto&& fn : insert_table_fns) {
      runner([=] {
        Status st = fn();
        stc->AddStatus(st);
      });
    }
    stc->Start();
  };
  if (clear) {
    table->Clear(after_clear);
  } else {
    std::vector<string> children_names = ChildrenInCheckpoints(
        reader, table_names);
    table->ClearChildren(children_names, after_clear);
  }
}

namespace {

// Tensors larger than this threshold will be restored from a thread-pool.
const int64 kLargeShapeThreshold = 16 << 20;  // 16M

// A restore operation for a single tensor.  Small tensors may be restored
// directly from the op thread to improve read locality.  Large tensors can be
// restored from a thread pool: this requires creating a separate BundleReader
// for each restore.
struct RestoreOp {
  RestoreOp& operator=(const RestoreOp&) = delete;

  bool should_run_in_pool(BundleReader* reader) const {
    TensorShape restored_full_shape;

    // Ignore status here; we'll catch the error later.
    if (!reader->LookupTensorShape(tensor_name, &restored_full_shape).ok()) {
      return false;
    }

    return restored_full_shape.num_elements() > kLargeShapeThreshold;
  }

  // Run this restore operation using a new BundleReader.
  void run_with_new_reader() {
    BundleReader reader(Env::Default(), reader_prefix);
    if (!reader.status().ok()) {
      status = reader.status();
      return;
    }

    status = run(&reader);
  }

  Status run(BundleReader* reader) {
    TensorShape restored_full_shape;
    TF_RETURN_IF_ERROR(
        reader->LookupTensorShape(tensor_name, &restored_full_shape));

    VLOG(1) << "Restoring tensor " << idx << " : " << tensor_name << " : "
            << restored_full_shape.num_elements();
    Tensor* restored_tensor;
    if (shape_and_slice.empty()) {
      // Lookup the full tensor.
      TF_RETURN_IF_ERROR(
          context->allocate_output(idx, restored_full_shape, &restored_tensor));
      TF_RETURN_IF_ERROR(reader->Lookup(tensor_name, restored_tensor));
    } else {
      // Lookup the slice.
      TensorShape parsed_full_shape;
      TensorSlice parsed_slice;
      TensorShape parsed_slice_shape;

      TF_RETURN_IF_ERROR(
          checkpoint::ParseShapeAndSlice(shape_and_slice, &parsed_full_shape,
                                         &parsed_slice, &parsed_slice_shape));

      if (!restored_full_shape.IsSameSize(parsed_full_shape)) {
        return errors::InvalidArgument(
            "tensor_name = ", tensor_name, "; shape in shape_and_slice spec ",
            parsed_full_shape.DebugString(),
            " does not match the shape stored in checkpoint: ",
            restored_full_shape.DebugString());
      }
      TF_RETURN_IF_ERROR(
          context->allocate_output(idx, parsed_slice_shape, &restored_tensor));
      TF_RETURN_IF_ERROR(
          reader->LookupSlice(tensor_name, parsed_slice, restored_tensor));
    }
    return Status::OK();
  }

  OpKernelContext* context;
  size_t idx;
  string tensor_name;
  string shape_and_slice;
  string reader_prefix;

  ::tensorflow::Status status;
};

}  // namespace

Status RestoreTensorsV2(OpKernelContext* context, const Tensor& prefix,
                        const Tensor& tensor_names,
                        const Tensor& shape_and_slices,
                        gtl::ArraySlice<DataType> dtypes) {
  const string& prefix_string = prefix.scalar<tstring>()();

  const auto& tensor_names_flat = tensor_names.flat<tstring>();
  const auto& shape_and_slices_flat = shape_and_slices.flat<tstring>();

  // Sort lookup keys to improve locality when reading multiple tensors.
  std::vector<size_t> sorted_name_idx(tensor_names_flat.size());
  std::iota(sorted_name_idx.begin(), sorted_name_idx.end(), 0);
  std::sort(sorted_name_idx.begin(), sorted_name_idx.end(),
            [&tensor_names_flat](size_t a, size_t b) {
              return tensor_names_flat(a) < tensor_names_flat(b);
            });

  std::vector<std::unique_ptr<RestoreOp> > pool_restore_ops;
  std::vector<std::unique_ptr<RestoreOp> > direct_restore_ops;

  BundleReader default_reader(Env::Default(), prefix_string);
  TF_RETURN_IF_ERROR(default_reader.status());

  std::vector<string> mismatched_errors;
  for (const size_t i : sorted_name_idx) {
    TensorShape restored_full_shape;
    DataType original_dtype;
    const string& tensor_name = tensor_names_flat(i);
    TF_RETURN_IF_ERROR(default_reader.LookupDtypeAndShape(
        tensor_name, &original_dtype, &restored_full_shape));
    if (dtypes[i] != original_dtype) {
      string error_msg = strings::StrCat(
          "tensor_name = ", tensor_name, "; expected dtype ",
          DataTypeString(dtypes[i]), " does not equal original dtype ",
          DataTypeString(original_dtype));
      mismatched_errors.emplace_back(error_msg);
    }
  }
  if (!mismatched_errors.empty()) {
    const string error_msg = absl::StrJoin(mismatched_errors, "\n");
    return errors::InvalidArgument(error_msg);
  }

  for (auto i : sorted_name_idx) {
    const string& tensor_name = tensor_names_flat(i);
    const string& shape_and_slice = shape_and_slices_flat(i);
    auto op =
        new RestoreOp{context, i, tensor_name, shape_and_slice, prefix_string};
    if (op->should_run_in_pool(&default_reader)) {
      pool_restore_ops.emplace_back(op);
    } else {
      direct_restore_ops.emplace_back(op);
    }
  }

  {
    // Schedule any threaded operations first, skipping thread pool creation if
    // we don't have any expensive operations.
    std::unique_ptr<thread::ThreadPool> reader_pool;
    if (!pool_restore_ops.empty()) {
      reader_pool.reset(
          new thread::ThreadPool(Env::Default(), "restore_tensors", 8));
      for (auto& op : pool_restore_ops) {
        reader_pool->Schedule([&op]() { op->run_with_new_reader(); });
      }
    }

    // Read small tensors from the op thread
    for (auto& op : direct_restore_ops) {
      TF_RETURN_IF_ERROR(op->run(&default_reader));
    }
  }

  // Check status of pool ops; this must come after the pool shuts down.
  for (auto& op : pool_restore_ops) {
    TF_RETURN_IF_ERROR(op->status);
  }

  for (auto i : sorted_name_idx) {
    const string& tensor_name = tensor_names_flat(i);
    if (dtypes[i] != context->mutable_output(i)->dtype()) {
      return errors::InvalidArgument(
          "tensor_name = ", tensor_name, "; expected dtype ",
          DataTypeString(dtypes[i]), " does not equal restored dtype ",
          DataTypeString(context->mutable_output(i)->dtype()));
    }
  }

  return Status::OK();
}

}  // namespace tensorflow
