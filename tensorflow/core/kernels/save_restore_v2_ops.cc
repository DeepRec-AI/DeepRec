/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/io_ops.cc.

#include <string>
#include <vector>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/kv_variable_ops.h"
#include "tensorflow/core/kernels/save_restore_tensor.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/core/util/tensor_slice_reader.h"
#include "tensorflow/core/framework/hash_table/tensible_variable.h"
#include "tensorflow/core/framework/hash_table/hash_table.h"

namespace tensorflow {

namespace {

template<typename T>
bool IsHandle(const ResourceHandle& handle) {
  return handle.hash_code() == MakeTypeIndex<T>().hash_code();
}

// Shared validations of the inputs to the SaveV2 and RestoreV2 ops.
void ValidateInputs(bool is_save_op, OpKernelContext* context,
                    const Tensor& prefix, const Tensor& tensor_names,
                    const Tensor& shape_and_slices,
                    const int kFixedInputs) {
  const int num_tensors = static_cast<int>(tensor_names.NumElements());
  OP_REQUIRES(
      context, prefix.NumElements() == 1,
      errors::InvalidArgument("Input prefix should have a single element, got ",
                              prefix.NumElements(), " instead."));
  OP_REQUIRES(context,
              TensorShapeUtils::IsVector(tensor_names.shape()) &&
                  TensorShapeUtils::IsVector(shape_and_slices.shape()),
              errors::InvalidArgument(
                  "Input tensor_names and shape_and_slices "
                  "should be an 1-D tensors, got ",
                  tensor_names.shape().DebugString(), " and ",
                  shape_and_slices.shape().DebugString(), " instead."));
  OP_REQUIRES(context,
              tensor_names.NumElements() == shape_and_slices.NumElements(),
              errors::InvalidArgument("tensor_names and shape_and_slices "
                                      "have different number of elements: ",
                                      tensor_names.NumElements(), " vs. ",
                                      shape_and_slices.NumElements()));
  OP_REQUIRES(context,
              FastBoundsCheck(tensor_names.NumElements() + kFixedInputs,
                              std::numeric_limits<int>::max()),
              errors::InvalidArgument("Too many inputs to the op"));
  OP_REQUIRES(
      context, shape_and_slices.NumElements() == num_tensors,
      errors::InvalidArgument("Expected ", num_tensors,
                              " elements in shapes_and_slices, but got ",
                              context->input(2).NumElements()));
  if (is_save_op) {
    OP_REQUIRES(context, context->num_inputs() == num_tensors + kFixedInputs,
                errors::InvalidArgument(
                    "Got ", num_tensors, " tensor names but ",
                    context->num_inputs() - kFixedInputs, " tensors."));
    OP_REQUIRES(context, context->num_inputs() == num_tensors + kFixedInputs,
                errors::InvalidArgument(
                    "Expected a total of ", num_tensors + kFixedInputs,
                    " inputs as input #1 (which is a string "
                    "tensor of saved names) contains ",
                    num_tensors, " names, but received ", context->num_inputs(),
                    " inputs"));
  }
}

}  // namespace

// Saves a list of named tensors using the tensor bundle library.
class SaveV2 : public OpKernel {
 public:
  explicit SaveV2(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtypes", &tensor_types_));
    OP_REQUIRES_OK(context, context->GetAttr("ev_key_types", &ev_key_types_));
    OP_REQUIRES_OK(context, context->GetAttr("has_ev", &has_ev_));
  }

  template <typename TKey, typename TValue>
  void DumpEvWithGlobalStep(OpKernelContext* context, int variable_index,
      const string& tensor_name, BundleWriter& writer,
      DataType global_step_type) {
    if (global_step_type == DT_INT32) {
      DumpEv<TKey, TValue, int32>(context, variable_index,
          tensor_name, writer);
    } else {
      DumpEv<TKey, TValue, int64>(context, variable_index,
          tensor_name, writer);
    }
  }

  template <typename TKey, typename TValue, typename TGlobalStep>
  void DumpEv(OpKernelContext* context, int variable_index,
      const string& tensor_name, BundleWriter& writer) {
    EmbeddingVar<TKey, TValue>* variable = nullptr;
    OP_REQUIRES_OK(context,
                   LookupResource(context,
                     HandleFromInput(context, variable_index), &variable));
    const Tensor& global_step = context->input(3);
    Tensor part_offset_tensor;
    context->allocate_temp(DT_INT32,
                           TensorShape({kSavedPartitionNum + 1}),
                           &part_offset_tensor);
    TGlobalStep global_step_scalar = global_step.scalar<TGlobalStep>()();
    core::ScopedUnref s(variable);
    embedding::ShrinkArgs shrink_args;
    shrink_args.global_step = global_step_scalar;
    OP_REQUIRES_OK(context, variable->Shrink(shrink_args));
    const Tensor& prefix = context->input(0);
    const string& prefix_string = prefix.scalar<tstring>()();
    OP_REQUIRES_OK(context, DumpEmbeddingValues(variable, tensor_name,
        &writer, &part_offset_tensor, prefix_string));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& prefix = context->input(0);
    const Tensor& tensor_names = context->input(1);
    const Tensor& shape_and_slices = context->input(2);
    const int kFixedInputs = 3;  // Prefix, tensor names, shape_and_slices.
    ValidateInputs(true /* is save op */, context, prefix, tensor_names,
                   shape_and_slices, kFixedInputs);
    if (!context->status().ok()) return;

    const int num_tensors = static_cast<int>(tensor_names.NumElements());
    const string& prefix_string = prefix.scalar<tstring>()();
    const auto& tensor_names_flat = tensor_names.flat<tstring>();
    const auto& shape_and_slices_flat = shape_and_slices.flat<tstring>();

    BundleWriter writer(Env::Default(), prefix_string);
    OP_REQUIRES_OK(context, writer.status());
    VLOG(1) << "BundleWriter, prefix_string: " << prefix_string;

    int start_index = 0;
    if (has_ev_) {
      start_index = 1;
    }

    int start_ev_key_index = 0;

    for (int i = start_index; i < num_tensors; ++i) {
      const string& tensor_name = tensor_names_flat(i);
      if (tensor_types_[i] == DT_RESOURCE) {
        auto& handle = HandleFromInput(context, i + kFixedInputs);
        if (IsHandle<EmbeddingVar<int64, float>>(handle)) {
          if (ev_key_types_[start_ev_key_index] == DT_INT32) {
            DumpEvWithGlobalStep<int32, float>(context,
                i + kFixedInputs, tensor_name, writer, tensor_types_[0]);
          } else if (ev_key_types_[start_ev_key_index] == DT_INT64) {
            DumpEvWithGlobalStep<int64, float>(context,
                i + kFixedInputs, tensor_name, writer, tensor_types_[0]);
          }
        } else if (IsHandle<HashTableResource>(handle)) {
          auto handles = context->input(i + kFixedInputs).flat<ResourceHandle>();
          int tensible_size = handles.size() - 1;
          std::vector<core::ScopedUnref> unrefs;
          HashTable* hashtable;
          std::vector<TensibleVariable*> tensibles;

          HashTableResource* htr;
          OP_REQUIRES_OK(context,
              LookupResource(context, handles(0), &htr));
          unrefs.emplace_back(htr);
          hashtable = htr->Internal();

          for (int j = 0; j < tensible_size; j++) {
            TensibleVariableResource* tvr;
            OP_REQUIRES_OK(context,
                LookupResource(context, handles(j + 1), &tvr));
            unrefs.emplace_back(tvr);
            tensibles.push_back(tvr->Internal());
          }

          string shape_spec = shape_and_slices_flat(i);
          TensorShape shape;
          TensorSlice slice(1);
          TensorShape slice_shape;

          OP_REQUIRES_OK(context, checkpoint::ParseShapeAndSlice(
              shape_spec, &shape, &slice, &slice_shape));

          std::vector<string> names_lst = str_util::Split(tensor_name, '|');
          for (auto&& name : names_lst) {
            std::vector<string> tensor_name_x =
                str_util::Split(name, ';');
            OP_REQUIRES(context, tensor_name_x.size() == tensible_size + 1,
                errors::InvalidArgument("save tensor name error", tensor_name));
            string table_name = tensor_name_x[0];
            std::vector<string> tensible_name(
                tensor_name_x.begin() + 1, tensor_name_x.end());
            OP_REQUIRES_OK(context, SaveHashTable(
                  &writer, hashtable, tensibles, table_name, tensible_name,
                  slice.start(0), slice.length(0), slice_shape.dim_size(0)));
          }
        } else if (IsHandle<HashTableAdmitStrategyResource>(handle)) {
          HashTableAdmitStrategyResource* resource;
          OP_REQUIRES_OK(context,
              LookupResource(context,
                HandleFromInput(context, i + kFixedInputs), &resource));
          HashTableAdmitStrategy* strategy = resource->Internal();
          BloomFilterAdmitStrategy* bf =
            dynamic_cast<BloomFilterAdmitStrategy*>(strategy);
          CHECK(bf != nullptr) << "Cannot save Non-BloomFilterAdmitStrategy!";

          string shape_spec = shape_and_slices_flat(i);
          TensorShape shape;
          TensorSlice slice(1);
          TensorShape slice_shape;
          OP_REQUIRES_OK(context, checkpoint::ParseShapeAndSlice(
              shape_spec, &shape, &slice, &slice_shape));

          OP_REQUIRES_OK(context, SaveBloomFilter(
              &writer, bf, tensor_name, slice.start(0),
              slice.length(0), slice_shape.dim_size(0)));
        }
        start_ev_key_index++;
      } else {
        const Tensor& tensor = context->input(i + kFixedInputs);

        if (!shape_and_slices_flat(i).empty()) {
          const string& shape_spec = shape_and_slices_flat(i);
          TensorShape shape;
          TensorSlice slice(tensor.dims());
          TensorShape slice_shape;

          OP_REQUIRES_OK(context, checkpoint::ParseShapeAndSlice(
                             shape_spec, &shape, &slice, &slice_shape));
          OP_REQUIRES(context, slice_shape.IsSameSize(tensor.shape()),
              errors::InvalidArgument("Slice in shape_and_slice "
                                      "specification does not match the "
                                      "shape of the tensor to  save: ",
                                      shape_spec, ", tensor: ",
                                      tensor.shape().DebugString()));

          OP_REQUIRES_OK(context,
                         writer.AddSlice(tensor_name, shape, slice, tensor));
        } else {
          OP_REQUIRES_OK(context, writer.Add(tensor_name, tensor));
        }
      }
    }
    OP_REQUIRES_OK(context, writer.Finish());
  }
 private:
  DataTypeVector tensor_types_;
  DataTypeVector ev_key_types_;
  bool has_ev_;
};
REGISTER_KERNEL_BUILDER(Name("SaveV2").Device(DEVICE_CPU), SaveV2);

class SaveV3 : public OpKernel {
 public:
  explicit SaveV3(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtypes", &tensor_types_));
    OP_REQUIRES_OK(context, context->GetAttr("ev_key_types", &ev_key_types_));
    OP_REQUIRES_OK(context, context->GetAttr("has_ev", &has_ev_));
  }

  template <typename TKey, typename TValue>
  void DumpEvWithGlobalStep(
      OpKernelContext* context,
      const string& tensor_name,
      EmbeddingVar<TKey, TValue>* ev,
      BundleWriter& writer,
      DataType global_step_type) {
    if (global_step_type == DT_INT32) {
      DumpEv<TKey, TValue, int32>(context, ev, tensor_name, writer);
    } else {
      DumpEv<TKey, TValue, int64>(context, ev, tensor_name, writer);
    }
  }

  template <typename TKey, typename TValue, typename TGlobalStep>
  void DumpEv(
      OpKernelContext* context,
      EmbeddingVar<TKey, TValue>* variable,
      const string& tensor_name, BundleWriter& writer) {
    const Tensor& global_step = context->input(5);
    Tensor part_offset_tensor;
    context->allocate_temp(DT_INT32,
                           TensorShape({kSavedPartitionNum + 1}),
                           &part_offset_tensor);
    TGlobalStep global_step_scalar = global_step.scalar<TGlobalStep>()();
    core::ScopedUnref s(variable);
    embedding::ShrinkArgs shrink_args;
    shrink_args.global_step = global_step_scalar;
    OP_REQUIRES_OK(context, variable->Shrink(shrink_args));
    const Tensor& prefix = context->input(0);
    const string& prefix_string = prefix.scalar<tstring>()();
    OP_REQUIRES_OK(context, DumpEmbeddingValues(variable, tensor_name,
          &writer, &part_offset_tensor, prefix_string));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& prefix = context->input(0);
    const Tensor& tensor_names = context->input(1);
    const Tensor& shape_and_slices = context->input(2);
    const Tensor& ev_names = context->input(3);
    const Tensor& ev_resources = context->input(4);
    const int kFixedInputs = 5;
    ValidateInputs(true /* is save op */, context, prefix, tensor_names,
                   shape_and_slices, kFixedInputs);
    if (!context->status().ok()) return;
    // Prefix, tensor names, shape_and_slices, ev names, ev resources.
    const int num_tensors = static_cast<int>(tensor_names.NumElements());
    const int num_ev = static_cast<int>(ev_names.NumElements());
    const string& prefix_string = prefix.scalar<tstring>()();
    const auto& tensor_names_flat = tensor_names.flat<tstring>();
    const auto& ev_names_flat = ev_names.flat<tstring>();
    const auto& ev_resources_flat = ev_resources.flat<int64>();
    const auto& shape_and_slices_flat = shape_and_slices.flat<tstring>();

    BundleWriter writer(Env::Default(), prefix_string);
    OP_REQUIRES_OK(context, writer.status());
    VLOG(1) << "BundleWriter, prefix_string: " << prefix_string;

    int start_index = 0;
    if (has_ev_) {
      start_index = 1;
    }

    for (int i = 0; i < num_ev; i++) {
      const string& ev_name = ev_names_flat(i);
      if (ev_key_types_[i] == DT_INT32) {
        EmbeddingVar<int32, float>* ev =
            reinterpret_cast<
                EmbeddingVar<int32, float>*>(ev_resources_flat(i));
        DumpEvWithGlobalStep(
            context, ev_name, ev, writer, tensor_types_[0]);
      } else if (ev_key_types_[i] == DT_INT64) {
        EmbeddingVar<int64, float>* ev =
            reinterpret_cast<
                EmbeddingVar<int64, float>*>(ev_resources_flat(i));
        DumpEvWithGlobalStep(
            context, ev_name, ev, writer, tensor_types_[0]);
      }
    }

    for (int i = start_index; i < num_tensors; ++i) {
      const string& tensor_name = tensor_names_flat(i);
      if (tensor_types_[i] == DT_RESOURCE) {
        auto& handle = HandleFromInput(context, i + kFixedInputs);
        if (IsHandle<HashTableResource>(handle)) {
          auto handles =
              context->input(i + kFixedInputs).flat<ResourceHandle>();
          int tensible_size = handles.size() - 1;
          std::vector<core::ScopedUnref> unrefs;
          HashTable* hashtable;
          std::vector<TensibleVariable*> tensibles;

          HashTableResource* htr;
          OP_REQUIRES_OK(context,
              LookupResource(context, handles(0), &htr));
          unrefs.emplace_back(htr);
          hashtable = htr->Internal();

          for (int j = 0; j < tensible_size; j++) {
            TensibleVariableResource* tvr;
            OP_REQUIRES_OK(context,
                LookupResource(context, handles(j + 1), &tvr));
            unrefs.emplace_back(tvr);
            tensibles.push_back(tvr->Internal());
          }

          string shape_spec = shape_and_slices_flat(i);
          TensorShape shape;
          TensorSlice slice(1);
          TensorShape slice_shape;

          OP_REQUIRES_OK(context, checkpoint::ParseShapeAndSlice(
              shape_spec, &shape, &slice, &slice_shape));

          std::vector<string> names_lst = str_util::Split(tensor_name, '|');
          for (auto&& name : names_lst) {
            std::vector<string> tensor_name_x =
                str_util::Split(name, ';');
            OP_REQUIRES(context, tensor_name_x.size() == tensible_size + 1,
                errors::InvalidArgument("save tensor name error", tensor_name));
            string table_name = tensor_name_x[0];
            std::vector<string> tensible_name(
                tensor_name_x.begin() + 1, tensor_name_x.end());
            OP_REQUIRES_OK(context, SaveHashTable(
                  &writer, hashtable, tensibles, table_name, tensible_name,
                  slice.start(0), slice.length(0), slice_shape.dim_size(0)));
          }
        } else if (IsHandle<HashTableAdmitStrategyResource>(handle)) {
          HashTableAdmitStrategyResource* resource;
          OP_REQUIRES_OK(context,
              LookupResource(context,
                HandleFromInput(context, i + kFixedInputs), &resource));
          HashTableAdmitStrategy* strategy = resource->Internal();
          BloomFilterAdmitStrategy* bf =
            dynamic_cast<BloomFilterAdmitStrategy*>(strategy);
          CHECK(bf != nullptr) << "Cannot save Non-BloomFilterAdmitStrategy!";

          string shape_spec = shape_and_slices_flat(i);
          TensorShape shape;
          TensorSlice slice(1);
          TensorShape slice_shape;
          OP_REQUIRES_OK(context, checkpoint::ParseShapeAndSlice(
              shape_spec, &shape, &slice, &slice_shape));

          OP_REQUIRES_OK(context, SaveBloomFilter(
              &writer, bf, tensor_name, slice.start(0),
              slice.length(0), slice_shape.dim_size(0)));
        }
      } else {
        const Tensor& tensor = context->input(i + kFixedInputs);

        if (!shape_and_slices_flat(i).empty()) {
          const string& shape_spec = shape_and_slices_flat(i);
          TensorShape shape;
          TensorSlice slice(tensor.dims());
          TensorShape slice_shape;

          OP_REQUIRES_OK(context, checkpoint::ParseShapeAndSlice(
                             shape_spec, &shape, &slice, &slice_shape));
          OP_REQUIRES(context, slice_shape.IsSameSize(tensor.shape()),
              errors::InvalidArgument("Slice in shape_and_slice "
                                      "specification does not match the "
                                      "shape of the tensor to  save: ",
                                      shape_spec, ", tensor: ",
                                      tensor.shape().DebugString()));

          OP_REQUIRES_OK(context,
                         writer.AddSlice(tensor_name, shape, slice, tensor));
        } else {
          OP_REQUIRES_OK(context, writer.Add(tensor_name, tensor));
        }
      }
    }
    OP_REQUIRES_OK(context, writer.Finish());
  }
 private:
  DataTypeVector tensor_types_;
  DataTypeVector ev_key_types_;
  bool has_ev_;
};
REGISTER_KERNEL_BUILDER(Name("SaveV3").Device(DEVICE_CPU), SaveV3);

// Restores a list of named tensors from a tensor bundle (V2 checkpoint format).
class RestoreHashTableOp : public AsyncOpKernel {
 public:
  explicit RestoreHashTableOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("clear", &clear_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& prefix = context->input(0);
    const Tensor& tensor_names = context->input(1);
    const Tensor& shape_and_slices = context->input(2);
    const Tensor& handles = context->input(3);
    const string& prefix_string = prefix.scalar<string>()();
    const string& shape_and_slices_string =
      shape_and_slices.scalar<string>()();
    auto tensor_names_flat = tensor_names.flat<string>();
    auto handles_flat = handles.flat<ResourceHandle>();

    std::unique_ptr<BundleReader> reader(
        new BundleReader(Env::Default(), prefix_string));
    OP_REQUIRES_OK_ASYNC(context, reader->status(), done);

    TensorShape shape;
    TensorSlice slice(1);
    TensorShape slice_shape;

    OP_REQUIRES_OK_ASYNC(context, checkpoint::ParseShapeAndSlice(
        shape_and_slices_string, &shape, &slice, &slice_shape), done);

    std::unique_ptr<std::vector<core::ScopedUnref>> unrefs(
        new std::vector<core::ScopedUnref>);
    HashTable* hashtable;
    std::vector<TensibleVariable*> tensibles;

    HashTableResource* htr;
    {
      OP_REQUIRES_OK_ASYNC(context,
          LookupResource(context, handles_flat(0), &htr), done);
      unrefs->emplace_back(htr);
      hashtable = htr->Internal();
    }

    std::vector<TensibleVariableResource*> tvrs;
    for (int i = 1; i < handles_flat.size(); i++) {
      TensibleVariableResource* tvr;
      OP_REQUIRES_OK_ASYNC(context,
          LookupResource(context, handles_flat(i), &tvr), done);
      tvrs.emplace_back(tvr);
      unrefs->emplace_back(tvr);
      tensibles.push_back(tvr->Internal());
    }

    std::vector<core::ScopedUnref>* unrefs_x = unrefs.release();
    BundleReader* reader_x = reader.release();
    auto done_cb = [htr, tvrs, unrefs_x, reader_x, context, done](Status st){
      htr->SetInitialized(true);
      for (auto&& item : tvrs) {
        item->SetInitialized(true);
      }
      delete unrefs_x;
      delete reader_x;
      OP_REQUIRES_OK_ASYNC(context, st, done);
      done();
    };

    std::vector<string> table_names;
    std::vector<std::vector<string>> tensibles_names;
    for (size_t i = 0; i < tensor_names_flat.size(); ++i) {
      std::vector<string> tensor_names_x = str_util::Split(
          tensor_names_flat(i), ';');
      OP_REQUIRES_ASYNC(
          context, tensor_names_x.size() == handles_flat.size(),
          errors::InvalidArgument("tensor_name size should be same to handle"),
          done);
      table_names.push_back(tensor_names_x[0]);
      tensibles_names.emplace_back(tensor_names_x.begin() + 1,
                                   tensor_names_x.end());
    }

    CoalescedHashTable* coalesced_table =
        dynamic_cast<CoalescedHashTable*>(hashtable);
    if (coalesced_table == nullptr) {
      OP_REQUIRES_ASYNC(
          context, 1 == table_names.size(),
          errors::InvalidArgument("raw hash table should be 1"),
          done);
      RestoreHashTable(*context->runner(), reader_x, hashtable, tensibles,
                       table_names[0], tensibles_names[0], slice.start(0),
                       slice.length(0), shape.dim_size(0), done_cb);
    } else {
      RestoreCoalescedHashTable(*context->runner(), reader_x, coalesced_table,
                                tensibles, table_names, tensibles_names,
                                slice.start(0), slice.length(0),
                                shape.dim_size(0), clear_, done_cb);
    }
  }

 private:
  bool clear_;
};
REGISTER_KERNEL_BUILDER(Name("RestoreHashTable").Device(DEVICE_CPU),
    RestoreHashTableOp);

class RestoreBloomFilterOp : public AsyncOpKernel {
 public:
  explicit RestoreBloomFilterOp(OpKernelConstruction* context)
    : AsyncOpKernel(context) {}
  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& prefix = context->input(0);
    const Tensor& tensor_name = context->input(1);
    const Tensor& shape_and_slice = context->input(2);
    const Tensor& handle = context->input(3);
    const string& prefix_string = prefix.scalar<string>()();
    const string& shape_slice_string = shape_and_slice.scalar<string>()();
    auto tensor_name_flat = tensor_name.scalar<string>()();
    auto handle_flat = handle.scalar<ResourceHandle>()();

    std::unique_ptr<BundleReader> reader(
        new BundleReader(Env::Default(), prefix_string));
    OP_REQUIRES_OK_ASYNC(context, reader->status(), done);

    TensorShape shape;
    TensorSlice slice(1);
    TensorShape slice_shape;
    OP_REQUIRES_OK_ASYNC(context, checkpoint::ParseShapeAndSlice(
        shape_slice_string, &shape, &slice, &slice_shape), done);
    HashTableAdmitStrategyResource* resource = nullptr;
    BloomFilterAdmitStrategy* strategy = nullptr;
    {
      OP_REQUIRES_OK_ASYNC(
          context, LookupResource(context, handle_flat, &resource), done);
      strategy = dynamic_cast<BloomFilterAdmitStrategy*>(resource->Internal());
      CHECK(strategy != nullptr)
        << "Cannot restore BloomFilter from another strategy";
    }
    Status st = RestoreBloomFilter(
        reader.get(), strategy, tensor_name_flat, slice.start(0),
        slice.length(0), shape.dim_size(0));
    resource->SetInitialized(true);
    done();
  }
};
REGISTER_KERNEL_BUILDER(Name("RestoreBloomFilter").Device(DEVICE_CPU),
    RestoreBloomFilterOp);

// Restores a list of named tensors from a tensor bundle (V2 checkpoint format).
class RestoreV2 : public OpKernel {
 public:
  explicit RestoreV2(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtypes", &dtypes_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& prefix = context->input(0);
    const Tensor& tensor_names = context->input(1);
    const Tensor& shape_and_slices = context->input(2);
    OP_REQUIRES(context, tensor_names.NumElements() == dtypes_.size(),
                errors::InvalidArgument("Got ", tensor_names.NumElements(),
                                        " tensor names, but ", dtypes_.size(),
                                        " expected dtypes."));
    ValidateInputs(false /* not save op */, context, prefix, tensor_names,
                   shape_and_slices,
                   3 /*Prefix, tensor names, shape_and_slices.*/);
    if (!context->status().ok()) return;

    const string& prefix_string = prefix.scalar<tstring>()();

    // Intention: we plan to use the RestoreV2 op as a backward-compatible
    // reader as we upgrade to the V2 format.  This allows transparent upgrade.
    // We here attempt to read a V1 checkpoint, if "prefix_string" does not
    // refer to a V2 checkpoint.
    Env* env = Env::Default();
    std::vector<string> paths;
    if (!env->GetMatchingPaths(MetaFilename(prefix_string), &paths).ok() ||
        paths.empty()) {
      // Cannot find V2's metadata file, so "prefix_string" does not point to a
      // V2 checkpoint.  Invokes the V1 read path instead.
      for (size_t i = 0; i < tensor_names.NumElements(); ++i) {
        RestoreTensor(context, &checkpoint::OpenTableTensorSliceReader,
                      /* preferred_shard */ -1, /* restore_slice */ true,
                      /* restore_index */ i);
        if (!context->status().ok()) {
          return;
        }
      }
      return;
    }
    // If found, invokes the V2 reader.
    OP_REQUIRES_OK(context, RestoreTensorsV2(context, prefix, tensor_names,
                                             shape_and_slices, dtypes_));
  }

 private:
  // Expected dtypes of the to-restore tensors.
  std::vector<DataType> dtypes_;
};
REGISTER_KERNEL_BUILDER(Name("RestoreV2").Device(DEVICE_CPU), RestoreV2);

// The final step in saving sharded V2 checkpoints: merges metadata files.
class MergeV2Checkpoints : public OpKernel {
 public:
  explicit MergeV2Checkpoints(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("delete_old_dirs", &delete_old_dirs_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& checkpoint_prefixes = context->input(0);
    const Tensor& destination_prefix = context->input(1);
    OP_REQUIRES(context,
                TensorShapeUtils::IsVector(checkpoint_prefixes.shape()),
                errors::InvalidArgument(
                    "Input checkpoint_prefixes should be an 1-D tensor, got ",
                    checkpoint_prefixes.shape().DebugString(), " instead."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(destination_prefix.shape()),
                errors::InvalidArgument(
                    "Input destination_prefix should be a scalar tensor, got ",
                    destination_prefix.shape().DebugString(), " instead."));

    const gtl::ArraySlice<tstring> input_prefixes =
        gtl::ArraySlice<tstring>(checkpoint_prefixes.flat<tstring>());
    Env* env = Env::Default();
    const string& merged_prefix = destination_prefix.scalar<tstring>()();
    OP_REQUIRES_OK(
        context, tensorflow::MergeBundles(env, input_prefixes, merged_prefix));
    OP_REQUIRES_OK(
        context, MoveSsdFiles(env, input_prefixes, merged_prefix));

    if (delete_old_dirs_) {
      const string merged_dir(io::Dirname(merged_prefix));
      for (const string& input_prefix : input_prefixes) {
        const string dirname(io::Dirname(input_prefix));
        if (dirname == merged_dir) continue;
        Status status = env->DeleteDir(dirname);
        // For sharded save, only the first delete will go through and all
        // others will hit NotFound.  Use vlog to be less verbose.
        if (!status.ok()) VLOG(1) << status;
      }
    }
  }

 private:
  // On merge, whether or not to delete the input (temporary) directories.
  bool delete_old_dirs_;
};
REGISTER_KERNEL_BUILDER(Name("MergeV2Checkpoints").Device(DEVICE_CPU),
                        MergeV2Checkpoints);

}  // namespace tensorflow
