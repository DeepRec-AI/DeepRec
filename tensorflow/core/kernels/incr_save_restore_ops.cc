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

#include "tensorflow/core/kernels/incr_save_restore_ops.h"
#include "tensorflow/core/framework/resource_handle.h"

namespace tensorflow {

template <typename TIndex>
class RecordSparseIndicesOp: public OpKernel {
 public:
  explicit RecordSparseIndicesOp(OpKernelConstruction* context)
      : OpKernel(context)
      , auto_record_(false) {
     OP_REQUIRES_OK(context,
         context->GetAttr("var_name", &sparse_incr_res_name_));
     OP_REQUIRES_OK(context,
                    context->GetAttr("auto_record", &auto_record_));
  }

  void Compute(OpKernelContext* ctx) override {
    IndicesIncrRecorder<TIndex>* sparse_incr_res = nullptr;
    auto rm = ctx->resource_manager();
    OP_REQUIRES_OK(
        ctx,
        rm->LookupOrCreate<IndicesIncrRecorder<TIndex>>(
            "", sparse_incr_res_name_ + "_sparse_incr", &sparse_incr_res,
            [this](IndicesIncrRecorder<TIndex>** ptr) {
              *ptr = new IndicesIncrRecorder<TIndex>(sparse_incr_res_name_);
              if (auto_record_) {
                (*ptr)->UpdateGlobalVersion();
              }
              VLOG(2) << "sparse_incr_res created, name:"
                      << sparse_incr_res_name_;
              return Status::OK();
            }));
    sparse_incr_res->UpdateIndices(ctx->input(0), ctx);
  }

 private:
  string sparse_incr_res_name_;
  bool auto_record_;
};

REGISTER_KERNEL_BUILDER(Name("RecordSparseIndices")
    .Device(DEVICE_CPU)
    .TypeConstraint<int32>("TIndex"),
    RecordSparseIndicesOp<int32>);

REGISTER_KERNEL_BUILDER(Name("RecordSparseIndices")
    .Device(DEVICE_CPU)
    .TypeConstraint<int64>("TIndex"),
    RecordSparseIndicesOp<int64>);

class ActivateSparseRecorderOp : public OpKernel {
 public:
  explicit ActivateSparseRecorderOp(OpKernelConstruction* context)
    : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_names = context->input(0);
    const auto& tensor_names_flat = tensor_names.flat<string>();
    const int num_tensors = static_cast<int>(tensor_names.NumElements());

    auto rm = context->resource_manager();
    for (int i = 0; i < num_tensors; ++i) {
      const string& tensor_name = tensor_names_flat(i);
      // cast forcely to IndicesIncrRecorder for incr cpkt
      string incr_res_name = tensor_name + "_sparse_incr";
      IndicesIncrRecorder<int32>* sparse_incr_res = nullptr;
      rm->Lookup("", incr_res_name, &sparse_incr_res);
      if (sparse_incr_res != nullptr) {
        sparse_incr_res->UpdateGlobalVersion();
      } else {
        IndicesIncrRecorder<int64>* sparse_incr_res = nullptr;
        rm->Lookup("", incr_res_name, &sparse_incr_res);
        if (sparse_incr_res != nullptr) {
          sparse_incr_res->UpdateGlobalVersion();
        } else {
          LOG(WARNING) << tensor_name
                       << "_sparse_incr" << " Resource NOT FOUND";
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("ActivateSparseRecorder")
    .Device(DEVICE_CPU),
    ActivateSparseRecorderOp);

class IncrSaveOp: public OpKernel {
 public:
  explicit IncrSaveOp(OpKernelConstruction* context)
    : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtypes", &tensor_types_));
  }

  void Compute(OpKernelContext* context) override {
    const int kFixedInputs = 4;  // Prefix, tensor names, is_sparse
    const Tensor& prefix = context->input(0);
    const string& prefix_string = prefix.scalar<string>()();
    const Tensor& tensor_names = context->input(1);
    const Tensor& shape_and_slices = context->input(2);
    const Tensor& is_sparse = context->input(3);
    const int num_tensors = static_cast<int>(tensor_names.NumElements());
    const auto& tensor_names_flat = tensor_names.flat<string>();
    const auto& is_sparse_flat = is_sparse.flat<bool>();
    const auto& shape_and_slices_flat = shape_and_slices.flat<string>();
    LOG(INFO) <<  "prefix_string: " << prefix_string
              << "num tensors:" << num_tensors;
    auto rm = context->resource_manager();
    BundleWriter writer(Env::Default(), prefix_string);

    for (int i = 0; i < num_tensors; i++) {
      const string& tensor_name = tensor_names_flat(i);
      if (is_sparse_flat(i)) {
        IndicesIncrRecorder<int64>* sparse_incr_res = nullptr;
        rm->Lookup("", tensor_name + "_sparse_incr", &sparse_incr_res);
        if (sparse_incr_res != nullptr) {
          DumpIncrSparse<int64>(context, i, kFixedInputs,
              tensor_name, &writer, sparse_incr_res);
        } else {
          IndicesIncrRecorder<int32>* sparse_incr_res = nullptr;
          rm->Lookup("", tensor_name + "_sparse_incr",
              &sparse_incr_res);
          if (sparse_incr_res != nullptr) {
            DumpIncrSparse<int32>(context, i, kFixedInputs,
                tensor_name, &writer, sparse_incr_res);
          } else {
            LOG(WARNING) << tensor_name << "_sparse_incr"
                         << " Resource NOT FOUND";
          }
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
  template<typename T>
  void DumpIncrSparse(OpKernelContext* context, int i,
      const int& kFixedInputs, const string& tensor_name,
      BundleWriter* writer, IndicesIncrRecorder<T>* sparse_incr_res) {
    if (tensor_types_[i] == DT_RESOURCE) {
      // ev, must be sparse
      EmbeddingVar<T, float>* variable = nullptr;
      OP_REQUIRES_OK(context,
          LookupResource(context,
            HandleFromInput(context, i + kFixedInputs), &variable));
      core::ScopedUnref unref_variable(variable);
      OP_REQUIRES_OK(context,
          sparse_incr_res->DumpSparseEmbeddingTensor(
            tensor_name, variable, writer, context));
    } else {
      const Tensor& sparse_var  = context->input(i + kFixedInputs);
      OP_REQUIRES_OK(context,
          sparse_incr_res->DumpSparseNormalTensor(
            tensor_name, sparse_var, writer));
    }
  }

 private:
  DataTypeVector tensor_types_;
};

REGISTER_KERNEL_BUILDER(
    Name("IncrSave")
    .Device(DEVICE_CPU),
    IncrSaveOp);

class IncrRestoreOp: public OpKernel {
 public:
  explicit IncrRestoreOp(OpKernelConstruction* context)
    : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtypes", &tensor_types_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& prefix = context->input(0);
    const string& prefix_string = prefix.scalar<string>()();
    const Tensor& tensor_names = context->input(1);
    const Tensor& shape_and_slices = context->input(2);
    const Tensor& is_sparse_tensor = context->input(3);
    const bool& is_sparse = is_sparse_tensor.scalar<bool>()();
    const auto& shape_and_slices_flat = shape_and_slices.flat<string>();
    const int num_tensors = static_cast<int>(tensor_names.NumElements());
    if (is_sparse) {
      BundleReader reader(Env::Default(), prefix_string);
      OP_REQUIRES_OK(context, reader.status());
      VLOG(1) << "BundleReader incr, prefix_string: " << prefix_string;
      LOG(INFO) << "BundleReader incr, prefix_string: " << prefix_string;
      const auto& tensor_names_flat = tensor_names.flat<string>();
      if (num_tensors > 1) {
        // EV
        if (num_tensors != 3) {
          OP_REQUIRES_OK(context, errors::InvalidArgument(
              "Incr cpkt restore for ev must has 3 tensors, actually ",
              num_tensors, " given"));
        }

        const string& ev_keys_name = tensor_names_flat(0);
        string incr_tensor_name = ev_keys_name.substr(0,
            ev_keys_name.find("-keys"));
        // 1 read keys, values and versions
        TensorShape incr_shape;
        Tensor* incr_keys_tensor = nullptr;
        Tensor* incr_values_tensor = nullptr;
        Tensor* incr_versions_tensor = nullptr;
        OP_REQUIRES_OK(context,
            reader.LookupTensorShape(incr_tensor_name + "-sparse_incr_keys",
              &incr_shape));
        OP_REQUIRES_OK(context,
            context->allocate_output(0, incr_shape, &incr_keys_tensor));
        OP_REQUIRES_OK(context,
            reader.Lookup(incr_tensor_name + "-sparse_incr_keys",
              incr_keys_tensor));
        OP_REQUIRES_OK(context,
            reader.LookupTensorShape(incr_tensor_name + "-sparse_incr_values",
              &incr_shape));
        OP_REQUIRES_OK(context,
            context->allocate_output(1, incr_shape, &incr_values_tensor));
        OP_REQUIRES_OK(context,
            reader.Lookup(incr_tensor_name + "-sparse_incr_values",
              incr_values_tensor));

        OP_REQUIRES_OK(context,
            reader.LookupTensorShape(incr_tensor_name +
              "-sparse_incr_versions", &incr_shape));
        OP_REQUIRES_OK(context,
            context->allocate_output(2, incr_shape,
              &incr_versions_tensor));
        OP_REQUIRES_OK(context,
            reader.Lookup(incr_tensor_name + "-sparse_incr_versions",
              incr_versions_tensor));
      } else {
        // 1 Read keys from incr ckpt
        TensorShape keys_shape;
        Tensor keys_tensor;
        DataType key_type;

        const string& tensor_name = tensor_names_flat(0);
        OP_REQUIRES_OK(context,
            reader.LookupDtypeAndShape(tensor_name + "-sparse_incr_keys",
              &key_type, &keys_shape));

        OP_REQUIRES_OK(context,
            context->allocate_temp(key_type, keys_shape, &keys_tensor));

        OP_REQUIRES_OK(context,
            reader.Lookup(tensor_name + "-sparse_incr_keys",
              &keys_tensor));

        LOG(INFO) << "Finished restoring incr normal sparse keys tensor:"
          << tensor_name.data() << ", size:" << keys_tensor.TotalBytes();

        // 2 Read values from incr ckpt
        TensorShape values_shape;
        Tensor values_tensor;

        OP_REQUIRES_OK(context,
            reader.LookupTensorShape(tensor_name + "-sparse_incr_values",
              &values_shape));

        OP_REQUIRES_OK(context,
            context->allocate_temp(DT_FLOAT, values_shape, &values_tensor));

        OP_REQUIRES_OK(context,
            reader.Lookup(tensor_name + "-sparse_incr_values", &values_tensor));

        LOG(INFO) << "Finished restoring incr normal sparse values tensor:"
          << tensor_name.data() << ", size:" << values_tensor.TotalBytes();
        // 3 do incr update
        const Tensor& orig_sparse_tensor = context->input(4);
        Tensor* new_sparse_tensor = nullptr;
        OP_REQUIRES_OK(context,
            context->forward_input_or_allocate_output({4}, 0,
              orig_sparse_tensor.shape(), &new_sparse_tensor));

        // 3.1 update specific rows
        auto incr_values_flat = values_tensor.template matrix<float>();
        auto new_values_flat = new_sparse_tensor->template matrix<float>();
        auto limit = new_sparse_tensor->dim_size(1);

        for (auto i = 0; i < keys_tensor.NumElements(); i++) {
          if (key_type == DT_INT32) {
            auto incr_key =
              keys_tensor.flat<EnumToDataType<DT_INT32>::Type>()(i);
            if (incr_key >= new_sparse_tensor->dim_size(0))
                continue;
            for (auto j = 0; j < limit; j++) {
              new_values_flat(incr_key, j) = incr_values_flat(i, j);
            }
          } else {
            auto incr_key =
              keys_tensor.flat<EnumToDataType<DT_INT64>::Type>()(i);
            if (incr_key >= new_sparse_tensor->dim_size(0))
              continue;
            for (auto j = 0; j < limit; j++) {
              new_values_flat(incr_key, j) = incr_values_flat(i, j);
            }
          }
        }
        LOG(INFO) << "Finished restoring normal sparse tensor(full+incr):"
          << tensor_name.data() << ", size:" << new_sparse_tensor->TotalBytes();
      }
    } else {
      RestoreTensorsV2(context, prefix, tensor_names,
          shape_and_slices, tensor_types_);
    }
  }

 private:
  DataTypeVector tensor_types_;
};

REGISTER_KERNEL_BUILDER(
    Name("IncrRestore")
    .Device(DEVICE_CPU),
    IncrRestoreOp);

class CollectSparseIndicesOp : public OpKernel {
 public:
  explicit CollectSparseIndicesOp(OpKernelConstruction* context) :
    OpKernel(context),
    update_count_thd_(0) {
    string config_str;
    OP_REQUIRES_OK(context, context->GetAttr("config", &config_str));
    OP_REQUIRES_OK(context, ParseConfig(config_str));
    OP_REQUIRES_OK(context, context->GetAttr("ktype", &tensor_type_));
    OP_REQUIRES_OK(context, context->GetAttr("tensor_name", &tensor_name_));

    int64 part_idx, part_count, hash_bucket_size;
    OP_REQUIRES_OK(context, context->GetAttr("part_idx", &part_idx));
    OP_REQUIRES_OK(context, context->GetAttr("part_count", &part_count));
    OP_REQUIRES_OK(context, context->GetAttr("hash_bucket_size",
          &hash_bucket_size));

    if (part_count > 0 && hash_bucket_size > 0) {
      string part_mode_str;
      OP_REQUIRES_OK(context, context->GetAttr("part_mode", &part_mode_str));
      if (part_mode_str == "mod") {
        partitioner_ = std::move(std::unique_ptr<SparsePartitioner>(
              new ModSparsePartitioner(part_count, part_idx,
                                       hash_bucket_size)));
      } else {
        partitioner_ = std::move(std::unique_ptr<SparsePartitioner>(
              new DivSparsePartitioner(part_count, part_idx,
                                       hash_bucket_size)));
      }
    }
  }

  void Compute(OpKernelContext* context) override {
    if (tensor_type_ == DT_INT32) {
      OP_REQUIRES_OK(context, ExportSparseIndices<int32>(
            tensor_name_, context));
    } else if (tensor_type_ == DT_INT64) {
      OP_REQUIRES_OK(context, ExportSparseIndices<int64>(
            tensor_name_, context));
    } else {
      LOG(WARNING) << "Not support key type:"
                   << DataTypeString(tensor_type_);
    }
  }

 private:
  template <typename KeyType>
  Status ExportSparseIndices(const string &tensor_name,
      OpKernelContext *context) {
    auto rm = context->resource_manager();
    string resource_name = tensor_name + "_sparse_incr";
    IndicesIncrRecorder<KeyType>* sparse_incr_res = nullptr;
    rm->Lookup("", resource_name, &sparse_incr_res);
    if (sparse_incr_res == nullptr) {
      LOG(WARNING) << tensor_name << " Resource NOT FOUND";
      return Status::OK();
    }
    return DoExportSparseIndices(sparse_incr_res, context);
  }

  template <typename KeyType>
  Status DoExportSparseIndices(IndicesIncrRecorder<KeyType> *sparse_incr_res,
      OpKernelContext* ctx) {
    std::unordered_map<KeyType, uint64> indices;
    sparse_incr_res->SwapIndices(indices);
    std::vector<KeyType> filtered_indices;
    FilterIndices(indices, filtered_indices);

    Tensor *keys_out = nullptr;
    Tensor *global_keys_out = nullptr;
    TF_RETURN_IF_ERROR(ctx->allocate_output(0,
          TensorShape({(int64)filtered_indices.size()}), &keys_out));

    TF_RETURN_IF_ERROR(ctx->allocate_output(1,
          TensorShape({(int64)filtered_indices.size()}), &global_keys_out));

    auto keys_out_flat = keys_out->flat<KeyType>();
    auto global_keys_out_flat = global_keys_out->flat<KeyType>();
    for (size_t i = 0; i < filtered_indices.size(); i++) {
      KeyType k = filtered_indices[i];
      KeyType global_k = k;
      if (partitioner_) {
        global_k = (KeyType)partitioner_->CalcGlobalOffset(k);
        VLOG(2) << partitioner_->toString() << ", key:"
                << k << ", global key:" << global_k;
      }
      keys_out_flat(i) = k;
      global_keys_out_flat(i) = global_k;
    }
    return Status::OK();
  }

  template <typename KeyType>
  void FilterIndices(const std::unordered_map<KeyType, uint64>& indices,
      std::vector<KeyType>& filtered_indices) {
    filtered_indices.reserve(indices.size());
    for (const auto &it : indices) {
      const auto &key = it.first;
      uint64 update_count = it.second;
      if (update_count >= update_count_thd_) {
        filtered_indices.push_back(key);
      }
    }
  }

  Status ParseConfig(const string &config_str) {
    LOG(INFO) << "Collect sparse indices config:" << config_str;
    std::vector<string> configs = str_util::Split(config_str, ",");
    for (size_t i = 0; i < configs.size(); i++) {
      const string &s = configs[i];
      std::vector<string> kv = str_util::Split(s, "=");
      if (kv.size() < 2) {
        LOG(WARNING) << "invalid config:" << s;
        continue;
      }
      if (kv[0] == "update_count_thd") {
        if (!strings::safe_strtou64(kv[1], &update_count_thd_)) {
          LOG(WARNING) << "invalid config:" << s;
        }
      }
    }
 
    LOG(INFO) << "Parse collect sparse indices config success,"
              << "update_cound_thd=" << update_count_thd_;
 
    return Status::OK();
  }

 private:
  std::string tensor_name_;
  DataType tensor_type_;
  uint64 update_count_thd_;
  std::unique_ptr<SparsePartitioner> partitioner_;
};

REGISTER_KERNEL_BUILDER(
    Name("CollectSparseIndices")
    .Device(DEVICE_CPU),
    CollectSparseIndicesOp);

} // namespace tensorflow

