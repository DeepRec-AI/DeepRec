/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "serving/odl_processor/storage/redis_feature_store.h"
#include "serving/odl_processor/storage/feature_store_mgr.h"

namespace tensorflow {
namespace processor {

namespace {


template <typename TValue>
BatchGetCallback make_lookup_callback(
    OpKernelContext* ctx,
    int64 N, // indices count
    Tensor allocated_out_tensor_const,
    Tensor default_values,
    AsyncOpKernel::DoneCallback done) {
  return [ctx, N, allocated_out_tensor_const, default_values,
          done = std::move(done)](const Status& s) {
/*
    Tensor allocated_out_tensor = allocated_out_tensor_const;
    auto out_flat = allocated_out_tensor.shaped<TValue, 2>(
        {N, allocated_out_tensor.NumElements() / N});
    LOG(INFO) << "Return Tensor is: ";
    for (int i = 0; i < N; ++i) {
      LOG(INFO) << "Row " << i << " :";
      for (int j = 0; j < allocated_out_tensor.NumElements()/N; ++j) {
        LOG(INFO) << *((float*)&out_flat(i, j));
      }
    }

*/
    ctx->SetStatus(s);

    done();
  };
}

}  // namespace

template <typename TKey, typename TValue>
class KvLookupOp : public AsyncOpKernel {
 public:
  explicit KvLookupOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_name", &feature_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_name_to_id",
                                     &feature_name_to_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim_len", &dim_len_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const Tensor& indices = ctx->input(0);

    const int64 N = indices.NumElements();

    Tensor default_values(ctx->input(1));

    const Tensor& storage_pointer = ctx->input(2);
    const uint64 storage_pointer_value =
        storage_pointer.scalar<tensorflow::uint64>()();
    IFeatureStoreMgr* storageMgr =
        reinterpret_cast<IFeatureStoreMgr*>(storage_pointer_value);

    const Tensor& model_version = ctx->input(3);
    const uint64 model_version_value =
        model_version.scalar<tensorflow::uint64>()();

    TensorShape result_shape = indices.shape();
    TensorShape value_shape({dim_len_});
    result_shape.AppendShape(value_shape);

    // buffer will be pass to sparse storage
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, result_shape, &out));

    if (N <= 0) {
      done();
      return;
    }

    // CHECK has the same dim value
    OP_REQUIRES(ctx, dim_len_ == out->NumElements() / N,
        errors::InvalidArgument(
            "hashmap's value_len should same with output's dimension(1)",
            std::to_string(dim_len_), std::to_string(out->NumElements() / N)));

    Status s = storageMgr->GetValues(
        model_version_value,
        feature_name_to_id_,
        (const char*)indices.data(),
        (char*)out->data(), sizeof(TKey),
        sizeof(TValue) * dim_len_, N,
        (const char*)default_values.data(),
        make_lookup_callback<TValue>(
            ctx, N, *out, default_values,
            std::move(done)));

    if (!s.ok()) {
      ctx->SetStatus(s);
      done();
    }
  }

 private:
  std::string feature_name_;
  int64 feature_name_to_id_;
  int dim_len_;
};

#define REGISTER_KV_LOOKUP(dev, ktype, vtype)                  \
  REGISTER_KERNEL_BUILDER(Name("KvLookup")                     \
                              .Device(DEVICE_##dev)            \
                              .TypeConstraint<vtype>("dtype")  \
                              .TypeConstraint<ktype>("Tkeys"), \
                          KvLookupOp<ktype, vtype>)

#define REGISTER_KV_LOOKUP_ALL_KEY_TYPES(dev, type) \
  REGISTER_KV_LOOKUP(dev, int32, type);             \
  REGISTER_KV_LOOKUP(dev, int64, type)

#define REGISTER_KV_LOOKUP_CPU(type) \
    REGISTER_KV_LOOKUP_ALL_KEY_TYPES(CPU, type)

TF_CALL_ALL_TYPES(REGISTER_KV_LOOKUP_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_KV_LOOKUP_CPU);

#undef REGISTER_KV_LOOKUP_CPU
#undef REGISTER_KV_LOOKUP_ALL_KEY_TYPES
#undef REGISTER_KV_LOOKUP


namespace {

// TODO: magic-num 128MB
const static size_t BUFFER_SIZE = 128 << 20;

size_t LookupSegmentInternal(size_t key_size,
                             size_t value_bytes,
                             int64 total_keys_num,
                             const std::string& tensor_key,
                             const std::string& tensor_value,
                             char* key_buffer, char* value_buffer,
                             BundleReader* reader,
                             Status& s_read) {
  size_t read_key_num = std::min(BUFFER_SIZE / key_size,
                                 BUFFER_SIZE / value_bytes);
  read_key_num = std::min((int64)read_key_num, total_keys_num);

  size_t key_bytes_read = 0, value_bytes_read = 0;
  if (total_keys_num > 0) {
    reader->LookupSegment(tensor_key, read_key_num * key_size,
                          key_buffer, key_bytes_read);
    reader->LookupSegment(tensor_value, read_key_num * value_bytes,
                          value_buffer, value_bytes_read);

    if (key_bytes_read > 0) {
      read_key_num = key_bytes_read / key_size;
      if (read_key_num != value_bytes_read / value_bytes) {
        s_read = errors::Internal("read key num not euqal read value num.");
        return 0;
      }
      total_keys_num -= read_key_num;
    } else {
      read_key_num = 0;
    }
  }

  return read_key_num;
}

int64 GetTotalKeysNum(const std::string& part_var_name,
                      std::string& tensor_key,
                      std::string& tensor_value,
                      BundleReader* reader,
                      OpKernelContext* ctx,
                      int dim_len,
                      int& curr_part_index,
                      int partition_num,
                      size_t key_size, size_t value_size) {
  while(curr_part_index < partition_num) {
    TensorShape key_shape, value_shape;
    reader->LookupTensorShape(tensor_key, &key_shape);
    reader->LookupTensorShape(tensor_value, &value_shape);
    if (value_shape.dim_size(1) != dim_len) {
      ctx->CtxFailure(errors::InvalidArgument(
                          strings::StrCat("value_shape.dim_size(1) not equal "
                          "the dim_len attr value. ",
                          std::to_string(value_shape.dim_size(1)),
                          " vs ", std::to_string(dim_len))));
      return -1;
    }

    Status s = reader->LookupHeader(tensor_key,
        key_size * key_shape.dim_size(0));
    if (!s.ok()) {
      ctx->CtxFailure(s);
      return -1;
    }

    s = reader->LookupHeader(tensor_value,
        value_size * value_shape.dim_size(0) * dim_len);
    if (!s.ok()) {
      ctx->CtxFailure(s);
      return -1;
    }

    int64 total_keys_num = key_shape.dim_size(0);
    if (total_keys_num > 0) return total_keys_num;

    LOG(WARNING) << "Current variable partitions' key num is 0. "
                 << tensor_key << ", " << tensor_value;

    ++curr_part_index;
    // try next variable partition
    tensor_key = strings::StrCat(
        part_var_name, std::to_string(curr_part_index), "-keys");
    tensor_value = strings::StrCat(
        part_var_name, std::to_string(curr_part_index), "-values");
  }

  return 0;
}

BatchSetCallback make_import_callback(
    OpKernelContext* ctx,
    char* key_buffer,
    char* value_buffer,
    int64 total_keys_num,
    size_t dim_len,
    size_t key_size,
    size_t value_size,
    int part_index,
    int partition_num,
    const std::string& part_var_name,
    const std::string& tensor_key,
    const std::string& tensor_value,
    uint64_t feature_name_to_id,
    uint64_t model_version,
    BundleReader* reader,
    IFeatureStoreMgr* storageMgr,
    AsyncOpKernel::DoneCallback done);
 
Status InternalImportValues(
    OpKernelContext* ctx,
    size_t dim_len,
    size_t key_size,
    size_t value_size,
    int64 total_keys_num,
    int curr_part_index,
    int partition_num,
    const std::string& part_var_name,
    const std::string& tensor_key,
    const std::string& tensor_value,
    char* key_buffer, char* value_buffer,
    BundleReader* reader,
    uint64_t feature_name_to_id,
    uint64_t model_version,
    IFeatureStoreMgr* storageMgr,
    AsyncOpKernel::DoneCallback done) {
  Status s_read = tensorflow::Status::OK();
  size_t read_key_num = LookupSegmentInternal(
      key_size, value_size * dim_len,
      total_keys_num, tensor_key, tensor_value,
      key_buffer, value_buffer, reader, s_read);
  if (!s_read.ok()) {
    return s_read;
  }

  total_keys_num -= read_key_num;

  Status status = storageMgr->SetValues(
      model_version,
      feature_name_to_id,
      key_buffer, value_buffer,
      key_size, value_size * dim_len, read_key_num,
      make_import_callback(
          ctx, key_buffer, value_buffer,
          total_keys_num, dim_len, key_size,
          value_size, curr_part_index,
          partition_num, part_var_name,
          tensor_key, tensor_value,
          feature_name_to_id, model_version,
          reader, storageMgr, std::move(done)));

  return status;
}

BatchSetCallback make_import_callback(
    OpKernelContext* ctx,
    char* key_buffer,
    char* value_buffer,
    int64 total_keys_num,
    size_t dim_len,
    size_t key_size,
    size_t value_size,
    int part_index,
    int partition_num,
    const std::string& part_var_name,
    const std::string& tensor_key,
    const std::string& tensor_value,
    uint64_t feature_name_to_id,
    uint64_t model_version,
    BundleReader* reader,
    IFeatureStoreMgr* storageMgr,
    AsyncOpKernel::DoneCallback done) {
  return [ctx, key_buffer, value_buffer, key_size,
          value_size, left_keys_num = total_keys_num,
          dim_len, part_index, partition_num,
          part_var_name, tensor_key, tensor_value,
          feature_name_to_id, model_version,
          reader, storageMgr,
          done = std::move(done)](const Status& s) {

    if (!s.ok()) {
      ctx->SetStatus(s);
      done();
      return;
    }

    int curr_part_index = part_index;
    int64 total_keys_num = left_keys_num;
    bool new_part = false;
    std::string new_tensor_key;
    std::string new_tensor_value;
    if (total_keys_num <= 0) {
      ++curr_part_index;
      // All partitions have been imported
      if (curr_part_index >= partition_num) {
        delete []key_buffer;
        delete []value_buffer;
        delete reader;

        ctx->SetStatus(s);
        done();
        return;
      }

      new_tensor_key = strings::StrCat(
          part_var_name, std::to_string(curr_part_index), "-keys");
      new_tensor_value = strings::StrCat(
          part_var_name, std::to_string(curr_part_index), "-values");

      total_keys_num = GetTotalKeysNum(
          part_var_name, new_tensor_key, new_tensor_value,
          reader, ctx, dim_len, curr_part_index,
          partition_num, key_size, value_size);
      // All partitions' key num equal 0
      if (total_keys_num < 0 ||
          curr_part_index >= partition_num) {
        // NOTE: total_keys_num < 0, ctx will be set error
        //       status in GetTotalKeysNum func;
        //       curr_part_index >= partition_num: normal case.
        done();
        return;
      }

      new_part = true;
    }

    Status status = InternalImportValues(
        ctx, dim_len, key_size, value_size,
        total_keys_num, curr_part_index,
        partition_num, part_var_name,
        new_part ? new_tensor_key : tensor_key,
        new_part ? new_tensor_value : tensor_value,
        key_buffer, value_buffer, reader,
        feature_name_to_id, model_version,
        storageMgr, std::move(done));

    if (!status.ok()) {
      ctx->SetStatus(status);
      done();
      return;
    }
  };
}

}  // namespace

template <typename TKey, typename TValue>
class KvImportOp : public AsyncOpKernel {
 public:
  explicit KvImportOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_name", &feature_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_name_to_id",
                                     &feature_name_to_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim_len", &dim_len_));
  }

  // NOTE(jiankeng.pt):
  // In order to prevent excessive traffic pressure on the network,
  // here we send variable block by block(128MB).
  // The subsequent block will be sent after the storage return OK status.
  //
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    size_t key_size = sizeof(TKey);
    size_t value_size = sizeof(TValue);

    const Tensor& file_name = ctx->input(0);
    const std::string file_name_str = file_name.scalar<string>()();

    const Tensor& tensor_name = ctx->input(1);
    const std::string tensor_name_str = tensor_name.scalar<string>()();
    std::string part0_tensor_name_str =
        strings::StrCat(tensor_name_str, "/part_0");
    std::string part_tensor_name_str =
        strings::StrCat(tensor_name_str, "/part_");

    const Tensor& storage_pointer = ctx->input(2);
    const uint64 storage_pointer_value =
        storage_pointer.scalar<tensorflow::uint64>()();
    IFeatureStoreMgr* storageMgr =
        reinterpret_cast<IFeatureStoreMgr*>(storage_pointer_value);

    const Tensor& model_version = ctx->input(3);
    const uint64 model_version_value =
        model_version.scalar<tensorflow::uint64>()();

    const Tensor& t_incr_ckpt = ctx->input(4);
    const bool is_incr_ckpt = t_incr_ckpt.scalar<bool>()();

    // create for read from file
    BundleReader* reader = new BundleReader(Env::Default(), file_name_str);
    OP_REQUIRES_OK(ctx, reader->status());

    std::string tensor_key = strings::StrCat(tensor_name_str, "-keys");
    std::string tensor_value = strings::StrCat(tensor_name_str, "-values");
    if (is_incr_ckpt) {
      tensor_key = strings::StrCat(tensor_name_str, "-sparse_incr_keys");
      tensor_value = strings::StrCat(tensor_name_str, "-sparse_incr_values");
    }

    bool maybe_partition_var = false;
    int partition_num = 1;
    int curr_part_index = 0;

    // 1) check variable without partition
    TensorShape key_shape;
    Status key_status = reader->LookupTensorShape(tensor_key, &key_shape);
    if (!key_status.ok()) {
      // not found, check partition variable below
      if (errors::IsNotFound(key_status)) {
        maybe_partition_var = true;
      } else {
        OP_REQUIRES_OK(ctx, key_status);
      }
    }

    // 2) check variable with partition
    if (maybe_partition_var) {
      tensor_key = strings::StrCat(part0_tensor_name_str, "-keys");
      tensor_value = strings::StrCat(part0_tensor_name_str, "-values");
      key_status = reader->LookupTensorShape(tensor_key, &key_shape);
      OP_REQUIRES_OK(ctx, key_status);

      for (int i = 1; ; ++i) {
        std::string tmp_key = strings::StrCat(
            part_tensor_name_str, std::to_string(i), "-keys");
        key_status = reader->LookupTensorShape(tmp_key, &key_shape);
        if (errors::IsNotFound(key_status)) break;
        OP_REQUIRES_OK(ctx, key_status);
        ++partition_num;
      }
    }

    int64 total_keys_num = GetTotalKeysNum(
        part_tensor_name_str, tensor_key, tensor_value,
        reader, ctx, dim_len_, curr_part_index,
        partition_num, key_size, value_size);
    // All partitions' key num equal 0
    if (total_keys_num < 0 ||
        curr_part_index >= partition_num) {
      // NOTE: total_keys_num < 0, ctx will be set error
      //       status in GetTotalKeysNum func;
      //       curr_part_index >= partition_num: normal case.
      LOG(WARNING) << "All variable partitions' key num is 0. variable name:"
                   << tensor_name_str;
      done();
      return;
    }

    char* key_buffer = new char[BUFFER_SIZE];
    char* value_buffer = new char[BUFFER_SIZE];

    Status s = InternalImportValues(
        ctx, dim_len_, key_size, value_size,
        total_keys_num, curr_part_index,
        partition_num, part_tensor_name_str,
        tensor_key, tensor_value, key_buffer,
        value_buffer, reader, feature_name_to_id_,
        model_version_value, storageMgr, std::move(done));

    if (!s.ok()) {
      ctx->SetStatus(s);
      done();
    }
  }

 private:
  std::string feature_name_;
  int64 feature_name_to_id_;
  int dim_len_;
};

#define REGISTER_KV_IMPORT(dev, ktype, vtype)                \
  REGISTER_KERNEL_BUILDER(Name("KvImport")                   \
                            .Device(DEVICE_##dev)            \
                            .TypeConstraint<ktype>("Tkeys")  \
                            .TypeConstraint<vtype>("dtype"), \
                          KvImportOp<ktype, vtype>);
#define REGISTER_KV_IMPORT_ALL_KEY_TYPES(dev, type)   \
  REGISTER_KV_IMPORT(dev, int32, type);               \
  REGISTER_KV_IMPORT(dev, int64, type)

#define REGISTER_KV_IMPORT_CPU(type)   \
    REGISTER_KV_IMPORT_ALL_KEY_TYPES(CPU, type)

TF_CALL_ALL_TYPES(REGISTER_KV_IMPORT_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_KV_IMPORT_CPU);

#undef REGISTER_KV_IMPORT_CPU
#undef REGISTER_KV_IMPORT_ALL_KEY_TYPES
#undef REGISTER_KV_IMPORT

namespace {

// TODO: FIXME, function is just for testing
typedef std::function<void(const Status&)> InitCallback;

void FakeInit(std::vector<std::string>& feature_names,
              InitCallback callback) {
  for (auto name : feature_names) {
    LOG(INFO) << "name: " << name;
  }

  Status s;
  callback(s);
}

InitCallback make_init_callback(
    OpKernelContext* ctx,
    AsyncOpKernel::DoneCallback done) {
  return [ctx, done = std::move(done)](const Status& s) {
    ctx->SetStatus(s);
    done();   
  };
}

}

class KvInitOp : public AsyncOpKernel {
 public:
  explicit KvInitOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_names", &feature_names_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    FakeInit(feature_names_,
             make_init_callback(ctx, std::move(done)));
  }

 private:
  std::vector<std::string> feature_names_;
};

REGISTER_KERNEL_BUILDER(Name("KvInit").Device(DEVICE_CPU),
                        KvInitOp);


}  // namespace processor
}  // namespace tensorflow
