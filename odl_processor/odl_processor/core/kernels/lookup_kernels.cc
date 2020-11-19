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

namespace tensorflow {
namespace processor {

namespace {

// TODO: FIXME
typedef std::function<void(const Status&, // const Tensor& val,
                           std::vector<int> not_found_ids_offset)>
    LookupCallback;

void FakeLookup(OpKernelContext* ctx,
                int64 N, // indices count
                int64 dim_len, // len of dim 1
                Tensor allocated_out_tensor,
                const std::string& var_name,
                const std::string& version_str,
                Tensor indices, // indices
                LookupCallback callback) {
}

// TODO: FIXME
template <typename TValue>
LookupCallback make_lookup_callback(
    OpKernelContext* ctx,
    int64 N, // indices count
    int64 dim_len, // len of dim 1
    Tensor& default_values,
    Tensor& allocated_out_tensor,
    AsyncOpKernel::DoneCallback done) {
  return [ctx, N, dim_len, default_values_const = std::move(default_values),
          allocated_out_tensor_const = std::move(allocated_out_tensor),
          done = std::move(done)](const Status& s,
                                  //const Tensor& val,
                                  std::vector<int> not_found_ids_offset) {
    Tensor default_values = default_values_const;
    Tensor allocated_out_tensor = allocated_out_tensor_const;
    auto default_values_matrix = default_values.shaped<TValue, 2>(
        {default_values.NumElements()/dim_len, dim_len});

    // fill default value here
    if (not_found_ids_offset.size() > 0) {
      auto out_flat = allocated_out_tensor.shaped<TValue, 2>(
          {N, allocated_out_tensor.NumElements() / N});
      TValue* out_base = &out_flat(0, 0);
      for (size_t i = 0; i < not_found_ids_offset.size(); ++i) {
        TValue* default_v = &default_values_matrix(not_found_ids_offset[i], 0);
        TValue* pointer = out_base + not_found_ids_offset[i] * dim_len;
        for (int64 j = 0; j < dim_len; ++j) {
          *(pointer + j) = *(default_v + j);
        }
      }
    }

    ctx->SetStatus(s);

    done();
  };
}

}  // namespace

template <typename TKey, typename TValue>
class KvLookupOp : public AsyncOpKernel {
 public:
  explicit KvLookupOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim_len", &dim_len_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const Tensor& version = ctx->input(0);
    const std::string version_str = version.scalar<string>()();

    const Tensor& indices = ctx->input(1);
    const int64 N = indices.NumElements();

    Tensor default_values(ctx->input(2));

    TensorShape result_shape = indices.shape();
    TensorShape value_shape({dim_len_});
    result_shape.AppendShape(value_shape);

    // buffer will be pass to embedding service
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

    // TODO: FIXME call embedding service
    // Discuss the API args
    //
    FakeLookup(
        ctx, N, dim_len_, *out, var_name_, version_str, indices,
        make_lookup_callback<TValue>(ctx, N, dim_len_,
                                     default_values, *out,
                                     std::move(done)));
  }

 private:
  std::string var_name_;
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

const static size_t BUFFER_SIZE = 8 << 20; // 8MB

size_t LookupSegmentInternal(size_t key_size, size_t value_size,
                             int64 total_keys_num,
                             const std::string& tensor_key,
                             const std::string& tensor_value,
                             char* key_buffer, char* value_buffer,
                             BundleReader* reader,
                             Status& s_read) {
  size_t read_key_num = std::min(BUFFER_SIZE / key_size,
                                 BUFFER_SIZE / value_size);
  read_key_num = std::min((int64)read_key_num, total_keys_num);

  size_t key_bytes_read = 0, value_bytes_read = 0;
  if (total_keys_num > 0) {
    reader->LookupSegment(tensor_key, read_key_num * key_size,
                          key_buffer, key_bytes_read);
    reader->LookupSegment(tensor_value, read_key_num * value_size,
                          value_buffer, value_bytes_read);

    if (key_bytes_read > 0) {
      read_key_num = key_bytes_read / key_size;
      if (read_key_num != value_bytes_read / value_size) {
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

// TODO: FIXME for testing
typedef std::function<void(const Status&)> ImportCallback;

void FakeImport(char* key_buffer, char* value_buffer,
                size_t read_key_num, const std::string& var_name,
                const std::string& version_str, ImportCallback callback) {
  static int64 cur_done_num = 0;
  cur_done_num += read_key_num;
  LOG(INFO) << "KvImport: read key num = " << read_key_num << ", cur done num = " << cur_done_num;
  for (size_t i = 0; i < read_key_num; ++i) {
    LOG(INFO) << *((int64*)key_buffer+i) << " : " << *((float*)value_buffer+i);
  }

  Status s;
  callback(s);
}

ImportCallback make_import_callback(
    OpKernelContext* ctx,
    char* key_buffer,
    char* value_buffer,
    int64 total_keys_num,
    size_t key_size,
    size_t value_size,
    const std::string& tensor_key,
    const std::string& tensor_value,
    const std::string& var_name,
    const std::string& version_str,
    BundleReader* reader,
    AsyncOpKernel::DoneCallback done) {
  return [ctx, key_buffer, value_buffer, key_size, value_size,
          left_keys_num = total_keys_num, tensor_key, tensor_value, var_name,
          version_str, reader, done = std::move(done)](const Status& s) {

    int64 total_keys_num = left_keys_num;
    if (total_keys_num <= 0) {
      delete []key_buffer;
      delete []value_buffer;
      delete reader;

      ctx->SetStatus(s);
      done();
      return;
    }

    Status s_read = tensorflow::Status::OK();
    size_t read_key_num = LookupSegmentInternal(
        key_size, value_size, total_keys_num,
        tensor_key, tensor_value, key_buffer,
        value_buffer, reader, s_read);
    if (!s_read.ok()) {
      ctx->SetStatus(s_read);
      done();
      return;
    }
    total_keys_num -= read_key_num;

    // TODO: FIXME call embedding service here
    FakeImport(key_buffer, value_buffer,
               read_key_num, var_name,
               version_str, make_import_callback(
                   ctx, key_buffer, value_buffer,
                   total_keys_num, key_size,
                   value_size, tensor_key,
                   tensor_value, var_name, version_str,
                   reader, std::move(done)));

  };
}

}  // namespace

template <typename TKey, typename TValue>
class KvImportOp : public AsyncOpKernel {
 public:
  explicit KvImportOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("var_name", &var_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim_len", &dim_len_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    size_t key_size = sizeof(TKey);
    size_t value_size = sizeof(TValue);
    const Tensor& version = ctx->input(0);
    const std::string version_str = version.scalar<string>()();

    const Tensor& file_name = ctx->input(1);
    const std::string file_name_str = file_name.scalar<string>()();

    const Tensor& tensor_name = ctx->input(2);
    const std::string tensor_name_str = tensor_name.scalar<string>()();

    // create for read from file
    BundleReader* reader = new BundleReader(Env::Default(), file_name_str);
    OP_REQUIRES_OK(ctx, reader->status());

    std::string tensor_key = strings::StrCat(tensor_name_str, "-keys");
    std::string tensor_value = strings::StrCat(tensor_name_str, "-values");

    TensorShape key_shape, value_shape;
    reader->LookupTensorShape(tensor_key, &key_shape);
    reader->LookupTensorShape(tensor_value, &value_shape);

    Status s = reader->LookupHeader(tensor_key,
        key_size * key_shape.dim_size(0));
    OP_REQUIRES_OK(ctx, s);

    s = reader->LookupHeader(tensor_value,
        value_size * value_shape.dim_size(0) * value_shape.dim_size(1));
    OP_REQUIRES_OK(ctx, s);

    int64 total_keys_num = key_shape.dim_size(0);
    if (total_keys_num <= 0) {
      done();
      return;
    }

    char* key_buffer = new char[BUFFER_SIZE];
    char* value_buffer = new char[BUFFER_SIZE];
    size_t value_byte = value_size * value_shape.dim_size(1);

    Status s_read = tensorflow::Status::OK();
    size_t read_key_num = LookupSegmentInternal(
        key_size, value_byte, total_keys_num,
        tensor_key, tensor_value, key_buffer,
        value_buffer, reader, s_read);
    if (!s_read.ok()) {
      ctx->SetStatus(s_read);
      done();
      return;
    }

    total_keys_num -= read_key_num;

    // TODO: FIXME call embedding service here
    //  Discuss the API args
    //
    FakeImport(key_buffer, value_buffer,
               read_key_num, var_name_,
               version_str, make_import_callback(
                   ctx, key_buffer, value_buffer,
                   total_keys_num, key_size,
                   value_size, tensor_key,
                   tensor_value, var_name_, version_str,
                   reader, std::move(done)));
  }

 private:
  std::string var_name_;
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

}  // namespace processor
}  // namespace tensorflow
