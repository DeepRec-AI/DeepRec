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

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace processor {

namespace {

/*
// TODO: FIXME
template <typename TValue>
EmbeddingService::DoneCallback lookup_callback(
    OpKernelContext* ctx,
    int64 N, // indices count
    int64 dim_len, // len of dim 1
    Tensor* allocated_out_tensor,
    const std::string& var_name,
    const std::string& version_str,
    AsyncOpKernel::DoneCallback done) {
  return [ctx, default_values_matrix = std::move(default_values_matrix),
          done = std::move(done)](const Status& s,
                                  const Tensor& val,
                                  std::vector<int> not_found_ids_offset) {
    
    // fill default value here
    if (not_found_ids_offset.size() > 0) {
      auto out_flat = allocated_out_tensor->shaped<TValue, 2>(
          {N, allocated_out_tensor->NumElements() / N});
      TValue* out_base = &out_flat(0, 0);
      for (int i = 0; i < not_found_ids_offset.size(); ++i) {
        TValue* default_v = &default_values_matrix(not_found_ids_offset[i], 0);
        TValue* pointer = out_base + not_found_ids_offset[i] * dim_len;
        for (int j = 0; j < dim_len; ++j) {
          *(pointer+j) = *(default_v+j);
        }
      }
    }

    ctx->SetStatus(s);

    done();
  };
}
*/

}  // namespace

template <typename TKey, typename TValue>
class KvLookupOp : public AsyncOpKernel {
 public:
  explicit KvLookupOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("var_name", &var_name_));
    OP_REQUIRES_OK(context, context->GetAttr("dim_len", &dim_len_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const Tensor& version = ctx->input(0);
    const std::string version_str = version.scalar<string>()();

    const Tensor& indices = ctx->input(1);
    const int64 N = indices.NumElements();

    Tensor default_values(ctx->input(2));
    auto default_values_matrix = default_values.shaped<TValue, 2>(
        {default_values.NumElements()/dim_len_, dim_len_});

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
    // EmbeddingService.Lookup(
    //   N, dim_len_, out, var_name_, version_str,
    //   lookup_callback(ctx, std::move(default_values_matrix), std::move(done)));
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


class KvInsertOp : public AsyncOpKernel {
 public:
  using AsyncOpKernel::AsyncOpKernel;

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
  }
};

}  // namespace processor
}  // namespace tensorflow
