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

#include <string>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

namespace {

std::vector<string> Split(const string& str, const string& delimiter) {
  if (delimiter.size()) {
    return str_util::Split(str, delimiter, str_util::SkipEmpty());
  }
  // Empty delimiter means split the input character by character.
  std::vector<string> char_vector(str.size());
  for (size_t i = 0; i < str.size(); ++i) {
    char_vector[i] = str[i];
  }
  return char_vector;
}

}  // namespace

class StringSplitAndPadOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                errors::InvalidArgument("input must be a vector, got shape: ",
                                        input_tensor->shape().DebugString()));

    const auto input_vec = input_tensor->vec<string>();
    const int64 batch_size = input_vec.dimension(0);

    const Tensor* delimiter_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("delimiter", &delimiter_tensor));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(delimiter_tensor->shape()),
        errors::InvalidArgument("delimiter must scalar, got shape: ",
                                delimiter_tensor->shape().DebugString()));
    const auto delimiter_vec = delimiter_tensor->flat<string>();
    const string& delimiter = delimiter_vec(0);

    const Tensor* max_length_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("max_length", &max_length_tensor));
    int64 max_length = max_length_tensor->scalar<int64>()();

    const Tensor* default_value_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("default_value", &default_value_tensor));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(default_value_tensor->shape()),
        errors::InvalidArgument("default_value must scalar, got shape: ",
                                default_value_tensor->shape().DebugString()));
    const auto default_value_vec = default_value_tensor->flat<string>();
    const string& default_value = default_value_vec(0);

    Tensor* output_values_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({batch_size, max_length}),
                                  &output_values_tensor));
    auto output_values = output_values_tensor->matrix<string>();

    for (int64 i = 0; i < batch_size; ++i) {
      std::vector<string> parts = Split(input_vec(i), delimiter);
      int64 n_entries = max_length < parts.size() ? max_length : parts.size();
      int64 j = 0;
      for (; j < n_entries; ++j) {
        output_values(i, j) = parts[j];
      }
      for (; j < max_length; ++j) {
        output_values(i, j) = default_value;
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("StringSplitAndPad").Device(DEVICE_CPU),
                        StringSplitAndPadOp);

}  // namespace tensorflow
