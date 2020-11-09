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

#include "tensorflow/compiler/xla/service/gpu/cudnn_softmax_runner.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {
namespace {

struct CudnnSoftmaxParams {
  se::DeviceMemoryBase operand;
  se::DeviceMemoryBase output;
  se::dnn::BatchDescriptor operand_desc;
  bool log;
};

void AssignParams(const HloInstruction* softmax,
                        CudnnSoftmaxParams* params,
                        const se::DeviceMemoryBase& operand,
                        se::DeviceMemoryBase& output,
                        int64 feature_index, bool log) {
  const Shape& shape = softmax->shape();
  se::dnn::BatchDescriptor input_desc =
      MakeSoftmaxDescriptor(shape, feature_index);
  params->operand = operand;
  params->output = output;
  params->operand_desc = input_desc;
  params->log = log;
}

template <typename ElemType>
void RunCudnnSoftmaxImpl(
    CudnnSoftmaxParams* params, se::Stream* stream) {
  auto output_buf = se::DeviceMemory<ElemType>(params->output);
  stream->ThenSoftmax(
      se::DeviceMemory<ElemType>(params->operand),
      params->operand_desc,
      params->log,
      &output_buf);
}

}  // namespace

Status RunCudnnSoftmax(
    const HloInstruction* softmax, se::DeviceMemoryBase operand,
    se::DeviceMemoryBase output, int64 feature_index, bool log,
    se::Stream* stream) {
  CudnnSoftmaxParams params;
  AssignParams(softmax, &params, operand, output, feature_index, log);

  PrimitiveType output_primitive_type = softmax->shape().element_type();
  switch (output_primitive_type) {
    case F16:
      RunCudnnSoftmaxImpl<Eigen::half>(&params, stream);
      break;
    case F32:
      RunCudnnSoftmaxImpl<float>(&params, stream);
      break;
    default:
      return Unimplemented("Primitive type not implemented for \"%s\" ",
                           softmax->ToString());
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
