/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cudnn_softmax_thunk.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_softmax_runner.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

CudnnSoftmaxThunk::CudnnSoftmaxThunk(
    const BufferAllocation::Slice& operand, int64 feature_index, bool log,
    const BufferAllocation::Slice& output, const HloInstruction* hlo)
    : Thunk(Thunk::Kind::kCudnnSoftmax, hlo),
      operand_(operand),
      feature_index_(feature_index),
      log_(log),
      output_(output) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kCustomCall);
  CHECK_EQ(hlo->custom_call_target(), kCudnnSoftmaxCallTarget);
  CHECK(
      LayoutUtil::LayoutsInShapesEqual(hlo->shape(), hlo->operand(0)->shape()));
  CHECK(hlo->operand(0)->shape().element_type() == F32 ||
        hlo->operand(0)->shape().element_type() == F16);
}

Status CudnnSoftmaxThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  auto& buffer_allocations = *params.buffer_allocations;
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());
  se::DeviceMemoryBase output_base =
      buffer_allocations.GetDeviceAddress(output_);
  se::DeviceMemoryBase operand = buffer_allocations.GetDeviceAddress(operand_);
  auto& stream = *params.stream;
  TF_RETURN_IF_ERROR(RunCudnnSoftmax(
      hlo_instruction(), operand, output_base,
      feature_index_, log_, &stream));

  if (!stream.ok()) {
    return InternalError("Softmax call failed.");
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
