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

#include "tensorflow/compiler/xla/service/gpu/async_out_thunk.h"

#include "tensorflow/compiler/jit/kernels/async_io_rendezvous.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"

namespace xla {
namespace gpu {

AsyncOutSendThunk::AsyncOutSendThunk(
    const BufferAllocation::Slice& input_buffer,
    const HloInstruction* hlo_instruction, const Shape& async_out_send_shape,
    std::string key)
    : Thunk(Kind::kAsyncOutSend, hlo_instruction),
      input_buffer_(input_buffer),
      async_out_send_shape_(async_out_send_shape),
      key_(std::move(key)) {
  key_hash_ = tensorflow::AsyncIoRendezvous::GetRendezvousKeyHash(key_);
}

Status AsyncOutSendThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto& buffer_allocations = *params.buffer_allocations;

  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());
  se::DeviceMemoryBase addr =
      buffer_allocations.GetDeviceAddress(input_buffer_);

  VLOG(4) << "AsyncOutSendThunk on GPU: " << hlo_instruction()->ToString()
          << " with buf size " << addr.size() << " @" << addr.opaque()
          << ", key " << key_ << ", hash " << key_hash_;

  tensorflow::AsyncIoRendezvous::TensorPayload payload;
  payload.addr = addr;
  CHECK(ShapeUtil::Equal(async_out_send_shape_,
                         hlo_instruction()->operand(0)->shape()));
  payload.shape = hlo_instruction()->operand(0)->shape();
  tensorflow::GetXlaAsyncIORendezvous()->Send(key_hash_, payload);

  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
