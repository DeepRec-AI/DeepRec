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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ASYNC_OUT_SEND_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ASYNC_OUT_SEND_THUNK_H_

#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace gpu {

// A thunk that asynchronously sends out data. This thunk performs no copies
// at all but just passes the device pointer to a corresponding AsyncOutRecv op.
class AsyncOutSendThunk : public Thunk {
 public:
  // Constructs a AsyncOutSendThunk that sends out data to a corresponding
  // AsyncOutRecv on the same device.
  AsyncOutSendThunk(const BufferAllocation::Slice& input_buffer,
                    const HloInstruction* hlo_instruction,
                    const Shape& async_out_send_shape, std::string key);

  AsyncOutSendThunk(const AsyncOutSendThunk&) = delete;
  AsyncOutSendThunk& operator=(const AsyncOutSendThunk&) = delete;

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const BufferAllocation::Slice input_buffer_;
  Shape async_out_send_shape_;
  std::string key_;
  uint64 key_hash_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ASYNC_OUT_SEND_THUNK_H_
