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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_SOFTMAX_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_SOFTMAX_THUNK_H_

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace gpu {

// This file contains the thunks which calls into cudnn to run
// softmax: Softmax, known to cudnn as SoftmaxForward.
//
// As an alternative to using these thunks, XLA can decompose the softmax HLO
// into smaller components using the SoftmaxRewriter pass.  This can result in
// faster code because those individual components can fuse into their
// inputs/outputs, but it may also be slower if cudnn's softmax implementation
// outperforms the code XLA generates for these components.
//

class CudnnSoftmaxThunk : public Thunk {
 public:
  CudnnSoftmaxThunk(const BufferAllocation::Slice& operand,
                    int64 feature_index, bool log,
                    const BufferAllocation::Slice& output,
                    const HloInstruction* hlo);

  CudnnSoftmaxThunk(
      const CudnnSoftmaxThunk&) = delete;
  CudnnSoftmaxThunk& operator=(
      const CudnnSoftmaxThunk&) = delete;

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  BufferAllocation::Slice operand_;
  int64 feature_index_;
  bool log_;
  BufferAllocation::Slice output_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_SOFTMAX_THUNK_H_
