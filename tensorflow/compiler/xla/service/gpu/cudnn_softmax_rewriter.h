#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_SOFTMAX_REWRITER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_SOFTMAX_REWRITER_H_

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

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrites Softmax HLOs into calls into cudnn where possible.
//
// A call into cudnn for performing a softmax op is represented as a
// CustomCall HLO with custom_call_target equal kCudnnSoftmaxCallTarget
//
// A CustomCall created by this pass has the same operands corresponding
// softmax HLO, except the softmax_feature_index() and log() properties of the
// softmax HLO are converted into proper operands, added to the end of the
// CustomCall's operands list.
//
// This pass adds HLOs in front of / behind the CustomCalls to fix up the
// inputs/outputs as appropriate, and we rely on the AlgebraicSimplifier to
// remove these where possible.
//
// The GPU backend does not implement a lowering for the softmax HLOs -- it
// expects them to be lowered to cudnn calls via this pass or to HLO soup via
// SoftmaxExpander.
class CudnnSoftmaxRewriter : public HloModulePass {
 public:
  CudnnSoftmaxRewriter() {}
  absl::string_view name() const override { return "cudnn_softmax_rewriter"; }
  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_SOFTMAX_REWRITER_H_
