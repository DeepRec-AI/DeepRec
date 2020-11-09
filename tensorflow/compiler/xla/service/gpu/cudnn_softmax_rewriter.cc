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

#include "tensorflow/compiler/xla/service/gpu/cudnn_softmax_rewriter.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"

namespace xla {
namespace gpu {
namespace {

class Visitor : public DfsHloVisitorWithDefault {
 public:
  explicit Visitor(HloComputation* computation)
      : computation_(computation) {}

  static bool Run(HloComputation* computation) {
    Visitor visitor(computation);
    TF_CHECK_OK(computation->Accept(&visitor));
    return visitor.changed_;
  }

  Status DefaultAction(HloInstruction* /*hlo_instruction*/) override {
    return Status::OK();
  }

  Status HandleSoftmax(HloInstruction* softmax) override;

 private:
  bool changed_ = false;
  HloComputation* computation_;
};

Status Visitor::HandleSoftmax(HloInstruction* softmax) {
  VLOG(1) << softmax->ToString();
  if (softmax->operand(0)->shape().element_type() != F32 &&
      softmax->operand(0)->shape().element_type() != F16) {
    VLOG(1) << "Not rewriting op with non-F32 and non-F16 element type: "
            << softmax->ToString();
    return Status::OK();
  }
  // cudnn errors out on zero-sized inputs.
  if (ShapeUtil::ElementsIn(softmax->operand(0)->shape()) == 0) {
    return Status::OK();
  }

  HloInstruction* feature_index =
      computation_->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0(softmax->softmax_feature_index())));
  HloInstruction* log =
      computation_->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<bool>(softmax->log())));

  std::vector<HloInstruction*> operands(softmax->operands().begin(),
                                        softmax->operands().end());
  operands.push_back(feature_index);
  operands.push_back(log);

  auto softmax_shape = ShapeUtil::MakeShape(
      operands[0]->shape().element_type(), softmax->shape().dimensions());

  auto softmax_result = HloInstruction::CreateCustomCall(
      softmax_shape, operands, kCudnnSoftmaxCallTarget);

  TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
      softmax, std::move(softmax_result)));
  changed_ = true;
  return Status::OK();
}

}  // anonymous namespace

StatusOr<bool> CudnnSoftmaxRewriter::Run(HloModule* module) {
  VLOG(2) << "CudnnSoftmaxRewriter::Run(), before:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (Visitor::Run(comp)) {
      changed = true;
    }
  }

  VLOG(2) << "CudnnSoftmaxRewriter::Run(), after:";
  XLA_VLOG_LINES(2, module->ToString());
  return changed;
}

}  // namespace gpu
}  // namespace xla
