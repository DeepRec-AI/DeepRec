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

#include "tensorflow/compiler/xla/service/softmax_expander.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

namespace {

using absl::optional;

// BatchNormExpanderVisitor traverses the HLO computation and rewrites BatchNorm
// operations into smaller operations.
class SoftmaxExpanderVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleSoftmax(HloInstruction* softmax) override;

  // Runs the visitor on a computation.
  static bool Run(HloComputation* computation);

  ~SoftmaxExpanderVisitor() override = default;

 private:
  explicit SoftmaxExpanderVisitor(HloComputation* computation)
      : computation_(computation) {}

  HloComputation* GetOrCreateScalarMaxComputation(
      PrimitiveType primitive_type) {
    HloComputation::Builder b("scalar_max_computation");
    Shape shape = ShapeUtil::MakeShape(primitive_type, {});
    auto scalar_lhs = b.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "scalar_lhs"));
    auto scalar_rhs = b.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "scalar_rhs"));
    auto scalar_op = b.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kMaximum, scalar_lhs, scalar_rhs));
    return computation_->parent()->AddEmbeddedComputation(b.Build(scalar_op));
  }

  HloComputation* GetOrCreateScalarAddComputation(
      PrimitiveType primitive_type) {
    HloComputation::Builder b("scalar_add_computation");
    Shape shape = ShapeUtil::MakeShape(primitive_type, {});
    auto scalar_lhs = b.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "scalar_lhs"));
    auto scalar_rhs = b.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "scalar_rhs"));
    auto scalar_op = b.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, scalar_lhs, scalar_rhs));
    return computation_->parent()->AddEmbeddedComputation(b.Build(scalar_op));
  }
  // Current HloComputation instance the SoftmaxExpander is
  // traversing.
  HloComputation* computation_;
};

}  // namespace

bool SoftmaxExpanderVisitor::Run(HloComputation* computation) {
  SoftmaxExpanderVisitor visitor(computation);
  TF_CHECK_OK(computation->Accept(&visitor));
  return visitor.changed();
}

Status SoftmaxExpanderVisitor::HandleSoftmax(
    HloInstruction* softmax) {
  std::vector<HloInstruction*> added_instructions;
  auto add = [&](std::unique_ptr<HloInstruction> inst) {
    HloInstruction* added_inst = computation_->AddInstruction(std::move(inst));
    added_inst->set_metadata(softmax->metadata());
    added_instructions.push_back(added_inst);
    return added_inst;
  };
  auto add_binary = [&](const Shape& shape, const HloOpcode opcode,
                        HloInstruction* a, HloInstruction* b) {
    return add(HloInstruction::CreateBinary(shape, opcode, a, b));
  };
  int64 instruction_count_before = computation_->instruction_count();

  // Expand batch norm training into smaller HLO ops.
  HloInstruction* operand = softmax->mutable_operand(0);
  const Shape operand_shape = operand->shape();
  PrimitiveType ptype = operand_shape.element_type();
  int64 feature_index = softmax->softmax_feature_index();
  std::vector<int64> dimensions_without_feature;
  std::vector<int64> reduction_dimensions;

  for (int64 i = 0; i < operand_shape.rank(); ++i) {
    if (i != feature_index) {
      dimensions_without_feature.push_back(i);
      reduction_dimensions.push_back(operand_shape.dimensions(i));
    }
  }
  const Shape reduce_shape = ShapeUtil::MakeShape(ptype, reduction_dimensions);

  // max(operand) 
  auto minf_literal = LiteralUtil::MinValue(ptype);
  auto minf = add(HloInstruction::CreateConstant(std::move(minf_literal)));
  HloComputation* max_reduce_computation =
      GetOrCreateScalarMaxComputation(ptype);
  auto max = add(HloInstruction::CreateReduce(reduce_shape, operand, minf,
                                              {feature_index},
                                              max_reduce_computation));
  auto max_broadcasted =
      add(HloInstruction::CreateBroadcast(operand_shape, max,
                                          dimensions_without_feature));
  // operand - max(operand)
  auto sub =
      add_binary(operand_shape, HloOpcode::kSubtract, operand, max_broadcasted);
  // e^(operand - max(operand))
  auto exp = add(HloInstruction::CreateUnary(operand_shape, HloOpcode::kExp,
                                             sub));
  // sum(e^(operand - max(operand)))
  auto zero_literal = LiteralUtil::CreateR0(0.0f);
  TF_ASSIGN_OR_RETURN(zero_literal, zero_literal.Convert(ptype));
  auto zero = add(HloInstruction::CreateConstant(std::move(zero_literal)));
  HloComputation* add_reduce_computation =
      GetOrCreateScalarAddComputation(ptype);
  auto sum = add(HloInstruction::CreateReduce(reduce_shape, exp, zero,
                                              {feature_index},
                                              add_reduce_computation));
  auto sum_broadcasted =
      add(HloInstruction::CreateBroadcast(operand_shape, sum,
                                          dimensions_without_feature));
  // softmax result
  HloInstruction* result;
  if (softmax->log()) {
    auto log = add(HloInstruction::CreateUnary(operand_shape, HloOpcode::kLog,
                                               sum_broadcasted));
    result = add_binary(operand_shape, HloOpcode::kSubtract, sub, log);
  } else {
    result = add_binary(operand_shape, HloOpcode::kDivide, exp, sum_broadcasted);
  }
  int64 instruction_count_after = computation_->instruction_count();
  CHECK_EQ(instruction_count_after,
           instruction_count_before + added_instructions.size());

  TF_CHECK_OK(ReplaceInstruction(softmax, result));

  return Status::OK();
}

StatusOr<bool> SoftmaxExpander::Run(HloModule* module) {
  XLA_VLOG_LINES(2, "SoftmaxExpander::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (SoftmaxExpanderVisitor::Run(comp)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(2, "SoftmaxExpander::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
