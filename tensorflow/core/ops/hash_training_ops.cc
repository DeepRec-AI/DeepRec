/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

static ShapeHandle ShapeOrHandleShape(InferenceContext* c, int input) {
  auto* handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  return c->input(input);
}

// Handle the gradient and, if <sparse>, indices inputs.
// <s> is an input+output parameter, containing the current known input shape to
// the gradient.
static Status HandleGradAndIndicesInputs(InferenceContext* c,
                                         int grad_idx, ShapeHandle* s) {
  ShapeHandle grad = ShapeOrHandleShape(c, grad_idx);
  ShapeHandle indices = c->input(grad_idx + 1);
  if (c->RankKnown(indices) && c->RankKnown(*s) && c->RankKnown(grad)) {
    if (c->Rank(indices) + c->Rank(*s) - 1 != c->Rank(grad)) {
      return errors::InvalidArgument(
          "grad shape error ", c->DebugString(*s),
          c->DebugString(indices), c->DebugString(grad));
    }
  }
  if (c->RankKnown(indices)) {
    int64_t rank = c->Rank(indices);
    for (int i = 0; i < rank; i++) {
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, i), c->Dim(grad, i), &unused));
    }
  }
  if (c->RankKnown(*s) && c->RankKnown(grad)) {
    int64_t rank1 = c->Rank(*s), rank2 = c->Rank(grad);
    if (rank2 < rank1 - 1) {
      return errors::InvalidArgument(
          "grad shape error ", c->DebugString(*s),
          c->DebugString(indices), c->DebugString(grad));
    }
    for (int i = 0; i < rank1 - 1; i++) {
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(*s, i + 1), c->Dim(grad, i + (rank2 - rank1 + 1)), &unused));
    }
  }
  return Status::OK();
}

static Status OptimizerShapeFn(
    int64_t var, int64_t grad, const std::vector<int64_t>& slots,
    const std::vector<int64_t>& scalars, InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, var);
  for (auto slot : slots) {
    TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, slot), &s));
  }
  for (auto scalar : scalars) {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(scalar), 0, &unused));
  }
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, grad, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("TensibleVariableApplyGradientDescent")
    .Input("var: resource")
    .Input("alpha: T")
    .Input("delta: T")
    .Input("indices: int64")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
      [](InferenceContext* c){
        return OptimizerShapeFn(0, 2, {}, {1}, c);
      });

REGISTER_OP("TensibleVariableApplyProximalGradientDescent")
    .Input("var: resource")
    .Input("alpha: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("grad: T")
    .Input("indices: int64")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
      [](InferenceContext* c){
        return OptimizerShapeFn(0, 4, {}, {1, 2, 3}, c);
      });

REGISTER_OP("TensibleVariableApplyAdadelta")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("accum_update: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: int64")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
      [](InferenceContext* c){
        return OptimizerShapeFn(0, 6, {1, 2}, {3, 4, 5}, c);
      });

REGISTER_OP("TensibleVariableApplyAdagrad")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("grad: T")
    .Input("indices: int64")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
      [](InferenceContext* c){
        return OptimizerShapeFn(0, 3, {1}, {2}, c);
      });

REGISTER_OP("TensibleVariableApplyAdagradDA")
    .Input("var: resource")
    .Input("gradient_accumulator: resource")
    .Input("gradient_squared_accumulator: resource")
    .Input("grad: T")
    .Input("indices: int64")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("global_step: int64")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
      [](InferenceContext* c){
        return OptimizerShapeFn(0, 3, {1, 2}, {5, 6, 7, 8}, c);
      });

REGISTER_OP("TensibleVariableApplyProximalAdagrad")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("grad: T")
    .Input("indices: int64")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
      [](InferenceContext* c){
        return OptimizerShapeFn(0, 5, {1}, {2, 3, 4}, c);
      });

REGISTER_OP("TensibleVariableApplyFtrl")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("grad: T")
    .Input("indices: int64")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("lr_power: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
      [](InferenceContext* c){
        return OptimizerShapeFn(0, 3, {1, 2}, {5, 6, 7, 8}, c);
      });

REGISTER_OP("TensibleVariableApplyFtrlV2")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("linear: resource")
    .Input("grad: T")
    .Input("indices: int64")
    .Input("lr: T")
    .Input("l1: T")
    .Input("l2: T")
    .Input("l2_shrinkage: T")
    .Input("lr_power: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
      [](InferenceContext* c){
        return OptimizerShapeFn(0, 3, {1, 2}, {5, 6, 7, 8, 9}, c);
      });

REGISTER_OP("TensibleVariableApplyMomentum")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("lr: T")
    .Input("grad: T")
    .Input("indices: int64")
    .Input("momentum: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn(
      [](InferenceContext* c){
        return OptimizerShapeFn(0, 3, {1}, {2, 5}, c);
      });

REGISTER_OP("TensibleVariableApplyAdam")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: int64")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn(
      [](InferenceContext* c){
        return OptimizerShapeFn(0, 9, {1, 2}, {3, 4, 5, 6, 7, 8}, c);
      });

REGISTER_OP("TensibleVariableApplyRMSProp")
    .Input("var: resource")
    .Input("ms: resource")
    .Input("mom: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: int64")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
      [](InferenceContext* c){
        return OptimizerShapeFn(0, 7, {1, 2}, {3, 4, 5, 6}, c);
      });

REGISTER_OP("TensibleVariableApplyCenteredRMSProp")
    .Input("var: resource")
    .Input("mg: resource")
    .Input("ms: resource")
    .Input("mom: resource")
    .Input("lr: T")
    .Input("rho: T")
    .Input("momentum: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: int64")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(
      [](InferenceContext* c){
        return OptimizerShapeFn(0, 8, {1, 2, 3}, {4, 5, 6, 7}, c);
      });

}  // namespace tensorflow
