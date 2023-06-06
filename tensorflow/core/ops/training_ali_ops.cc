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
static Status HandleGradAndIndicesInputs(InferenceContext* c, bool sparse,
                                         int grad_idx, ShapeHandle* s) {
  ShapeHandle grad = ShapeOrHandleShape(c, grad_idx);
  if (!sparse) {
    TF_RETURN_IF_ERROR(c->Merge(*s, grad, s));
    return Status::OK();
  }
  // Indices is a vector where indices.dim[0].rank == grad[0].rank.
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(grad_idx + 1), 1, &indices));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused));

  // Trailing part of grad matches trailing part of *s.
  ShapeHandle grad_unknown_first;
  TF_RETURN_IF_ERROR(
      c->ReplaceDim(grad, 0, c->UnknownDim(), &grad_unknown_first));
  TF_RETURN_IF_ERROR(c->Merge(*s, grad_unknown_first, s));

  return Status::OK();
}

static Status HandleKvGradAndIndicesInputs(InferenceContext* c, bool sparse,
                                         int grad_idx, ShapeHandle* s) {
  ShapeHandle grad = ShapeOrHandleShape(c, grad_idx);
  if (!sparse) {
    TF_RETURN_IF_ERROR(c->Merge(*s, grad, s));
    return Status::OK();
  }
  // Indices is a vector where indices.dim[0].rank == grad[0].rank.
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(grad_idx + 1), 1, &indices));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused));

  // Trailing part of grad matches trailing part of *s.
  ShapeHandle grad_unknown_first;
  TF_RETURN_IF_ERROR(
      c->Subshape(grad, 1, &grad_unknown_first));
  TF_RETURN_IF_ERROR(c->Merge(*s, grad_unknown_first, s));

  return Status::OK();
}

static Status KvResourceApplyAdagradShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(
      HandleKvGradAndIndicesInputs(c, sparse, 3 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

#define REGISTER_OP_BY_NAME(name)              \
REGISTER_OP(name)                              \
    .Input("var: resource")                    \
    .Input("accum: resource")                  \
    .Input("lr: T")                            \
    .Input("grad: T")                          \
    .Input("indices: Tindices")                \
    .Input("global_step: Tstep")               \
    .Attr("T: numbertype")                     \
    .Attr("Tindices: {int32, int64}")          \
    .Attr("Tstep: {int32, int64}")             \
    .Attr("use_locking: bool = false")         \
    .SetShapeFn([](InferenceContext* c) {      \
      return KvResourceApplyAdagradShapeFn(c, true /* sparse */); \
    })                                         \
    .Doc(R"doc()doc")
REGISTER_OP_BY_NAME("KvResourceSparseApplyAdagrad");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyAdagrad");
#undef REGISTER_OP_BY_NAME

#define REGISTER_OP_BY_NAME(name)              \
REGISTER_OP(name)                              \
    .Input("var: resource")                    \
    .Input("accum: resource")                  \
    .Input("lr: T")                            \
    .Input("grad: T")                          \
    .Input("indices: Tindices")                \
    .Input("global_step: Tstep")               \
    .Input("indices_counts: int64")            \
    .Attr("T: numbertype")                     \
    .Attr("Tindices: {int32, int64}")          \
    .Attr("Tstep: {int32, int64}")             \
    .Attr("use_locking: bool = false")         \
    .SetShapeFn([](InferenceContext* c) {      \
      return KvResourceApplyAdagradShapeFn(c, true /* sparse */); \
    })                                         \
    .Doc(R"doc()doc")
REGISTER_OP_BY_NAME("KvResourceSparseApplyAdagradWithCounts");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyAdagradWithCounts");
#undef REGISTER_OP_BY_NAME

static Status KvResourceApplyFtrlShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // linear
  TF_RETURN_IF_ERROR(
      HandleKvGradAndIndicesInputs(c, sparse, 3 /* grad_idx */, &s));
  int idx = sparse ? 5 : 4;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // l2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(idx++), 0, &unused));  // lr_power
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

#define REGISTER_OP_BY_NAME(name)              \
REGISTER_OP(name)                              \
    .Input("var: resource")                    \
    .Input("accum: resource")                  \
    .Input("linear: resource")                 \
    .Input("grad: T")                          \
    .Input("indices: Tindices")                \
    .Input("lr: T")                            \
    .Input("l1: T")                            \
    .Input("l2: T")                            \
    .Input("lr_power: T")                      \
    .Attr("T: numbertype")                     \
    .Attr("Tindices: {int32, int64, string}")  \
    .Attr("use_locking: bool = false")         \
    .Attr("indices_as_pointer: bool = false")  \
    .SetShapeFn([](InferenceContext* c) {      \
      return KvResourceApplyFtrlShapeFn(c, true /* sparse */);\
    })                                         \
    .Doc(R"doc()doc")
REGISTER_OP_BY_NAME("KvResourceSparseApplyFtrl");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyFtrl");
#undef REGISTER_OP_BY_NAME

#define REGISTER_OP_BY_NAME(name)              \
REGISTER_OP(name)                              \
    .Input("var: resource")                    \
    .Input("accum: resource")                  \
    .Input("linear: resource")                 \
    .Input("grad: T")                          \
    .Input("indices: Tindices")                \
    .Input("lr: T")                            \
    .Input("l1: T")                            \
    .Input("l2: T")                            \
    .Input("lr_power: T")                      \
    .Input("indices_counts: int64")              \
    .Attr("T: numbertype")                     \
    .Attr("Tindices: {int32, int64, string}")  \
    .Attr("use_locking: bool = false")         \
    .Attr("indices_as_pointer: bool = false")  \
    .SetShapeFn([](InferenceContext* c) {      \
      return KvResourceApplyFtrlShapeFn(c, true /* sparse */);\
    })                                         \
    .Doc(R"doc()doc")
REGISTER_OP_BY_NAME("KvResourceSparseApplyFtrlWithCounts");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyFtrlWithCounts");
#undef REGISTER_OP_BY_NAME

#define REGISTER_OP_BY_NAME(name)              \
REGISTER_OP(name)                              \
    .Input("var: resource")                    \
    .Input("accum: resource")                  \
    .Input("linear: resource")                 \
    .Input("grad: T")                          \
    .Input("indices: Tindices")                \
    .Input("lr: T")                            \
    .Input("l1: T")                            \
    .Input("l2: T")                            \
    .Input("l2_shrinkage: T")                  \
    .Input("lr_power: T")                      \
    .Attr("T: numbertype")                     \
    .Attr("Tindices: {int32, int64, string}")  \
    .Attr("use_locking: bool = false")         \
    .Attr("indices_as_pointer: bool = false")  \
    .SetShapeFn([](InferenceContext* c) {      \
      return KvResourceApplyFtrlShapeFn(c, true /* sparse */);\
    })
REGISTER_OP_BY_NAME("KvResourceSparseApplyFtrlV2");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyFtrlV2");
#undef REGISTER_OP_BY_NAME

#define REGISTER_OP_BY_NAME(name)              \
REGISTER_OP(name)                              \
    .Input("var: resource")                    \
    .Input("accum: resource")                  \
    .Input("linear: resource")                 \
    .Input("grad: T")                          \
    .Input("indices: Tindices")                \
    .Input("lr: T")                            \
    .Input("l1: T")                            \
    .Input("l2: T")                            \
    .Input("l2_shrinkage: T")                  \
    .Input("lr_power: T")                      \
    .Input("indices_counts: int64")              \
    .Attr("T: numbertype")                     \
    .Attr("Tindices: {int32, int64, string}")  \
    .Attr("use_locking: bool = false")         \
    .Attr("indices_as_pointer: bool = false")  \
    .SetShapeFn([](InferenceContext* c) {      \
      return KvResourceApplyFtrlShapeFn(c, true /* sparse */);\
    })
REGISTER_OP_BY_NAME("KvResourceSparseApplyFtrlV2WithCounts");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyFtrlV2WithCounts");
#undef REGISTER_OP_BY_NAME

static Status ApplyAdagradDecayShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  // AdagradDecayOptimizer & AdagradDecayOptimizerV2 has different shape
  // of accum_decay_power. DO NOT check shape of input(2) here.
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // decay_step
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));       // decay_rate
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));       // baseline
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));       // global_step
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 8 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyAdagradDecay")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("accum_decay_power: Ref(Tstep)")
    .Input("lr: T")
    .Input("accum_decay_step: Tstep")
    .Input("accum_decay_rate: T")
    .Input("accum_baseline: T")
    .Input("global_step: Tstep")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tstep: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdagradDecayShapeFn(c, false /* sparse */);
    })
    .Doc(R"doc(
)doc");

REGISTER_OP("ResourceApplyAdagradDecay")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("accum_decay_power: resource")
    .Input("lr: T")
    .Input("accum_decay_step: Tstep")
    .Input("accum_decay_rate: T")
    .Input("accum_baseline: T")
    .Input("global_step: Tstep")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("Tstep: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdagradDecayShapeFn(c, false /* sparse */);
    })
    .Doc(R"doc(
)doc");

REGISTER_OP("SparseApplyAdagradDecay")
    .Input("var: Ref(T)")
    .Input("accum: Ref(T)")
    .Input("accum_decay_power: Ref(Tstep)")
    .Input("lr: T")
    .Input("accum_decay_step: Tstep")
    .Input("accum_decay_rate: T")
    .Input("accum_baseline: T")
    .Input("global_step: Tstep")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("Tstep: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdagradDecayShapeFn(c, true /* sparse */);
    })
    .Doc(R"doc(
)doc");

REGISTER_OP("ResourceSparseApplyAdagradDecay")
    .Input("var: resource")
    .Input("accum: resource")
    .Input("accum_decay_power: resource")
    .Input("lr: T")
    .Input("accum_decay_step: Tstep")
    .Input("accum_decay_rate: T")
    .Input("accum_baseline: T")
    .Input("global_step: Tstep")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("Tstep: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdagradDecayShapeFn(c, true /* sparse */);
    })
    .Doc(R"doc(
)doc");

static Status KvApplyAdagradDecayShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // accum
  // AdagradDecayOptimizer & AdagradDecayOptimizerV2 has different shape
  // of accum_decay_power. DO NOT check shape of input(2) here.
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // decay_step
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));       // decay_rate
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));       // baseline
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));       // global_step
  TF_RETURN_IF_ERROR(
      HandleKvGradAndIndicesInputs(c, sparse, 8 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

#define REGISTER_OP_BY_NAME(name)              \
REGISTER_OP(name)                              \
    .Input("var: resource")                    \
    .Input("accum: resource")                  \
    .Input("accum_decay_power: resource")      \
    .Input("lr: T")                            \
    .Input("accum_decay_step: Tstep")          \
    .Input("accum_decay_rate: T")              \
    .Input("accum_baseline: T")                \
    .Input("global_step: Tstep")               \
    .Input("grad: T")                          \
    .Input("indices: Tindices")                \
    .Attr("T: numbertype")                     \
    .Attr("Tindices: {int32, int64}")          \
    .Attr("Tstep: {int32, int64}")             \
    .Attr("use_locking: bool = false")         \
    .Attr("indices_as_pointer: bool = false")  \
    .SetShapeFn([](InferenceContext* c) {      \
      return KvApplyAdagradDecayShapeFn(c, true /* sparse */);\
    })                                         \
    .Doc(R"doc()doc")
REGISTER_OP_BY_NAME("KvResourceSparseApplyAdagradDecay");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyAdagradDecay");
#undef REGISTER_OP_BY_NAME

#define REGISTER_OP_BY_NAME(name)              \
REGISTER_OP(name)                              \
    .Input("var: resource")                    \
    .Input("accum: resource")                  \
    .Input("accum_decay_power: resource")      \
    .Input("lr: T")                            \
    .Input("accum_decay_step: Tstep")          \
    .Input("accum_decay_rate: T")              \
    .Input("accum_baseline: T")                \
    .Input("global_step: Tstep")               \
    .Input("grad: T")                          \
    .Input("indices: Tindices")                \
    .Input("indices_counts: int64")              \
    .Attr("T: numbertype")                     \
    .Attr("Tindices: {int32, int64}")          \
    .Attr("Tstep: {int32, int64}")             \
    .Attr("use_locking: bool = false")         \
    .Attr("indices_as_pointer: bool = false")  \
    .SetShapeFn([](InferenceContext* c) {      \
      return KvApplyAdagradDecayShapeFn(c, true /* sparse */);\
    })                                         \
    .Doc(R"doc()doc")
REGISTER_OP_BY_NAME("KvResourceSparseApplyAdagradDecayWithCounts");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyAdagradDecayWithCounts");
#undef REGISTER_OP_BY_NAME

static Status ApplyAdamAsyncShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));       // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));       // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));       // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));       // epsilon
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 9 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ApplyAdamAsync")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("beta1_power: Ref(T)")
    .Input("beta2_power: Ref(T)")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdamAsyncShapeFn(c, false /* sparse */);
    });

REGISTER_OP("SparseApplyAdamAsync")
    .Input("var: Ref(T)")
    .Input("m: Ref(T)")
    .Input("v: Ref(T)")
    .Input("beta1_power: Ref(T)")
    .Input("beta2_power: Ref(T)")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Attr("apply_sparse_rmsprop: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdamAsyncShapeFn(c, true /* sparse */);
    });

REGISTER_OP("ResourceApplyAdamAsync")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("beta1_power: resource")
    .Input("beta2_power: resource")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdamAsyncShapeFn(c, false /* sparse */);
    });

REGISTER_OP("ResourceSparseApplyAdamAsync")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("beta1_power: resource")
    .Input("beta2_power: resource")
    .Input("lr: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .Attr("apply_sparse_rmsprop: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdamAsyncShapeFn(c, true /* sparse */);
    });

static Status KvResourceApplyAdamShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));       // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));       // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));       // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));       // epsilon
  TF_RETURN_IF_ERROR(
      HandleKvGradAndIndicesInputs(c, sparse, 9 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

#define REGISTER_OP_BY_NAME(name)              \
REGISTER_OP(name)                              \
    .Input("var: resource")                    \
    .Input("m: resource")                      \
    .Input("v: resource")                      \
    .Input("beta1_power: T")                   \
    .Input("beta2_power: T")                   \
    .Input("lr: T")                            \
    .Input("beta1: T")                         \
    .Input("beta2: T")                         \
    .Input("epsilon: T")                       \
    .Input("grad: T")                          \
    .Input("indices: Tindices")                \
    .Input("global_step: Tstep")               \
    .Attr("T: numbertype")                     \
    .Attr("Tindices: {int32, int64, string}")  \
    .Attr("Tstep: {int32, int64}")             \
    .Attr("use_locking: bool = false")         \
    .Attr("indices_as_pointer: bool = false")  \
    .SetShapeFn([](InferenceContext* c) {      \
      return KvResourceApplyAdamShapeFn(c, true /* sparse */);\
    })                                         \
    .Doc(R"doc()doc")
REGISTER_OP_BY_NAME("KvResourceSparseApplyAdam");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyAdam");
#undef REGISTER_OP_BY_NAME

#define REGISTER_OP_BY_NAME(name)              \
REGISTER_OP(name)                              \
    .Input("var: resource")                    \
    .Input("m: resource")                      \
    .Input("v: resource")                      \
    .Input("beta1_power: T")                   \
    .Input("beta2_power: T")                   \
    .Input("lr: T")                            \
    .Input("beta1: T")                         \
    .Input("beta2: T")                         \
    .Input("epsilon: T")                       \
    .Input("grad: T")                          \
    .Input("indices: Tindices")                \
    .Input("global_step: Tstep")               \
    .Input("indices_counts: int64")              \
    .Attr("T: numbertype")                     \
    .Attr("Tindices: {int32, int64, string}")  \
    .Attr("Tstep: {int32, int64}")             \
    .Attr("use_locking: bool = false")         \
    .Attr("indices_as_pointer: bool = false")  \
    .SetShapeFn([](InferenceContext* c) {      \
      return KvResourceApplyAdamShapeFn(c, true /* sparse */);\
    })                                         \
    .Doc(R"doc()doc")
REGISTER_OP_BY_NAME("KvResourceSparseApplyAdamWithCounts");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyAdamWithCounts");
#undef REGISTER_OP_BY_NAME

static Status KvApplyAdamAsyncShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));       // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));       // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));       // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));       // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));       // epsilon
  TF_RETURN_IF_ERROR(
      HandleKvGradAndIndicesInputs(c, sparse, 9 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

#define REGISTER_OP_BY_NAME(name)              \
REGISTER_OP(name)                              \
    .Input("var: resource")                    \
    .Input("m: resource")                      \
    .Input("v: resource")                      \
    .Input("beta1_power: resource")            \
    .Input("beta2_power: resource")            \
    .Input("lr: T")                            \
    .Input("beta1: T")                         \
    .Input("beta2: T")                         \
    .Input("epsilon: T")                       \
    .Input("grad: T")                          \
    .Input("indices: Tindices")                \
    .Input("global_step: Tstep")               \
    .Attr("T: numbertype")                     \
    .Attr("Tindices: {int32, int64}")          \
    .Attr("Tstep: {int32, int64}")             \
    .Attr("use_locking: bool = false")         \
    .Attr("apply_sparse_rmsprop: bool = false")\
    .Attr("indices_as_pointer: bool = false")  \
    .SetShapeFn([](InferenceContext* c) {      \
      return KvApplyAdamAsyncShapeFn(c, true /* sparse */);\
    })
REGISTER_OP_BY_NAME("KvResourceSparseApplyAdamAsync");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyAdamAsync");
#undef REGISTER_OP_BY_NAME

#define REGISTER_OP_BY_NAME(name)              \
REGISTER_OP(name)                              \
    .Input("var: resource")                    \
    .Input("m: resource")                      \
    .Input("v: resource")                      \
    .Input("beta1_power: resource")            \
    .Input("beta2_power: resource")            \
    .Input("lr: T")                            \
    .Input("beta1: T")                         \
    .Input("beta2: T")                         \
    .Input("epsilon: T")                       \
    .Input("grad: T")                          \
    .Input("indices: Tindices")                \
    .Input("global_step: Tstep")               \
    .Input("indices_counts: int64")              \
    .Attr("T: numbertype")                     \
    .Attr("Tindices: {int32, int64}")          \
    .Attr("Tstep: {int32, int64}")             \
    .Attr("use_locking: bool = false")         \
    .Attr("apply_sparse_rmsprop: bool = false")\
    .Attr("indices_as_pointer: bool = false")  \
    .SetShapeFn([](InferenceContext* c) {      \
      return KvApplyAdamAsyncShapeFn(c, true /* sparse */);\
    })
REGISTER_OP_BY_NAME("KvResourceSparseApplyAdamAsyncWithCounts");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyAdamAsyncWithCounts");
#undef REGISTER_OP_BY_NAME

static Status KvApplyGradientDescentShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));  // alpha
  ShapeHandle grad = ShapeOrHandleShape(c, 2);
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &indices));
  DimensionHandle unused2;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused2));
  return Status::OK();
}

#define REGISTER_OP_BY_NAME(name)              \
REGISTER_OP(name)                              \
    .Input("var: resource")                    \
    .Input("alpha: T")                         \
    .Input("grad: T")                          \
    .Input("indices: Tindices")                \
    .Input("global_step: Tstep")               \
    .Attr("T: numbertype")                     \
    .Attr("Tindices: {int32, int64}")          \
    .Attr("Tstep: {int32, int64}")             \
    .Attr("use_locking: bool = false")         \
    .Attr("indices_as_pointer: bool = false")  \
    .SetShapeFn(KvApplyGradientDescentShapeFn)
REGISTER_OP_BY_NAME("KvResourceSparseApplyGradientDescent");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyGradientDescent");
#undef REGISTER_OP_BY_NAME

#define REGISTER_OP_BY_NAME(name)              \
REGISTER_OP(name)                              \
    .Input("var: resource")                    \
    .Input("alpha: T")                         \
    .Input("grad: T")                          \
    .Input("indices: Tindices")                \
    .Input("global_step: Tstep")               \
    .Input("counts: int64")                    \
    .Attr("T: numbertype")                     \
    .Attr("Tindices: {int32, int64}")          \
    .Attr("Tstep: {int32, int64}")             \
    .Attr("use_locking: bool = false")         \
    .Attr("indices_as_pointer: bool = false")  \
    .SetShapeFn(KvApplyGradientDescentShapeFn)
REGISTER_OP_BY_NAME("KvResourceSparseApplyGradientDescentWithCounts");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyGradientDescentWithCounts");
#undef REGISTER_OP_BY_NAME

#define REGISTER_OP_BY_NAME(name)              \
REGISTER_OP(name)                              \
    .Input("var: resource")                    \
    .Input("m: resource")                      \
    .Input("v: resource")                      \
    .Input("beta1_power: T")                   \
    .Input("beta2_power: T")                   \
    .Input("lr: T")                            \
    .Input("beta1: T")                         \
    .Input("beta2: T")                         \
    .Input("epsilon: T")                       \
    .Input("grad: T")                          \
    .Input("indices: Tindices")                \
    .Input("global_step: Tstep")               \
    .Input("weight_decay: T")                  \
    .Attr("T: numbertype")                     \
    .Attr("Tindices: {int32, int64, string}")  \
    .Attr("Tstep: {int32, int64}")             \
    .Attr("use_locking: bool = false")         \
    .Attr("indices_as_pointer: bool = false")  \
    .SetShapeFn([](InferenceContext* c) {      \
      return KvResourceApplyAdamShapeFn(c, true /* sparse */);\
    })                                         \
    .Doc(R"doc()doc")
REGISTER_OP_BY_NAME("KvResourceSparseApplyAdamW");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyAdamW");
#undef REGISTER_OP_BY_NAME

#define REGISTER_OP_BY_NAME(name)              \
REGISTER_OP(name)                              \
    .Input("var: resource")                    \
    .Input("m: resource")                      \
    .Input("v: resource")                      \
    .Input("beta1_power: T")                   \
    .Input("beta2_power: T")                   \
    .Input("lr: T")                            \
    .Input("beta1: T")                         \
    .Input("beta2: T")                         \
    .Input("epsilon: T")                       \
    .Input("grad: T")                          \
    .Input("indices: Tindices")                \
    .Input("global_step: Tstep")               \
    .Input("weight_decay: T")                  \
    .Input("indices_counts: int64")              \
    .Attr("T: numbertype")                     \
    .Attr("Tindices: {int32, int64, string}")  \
    .Attr("Tstep: {int32, int64}")             \
    .Attr("use_locking: bool = false")         \
    .Attr("indices_as_pointer: bool = false")  \
    .SetShapeFn([](InferenceContext* c) {      \
      return KvResourceApplyAdamShapeFn(c, true /* sparse */);\
    })                                         \
    .Doc(R"doc()doc")
REGISTER_OP_BY_NAME("KvResourceSparseApplyAdamWWithCounts");
REGISTER_OP_BY_NAME("_OPT_KvResourceSparseApplyAdamWWithCounts");
#undef REGISTER_OP_BY_NAME

}  // namespace tensorflow
