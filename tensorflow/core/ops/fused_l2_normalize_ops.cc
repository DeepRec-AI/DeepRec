#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("FusedL2Normalize")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float}")
    .Attr("axis: int = 1")
    .Attr("epsilon: float = 1e-12")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
                    c->set_output(0, c->input(0));
                    return Status::OK();
                });
//     .Doc(R"doc(
// FusedL2Normalize ops.
//     )doc");

REGISTER_OP("FusedL2NormalizeGrad")
    .Input("y_grad: T")
    .Input("x: T")
    .Output("x_grad: T")
    .Attr("T: {float}")
    .Attr("axis: int = 1")
    .Attr("epsilon: float = 1e-12")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
                    c->set_output(0, c->input(0));
                    return Status::OK();
                });
//     .Doc(R"doc(
// FusedL2NormalizeGrad ops.
//     )doc");

}  // namespace tensorflow
