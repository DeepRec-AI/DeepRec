#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// Dice fusion op for inference
REGISTER_OP("Dice")
    .Input("x: T")
    .Input("mean: T")
    .Input("rvar: T")
    .Input("gamma: T")
    .Output("y: T")
    .Attr("T: {float}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
                    c->set_output(0, c->input(0));
                    return Status::OK();
                });

}  // namespace tensorflow
