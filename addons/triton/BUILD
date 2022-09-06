# Description:
# TRITON API.

cc_library(
    name = "triton_tf",
    visibility = ["//visibility:public"],
    srcs = [
        "tensorflow_backend_tf.cc"
    ],
    hdrs = [
        "tensorflow_backend_tf.h"
    ],
    deps = [
        "//tensorflow/core:core",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:tag_constants",
    ],
)
