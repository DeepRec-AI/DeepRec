package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/odl.bzl",
    "tf_io_copts",
)

load(
    "@org_tensorflow//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc"
)

tf_proto_library_cc(
    name = "tf_predict_proto",
    srcs = ["tf_predict.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tf_processor",
    srcs = ["model_serving.cc",
            "model_config.cc"],
    hdrs = ["model_serving.h",
            "model_config.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
        "tf_predict_proto_cc",
        "//odl_processor/core:graph_optimizer",
    ],
)

cc_library(
    name = "tf_processor_prev",
    srcs = ["freeze_bundle.cc",
            "run_predict.cc",
            "tf_processor.cc",
            "saved_model_loader.cc"],
    hdrs = ["freeze_bundle.h",
            "run_predict.h",
            "saved_model_loader.h",
            "tf_processor.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
        "@jsoncpp_git//:jsoncpp",
        "tf_predict_proto_cc",
    ],
)

