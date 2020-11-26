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
    srcs = ["tf_processor.cc"],
    hdrs = ["tf_processor.h"],
    deps = [
            "tf_predict_proto_cc",
            "model_serving"],
)

cc_library(
    name = "model_serving",
    srcs = ["model_serving.cc",
            "model_impl.cc",
            "model_config.cc",
            "model_instance.cc",
            "model_session.cc"],
    hdrs = ["model_serving.h",
            "model_impl.h",
            "model_config.h",
            "model_instance.h",
            "model_storage.h",
            "tensor_coding.h",
            "model_session.h",
            "model_version.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
        "@jsoncpp_git//:jsoncpp",
        "//odl_processor/core:graph_optimizer",
    ],
)
