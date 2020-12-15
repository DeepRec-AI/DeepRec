package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/odl.bzl",
    "tf_io_copts",
)

load(
    "@org_tensorflow//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc"
)

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_shared_object"
)

tf_proto_library_cc(
    name = "tf_predict_proto",
    srcs = ["tf_predict.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

tf_cc_shared_object(
    name = "libtf_processor.so",
    srcs = ["tf_processor.cc",
            "message_coding.cc",
            "tf_processor.h",
            "message_coding.h"],
    deps = [
            "tf_predict_proto_cc",
            "model_serving"],
)

cc_library(
    name = "model_config",
    srcs = ["model_config.cc"],
    hdrs = ["model_config.h"],
    deps = [
        "@jsoncpp_git//:jsoncpp",
        "@org_tensorflow//tensorflow/core:framework",
        ],
)

cc_test(
    name = "model_config_test",
    srcs = ["model_config_test.cc",],
    deps = [":model_config",
            "@com_google_googletest//:gtest",
            "@com_google_googletest//:gtest_main",],
)

cc_library(
    name = "model_message",
    hdrs = ["model_message.h",],
)

cc_library(
    name = "model_session",
    srcs = ["model_session.cc"],
    hdrs = ["model_session.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
        "//odl_processor/framework:graph_optimizer",
        "//odl_processor/framework:model_version",
        "//odl_processor/storage:model_store",
        "model_config",
        "model_message"],
)

cc_test(
    name = "model_session_test",
    srcs = ["model_session_test.cc",],
    deps = [":model_session",
            "@com_google_googletest//:gtest",
            "@com_google_googletest//:gtest_main",],
)

cc_library(
    name = "model_serving",
    srcs = ["model_serving.cc",
            "model_impl.cc",
            "model_instance.cc"],
    hdrs = ["model_serving.h",
            "model_impl.h",
            "model_instance.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
        "//odl_processor/framework:graph_optimizer",
        "//odl_processor/framework:model_version",
        "//odl_processor/storage:model_store",
        "model_config",
        "model_session",
        "model_message",
    ],
)
