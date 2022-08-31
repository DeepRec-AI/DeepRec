package(default_visibility = ["//visibility:public"])

load(
    "//serving/processor:configure/processor.bzl",
    "tf_io_copts",
)

load(
    "//tensorflow/core/platform:default/build_config.bzl",
    "tf_proto_library_cc"
)

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_shared_object"
)

tf_proto_library_cc(
    name = "predict_proto",
    srcs = ["predict.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

tf_cc_shared_object(
    name = "libserving_processor.so",
    srcs = ["processor.cc",
            "processor.h",],
    deps = [
            "model_serving"],
)

cc_library(
    name = "serving_processor_internal",
    srcs = ["processor.cc",
            "processor.h",],
    copts = ["-g"],
    deps = [
            "model_serving"],
)

cc_library(
    name = "utils",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "model_message",
        "predict_proto_cc",
        "//serving/processor/framework:graph_optimizer",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:core_cpu",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:reader",
        "//tensorflow/cc/saved_model:constants",
        "//tensorflow/cc/saved_model:signature_constants",
        "//tensorflow/cc/saved_model:tag_constants",
        ],
)

cc_library(
    name = "tracer",
    hdrs = ["tracer.h"],
    deps = [
        "@aliyun_oss_c_sdk",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:framework",
        ],
)

cc_library(
    name = "model_config",
    srcs = ["model_config.cc"],
    hdrs = ["model_config.h"],
    deps = [
        "@jsoncpp_git//:jsoncpp",
        "//tensorflow/core:framework",
        "tracer",],
)

cc_library(
    name = "model_partition",
    hdrs = ["model_partition.h"],
    deps = ["model_config",],
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
    srcs = ["model_message.cc",],
    hdrs = ["model_message.h",],
    deps = [
        "//tensorflow/core:framework",
        ],
)

cc_library(
    name = "model_session",
    srcs = ["model_session.cc"],
    hdrs = ["model_session.h"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:core_cpu",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:signature_constants",
        "//tensorflow/cc/saved_model:tag_constants",
        "//serving/processor/framework:graph_optimizer",
        "//serving/processor/framework:model_version",
        "//serving/processor/storage:model_store",
        "model_config",
        "model_message",
        "predict_proto_cc",
        "utils",
        "tracer"],
)

cc_test(
    name = "model_session_test",
    srcs = ["model_session_test.cc",],
    deps = [":model_session",
            "@com_google_googletest//:gtest",
            "@com_google_googletest//:gtest_main",],
)

cc_library(
    name = "model_instance",
    srcs = ["model_instance.cc",],
    hdrs = ["model_instance.h"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:core_cpu",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:signature_constants",
        "//tensorflow/cc/saved_model:tag_constants",
        "//serving/processor/framework:graph_optimizer",
        "//serving/processor/framework:model_version",
        "//serving/processor/storage:model_store",
        "model_config",
        "model_partition",
        "model_session",
        "model_message",
        "predict_proto_cc",
        "utils",
    ],
)

cc_library(
    name = "model_serving",
    srcs = ["model_serving.cc",
            "message_coding.cc",
            "model_impl.cc",],
    hdrs = ["model_serving.h",
            "message_coding.h",
            "model_impl.h",],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//serving/processor/framework:model_version",
        "//serving/processor/storage:model_store",
        "model_config",
        "model_session",
        "model_message",
        "model_instance",
        "predict_proto_cc",
    ],
)
