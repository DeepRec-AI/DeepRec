package(default_visibility = ["//visibility:public"])

load(
    "//serving:tools/build/odl.bzl",
    "tf_io_copts",
)

cc_library(
    name = "model_version",
    hdrs = ["model_version.h"],
    deps = [],
)

cc_library(
    name = "lookup_ops",
    srcs = [
        "kernels/lookup_kernels.cc",
        "ops/lookup_ops.cc",
    ],
    deps = [
        "//serving/odl_processor/storage:redis_store",
        "//serving/odl_processor/storage:feature_store_mgr",
        "//tensorflow/core:framework",
        "//tensorflow/core/util/tensor_bundle",
    ],
)

cc_test(
    name = "lookup_ops_test",
    srcs = ["kernels/lookup_kernels_test.cc"],
    deps = [
        ":lookup_ops",
        ":graph_optimizer",
        "//serving/odl_processor/framework/filesystem:oss_filesystem",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],   
)

cc_library(
    name = "utils",
    srcs = ["util/utils.cc"],
    hdrs = [
        "util/utils.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["util/utils_test.cc"],
    deps = [
        ":utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:core_cpu",
    ],
)

cc_library(
    name = "graph_optimizer",
    srcs = ["graph_optimizer.cc"],
    hdrs = [
        "graph_optimizer.h",
    ],
    deps = [
        ":lookup_ops",
        ":utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/core:ops",
        "//tensorflow/cc/saved_model:loader",
    ],
)

cc_test(
    name = "graph_optimizer_test",
    srcs = ["graph_optimizer_test.cc"],
    deps = [
        ":graph_optimizer",
        ":utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:core_cpu",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/core:ops",
    ],
)

cc_test(
    name = "lookup_manual_test",
    srcs = ["kernels/lookup_kernels_manual_test.cc"],
    deps = [
        ":lookup_ops",
        ":graph_optimizer",
        "//serving/odl_processor/framework/filesystem:oss_filesystem",
    ],
)
