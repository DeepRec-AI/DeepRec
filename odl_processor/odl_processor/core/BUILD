package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/odl.bzl",
    "tf_io_copts",
)

cc_library(
    name = "lookup_ops",
    srcs = [
        "kernels/lookup_kernels.cc",
        "ops/lookup_ops.cc",
    ],
    deps = [
        "//odl_processor/model_store:redis_modelstore",
        "//odl_processor/model_store:sparse_storage_manager",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core/util/tensor_bundle",
    ],
)

cc_test(
    name = "lookup_ops_test",
    srcs = ["kernels/lookup_kernels_test.cc"],
    deps = [
        ":lookup_ops",
        ":graph_optimizer",
        "//odl_processor/storage/oss:oss_ops",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
        "@org_tensorflow//tensorflow/core/kernels:ops_util",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],   
)

cc_library(
    name = "utils",
    srcs = ["util/utils.cc"],
    hdrs = [
        "util/utils.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["util/utils_test.cc"],
    deps = [
        ":utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:core_cpu",
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
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:ops",
        "@org_tensorflow//tensorflow/core:ops",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
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
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:ops",
        "@org_tensorflow//tensorflow/core:ops",
    ],
)

cc_test(
    name = "lookup_manual_test",
    srcs = ["kernels/lookup_kernels_manual_test.cc"],
    deps = [
        ":lookup_ops",
        ":graph_optimizer",
        "//odl_processor/storage/oss:oss_ops",
    ],
)
