package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/odl.bzl",
    "tf_io_copts",
)

cc_library(
    name = "graph_optimizer",
    srcs = ["graph_optimizer.cc"],
    hdrs = [
        "graph_optimizer.h",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    deps = [
        "utils",
    ],
)

cc_test(
    name = "graph_optimizer_test",
    srcs = ["graph_optimizer_test.cc"],
    deps = [
        "graph_optimizer",
        "utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:core_cpu",
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
        "utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:core_cpu",
    ],
)

