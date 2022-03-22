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
    ],
    deps = [
        "utils",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
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
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
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
        #"@org_tensorflow//tensorflow/core:graph",
        #"@org_tensorflow//tensorflow/core:platform_base",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        #"@local_config_tf//:libtensorflow_framework",
        #"@local_config_tf//:tf_header_lib",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["util/utils_test.cc"],
    deps = [
        "utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

