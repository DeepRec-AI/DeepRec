package(default_visibility = ["//visibility:public"])

load(
    "//serving/processor:configure/processor.bzl",
    "tf_io_copts",
)

cc_library(
    name = "oss_filesystem",
    srcs = [
        "oss_file_system.cc",
        "oss_file_system.h",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "@aliyun_oss_c_sdk",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)
