package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/odl.bzl",
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
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)
