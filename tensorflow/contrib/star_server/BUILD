licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cuda_library",
    "tf_cc_test",
)
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_tests")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

# For platform specific build config
load(
    "//tensorflow/core/platform:default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
)
load(
    "//tensorflow/core/platform:default/build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(default_visibility = [
    "//tensorflow:internal",
])

tf_cuda_library(
    name = "tls_worker_service",
    srcs = select({"//tensorflow:with_star_support": ["tls_worker.cc",
                                                      "tls_worker_service.cc",
                                                      "tls_session_mgr.cc"],
                   "//conditions:default": []}),
    hdrs = [
        "tls_worker.h",
        "tls_worker_service.h",
        "tls_session_mgr.h",
    ],
    linkstatic = 1,
    copts = [
        "-std=gnu++1y",
        "-DFMT_HEADER_ONLY",
        "-DNO_EXCEPTION_HACK",
        "-DNO_EXCEPTION_INTERCEPT",
        "-DHAVE_LZ4_COMPRESS_DEFAULT",
    ],
    deps = select({"//tensorflow:with_star_support": ["@seastar_repo//:seastar",
        "//tensorflow/contrib/star:star_worker_service"],
                   "//conditions:default": []})
    + [
        "//tensorflow/contrib/verbs:verbs_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core:worker_proto_cc",
        "//tensorflow/core/distributed_runtime:worker",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core:gpu_runtime",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tls_rendezvous_mgr",
    srcs = select({"//tensorflow:with_star_support": ["tls_rendezvous_mgr.cc"],
                   "//conditions:default": []}),
    hdrs = ["tls_rendezvous_mgr.h"],
    linkstatic = 1,
    copts = [
        "-std=gnu++1y",
        "-DFMT_HEADER_ONLY",
        "-DNO_EXCEPTION_HACK",
        "-DNO_EXCEPTION_INTERCEPT",
        "-DHAVE_LZ4_COMPRESS_DEFAULT",
    ],
    deps = select({"//tensorflow:with_star_support": [
        "//tensorflow/contrib/star:star_rendezvous_mgr"],
        "//conditions:default": []})
    + [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:base_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime:worker_cache",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime:worker_interface",
    ],
    alwayslink = 1,
)

cc_library(
    name = "star_server_lib",
    srcs = select({"//tensorflow:with_star_support": ["star_server_lib.cc"],
                   "//conditions:default": []}),
    hdrs = select({"//tensorflow:with_star_support": ["star_server_lib.h"],
                   "//conditions:default": []}),
    linkstatic = 1,
    copts = [
        "-std=gnu++1y",
        "-DFMT_HEADER_ONLY",
        "-DNO_EXCEPTION_HACK",
        "-DNO_EXCEPTION_INTERCEPT",
        "-DHAVE_LZ4_COMPRESS_DEFAULT",
    ],
    deps = select({"//tensorflow:with_star_support": [":tls_rendezvous_mgr", 
                   ":tls_worker_service",
                   "//tensorflow/contrib/star:star_server_base_lib"],
                   "//conditions:default": []})
    + [
        "//tensorflow/core/distributed_runtime:local_master",
        "//tensorflow/core/distributed_runtime:master",
        "//tensorflow/core/distributed_runtime:master_env",
        "//tensorflow/core/distributed_runtime:master_session",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_master_service",
    ],
    alwayslink = 1,
)

