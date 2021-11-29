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

COMMON_COPTS = [
	"-std=gnu++14",
    "-DFMT_HEADER_ONLY",
    "-DNO_EXCEPTION_HACK",
    "-DNO_EXCEPTION_INTERCEPT",
    "-DHAVE_LZ4_COMPRESS_DEFAULT",
]

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

cc_library(
    name = "star_tensor_coding",
    srcs = select({"//tensorflow:with_star_support": ["star_tensor_coding.cc",
                                                      "star_message.cc"],
                   "//conditions:default": []}),
    hdrs = select({"//tensorflow:with_star_support": ["star_tensor_coding.h",
                                                      "star_message.h",
                                                      "star_worker_interface.h"],
                   "//conditions:default": []}),
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core/distributed_runtime:call_options",
    ],
)

cc_library(
    name = "star_rendezvous_mgr",
    srcs = select({"//tensorflow:with_star_support": ["star_rendezvous_mgr.cc"],
                   "//conditions:default": []}),
    hdrs = select({"//tensorflow:with_star_support": ["star_rendezvous_mgr.h"],
                   "//conditions:default": []}),
    linkstatic = 1,
    copts = COMMON_COPTS,
    deps = select({"//tensorflow:with_star_support": [":star_tensor_coding"],
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
    name = "star_channel_spec",
    srcs = select({"//tensorflow:with_star_support": ["star_channel_spec.cc"],
                   "//conditions:default": []}),
    hdrs = [
        "star_channel_spec.h",
    ],
    linkstatic = 1,
    copts = COMMON_COPTS,
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:framework",
        "//tensorflow/core/distributed_runtime:worker_cache_logger",
        "//tensorflow/core/distributed_runtime:worker_cache_partial",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "star_worker_service",
    srcs = select({"//tensorflow:with_star_support": ["star_worker_service.cc",
                                                      "star_client_tag.cc",
                                                      "star_server_tag.cc",
                                                      "seastar/seastar_tag_factory.cc",
                                                      "seastar/seastar_client_tag.cc",
                                                      "seastar/seastar_server_tag.cc",
                                                  ],
                   "//conditions:default": []}),
    hdrs = [
        "star_worker_service_method.h",
        "star_worker_service.h",
        "star_client_tag.h",
        "star_server_tag.h",
        "seastar/seastar_client_tag.h",
        "seastar/seastar_server_tag.h",
        "seastar/seastar_tag_factory.h",
        "seastar/seastar_header.h",
    ],
    linkstatic = 1,
    copts = COMMON_COPTS,
    deps = select({"//tensorflow:with_star_support": ["@seastar_repo//:seastar",
                                                      ":star_tensor_coding"],
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
    name = "seastar_engine",
    srcs = select({"//tensorflow:with_star_support": ["seastar/seastar_engine.cc",
                                                      "seastar/seastar_client.cc",
                                                      "seastar/seastar_server.cc",
                                                      "seastar/seastar_cpuset.cc",
                                                      "seastar/seastar_stat.cc"],
                   "//conditions:default": []}),
    hdrs = [
        "seastar/seastar_engine.h",
        "seastar/seastar_client.h",
        "seastar/seastar_server.h",
        "seastar/seastar_cpuset.h",
        "seastar/seastar_stat.h",
        "seastar/seastar_header.h",
    ],
    linkstatic = 1,
    copts = COMMON_COPTS,
    deps = select({"//tensorflow:with_star_support": [":star_worker_service"],
                   "//conditions:default": []}),
    alwayslink = 1,
)

cc_library(
    name = "seastar_worker_cache",
    srcs = select({"//tensorflow:with_star_support": ["seastar/seastar_remote_worker.cc",
                                                      "seastar/seastar_worker_cache.cc",
                                                      "seastar/seastar_channel_cache.cc"],
                   "//conditions:default": []}),
    hdrs = [
        "seastar/seastar_remote_worker.h",
        "seastar/seastar_worker_cache.h",
        "seastar/seastar_channel_cache.h",
        "seastar/seastar_header.h",
    ],
    linkstatic = 1,
    copts = COMMON_COPTS,
    deps = select({"//tensorflow:with_star_support": [":star_channel_spec",
                                                      ":star_worker_service",
                                                      ":seastar_engine"],
                   "//conditions:default": []})
    + [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:framework",
        "//tensorflow/core/distributed_runtime:worker_cache_logger",
        "//tensorflow/core/distributed_runtime:worker_cache_partial",
    ],
    alwayslink = 1,
)

cc_library(
    name = "star_server_base_lib",
    srcs = select({"//tensorflow:with_star_support": ["star_server_base_lib.cc",
                                                      "seastar/seastar_server_lib.cc"],
                   "//conditions:default": []}),
    hdrs = [
        "star_server_base_lib.h",
        "seastar/seastar_server_lib.h",
        "seastar/seastar_header.h",
    ],
    linkstatic = 1,
    copts = COMMON_COPTS,
    deps = select({"//tensorflow:with_star_support": [":star_rendezvous_mgr",
                                                      ":seastar_worker_cache"],
                   "//conditions:default": []})
    + [
        "//tensorflow/core/distributed_runtime:local_master",
        "//tensorflow/core/distributed_runtime:master",
        "//tensorflow/core/distributed_runtime:master_env",
        "//tensorflow/core/distributed_runtime:master_session",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime:worker_resource",
        "//tensorflow/core/distributed_runtime/rpc:grpc_master_service",
    ],
    alwayslink = 1,
)

