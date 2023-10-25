licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = [
    "//tensorflow:internal",
])

load(
    "//tensorflow:tensorflow.bzl", "tf_cc_test",
)

cc_library(
    name = "elastic_grpc_server_lib",
    srcs = select({"//tensorflow:with_elastic_support": ["elastic_service.cc",
                                                     "elastic_grpc_server_lib.cc"],
		   "//conditions:default": []}),
    hdrs = ["elastic_service.h",
            "elastic_grpc_server_lib.h"],
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
    deps = [
        "//tensorflow/core:elastic_service_proto_cc",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:async_service_interface",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_master_service",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_cache",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_service",
        "//tensorflow/core/distributed_runtime/rpc:rpc_rendezvous_mgr",
        "//tensorflow:grpc",
        "//tensorflow:grpc++",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/distributed_runtime:collective_param_resolver_distributed",
        "//tensorflow/core/distributed_runtime:device_resolver_distributed",
        "//tensorflow/core/distributed_runtime:graph_mgr",
        "//tensorflow/core/distributed_runtime:local_master",
        "//tensorflow/core/distributed_runtime:master",
        "//tensorflow/core/distributed_runtime:master_env",
        "//tensorflow/core/distributed_runtime:master_session",
        "//tensorflow/core/distributed_runtime:rpc_collective_executor_mgr",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime:session_mgr",
        "//tensorflow/core/distributed_runtime:worker_cache_wrapper",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime:worker_resource",
        "//tensorflow/core/distributed_runtime/rpc/eager:grpc_eager_service_impl",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "elastic_grpc_test",
    size = "small",
    srcs = ["elastic_grpc_server_lib_test.cc"],
    deps = [
        ":elastic_grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow:grpc",
        "//tensorflow:grpc++",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:worker_proto_cc",
    ],
    linkstatic = 1, 
)
