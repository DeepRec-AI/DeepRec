load(
  "//tensorflow:tensorflow.bzl",
  "tf_cuda_library",
  "tf_copts",
  "cc_header_only_library",
  "tf_cc_test")
    
package(
    default_visibility = [
        "//tensorflow/compiler/tf2xla:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "xla_ops",
    srcs = [
        "async_io_ops.cc",
        "xla_ops.cc",
    ],
    hdrs = [
        "async_io_ops.h",
        "xla_ops.h",
    ],
    deps = [
        ":async_io_rendezvous",
        "//tensorflow/compiler/jit:common",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/jit:xla_activity_listener",
        "//tensorflow/compiler/jit:xla_activity_proto_cc",
        "//tensorflow/compiler/jit:xla_compilation_cache",
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/jit:xla_launch_util",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:state_ops_op_lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/stream_executor:tf_allocator_adapter",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "cuda_graph_mode_ops",
    srcs = [
        "cuda_graph_mode_ops.cc",
    ],
    hdrs = [
        "cuda_graph_mode_ops.h",
        "//tensorflow/core:common_runtime/gpu/gpu_cuda_graph_bfc_allocator.h",
        "//tensorflow/core:common_runtime/gpu/gpu_cuda_graph_mode_mem.h",
        "//tensorflow/core:common_runtime/gpu/gpu_device.h",
        "//tensorflow/stream_executor/cuda:cuda_driver.h"
    ],
    copts = tf_copts() + ["-fexceptions"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/jit:common",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:state_ops_op_lib",
        "//tensorflow/core:stream_executor_no_cuda",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/stream_executor:tf_allocator_adapter",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)

cc_library(
    name = "async_io_rendezvous",
    srcs = [
        "async_io_rendezvous.cc",
    ],
    hdrs = [
        "async_io_rendezvous.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
	"//tensorflow/stream_executor:device_memory",
    ],
)

tf_cc_test(
    name = "async_io_ops_test",
    srcs = [
        "async_io_ops_test.cc",
    ],
    deps = [
        ":xla_ops",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "//tensorflow/core:test_main",
    ],
)
