package(default_visibility = ["//visibility:public"])
load("//third_party/flatbuffers:build_defs.bzl", "flatbuffer_cc_library")

cc_test(
    name = "graph_optimizer_e2e_test",
    srcs = ["graph_optimizer_test.cc"],
    deps = [
        "//serving/processor/framework:utils",
        "//serving/processor/framework:graph_optimizer",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:core_cpu",
    ],
)

flatbuffer_cc_library(
    name = "request_fbs",
    srcs = ["flatbuf_test/cpp/request.fbs"],
    flatc_args = [
        "--cpp",
    ],
)

cc_binary(
  name = "flat_test",
  srcs = [
    "flatbuf_test/cpp/benchmark.cc",
  ],
  linkstatic = 1,
  deps = [
      "@flatbuffers",
      ":request_fbs",
      "//serving/processor/serving:predict_proto_cc",
      "//tensorflow/core:framework",
      "//tensorflow/core:protos_all_cc",
      "//tensorflow/core:core_cpu",
      "//tensorflow/core:lib",
      "//tensorflow/cc:cc_ops",
      "//tensorflow/cc:ops",
      "//tensorflow/core:ops",
  ],
)

cc_binary(
  name = "demo",
  srcs = ["end2end/demo.cc",],
  deps = [
          "//serving/processor/serving:serving_processor_internal"],
  includes = ["serving/processor/serving"],
)

