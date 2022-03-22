package(default_visibility = ["//visibility:public"])
load("//third_party/flatbuffers:build_defs.bzl", "flatbuffer_cc_library")

cc_test(
    name = "graph_optimizer_e2e_test",
    srcs = ["graph_optimizer_test.cc"],
    deps = [
        "//odl_processor/framework:utils",
        "//odl_processor/framework:graph_optimizer",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:core_cpu",
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
      "//odl_processor/serving:tf_predict_proto_cc",
      "@org_tensorflow//tensorflow/core:framework",
      "@org_tensorflow//tensorflow/core:protos_all_cc",
      "@org_tensorflow//tensorflow/core:core_cpu",
      "@org_tensorflow//tensorflow/core:lib",
      "@org_tensorflow//tensorflow/cc:cc_ops",
      "@org_tensorflow//tensorflow/cc:ops",
      "@org_tensorflow//tensorflow/core:ops",
  ],
)

cc_binary(
  name = "demo",
  srcs = ["end2end/demo.cc",],
  deps = [
          "//odl_processor/serving:tf_processor_internal"],
  includes = ["odl_processor/serving"],
)

