package(default_visibility = ["//visibility:public"])

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

