package(default_visibility = ["//visibility:public"])

cc_library(
  name = "redis_store",
  srcs = [
    'redis_sparse_store.cc',
  ],
  linkopts = ["-levent", "-L/usr/local/lib"],
  linkstatic = 1,
  hdrs = [
    'sparse_storage_interface.h',
    'redis_sparse_store.h',
    'sparse_storage_factory.h',
  ],
  deps = [
      "@hiredis//:hiredis",
      "@org_tensorflow//tensorflow/core:lib",
      "@org_tensorflow//tensorflow/core:framework",
  ],
)

cc_binary(
  name = "redis_test",
  srcs = [
    "redis_test.cc",
  ],
  linkopts = ["-levent", "-L/usr/local/lib"],
  linkstatic = 1,
  deps = [
      "@hiredis//:hiredis",
  ],
)

cc_binary(
  name = "redis_perf_test",
  srcs = [
    "redis_perf_test.cc",
  ],
  linkopts = ["-levent", "-L/usr/local/lib"],
  linkstatic = 1,
  deps = [
      "@hiredis//:hiredis",
  ],
)

cc_library(
    name = "sparse_storage",
    srcs = [
        "sparse_storage.cc",
    ],
    hdrs = [
        "sparse_storage.h",
    ],
    linkstatic = True,
    deps = [
        ":redis_store",
        "@com_google_absl//absl/synchronization",
        "@concurrent_queue",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "model_storage",
    srcs = ["model_storage.cc"],
    hdrs = ["model_storage.h"],
    linkstatic = True,
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "//odl_processor/serving:model_version",
    ],
    alwayslink = 1,
)

cc_test(
    name = "sparse_storage_test",
    srcs = ["sparse_storage_test.cc"],
    deps = [
        ":sparse_storage",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],   
)

