package(default_visibility = ["//visibility:public"])

cc_library(
  name = "redis_modelstore",
  srcs = [
    'redis_model_store.cc',
  ],
  linkopts = ["-levent", "-L/usr/local/lib"],
  linkstatic = 1,
  hdrs = [
    'abs_model_store.h',
    'redis_model_store.h',
    'model_store_factory.h',
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
    name = "sparse_storage_manager",
    srcs = [
        "sparse_storage_manager.cc",
    ],
    hdrs = [
        "sparse_storage_manager.h",
    ],
    linkstatic = True,
    deps = [
        ":redis_modelstore",
        "@com_google_absl//absl/synchronization",
        "@concurrent_queue",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "sparse_storage_manager_test",
    srcs = ["sparse_storage_manager_test.cc"],
    deps = [
        ":sparse_storage_manager",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],   
)

