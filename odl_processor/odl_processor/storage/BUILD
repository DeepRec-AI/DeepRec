package(default_visibility = ["//visibility:public"])

load("@org_tensorflow//tensorflow:tensorflow.bzl",
     "tf_cc_test")

cc_library(
  name = "redis_store",
  srcs = [
    'redis_feature_store.cc',
  ],
  linkstatic = 1,
  hdrs = [
    'feature_store.h',
    'redis_feature_store.h',
  ],
  deps = [
      "@hiredis//:hiredis",
      "@libevent//:libevent",
      "@org_tensorflow//tensorflow/core:lib",
      "@org_tensorflow//tensorflow/core:framework",
  ],
)

cc_binary(
  name = "redis_test",
  srcs = [
    "redis_test.cc",
  ],
  linkstatic = 1,
  deps = [
      "@hiredis//:hiredis",
      "@libevent//:libevent",
  ],
)

cc_binary(
  name = "redis_perf_test",
  srcs = [
    "redis_perf_test.cc",
  ],
  linkstatic = 1,
  deps = [
      "@hiredis//:hiredis",
      "@libevent//:libevent",
  ],
)

cc_library(
    name = "feature_store_mgr",
    srcs = [
        "feature_store_mgr.cc",
    ],
    hdrs = [
        "feature_store_mgr.h",
    ],
    linkstatic = True,
    deps = [
        ":redis_store",
        "//odl_processor/serving:model_config",
        "@com_google_absl//absl/synchronization",
        "@concurrent_queue",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "model_store",
    srcs = ["model_store.cc"],
    hdrs = ["model_store.h"],
    linkstatic = True,
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "//odl_processor/framework:model_version",
        "//odl_processor/framework/filesystem:oss_filesystem",
        "//odl_processor/serving:model_config",
        "feature_store_mgr",
    ],
    alwayslink = 1,
)

cc_test(
    name = "model_store_test",
    srcs = ["model_store_test.cc",],
    deps = [":model_store",
            "//odl_processor/serving:model_config",
            "@com_google_googletest//:gtest",
            "@com_google_googletest//:gtest_main",],
)

cc_test(
    name = "feature_store_test",
    srcs = ["feature_store_test.cc"],
    deps = [
        ":feature_store_mgr",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],   
)
