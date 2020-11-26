package(default_visibility = ["//visibility:public"])

cc_library(
  name = "redis_modelstore",
  srcs = [
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
