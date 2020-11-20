package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "redis_test",
  srcs = [
    "redis_test.cc",
  ],
  linkopts = ["-levent", "-L/usr/local/lib"],
  linkstatic = 1,
  deps = [
      "@hiredis//:hiredis",
      "@mxml//:mxml",
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
