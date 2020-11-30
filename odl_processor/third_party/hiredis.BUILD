package(default_visibility = ["//visibility:public"])

licenses(["notice"]) # # BSD 3-Clause

cc_library(
  name = "hiredis",
  srcs = [
          "alloc.c",
          "async.c",
          "hiredis.c",
          "net.c",
          "read.c",
          "sds.c",
          "sockcompat.c",
          # for linking error
          # "ssl.c",
  ],
  hdrs = [
          "alloc.h",
          "async.h",
          "async_private.h",
          "dict.h",
          "dict.c",
          "fmacros.h",
          "hiredis.h",
          "hiredis_ssl.h",
          "net.h",
          "read.h",
          "sdsalloc.h",
          "sds.h",
          "sockcompat.h",
          "win32.h",
  ],
  includes = ["."],
  visibility = ["//visibility:public"],
)
