licenses(["notice"])  # Apache 2.0

cc_library(
    name = "mxml",
    srcs = [
        "config.h",
        "mxml-attr.c",
        "mxml-entity.c",
        "mxml-file.c",
        "mxml-get.c",
        "mxml-index.c",
        "mxml-node.c",
        "mxml-private.c",
        "mxml-private.h",
        "mxml-search.c",
        "mxml-set.c",
        "mxml-string.c",
    ],
    hdrs = [
        "mxml.h",
    ],
    copts = ["-pthread"],
    defines = [
        "_GNU_SOURCE",
        "_THREAD_SAFE",
        "_REENTRANT",
    ],
    includes = [
        ".",
    ],
    linkopts = ["-lpthread"],
    visibility = ["//visibility:public"],
)
