licenses(["notice"])  # BSD 2-clause

exports_files(["LICENSE"])

cc_library(
    name = "lz4",
    srcs = glob(["*.c"]),
    hdrs = glob(["*.h"]),
    includes = ["."],
    textual_hdrs = ["lz4.c"],
    visibility = ["//visibility:public"],
)
