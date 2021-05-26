licenses(["notice"])  # MIT

exports_files(["LICENSE"])

cc_library(
    name = "yaml-cpp_internal",
    hdrs = glob(["src/**/*.h"]),
    strip_include_prefix = "src",
)

cc_library(
    name = "yaml-cpp",
    srcs = glob([
        "src/**/*.cpp",
        "src/**/*.h",
    ]),
    hdrs = glob(["include/**/*.h"]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
