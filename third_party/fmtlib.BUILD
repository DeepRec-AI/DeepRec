licenses(["notice"])  # MIT

exports_files(["LICENSE.rst"])

cc_library(
    name = "fmtlib",
    srcs = [
        "src/format.cc",
        "src/posix.cc",
    ],
    hdrs = [
        "include/fmt/color.h",
        "include/fmt/core.h",
        "include/fmt/format.h",
        "include/fmt/format-inl.h",
        "include/fmt/ostream.h",
        "include/fmt/posix.h",
        "include/fmt/printf.h",
        "include/fmt/ranges.h",
        "include/fmt/time.h",
    ],
    defines = ["FMT_HEADER_ONLY"],
    includes = [
        "include",
    ],
    visibility = ["//visibility:public"],
)
