# Description:
#   AWS C CAL

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "aws_c_cal",
    srcs = glob([
        "include/aws/cal/**/*.h",
        "source/*.c",
        "source/unix/*.c",
    ]),
    includes = [
        "include/",
    ],
    deps = [
        "@aws_c_common",
    ],
)
