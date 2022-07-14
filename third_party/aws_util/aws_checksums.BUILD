# Description:
#   AWS CHECKSUMS

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "aws_checksums",
    srcs = glob([
        "include/aws/checksums/**/*.h",
        "source/*.c",
        "source/intel/asm/*.c",
    ]),
    includes = [
        "include/",
    ],
    deps = [
        "@aws_c_common",
    ],
)
