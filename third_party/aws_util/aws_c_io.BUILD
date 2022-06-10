# Description:
#   AWS C IO

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "aws_c_io",
    srcs = glob([
        "include/aws/io/**/*.h",
        "source/*.c",
        "source/pkcs11/v2.40/*.h",
        "source/pkcs11_private.h",
        "source/posix/*.c",
        "source/linux/*.c",
    ]),
    includes = [
        "include/",
        "source/",
    ],
    deps = [
        "@aws_c_common",
        "@aws_c_cal"
    ],
    defines = [
        "BYO_CRYPTO",
    ],
)
