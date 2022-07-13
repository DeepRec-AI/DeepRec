# Description:
#   AWS C COMMON

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("@org_tensorflow//third_party:common.bzl", "template_rule")

cc_library(
    name = "aws_c_common",
    srcs = glob([
        "include/aws/common/**/*.h",
        "include/aws/common/**/*.inl",
        "source/*.c",
        "source/posix/*.c",
        "source/arch/intel/cpuid.c",
        "source/arch/intel/asm/cpuid.c",
    ]),
    hdrs = [
        "include/aws/common/config.h",
    ],
    includes = [
        "include/",
    ],
    defines = [
        "AWS_AFFINITY_METHOD=AWS_AFFINITY_METHOD_PTHREAD_ATTR",
    ],
)

template_rule(
    name = "COMMONConfig_h",
    src = "include/aws/common/config.h.in",
    out = "include/aws/common/config.h",
    substitutions = {
        "cmakedefine": "define",
    },
)
