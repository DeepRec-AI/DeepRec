# Description:
#   AWS C EVENT STREAM

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "aws_c_event_stream",
    srcs = glob([
        "include/aws/event-stream/*.h",
        "source/event_stream.c",
    ]),
    includes = [
        "include/",
    ],
    deps = [
        "@aws_c_common",
        "@aws_checksums",
        "@aws_c_io",
    ],
)
