load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-1.10.0",
        sha256 = "3714e3db8c51e43028e10ad7adffb9a36fc4aa5b1a363c2d0c4303dd1be59a7c",
        urls = [
            "https://github.com/google/flatbuffers/archive/v1.10.0.tar.gz",
        ],
        build_file = "//third_party/flatbuffers:BUILD.bazel",
    )
