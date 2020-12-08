load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-1f5eae5d6a135ff6811724f6c57f911d1f46bb15",
        sha256 = "b2bb0311ca40b12ebe36671bdda350b10c7728caf0cfe2d432ea3b6e409016f3",
        urls = [
            "http://gitlab.alibaba-inc.com/odps_tensorflow/other/raw/master/mirror.bazel.build/github.com/google/flatbuffers/archive/1f5eae5d6a135ff6811724f6c57f911d1f46bb15.tar.gz",
            "http://gitlab.alibaba-inc.com/odps_tensorflow/other/raw/master/github.com/google/flatbuffers/archive/1f5eae5d6a135ff6811724f6c57f911d1f46bb15.tar.gz",
        ],
        build_file = "//third_party/flatbuffers:BUILD.bazel",
    )
