load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//tensorflow:version_check.bzl", "check_bazel_version_at_least")
load("//tensorflow:workspace.bzl", "tf_workspace")

def tf_repositories():
    check_bazel_version_at_least("0.15.0")
    tf_workspace()
