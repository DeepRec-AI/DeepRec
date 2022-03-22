load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

def tf_repositories():
    check_bazel_version_at_least("0.15.0")
    tf_workspace()
