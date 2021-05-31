licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "core",
    srcs = ["python/core.py",
            "python/utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

py_test(
    name = "core_test",
    srcs = ["python/core_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:private"],
    deps = [
        ":structured_model",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["python/utils_test.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:private"],
    deps = [
        ":structured_model",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "structured_model",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":core",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
