# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.
#
# Public targets:
#
# ":sdk_package" - Package the tensorflow dynamic library and necessry
#     headers for developing. The script should be executed manually
#     after 'bazel build'.

package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "transitive_hdrs", "tf_binary_additional_srcs")
load("//tensorflow/core/platform:default/build_config_root.bzl",
     "tf_additional_plugin_deps")

transitive_hdrs(
    name = "sdk_headers",
    deps = [
        # Need to check definition of //tensorflow:libtensorflow_cc.so
        # for updates.
        "//tensorflow/c:c_api",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:scope",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:signature_constants",
        "//tensorflow/cc/saved_model:tag_constants",
        "//tensorflow/contrib/session_bundle:bundle_shim",
    ] + tf_additional_plugin_deps(),
    tags = ["manual"],
)

sh_binary(
    name = "build_sdk_package",
    srcs = ["build_sdk_package.sh"],
    data = [
        ":sdk_headers",
        "@com_google_protobuf//:protoc",
        "//tensorflow:libtensorflow_cc.so",
    ] + tf_binary_additional_srcs(),
    tags = ["manual"],
)
