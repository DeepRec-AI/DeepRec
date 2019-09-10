#Description : NVIDIA Tools Extension (NVTX) library for adding profiling annotations to applications.

package(
    default_visibility = ["//visibility:public"],
)

# TODO(benbarsdell): Check this.
licenses(["restricted"])  # NVIDIA proprietary license

#exports_files(["LICENSE.TXT"])

## TODO(benbarsdell): Needed?
#load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts", "if_cuda")

filegroup(
    name = "nvtx_header_files",
    srcs = glob([
        #"usr/local/cuda-*/targets/x86_64-linux/include/**",
        "nvtx3/**",
    ]),
)

cc_library(
    name = "nvtx",
    #hdrs = if_cuda([":nvtx_header_files"]),
    hdrs = [":nvtx_header_files"],
    include_prefix = "third_party",
    deps = [
        #"@local_config_cuda//cuda:cuda_headers",
    ],
)
