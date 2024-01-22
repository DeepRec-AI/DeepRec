# -*- Python -*-
"""Repository rule for TensorRT configuration.

`tensorrt_configure` depends on the following environment variables:

  * `TF_TENSORRT_VERSION`: The TensorRT libnvinfer version.
  * `TENSORRT_INSTALL_PATH`: The installation path of the TensorRT library.
"""

load(
    "//third_party/gpus:cuda_configure.bzl",
    "find_cuda_config",
    "get_cpu_value",
    "lib_name",
    "make_copy_files_rule",
)

_TENSORRT_INSTALL_PATH = "TENSORRT_INSTALL_PATH"
_TF_TENSORRT_STATIC_PATH = "TF_TENSORRT_STATIC_PATH"
_TF_TENSORRT_CONFIG_REPO = "TF_TENSORRT_CONFIG_REPO"
_TF_TENSORRT_VERSION = "TF_TENSORRT_VERSION"
_TF_NEED_TENSORRT = "TF_NEED_TENSORRT"

_TF_TENSORRT_LIBS = ["nvinfer", "nvinfer_plugin"]
_TF_TENSORRT_HEADERS = ["NvInfer.h", "NvUtils.h", "NvInferPlugin.h"]
_TF_TENSORRT_HEADERS_V6 = [
    "NvInfer.h",
    "NvUtils.h",
    "NvInferPlugin.h",
    "NvInferVersion.h",
    "NvInferRuntime.h",
    "NvInferRuntimeCommon.h",
    "NvInferPluginUtils.h",
]
_TF_TENSORRT_HEADERS_V8 = [
    "NvInfer.h",
    "NvInferLegacyDims.h",
    "NvInferImpl.h",
    "NvUtils.h",
    "NvInferPlugin.h",
    "NvInferVersion.h",
    "NvInferRuntime.h",
    "NvInferRuntimeCommon.h",
    "NvInferPluginUtils.h",
]

_DEFINE_TENSORRT_SONAME_MAJOR = "#define NV_TENSORRT_SONAME_MAJOR"
_DEFINE_TENSORRT_SONAME_MINOR = "#define NV_TENSORRT_SONAME_MINOR"
_DEFINE_TENSORRT_SONAME_PATCH = "#define NV_TENSORRT_SONAME_PATCH"

def _at_least_version(actual_version, required_version):
    actual = [int(v) for v in actual_version.split(".")]
    required = [int(v) for v in required_version.split(".")]
    return actual >= required

def _get_tensorrt_headers(tensorrt_version):
    if _at_least_version(tensorrt_version, "8"):
        return _TF_TENSORRT_HEADERS_V8
    if _at_least_version(tensorrt_version, "6"):
        return _TF_TENSORRT_HEADERS_V6
    return _TF_TENSORRT_HEADERS

def _tpl(repository_ctx, tpl, substitutions):
    repository_ctx.template(
        tpl,
        Label("//third_party/tensorrt:%s.tpl" % tpl),
        substitutions,
    )

def _create_dummy_repository(repository_ctx):
    """Create a dummy TensorRT repository."""
    _tpl(repository_ctx, "build_defs.bzl", {"%{if_tensorrt}": "if_false"})
    _tpl(repository_ctx, "BUILD", {
        "%{copy_rules}": "",
        "\":tensorrt_include\"": "",
        "\":tensorrt_lib\"": "",
    })
    _tpl(repository_ctx, "tensorrt/include/tensorrt_config.h", {
        "%{tensorrt_version}": "",
    })

def enable_tensorrt(repository_ctx):
    """Returns whether to build with TensorRT support."""
    return int(repository_ctx.os.environ.get(_TF_NEED_TENSORRT, False))

def get_host_environ(repository_ctx, env):
    if env in repository_ctx.os.environ:
        version = repository_ctx.os.environ[env].strip()
        return version
    else:
        return ""

def _get_tensorrt_static_path(repository_ctx):
    """Returns the path for TensorRT static libraries."""
    return get_host_environ(repository_ctx, _TF_TENSORRT_STATIC_PATH)

def _get_tensorrt_full_version(repository_ctx):
    """Returns the full version for TensorRT."""
    return get_host_environ(repository_ctx, _TF_TENSORRT_VERSION)

def _tensorrt_configure_impl(repository_ctx):
    """Implementation of the tensorrt_configure repository rule."""
    if _TF_TENSORRT_CONFIG_REPO in repository_ctx.os.environ:
        # Forward to the pre-configured remote repository.
        remote_config_repo = repository_ctx.os.environ[_TF_TENSORRT_CONFIG_REPO]
        repository_ctx.template("BUILD", Label(remote_config_repo + ":BUILD"), {})
        repository_ctx.template(
            "build_defs.bzl",
            Label(remote_config_repo + ":build_defs.bzl"),
            {},
        )
        repository_ctx.template(
            "tensorrt/include/tensorrt_config.h",
            Label(remote_config_repo + ":tensorrt/include/tensorrt_config.h"),
            {},
        )
        repository_ctx.template(
            "LICENSE",
            Label(remote_config_repo + ":LICENSE"),
            {},
        )
        return

    # Copy license file in non-remote build.
    repository_ctx.template(
        "LICENSE",
        Label("//third_party/tensorrt:LICENSE"),
        {},
    )

    if not enable_tensorrt(repository_ctx):
        _create_dummy_repository(repository_ctx)
        return

    config = find_cuda_config(repository_ctx, ["cuda", "tensorrt"])
    cuda_version = config["cuda_version"]
    cuda_library_path = config["cuda_library_dir"] + "/"
    trt_version = config["tensorrt_version"]
    trt_full_version = _get_tensorrt_full_version(repository_ctx)
    cpu_value = get_cpu_value(repository_ctx)

    # Copy the library and header files.
    libraries = [lib_name(lib, cpu_value, trt_version) for lib in _TF_TENSORRT_LIBS]
    library_dir = config["tensorrt_library_dir"] + "/"
    headers = _get_tensorrt_headers(trt_version)
    include_dir = config["tensorrt_include_dir"] + "/"
    copy_rules = [
        make_copy_files_rule(
            repository_ctx,
            name = "tensorrt_lib",
            srcs = [library_dir + library for library in libraries],
            outs = ["tensorrt/lib/" + library for library in libraries],
        ),
        make_copy_files_rule(
            repository_ctx,
            name = "tensorrt_include",
            srcs = [include_dir + header for header in headers],
            outs = ["tensorrt/include/" + header for header in headers],
        ),
    ]

    tensorrt_static_path = _get_tensorrt_static_path(repository_ctx)
    if tensorrt_static_path:
        tensorrt_static_path = tensorrt_static_path + "/"
        if _at_least_version(trt_full_version, "8.4.1"):
            raw_static_library_names = _TF_TENSORRT_LIBS
            nvrtc_ptxjit_static_raw_names = ["nvrtc", "nvrtc-builtins", "nvptxcompiler"]
            nvrtc_ptxjit_static_names = ["%s_static" % name for name in nvrtc_ptxjit_static_raw_names]
            nvrtc_ptxjit_static_libraries = [lib_name(lib, cpu_value, trt_version, static = True) for lib in nvrtc_ptxjit_static_names]
        elif _at_least_version(trt_version, "8"):
            raw_static_library_names = _TF_TENSORRT_LIBS
            nvrtc_ptxjit_static_libraries = []
        else:
            raw_static_library_names = _TF_TENSORRT_LIBS + ["nvrtc", "myelin_compiler", "myelin_executor", "myelin_pattern_library", "myelin_pattern_runtime"]
            nvrtc_ptxjit_static_libraries = []
        static_library_names = ["%s_static" % name for name in raw_static_library_names]
        static_libraries = [lib_name(lib, cpu_value, trt_version, static = True) for lib in static_library_names]
        copy_rules = copy_rules + [
            make_copy_files_rule(
                repository_ctx,
                name = "tensorrt_static_lib",
                srcs = [tensorrt_static_path + library for library in static_libraries] +
                       [cuda_library_path + library for library in nvrtc_ptxjit_static_libraries],
                outs = ["tensorrt/lib/" + library for library in static_libraries] +
                       ["tensorrt/lib/" + library for library in nvrtc_ptxjit_static_libraries],
            ),
        ]

    # Set up config file.
    _tpl(repository_ctx, "build_defs.bzl", {"%{if_tensorrt}": "if_true"})

    # Set up BUILD file.
    _tpl(repository_ctx, "BUILD", {
        "%{copy_rules}": "\n".join(copy_rules),
    })

    # Set up tensorrt_config.h, which is used by
    # tensorflow/stream_executor/dso_loader.cc.
    _tpl(repository_ctx, "tensorrt/include/tensorrt_config.h", {
        "%{tensorrt_version}": trt_version,
    })

tensorrt_configure = repository_rule(
    implementation = _tensorrt_configure_impl,
    environ = [
        _TENSORRT_INSTALL_PATH,
        _TF_TENSORRT_VERSION,
        _TF_TENSORRT_CONFIG_REPO,
        _TF_NEED_TENSORRT,
        _TF_TENSORRT_STATIC_PATH,
        "TF_CUDA_PATHS",
    ],
)
"""Detects and configures the local CUDA toolchain.

Add the following to your WORKSPACE FILE:

```python
tensorrt_configure(name = "local_config_tensorrt")
```

Args:
  name: A unique name for this workspace rule.
"""
