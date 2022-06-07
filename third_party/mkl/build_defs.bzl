# -*- Python -*-
"""Skylark macros for OneDNN.

if_mkl is a conditional to check if we are building with OneDNN.
if_mkl_ml is a conditional to check if we are building with MKL-ML.
if_mkl_ml_only is a conditional to check for MKL-ML-only (no OneDNN) mode.
if_mkl_lnx_x64 is a conditional to check for OneDNN
if_enable_mkl is a conditional to check if building with OneDNN and OneDNN is enabled.

mkl_repository is a repository rule for creating OneDNN repository rule that can
be pointed to either a local folder, or download it from the internet.
mkl_repository depends on the following environment variables:
  * `TF_MKL_ROOT`: The root folder where a copy of libmkl is located.
"""

_TF_MKL_ROOT = "TF_MKL_ROOT"

def if_mkl(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with OneDNN.

    Args:
      if_true: expression to evaluate if building with OneDNN.
      if_false: expression to evaluate if building without OneDNN.

    Returns:
      a select evaluating to either if_true or if_false as appropriate.
    """
    return select({
        str(Label("//third_party/mkl:build_with_mkl")): if_true,
        "//conditions:default": if_false,
    })

def if_mkl_ml(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with MKL-ML.

    Args:
      if_true: expression to evaluate if building with MKL-ML.
      if_false: expression to evaluate if building without MKL-ML
        (i.e. without OneDNN at all, or with OneDNN only).

    Returns:
      a select evaluating to either if_true or if_false as appropriate.
    """
    return select({
        str(Label("//third_party/mkl_dnn:build_with_mkl_opensource")): if_false,
        str(Label("//third_party/mkl:build_with_mkl")): if_true,
        "//conditions:default": if_false,
    })

def if_mkl_lnx_x64(if_true, if_false = []):
    """Shorthand to select() if building with OneDNN and the target is Linux x86-64.

    Args:
      if_true: expression to evaluate if building with OneDNN is enabled and the
        target platform is Linux x86-64.
      if_false: expression to evaluate if building without OneDNN or for a
        different platform.

    Returns:
      a select evaluating to either if_true or if_false as appropriate.
    """
    return select({
        str(Label("//third_party/mkl:build_with_mkl_lnx_x64")): if_true,
        "//conditions:default": if_false,
    })

def if_enable_mkl(if_true, if_false = []):
    """Shorthand to select() if we are building with OneDNN and OneDNN is enabled.

    This is only effective when built with OneDNN.

    Args:
      if_true: expression to evaluate if building with OneDNN and OneDNN is enabled
      if_false: expression to evaluate if building without OneDNN or OneDNN is not enabled.

    Returns:
      A select evaluating to either if_true or if_false as appropriate.
    """
    return select({
        str(Label("//third_party/mkl:enable_mkl")): if_true,
        "//conditions:default": if_false,
    })

def mkl_deps():
    """Shorthand for select() to pull in the correct set of OneDNN library deps.

    Can pull in MKL-ML, OneDNN, both, or neither depending on config settings.

    Returns:
      a select evaluating to a list of library dependencies, suitable for
      inclusion in the deps attribute of rules.
    """
    return select({
        "@org_tensorflow//third_party/mkl:build_with_mkl": ["@mkl_dnn_v1//:mkl_dnn"],
        "//conditions:default": [],
    })

def _enable_local_mkl(repository_ctx):
    return _TF_MKL_ROOT in repository_ctx.os.environ

def _mkl_autoconf_impl(repository_ctx):
    """Implementation of the local_mkl_autoconf repository rule."""

    if _enable_local_mkl(repository_ctx):
        # Symlink lib and include local folders.
        mkl_root = repository_ctx.os.environ[_TF_MKL_ROOT]
        mkl_lib_path = "%s/lib" % mkl_root
        repository_ctx.symlink(mkl_lib_path, "lib")
        mkl_include_path = "%s/include" % mkl_root
        repository_ctx.symlink(mkl_include_path, "include")
        mkl_license_path = "%s/license.txt" % mkl_root
        repository_ctx.symlink(mkl_license_path, "license.txt")
    else:
        # setup remote OneDNN repository.
        repository_ctx.download_and_extract(
            repository_ctx.attr.urls,
            sha256 = repository_ctx.attr.sha256,
            stripPrefix = repository_ctx.attr.strip_prefix,
        )

    # Also setup BUILD file.
    repository_ctx.symlink(repository_ctx.attr.build_file, "BUILD")

mkl_repository = repository_rule(
    implementation = _mkl_autoconf_impl,
    environ = [
        _TF_MKL_ROOT,
    ],
    attrs = {
        "build_file": attr.label(),
        "urls": attr.string_list(default = []),
        "sha256": attr.string(default = ""),
        "strip_prefix": attr.string(default = ""),
    },
)
