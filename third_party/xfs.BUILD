licenses(["permissive"])  # LGPL headers only

exports_files(["LICENSES/LGPL-2.1"])

cc_library(
    name = "libxfs",
    hdrs = [
        "libxfs/xfs_da_format.h",
        "libxfs/xfs_format.h",
        "libxfs/xfs_fs.h",
        "libxfs/xfs_log_format.h",
        "libxfs/xfs_types.h",
    ],
    include_prefix = "xfs",
    strip_include_prefix = "libxfs",
)

cc_library(
    name = "xfs",
    hdrs = [
        "include/handle.h",
        "include/jdm.h",
        "include/linux.h",
        "include/xfs.h",
        "include/xfs_arch.h",
        "include/xqm.h",
    ],
    include_prefix = "xfs",
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = [
        ":libxfs",
        "@uuid",
    ],
)
