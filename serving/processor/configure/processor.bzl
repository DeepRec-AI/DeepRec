load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def tf_io_copts():
    return (
        select({
            "@bazel_tools//src/conditions:windows": [
                "/DEIGEN_STRONG_INLINE=inline",
                "-DTENSORFLOW_MONOLITHIC_BUILD",
                "/DPLATFORM_WINDOWS",
                "/DEIGEN_HAS_C99_MATH",
                "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
                "/DEIGEN_AVOID_STL_ARRAY",
                "/Iexternal/gemmlowp",
                "/wd4018",
                "/wd4577",
                "/DNOGDI",
                "/UTF_COMPILE_LIBRARY",
                "/DNDEBUG",
            ],
            "@bazel_tools//src/conditions:darwin": [
                "-DNDEBUG",
            ],
            "//conditions:default": [
                "-DNDEBUG",
                "-pthread",
            ],
        })
    )
