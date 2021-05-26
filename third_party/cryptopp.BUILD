licenses(["notice"])  # Boost

exports_files(["License.txt"])

cc_library(
    name = "cryptopp_internal",
    srcs = glob(["*.cpp"]) + glob(["*.h"]),
    copts = [
        "-fopenmp",
        "-msha",
        "-maes",
        "-mavx2",
        "-mpclmul",
    ],
    textual_hdrs = [
        "algebra.cpp",
        "strciphr.cpp",
        "eprecomp.cpp",
        "polynomi.cpp",
        "eccrypto.cpp",
    ],
)

cc_library(
    name = "cryptopp",
    hdrs = glob(["*.h"]),
    include_prefix = "cryptopp",
    visibility = ["//visibility:public"],
    deps = [":cryptopp_internal"],
)
