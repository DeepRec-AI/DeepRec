licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "libaio",
    srcs = glob(["src/*.c"]),
    hdrs = glob(["src/*.h"]),
    includes = ["./src"],
    visibility = ["//visibility:public"],
    copts = [
      "-fomit-frame-pointer",
      "-O2",
      "-Wall",
      "-fPIC",
      "-shared",
    ],

    deps=['libaio.lds'],
    linkopts = ["-Wl,--version-script=$(location libaio.lds)"]
)

