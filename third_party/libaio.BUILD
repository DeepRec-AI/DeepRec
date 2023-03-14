licenses(["notice"])

exports_files(["LICENSE"])

genrule(
    name = "libaio_lds",
    outs = ["libaio.lds"],
    cmd = """
      echo "LIBAIO_0.1 {                \
              global:                   \
                      io_queue_init;    \
                      io_queue_run;     \
                      io_queue_wait;    \
                      io_queue_release; \
                      io_cancel;        \
                      io_submit;        \
                      io_getevents;     \
              local:                    \
                      *;                \
      };                                \
      LIBAIO_0.4 {                      \
              global:                   \
                      io_setup;         \
                      io_destroy;       \
                      io_cancel;        \
                      io_getevents;     \
                      io_queue_wait;    \
      } LIBAIO_0.1;                     \
      LIBAIO_0.5 {                      \
              global:                   \
                      io_pgetevents;    \
      } LIBAIO_0.4;"  > $@
    """,
    stamp = 1,
)

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

    deps=[':libaio.lds'],
    linkopts = ["-Wl,--version-script=$(location libaio.lds)"]
)

