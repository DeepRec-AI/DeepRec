licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "seastar",
    srcs = glob(
        ["src/**/*.cc"],
        exclude = [
            "src/testing/*.cc",
			"demos/**",
			"tests/**",
			"build/**",
			"apps/**",
			"src/rpc/**",
			"src/core/prometheus.cc",
			"src/net/tls.cc",
			"src/core/dpdk_rte.cc",
			"hack-build/release/gen/proto/**"
        ],
    ) + glob(
        ["src/**/*.hh"],
    ),
    hdrs = glob(
        ["include/seastar/**/*.hh"],
        exclude = [
            "include/seastar/testing/*.hh",
        ],
    ) + [
        "include/seastar/http/request_parser.hh",
        "include/seastar/http/response_parser.hh",
    ],
    copts = [
        "-DSEASTAR_NO_EXCEPTION_HACK",
        "-DNO_EXCEPTION_INTERCEPT",
        "-DSEASTAR_DEFAULT_ALLOCATOR",
        "-DSEASTAR_HAVE_NUMA",
    ],
    includes = [
        "src",
    ],
    linkopts = [
        "-ldl",
        "-lm",
        "-lrt",
        "-lstdc++fs",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = [
        "@boost//:asio",
        "@boost//:filesystem",
        "@boost//:fusion",
        "@boost//:lockfree",
        "@boost//:program_options",
        "@boost//:system",
        "@boost//:thread",
        "@boost//:variant",
        "@cares",
        "@cryptopp",
        "@fmtlib",
        "@lz4",
        "@org_lzma_lzma//:lzma",
        "@readerwriterqueue",
        "@sctp",
        "@systemtap-sdt",
        "@xfs",
        "@yaml-cpp",
    ],
)

genrule(
    name = "generate_http_request_parser",
    srcs = ["src/http/request_parser.rl"],
    outs = ["include/seastar/http/request_parser.hh"],
    cmd = "\n".join([
        "$(location @ragel//:ragelc) -G2 -o $@ $<",
        "sed -i -e '1h;2,$$H;$$!d;g' -re 's/static const char _nfa[^;]*;//g' $@",
    ]),
    tools = ["@ragel//:ragelc"],
)

genrule(
    name = "generate_http_response_parser",
    srcs = ["src/http/response_parser.rl"],
    outs = ["include/seastar/http/response_parser.hh"],
    cmd = "\n".join([
        "$(location @ragel//:ragelc) -G2 -o $@ $<",
        "sed -i -e '1h;2,$$H;$$!d;g' -re 's/static const char _nfa[^;]*;//g' $@",
    ]),
    tools = ["@ragel//:ragelc"],
)
