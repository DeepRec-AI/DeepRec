licenses(["notice"])  # MIT

exports_files(["COPYING"])

cc_library(
    name = "config_h",
    hdrs = glob(["stub-config/*.h"]),
    includes = ["stub-config"],
    visibility = ["//bin:__pkg__"],
)

cc_library(
    name = "ragel_aapl",
    hdrs = glob(["aapl/*.h"]),
    strip_include_prefix = "aapl",
)

genrule(
    name = "gen_rlreduce",
    srcs = [
        "src/rlparse.lm",
        "src/ragel.lm",
        "src/reducer.lm",
    ],
    outs = [
        "src/parse.c",
        "src/rlreduce.cc",
    ],
    cmd = """
$(location @colm) -c -b rl_parse \
    -o $(location :src/parse.c) \
    -m $(location :src/rlreduce.cc) \
    -I $$(dirname $(location :src/rlparse.lm)) \
    $(location :src/rlparse.lm)

# http://www.colm.net/pipermail/colm-users/2018-October/000204.html
# https://trac.macports.org/ticket/57242
sed -i.bak 's/#include <ext.stdio_filebuf.h>//' $(location :src/rlreduce.cc)
""",
    tools = ["@colm"],
)

cc_library(
    name = "ragel_lib",
    srcs = glob([
        "src/*.cc",
        "src/*.h",
    ]) + [
        "src/parse.c",
        "src/rlreduce.cc",
    ],
    copts = ['-DBINDIR=""'],
    features = ["no_copts_tokenization"],
    includes = ["src"],
    visibility = ["//bin:__pkg__"],
    deps = [
        ":config_h",
        ":ragel_aapl",
        "@colm//:runtime",
    ],
)

cc_binary(
    name = "ragelc",
    visibility = ["//visibility:public"],
    deps = ["//:ragel_lib"],
)
