licenses(["notice"])  # MIT

exports_files(["COPYING"])

cc_library(
    name = "aapl",
    hdrs = glob(["aapl/*.h"]),
    strip_include_prefix = "aapl",
)

_RUNTIME_SRCS = [
    "src/map.c",
    "src/pdarun.c",
    "src/list.c",
    "src/input.c",
    "src/debug.c",
    "src/codevect.c",
    "src/pool.c",
    "src/string.c",
    "src/tree.c",
    "src/iter.c",
    "src/bytecode.c",
    "src/program.c",
    "src/struct.c",
    "src/commit.c",
    "src/print.c",
]

_RUNTIME_HDRS = [
    "src/bytecode.h",
    "src/config.h",
    "src/defs.h",
    "src/debug.h",
    "src/pool.h",
    "src/input.h",
    "src/pdarun.h",
    "src/map.h",
    "src/type.h",
    "src/tree.h",
    "src/struct.h",
    "src/program.h",
    "src/colm.h",
    "src/internal.h",
]

_PROG_SRCS = [
    "src/resolve.cc",
    "src/lookup.cc",
    "src/synthesis.cc",
    "src/parsetree.cc",
    "src/fsmstate.cc",
    "src/fsmbase.cc",
    "src/fsmattach.cc",
    "src/fsmmin.cc",
    "src/fsmgraph.cc",
    "src/pdagraph.cc",
    "src/pdabuild.cc",
    "src/pdacodegen.cc",
    "src/fsmcodegen.cc",
    "src/redfsm.cc",
    "src/fsmexec.cc",
    "src/redbuild.cc",
    "src/closure.cc",
    "src/fsmap.cc",
    "src/dotgen.cc",
    "src/pcheck.cc",
    "src/ctinput.cc",
    "src/declare.cc",
    "src/codegen.cc",
    "src/exports.cc",
    "src/compiler.cc",
    "src/parser.cc",
    "src/reduce.cc",
]

_PROG_HDRS = [
    "src/buffer.h",
    "src/bytecode.h",
    "src/colm.h",
    "src/debug.h",
    "src/dotgen.h",
    "src/fsmcodegen.h",
    "src/fsmgraph.h",
    "src/input.h",
    "src/keyops.h",
    "src/map.h",
    "src/compiler.h",
    "src/parsetree.h",
    "src/pcheck.h",
    "src/pdacodegen.h",
    "src/pdagraph.h",
    "src/pdarun.h",
    "src/pool.h",
    "src/redbuild.h",
    "src/redfsm.h",
    "src/rtvector.h",
    "src/tree.h",
    "src/version.h",
    "src/global.h",
    "src/parser.h",
    "src/cstring.h",
]

cc_library(
    name = "runtime_isystem",
    hdrs = _RUNTIME_HDRS,
    include_prefix = "colm",
    includes = ["src"],
    strip_include_prefix = "src",
)

cc_library(
    name = "runtime_iquote",
    hdrs = _RUNTIME_HDRS,
    strip_include_prefix = "src",
    deps = [":runtime_isystem"],
)

cc_library(
    name = "prog_isystem",
    hdrs = _PROG_HDRS,
    include_prefix = "colm",
    includes = ["src"],
    strip_include_prefix = "src",
)

cc_library(
    name = "prog_iquote",
    hdrs = _PROG_HDRS,
    strip_include_prefix = "src",
    deps = [":prog_isystem"],
)

cc_library(
    name = "runtime",
    srcs = _RUNTIME_SRCS,
    visibility = ["//visibility:public"],
    deps = [
        ":prog_iquote",
        ":runtime_iquote",
    ],
)

cc_library(
    name = "prog",
    srcs = _PROG_SRCS,
    deps = [
        "runtime_iquote",
        ":aapl",
        ":prog_iquote",
    ],
)

cc_binary(
    name = "bootstrap0",
    srcs = [
        "src/consinit.cc",
        "src/consinit.h",
        "src/main.cc",
    ],
    copts = [
        '-DPREFIX=""',
        "-DCONS_INIT",
    ],
    features = ["no_copts_tokenization"],
    deps = [
        ":prog",
        ":runtime",
    ],
)

genrule(
    name = "run_bootstrap0",
    outs = [
        "src/gen/parse1.c",
        "src/gen/if1.h",
        "src/gen/if1.cc",
    ],
    cmd = """
mkdir -p gen
$(location :bootstrap0) -c -o gen/parse1.c -e gen/if1.h -x gen/if1.cc
mv gen/parse1.c $(location :src/gen/parse1.c)
mv gen/if1.h $(location :src/gen/if1.h)
mv gen/if1.cc $(location :src/gen/if1.cc)
""",
    tools = [":bootstrap0"],
)

cc_library(
    name = "gen_if1",
    srcs = [
        "src/gen/if1.cc",
        "src/gen/parse1.c",
    ],
    hdrs = ["src/gen/if1.h"],
    strip_include_prefix = "src",
    deps = [
        "runtime_iquote",
        ":aapl",
        ":prog_iquote",
    ],
)

cc_binary(
    name = "bootstrap1",
    srcs = [
        "src/loadinit.cc",
        "src/loadinit.h",
        "src/main.cc",
    ],
    copts = [
        '-DPREFIX=""',
        "-DLOAD_INIT",
    ],
    features = ["no_copts_tokenization"],
    deps = [
        ":gen_if1",
        ":prog",
        ":runtime",
    ],
)

genrule(
    name = "run_bootstrap1",
    srcs = ["src/colm.lm"],
    outs = [
        "src/gen/parse2.c",
        "src/gen/if2.h",
        "src/gen/if2.cc",
    ],
    cmd = """
mkdir -p gen
$(location :bootstrap1) -c -o gen/parse2.c -e gen/if2.h -x gen/if2.cc $(location :src/colm.lm)
mv gen/parse2.c $(location :src/gen/parse2.c)
mv gen/if2.h $(location :src/gen/if2.h)
mv gen/if2.cc $(location :src/gen/if2.cc)
""",
    tools = [":bootstrap1"],
)

cc_library(
    name = "gen_if2",
    srcs = [
        "src/gen/if2.cc",
        "src/gen/parse2.c",
    ],
    hdrs = ["src/gen/if2.h"],
    strip_include_prefix = "src",
    deps = [
        "runtime_iquote",
        ":aapl",
        ":prog_iquote",
    ],
)

cc_binary(
    name = "colm",
    srcs = [
        "src/loadcolm.cc",
        "src/loadcolm.h",
        "src/main.cc",
    ],
    copts = [
        '-DPREFIX=""',
        "-DLOAD_COLM",
    ],
    features = ["no_copts_tokenization"],
    visibility = ["//visibility:public"],
    deps = [
        ":gen_if2",
        ":prog",
        ":runtime",
    ],
)
