# Description:
#   Expat library

licenses(["notice"])

exports_files(["COPYING"])

cc_library(
    name = "libexpat",
    srcs = [
        "lib/xmlparse.c",
        "lib/xmlrole.c",
        "lib/xmltok.c",
    ],
    hdrs = glob([
        "lib/*.h",
    ]) + [
        "lib/xmltok_impl.c",
        "lib/xmltok_ns.c",
    ],
    copts = [
        "-DHAVE_MEMMOVE",
        "-DXML_POOR_ENTROPY",
    ],
    includes = [
        "lib",
    ],
    visibility = ["//visibility:public"],
    deps = [],
)
