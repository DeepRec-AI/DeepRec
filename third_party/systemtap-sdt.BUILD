licenses(["unencumbered"])  # CC0 1.0 Public Domain

load("@//third_party:common.bzl", "template_rule")

cc_library(
    name = "systemtap-sdt",
    hdrs = [
        "includes/sys/sdt.h",
        "includes/sys/sdt-config.h",
    ],
    strip_include_prefix = "includes",
    visibility = ["//visibility:public"],
)

template_rule(
    name = "sdt_config_h",
    src = "includes/sys/sdt-config.h.in",
    out = "includes/sys/sdt-config.h",
    substitutions = {
        "#define _SDT_ASM_SECTION_AUTOGROUP_SUPPORT @support_section_question@": "#define _SDT_ASM_SECTION_AUTOGROUP_SUPPORT 1",
    },
)
