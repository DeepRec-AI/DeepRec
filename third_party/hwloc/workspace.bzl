"""loads the hwloc library, used by TF."""

load("//third_party:repo.bzl", "third_party_http_archive")

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def repo():
    third_party_http_archive(
        name = "hwloc",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/download.open-mpi.org/release/hwloc/v2.0/hwloc-2.0.3.tar.gz",
            "https://download.open-mpi.org/release/hwloc/v2.0/hwloc-2.0.3.tar.gz",
        ],
        sha256 = "64def246aaa5b3a6e411ce10932a22e2146c3031b735c8f94739534f06ad071c",
        strip_prefix = "hwloc-2.0.3",
        patch_file = [clean_dep("//third_party/hwloc:hwloc_fix.patch")],
        build_file = clean_dep("//third_party/hwloc:BUILD.bazel"),
        system_build_file = "//third_party/hwloc:BUILD.system",
    )
