"""loads the aws library, used by TF."""

load("//third_party:repo.bzl", "third_party_http_archive")

# NOTE: version updates here should also update the major, minor, and patch variables declared in
# the  copts field of the //third_party/aws:aws target

def repo():
    third_party_http_archive(
        name = "aws",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/aws/aws-sdk-cpp/archive/1.8.0.tar.gz",
            "https://github.com/aws/aws-sdk-cpp/archive/1.8.0.tar.gz",
        ],
        sha256 = "2a69fb2d1a5effe2f053adafcf820535dc6d04bf37e2501cc8c0a8243b8c1f09",
        strip_prefix = "aws-sdk-cpp-1.8.0",
        build_file = "//third_party/aws:BUILD.bazel",
    )
