workspace(name = "org_tensorflow")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

http_archive(
    name="io_bazel_rules_docker",
    sha256="e2674bb36d5c39e3dfd28c18fb6f0568083c98209f0c5a0ee8eaf35ab4766f1d",
    strip_prefix="rules_docker-251f6a68b439744094faff800cd029798edf9faa",
    urls=[
      "https://github.com/bazelbuild/rules_docker/archive/251f6a68b439744094faff800cd029798edf9faa.tar.gz"
    ],
)

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

# Load tf_repositories() before loading dependencies for other repository so
# that dependencies like com_google_protobuf won't be overridden.
load("//tensorflow:workspace.bzl", "tf_repositories")
# Please add all new TensorFlow dependencies in workspace.bzl.
tf_repositories()

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")

closure_repositories()

load("//third_party/toolchains/preconfig/generate:archives.bzl",
     "bazel_toolchains_archive")

bazel_toolchains_archive()

load(
    "@bazel_toolchains//repositories:repositories.bzl",
    bazel_toolchains_repositories = "repositories",
)

bazel_toolchains_repositories()

load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)

container_repositories()

load("//third_party/toolchains/preconfig/generate:workspace.bzl",
     "remote_config_workspace")

remote_config_workspace()

# Apple and Swift rules.
http_archive(
    name = "build_bazel_rules_apple",
    sha256 = "6efdde60c91724a2be7f89b0c0a64f01138a45e63ba5add2dca2645d981d23a1",
    urls = ["https://github.com/bazelbuild/rules_apple/releases/download/0.17.2/rules_apple.0.17.2.tar.gz"],
)  # https://github.com/bazelbuild/rules_apple/releases
http_archive(
    name = "build_bazel_rules_swift",
    sha256 = "96a86afcbdab215f8363e65a10cf023b752e90b23abf02272c4fc668fcb70311",
    urls = ["https://github.com/bazelbuild/rules_swift/releases/download/0.11.1/rules_swift.0.11.1.tar.gz"],
)  # https://github.com/bazelbuild/rules_swift/releases
http_archive(
    name = "build_bazel_apple_support",
    sha256 = "7356dbd44dea71570a929d1d4731e870622151a5f27164d966dda97305f33471",
    urls = ["https://github.com/bazelbuild/apple_support/releases/download/0.6.0/apple_support.0.6.0.tar.gz"],
)  # https://github.com/bazelbuild/apple_support/releases
http_archive(
    name = "bazel_skylib",
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.8.0/bazel-skylib.0.8.0.tar.gz"],
)  # https://github.com/bazelbuild/bazel-skylib/releases
http_archive(
    name = "com_github_apple_swift_swift_protobuf",
    type = "zip",
    strip_prefix = "swift-protobuf-1.5.0/",
    urls = ["https://github.com/apple/swift-protobuf/archive/1.5.0.zip"],
)  # https://github.com/apple/swift-protobuf/releases
http_file(
    name = "xctestrunner",
    executable = 1,
    urls = ["https://github.com/google/xctestrunner/releases/download/0.2.7/ios_test_runner.par"],
)  # https://github.com/google/xctestrunner/releases
# Use `swift_rules_dependencies` to fetch the toolchains. With the
# `git_repository` rules above, the following call will skip redefining them.
load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")
swift_rules_dependencies()

# We must check the bazel version before trying to parse any other BUILD
# files, in case the parsing of those build files depends on the bazel
# version we require here.
load("//tensorflow:version_check.bzl", "check_bazel_version_at_least")
check_bazel_version_at_least("0.19.0")

load("//third_party/android:android_configure.bzl", "android_configure")
android_configure(name="local_config_android")
load("@local_config_android//:android.bzl", "android_workspace")
android_workspace()

# If a target is bound twice, the later one wins, so we have to do tf bindings
# at the end of the WORKSPACE file.
load("//tensorflow:workspace.bzl", "tf_bind")
tf_bind()

http_archive(
    name = "inception_v1",
    build_file = "//:models.BUILD",
    sha256 = "7efe12a8363f09bc24d7b7a450304a15655a57a7751929b2c1593a71183bb105",
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/models/inception_v1.zip",
    ],
)

http_archive(
    name = "mobile_ssd",
    build_file = "//:models.BUILD",
    sha256 = "bddd81ea5c80a97adfac1c9f770e6f55cbafd7cce4d3bbe15fbeb041e6b8f3e8",
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_android_export.zip",
    ],
)

http_archive(
    name = "mobile_multibox",
    build_file = "//:models.BUILD",
    sha256 = "859edcddf84dddb974c36c36cfc1f74555148e9c9213dedacf1d6b613ad52b96",
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/models/mobile_multibox_v1a.zip",
    ],
)

http_archive(
    name = "stylize",
    build_file = "//:models.BUILD",
    sha256 = "3d374a730aef330424a356a8d4f04d8a54277c425e274ecb7d9c83aa912c6bfa",
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/models/stylize_v1.zip",
    ],
)

http_archive(
    name = "speech_commands",
    build_file = "//:models.BUILD",
    sha256 = "c3ec4fea3158eb111f1d932336351edfe8bd515bb6e87aad4f25dbad0a600d0c",
    urls = [
        "https://storage.googleapis.com/download.tensorflow.org/models/speech_commands_v0.01.zip",
    ],
)

http_archive(
    name = "com_github_nelhage_rules_boost",
    sha256 = "4031539fe0af832c6b6ed6974d820d350299a291ba7337d6c599d4854e47ed88",
    strip_prefix = "rules_boost-4ee400beca08f524e7ea3be3ca41cce34454272f",
    urls = [
        "https://github.com/nelhage/rules_boost/archive/4ee400beca08f524e7ea3be3ca41cce34454272f.tar.gz",
    ],
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

http_archive(
    name = "colm",
    build_file = "//third_party:colm.BUILD",
    sha256 = "4644956dd82bedf3795bb1a6fdf9ee8bdd33bd1e7769ef81ffdaa3da70c5a349",
    strip_prefix = "colm-0.13.0.6",
    urls = ["http://www.colm.net/files/colm/colm-0.13.0.6.tar.gz"],
)

http_archive(
    name = "ragel",
    build_file = "//third_party:ragel.BUILD",
    sha256 = "08bac6ff8ea9ee7bdd703373fe4d39274c87fecf7ae594774dfdc4f4dd4a5340",
    strip_prefix = "ragel-7.0.0.11",
    urls = ["http://www.colm.net/files/ragel/ragel-7.0.0.11.tar.gz"],
)

http_archive(
    name = "sctp",
    build_file = "//third_party:sctp.BUILD",
    sha256 = "3e9ab5b3844a8b65fc8152633aafe85f406e6da463e53921583dfc4a443ff03a",
    strip_prefix = "lksctp-tools-1.0.18",
    urls = ["https://github.com/sctp/lksctp-tools/archive/v1.0.18.tar.gz"],
)

http_archive(
    name = "uuid",
    build_file = "//third_party:uuid.BUILD",
    sha256 = "37ac05d82c6410d89bc05d43cee101fefc8fe6cf6090b3ce7a1409a6f35db606",
    strip_prefix = "util-linux-2.35.1",
    urls = ["https://mirrors.edge.kernel.org/pub/linux/utils/util-linux/v2.35/util-linux-2.35.1.tar.gz"],
)

http_archive(
    name = "xfs",
    build_file = "//third_party:xfs.BUILD",
    sha256 = "cfbb0b136799c48cb79435facd0969c5a60a587a458e2d16f9752771027efbec",
    strip_prefix = "xfsprogs-5.5.0",
    urls = ["https://mirrors.edge.kernel.org/pub/linux/utils/fs/xfs/xfsprogs/xfsprogs-5.5.0.tar.xz"],
)

http_archive(
    name = "libaio",
    build_file = "//third_party:libaio.BUILD",
    sha256 = "b7cf93b29bbfb354213a0e8c0e82dfcf4e776157940d894750528714a0af2272",
    strip_prefix = "libaio-libaio-0.3.112",
    patches = [
        "//third_party:libaio.patch",
    ],
    urls = ["https://pagure.io/libaio/archive/libaio-0.3.112/libaio-libaio-0.3.112.tar.gz"],
)

http_archive(
    name = "aliyun_oss_c_sdk",
    build_file = "//serving/third_party:oss_c_sdk.BUILD",
    sha256 = "6450d3970578c794b23e9e1645440c6f42f63be3f82383097660db5cf2fba685",
    strip_prefix = "aliyun-oss-c-sdk-3.7.0",
    urls = [
        "http://pythonrun.oss-cn-zhangjiakou.aliyuncs.com/tensorflow_io/github.com/aliyun/aliyun-oss-c-sdk/archive/3.7.0.tar.gz",
    ],
)

http_archive(
    name = "concurrent_queue",
    build_file = "//serving/third_party:concurrent_queue.BUILD",
    sha256 = "c3aeb97c97169f743a53ca33812ea2ab61dd06dfd28319ca3f0a0771372fc7fc",
    strip_prefix = "concurrentqueue-1.0.2",
    urls = [
        "https://github.com/cameron314/concurrentqueue/archive/v1.0.2.tar.gz",
    ],
)


http_archive(
    name = "libevent",
    build_file = "//serving/third_party:libevent.BUILD",
    sha256 = "7180a979aaa7000e1264da484f712d403fcf7679b1e9212c4e3d09f5c93efc24",
    patches = [
        "//serving/third_party:libevent1.patch",
        "//serving/third_party:libevent.patch",
    ],
    urls = [
        "https://github.com/libevent/libevent/archive/release-2.1.12-stable.tar.gz",
    ],
)

http_archive(
    name = "hiredis",
    build_file = "//serving/third_party:hiredis.BUILD",
    sha256 = "e0ab696e2f07deb4252dda45b703d09854e53b9703c7d52182ce5a22616c3819",
    strip_prefix = "hiredis-1.0.2",
    urls = [
        "https://github.com/redis/hiredis/archive/v1.0.2.tar.gz",
    ],
)

http_archive(
    name = "libexpat",
    build_file = "//serving/third_party:libexpat.BUILD",
    sha256 = "574499cba22a599393e28d99ecfa1e7fc85be7d6651d543045244d5b561cb7ff",
    strip_prefix = "libexpat-R_2_2_6/expat",
    urls = [
        "http://pythonrun.oss-cn-zhangjiakou.aliyuncs.com/tensorflow_io/github.com/libexpat/libexpat/archive/R_2_2_6.tar.gz",
    ],
)

http_archive(
    name = "libapr1",
    build_file = "//serving/third_party:libapr1.BUILD",
    sha256 = "1a0909a1146a214a6ab9de28902045461901baab4e0ee43797539ec05b6dbae0",
    strip_prefix = "apr-1.6.5",
    patches = [
        "//serving/third_party:libapr1.patch",
    ],
    urls = [
        "http://pythonrun.oss-cn-zhangjiakou.aliyuncs.com/tensorflow_io/github.com/apache/apr/archive/1.6.5.tar.gz",
    ],
)

http_archive(
    name = "libaprutil1",
    build_file = "//serving/third_party:libaprutil1.BUILD",
    sha256 = "4c9ae319cedc16890fc2776920e7d529672dda9c3a9a9abd53bd80c2071b39af",
    strip_prefix = "apr-util-1.6.1",
    patches = [
        "//serving/third_party:libaprutil1.patch",
    ],
    urls = [
        "http://pythonrun.oss-cn-zhangjiakou.aliyuncs.com/tensorflow_io/github.com/apache/apr-util/archive/1.6.1.tar.gz",
    ],
)

http_archive(
    name = "mxml",
    build_file = "//serving/third_party:mxml.BUILD",
    sha256 = "4d850d15cdd4fdb9e82817eb069050d7575059a9a2729c82b23440e4445da199",
    strip_prefix = "mxml-2.12",
    patches = [
        "//serving/third_party:mxml.patch",
    ],
    urls = [
        "http://pythonrun.oss-cn-zhangjiakou.aliyuncs.com/tensorflow_io/github.com/michaelrsweet/mxml/archive/v2.12.tar.gz",
    ],
)

