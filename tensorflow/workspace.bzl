# TensorFlow external dependencies that can be loaded in WORKSPACE files.

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("//third_party/gpus:rocm_configure.bzl", "rocm_configure")
load("//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")
load("//third_party/nccl:nccl_configure.bzl", "nccl_configure")
load("//third_party/mkl:build_defs.bzl", "mkl_repository")
load("//third_party/git:git_configure.bzl", "git_configure")
load("//third_party/py:python_configure.bzl", "python_configure")
load("//third_party/sycl:sycl_configure.bzl", "sycl_configure")
load("//third_party/systemlibs:syslibs_configure.bzl", "syslibs_configure")
load("//third_party/toolchains/remote:configure.bzl", "remote_execution_configure")
load("//third_party/toolchains/clang6:repo.bzl", "clang6_configure")
load("//third_party/toolchains/cpus/arm:arm_compiler_configure.bzl", "arm_compiler_configure")
load("//third_party:repo.bzl", "tf_http_archive")
load("//third_party/clang_toolchain:cc_configure_clang.bzl", "cc_download_clang_toolchain")
load("@io_bazel_rules_closure//closure/private:java_import_external.bzl", "java_import_external")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@io_bazel_rules_closure//closure:defs.bzl", "filegroup_external")
load(
    "//tensorflow/tools/def_file_filter:def_file_filter_configure.bzl",
    "def_file_filter_configure",
)
load("//third_party/FP16:workspace.bzl", FP16 = "repo")
load("//third_party/aws:workspace.bzl", aws = "repo")
load("//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
load("//third_party/highwayhash:workspace.bzl", highwayhash = "repo")
load("//third_party/hwloc:workspace.bzl", hwloc = "repo")
load("//third_party/icu:workspace.bzl", icu = "repo")
load("//third_party/jpeg:workspace.bzl", jpeg = "repo")
load("//third_party/nasm:workspace.bzl", nasm = "repo")
load("//third_party/opencl_headers:workspace.bzl", opencl_headers = "repo")
load("//third_party/kissfft:workspace.bzl", kissfft = "repo")
load("//third_party/keras_applications_archive:workspace.bzl", keras_applications = "repo")
load("//third_party/pasta:workspace.bzl", pasta = "repo")

def initialize_third_party():
    """ Load third party repositories.  See above load() statements. """
    FP16()
    aws()
    flatbuffers()
    highwayhash()
    hwloc()
    icu()
    keras_applications()
    kissfft()
    jpeg()
    nasm()
    opencl_headers()
    pasta()

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
    return str(Label(dep))

# If TensorFlow is linked as a submodule.
# path_prefix is no longer used.
# tf_repo_name is thought to be under consideration.
def tf_workspace(path_prefix = "", tf_repo_name = ""):
    tf_repositories(path_prefix, tf_repo_name)
    tf_bind()

# Define all external repositories required by TensorFlow
def tf_repositories(path_prefix = "", tf_repo_name = ""):
    """All external dependencies for TF builds."""

    # Note that we check the minimum bazel version in WORKSPACE.
    clang6_configure(name = "local_config_clang6")
    cc_download_clang_toolchain(name = "local_config_download_clang")
    cuda_configure(name = "local_config_cuda")
    tensorrt_configure(name = "local_config_tensorrt")
    nccl_configure(name = "local_config_nccl")
    git_configure(name = "local_config_git")
    sycl_configure(name = "local_config_sycl")
    syslibs_configure(name = "local_config_syslibs")
    python_configure(name = "local_config_python")
    rocm_configure(name = "local_config_rocm")
    remote_execution_configure(name = "local_config_remote_execution")

    initialize_third_party()

    # For windows bazel build
    # TODO: Remove def file filter when TensorFlow can export symbols properly on Windows.
    def_file_filter_configure(name = "local_config_def_file_filter")

    # Point //external/local_config_arm_compiler to //external/arm_compiler
    arm_compiler_configure(
        name = "local_config_arm_compiler",
        build_file = clean_dep("//third_party/toolchains/cpus/arm:BUILD"),
        remote_config_repo = "../arm_compiler",
    )

    mkl_repository(
        name = "mkl_linux",
        build_file = clean_dep("//third_party/mkl:mkl.BUILD"),
        sha256 = "a936d6b277a33d2a027a024ea8e65df62bd2e162c7ca52c48486ed9d5dc27160",
        strip_prefix = "mklml_lnx_2019.0.5.20190502",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/intel/mkl-dnn/releases/download/v0.21/mklml_lnx_2019.0.5.20190502.tgz",
            "https://github.com/intel/mkl-dnn/releases/download/v0.21/mklml_lnx_2019.0.5.20190502.tgz",
        ],
    )
    mkl_repository(
        name = "mkl_windows",
        build_file = clean_dep("//third_party/mkl:mkl.BUILD"),
        sha256 = "33cc27652df3b71d7cb84b26718b5a2e8965e2c864a502347db02746d0430d57",
        strip_prefix = "mklml_win_2020.0.20190813",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/intel/mkl-dnn/releases/download/v0.21/mklml_win_2020.0.20190813.zip",
            "https://github.com/intel/mkl-dnn/releases/download/v0.21/mklml_win_2020.0.20190813.zip",
        ],
    )
    mkl_repository(
        name = "mkl_darwin",
        build_file = clean_dep("//third_party/mkl:mkl.BUILD"),
        sha256 = "2fbb71a0365d42a39ea7906568d69b1db3bfc9914fee75eedb06c5f32bf5fa68",
        strip_prefix = "mklml_mac_2019.0.5.20190502",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/intel/mkl-dnn/releases/download/v0.21/mklml_mac_2019.0.5.20190502.tgz",
            "https://github.com/intel/mkl-dnn/releases/download/v0.21/mklml_mac_2019.0.5.20190502.tgz",
        ],
    )

    if path_prefix:
        print("path_prefix was specified to tf_workspace but is no longer used " +
              "and will be removed in the future.")

    tf_http_archive(
        name = "cudnn_frontend_archive",
        build_file = clean_dep("//third_party:cudnn_frontend.BUILD"),
        patch_file = clean_dep("//third_party:cudnn_frontend_header_fix.patch"),
        sha256 = "314569f65d5c7d05fb7e90157a838549db3e2cfb464c80a6a399b39a004690fa",
        strip_prefix = "cudnn-frontend-0.6.2",
        urls = [
            "https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v0.6.2.zip",
            "https://github.com/AlibabaPAI/cudnn-frontend/archive/refs/tags/v0.6.2.zip",
        ],
    )

    # Important: If you are upgrading MKL-DNN, then update the version numbers
    # in third_party/mkl_dnn/mkldnn.BUILD. In addition, the new version of
    # MKL-DNN might require upgrading MKL ML libraries also. If they need to be
    # upgraded then update the version numbers on all three versions above
    # (Linux, Mac, Windows).
    tf_http_archive(
        name = "mkl_dnn",
        build_file = clean_dep("//third_party/mkl_dnn:mkldnn.BUILD"),
        sha256 = "a0211aeb5e7dad50b97fa5dffc1a2fe2fe732572d4164e1ee8750a2ede43fbec",
        strip_prefix = "oneDNN-0.21.3",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/oneapi-src/oneDNN/archive/v0.21.3.tar.gz",
            "https://github.com/oneapi-src/oneDNN/archive/v0.21.3.tar.gz",
        ],
    )

    tf_http_archive(
        name = "mkl_dnn_v1",
        build_file = clean_dep("//third_party/mkl_dnn:mkldnn_v1.BUILD"),
        patch_file = clean_dep("//third_party/mkl_dnn:oneDNN-v2.3.2-expose-bf16-tp.patch"),
        sha256 = "9695640f55acd833ddcef4776af15e03446c4655f9296e5074b1b178dd7a4fb2",
        strip_prefix = "oneDNN-2.6",
        urls = [
            "https://github.com/oneapi-src/oneDNN/archive/v2.6.tar.gz",
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/oneapi-src/oneDNN/archive/v2.6.tar.gz",
        ],
    )

    tf_http_archive(
        name = "sparsehash_c11",
        patch_file = clean_dep("//third_party:0001-Avoid-fetching-nullptr-when-use-featrue-filter.patch"),
        build_file = clean_dep("//third_party:sparsehash_c11.BUILD"),
        sha256 = "d4a43cad1e27646ff0ef3a8ce3e18540dbcb1fdec6cc1d1cb9b5095a9ca2a755",
        strip_prefix = "sparsehash-c11-2.11.1",
        urls = [
            "https://github.com/sparsehash/sparsehash-c11/archive/v2.11.1.tar.gz",
            "https://github.com/sparsehash/sparsehash-c11/archive/v2.11.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "cuCollections",
        patch_file = clean_dep("//third_party:0001-cuco-modification-for-deeprec.patch"),
        build_file = clean_dep("//third_party:cuco.BUILD"),
        sha256 = "c5c77a1f96b439b67280e86483ce8d5994aa4d14b7627b1d3bd7880be6be23fa",
        strip_prefix = "cuCollections-193de1aa74f5721717f991ca757dc610c852bb17",
        urls = [
            "https://github.com/NVIDIA/cuCollections/archive/193de1aa74f5721717f991ca757dc610c852bb17.zip",
            "https://github.com/NVIDIA/cuCollections/archive/193de1aa74f5721717f991ca757dc610c852bb17.zip",
        ],
    )

    tf_http_archive(
        name = "com_google_absl",
        patch_file = clean_dep("//third_party:string_view_h.patch"),
        build_file = clean_dep("//third_party:com_google_absl.BUILD"),
        sha256 = "acd93f6baaedc4414ebd08b33bebca7c7a46888916101d8c0b8083573526d070",  # SHARED_ABSL_SHA
        strip_prefix = "abseil-cpp-43ef2148c0936ebf7cb4be6b19927a9d9d145b8f",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/43ef2148c0936ebf7cb4be6b19927a9d9d145b8f.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/43ef2148c0936ebf7cb4be6b19927a9d9d145b8f.tar.gz",
        ],
    )

    tf_http_archive(
        name = "eigen_archive",
        build_file = clean_dep("//third_party:eigen.BUILD"),
        patch_file = clean_dep("//third_party/eigen3:eigen.patch"),
        sha256 = "2f046557f4093becf51b44c6339873c18e2f1ea55c4b3f3a08b7d15a1d9c6e5b",  # SHARED_EIGEN_SHA
        strip_prefix = "eigen-4fd5d1477b221fc7daf2b7f1c7e4ee4f04ceaced",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/4fd5d1477b221fc7daf2b7f1c7e4ee4f04ceaced/eigen-4fd5d1477b221fc7daf2b7f1c7e4ee4f04ceaced.tar.gz",
            "https://gitlab.com/libeigen/eigen/-/archive/4fd5d1477b221fc7daf2b7f1c7e4ee4f04ceaced/eigen-4fd5d1477b221fc7daf2b7f1c7e4ee4f04ceaced.tar.gz",
        ],
    )

    tf_http_archive(
        name = "arm_compiler",
        build_file = clean_dep("//:arm_compiler.BUILD"),
        sha256 = "b9e7d50ffd9996ed18900d041d362c99473b382c0ae049b2fce3290632d2656f",
        strip_prefix = "rpi-newer-crosstools-eb68350c5c8ec1663b7fe52c742ac4271e3217c5/x64-gcc-6.5.0/arm-rpi-linux-gnueabihf/",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/rvagg/rpi-newer-crosstools/archive/eb68350c5c8ec1663b7fe52c742ac4271e3217c5.tar.gz",
            "https://github.com/rvagg/rpi-newer-crosstools/archive/eb68350c5c8ec1663b7fe52c742ac4271e3217c5.tar.gz",
        ],
    )

    tf_http_archive(
        name = "libxsmm_archive",
        build_file = clean_dep("//third_party:libxsmm.BUILD"),
        sha256 = "5fc1972471cd8e2b8b64ea017590193739fc88d9818e3d086621e5c08e86ea35",
        strip_prefix = "libxsmm-1.11",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/hfp/libxsmm/archive/1.11.tar.gz",
            "https://github.com/hfp/libxsmm/archive/1.11.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "d070e2ffc5476c496a6a872a6f246bfddce8e7797d6ba605a7c8d72866743bf9",
        strip_prefix = "re2-506cfa4bffd060c06ec338ce50ea3468daa6c814",
        system_build_file = clean_dep("//third_party/systemlibs:re2.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/re2/archive/506cfa4bffd060c06ec338ce50ea3468daa6c814.tar.gz",
            "https://github.com/google/re2/archive/506cfa4bffd060c06ec338ce50ea3468daa6c814.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_github_googlecloudplatform_google_cloud_cpp",
        sha256 = "fd0c3e3b50f32af332b53857f8cd1bfa009e33d1eeecabc5c79a4825d906a90c",
        strip_prefix = "google-cloud-cpp-0.10.0",
        system_build_file = clean_dep("//third_party/systemlibs:google_cloud_cpp.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:google_cloud_cpp.google.cloud.bigtable.BUILD": "google/cloud/bigtable/BUILD",
        },
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/googleapis/google-cloud-cpp/archive/v0.10.0.tar.gz",
            "https://github.com/googleapis/google-cloud-cpp/archive/v0.10.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_github_googleapis_googleapis",
        build_file = clean_dep("//third_party:googleapis.BUILD"),
        sha256 = "824870d87a176f26bcef663e92051f532fac756d1a06b404055dc078425f4378",
        strip_prefix = "googleapis-f81082ea1e2f85c43649bee26e0d9871d4b41cdb",
        system_build_file = clean_dep("//third_party/systemlibs:googleapis.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/googleapis/googleapis/archive/f81082ea1e2f85c43649bee26e0d9871d4b41cdb.zip",
            "https://github.com/googleapis/googleapis/archive/f81082ea1e2f85c43649bee26e0d9871d4b41cdb.zip",
        ],
    )

    tf_http_archive(
        name = "gemmlowp",
        sha256 = "6678b484d929f2d0d3229d8ac4e3b815a950c86bb9f17851471d143f6d4f7834",  # SHARED_GEMMLOWP_SHA
        strip_prefix = "gemmlowp-12fed0cd7cfcd9e169bf1925bc3a7a58725fdcc3",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/gemmlowp/archive/12fed0cd7cfcd9e169bf1925bc3a7a58725fdcc3.zip",
            "https://github.com/google/gemmlowp/archive/12fed0cd7cfcd9e169bf1925bc3a7a58725fdcc3.zip",
        ],
    )

    tf_http_archive(
        name = "farmhash_archive",
        build_file = clean_dep("//third_party:farmhash.BUILD"),
        sha256 = "6560547c63e4af82b0f202cb710ceabb3f21347a4b996db565a411da5b17aba0",  # SHARED_FARMHASH_SHA
        strip_prefix = "farmhash-816a4ae622e964763ca0862d9dbd19324a1eaf45",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
            "https://github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
        ],
    )

    tf_http_archive(
        name = "png_archive",
        build_file = clean_dep("//third_party:png.BUILD"),
        patch_file = clean_dep("//third_party:png_fix_rpi.patch"),
        sha256 = "ca74a0dace179a8422187671aee97dd3892b53e168627145271cad5b5ac81307",
        strip_prefix = "libpng-1.6.37",
        system_build_file = clean_dep("//third_party/systemlibs:png.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/glennrp/libpng/archive/v1.6.37.tar.gz",
            "https://github.com/glennrp/libpng/archive/v1.6.37.tar.gz",
        ],
    )

    tf_http_archive(
        name = "org_sqlite",
        build_file = clean_dep("//third_party:sqlite.BUILD"),
        sha256 = "8ff0b79fd9118af7a760f1f6a98cac3e69daed325c8f9f0a581ecb62f797fd64",
        strip_prefix = "sqlite-amalgamation-3340000",
        system_build_file = clean_dep("//third_party/systemlibs:sqlite.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/www.sqlite.org/2020/sqlite-amalgamation-3340000.zip",
            "https://www.sqlite.org/2020/sqlite-amalgamation-3340000.zip",
        ],
    )

    tf_http_archive(
        name = "gif_archive",
        build_file = clean_dep("//third_party:gif.BUILD"),
        patch_file = clean_dep("//third_party:gif_fix_strtok_r.patch"),
        sha256 = "31da5562f44c5f15d63340a09a4fd62b48c45620cd302f77a6d9acf0077879bd",
        strip_prefix = "giflib-5.2.1",
        system_build_file = clean_dep("//third_party/systemlibs:gif.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pilotfiber.dl.sourceforge.net/project/giflib/giflib-5.2.1.tar.gz",
            "https://pilotfiber.dl.sourceforge.net/project/giflib/giflib-5.2.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "six_archive",
        build_file = clean_dep("//third_party:six.BUILD"),
        sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
        strip_prefix = "six-1.10.0",
        system_build_file = clean_dep("//third_party/systemlibs:six.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
            "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "astor_archive",
        build_file = clean_dep("//third_party:astor.BUILD"),
        sha256 = "95c30d87a6c2cf89aa628b87398466840f0ad8652f88eb173125a6df8533fb8d",
        strip_prefix = "astor-0.7.1",
        system_build_file = clean_dep("//third_party/systemlibs:astor.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/99/80/f9482277c919d28bebd85813c0a70117214149a96b08981b72b63240b84c/astor-0.7.1.tar.gz",
            "https://pypi.python.org/packages/99/80/f9482277c919d28bebd85813c0a70117214149a96b08981b72b63240b84c/astor-0.7.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "astunparse_archive",
        build_file = clean_dep("//third_party:astunparse.BUILD"),
        sha256 = "5ad93a8456f0d084c3456d059fd9a92cce667963232cbf763eac3bc5b7940872",
        strip_prefix = "astunparse-1.6.3/lib",
        system_build_file = clean_dep("//third_party/systemlibs:astunparse.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/files.pythonhosted.org/packages/f3/af/4182184d3c338792894f34a62672919db7ca008c89abee9b564dd34d8029/astunparse-1.6.3.tar.gz",
            "https://files.pythonhosted.org/packages/f3/af/4182184d3c338792894f34a62672919db7ca008c89abee9b564dd34d8029/astunparse-1.6.3.tar.gz",
        ],
    )

    filegroup_external(
        name = "astunparse_license",
        licenses = ["notice"],  # PSFL
        sha256_urls = {
            "92fc0e4f4fa9460558eedf3412b988d433a2dcbb3a9c45402a145a4fab8a6ac6": [
                "http://mirror.tensorflow.org/raw.githubusercontent.com/simonpercivall/astunparse/v1.6.2/LICENSE",
                "https://raw.githubusercontent.com/simonpercivall/astunparse/v1.6.2/LICENSE",
            ],
        },
    )

    tf_http_archive(
        name = "functools32_archive",
        build_file = clean_dep("//third_party:functools32.BUILD"),
        sha256 = "f6253dfbe0538ad2e387bd8fdfd9293c925d63553f5813c4e587745416501e6d",
        strip_prefix = "functools32-3.2.3-2",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/c5/60/6ac26ad05857c601308d8fb9e87fa36d0ebf889423f47c3502ef034365db/functools32-3.2.3-2.tar.gz",
            "https://pypi.python.org/packages/c5/60/6ac26ad05857c601308d8fb9e87fa36d0ebf889423f47c3502ef034365db/functools32-3.2.3-2.tar.gz",
        ],
    )

    tf_http_archive(
        name = "gast_archive",
        build_file = clean_dep("//third_party:gast.BUILD"),
        sha256 = "b881ef288a49aa81440d2c5eb8aeefd4c2bb8993d5f50edae7413a85bfdb3b57",
        strip_prefix = "gast-0.3.3",
        system_build_file = clean_dep("//third_party/systemlibs:gast.BUILD"),
        urls = [
            "http://mirror.tensorflow.org/files.pythonhosted.org/packages/12/59/eaa15ab9710a20e22225efd042cd2d6a0b559a0656d5baba9641a2a4a921/gast-0.3.3.tar.gz",
            "https://files.pythonhosted.org/packages/12/59/eaa15ab9710a20e22225efd042cd2d6a0b559a0656d5baba9641a2a4a921/gast-0.3.3.tar.gz",
        ],
    )

    tf_http_archive(
        name = "termcolor_archive",
        build_file = clean_dep("//third_party:termcolor.BUILD"),
        sha256 = "1d6d69ce66211143803fbc56652b41d73b4a400a2891d7bf7a1cdf4c02de613b",
        strip_prefix = "termcolor-1.1.0",
        system_build_file = clean_dep("//third_party/systemlibs:termcolor.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/8a/48/a76be51647d0eb9f10e2a4511bf3ffb8cc1e6b14e9e4fab46173aa79f981/termcolor-1.1.0.tar.gz",
            "https://pypi.python.org/packages/8a/48/a76be51647d0eb9f10e2a4511bf3ffb8cc1e6b14e9e4fab46173aa79f981/termcolor-1.1.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "opt_einsum_archive",
        build_file = clean_dep("//third_party:opt_einsum.BUILD"),
        sha256 = "d3d464b4da7ef09e444c30e4003a27def37f85ff10ff2671e5f7d7813adac35b",
        strip_prefix = "opt_einsum-2.3.2",
        system_build_file = clean_dep("//third_party/systemlibs:opt_einsum.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/f6/d6/44792ec668bcda7d91913c75237314e688f70415ab2acd7172c845f0b24f/opt_einsum-2.3.2.tar.gz",
            "https://pypi.python.org/packages/f6/d6/44792ec668bcda7d91913c75237314e688f70415ab2acd7172c845f0b24f/opt_einsum-2.3.2.tar.gz",
        ],
    )

    tf_http_archive(
        name = "absl_py",
        sha256 = "603febc9b95a8f2979a7bdb77d2f5e4d9b30d4e0d59579f88eba67d4e4cc5462",
        strip_prefix = "abseil-py-pypi-v0.9.0",
        system_build_file = clean_dep("//third_party/systemlibs:absl_py.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:absl_py.absl.BUILD": "absl/BUILD",
            "//third_party/systemlibs:absl_py.absl.flags.BUILD": "absl/flags/BUILD",
            "//third_party/systemlibs:absl_py.absl.testing.BUILD": "absl/testing/BUILD",
        },
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-py/archive/pypi-v0.9.0.tar.gz",
            "https://github.com/abseil/abseil-py/archive/pypi-v0.9.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "enum34_archive",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/bf/3e/31d502c25302814a7c2f1d3959d2a3b3f78e509002ba91aea64993936876/enum34-1.1.6.tar.gz",
            "https://pypi.python.org/packages/bf/3e/31d502c25302814a7c2f1d3959d2a3b3f78e509002ba91aea64993936876/enum34-1.1.6.tar.gz",
        ],
        sha256 = "8ad8c4783bf61ded74527bffb48ed9b54166685e4230386a9ed9b1279e2df5b1",
        build_file = clean_dep("//third_party:enum34.BUILD"),
        system_build_file = clean_dep("//third_party/systemlibs:enum34.BUILD"),
        strip_prefix = "enum34-1.1.6/enum",
    )

    tf_http_archive(
        name = "org_python_pypi_backports_weakref",
        build_file = clean_dep("//third_party:backports_weakref.BUILD"),
        sha256 = "8813bf712a66b3d8b85dc289e1104ed220f1878cf981e2fe756dfaabe9a82892",
        strip_prefix = "backports.weakref-1.0rc1/src",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/bc/cc/3cdb0a02e7e96f6c70bd971bc8a90b8463fda83e264fa9c5c1c98ceabd81/backports.weakref-1.0rc1.tar.gz",
            "https://pypi.python.org/packages/bc/cc/3cdb0a02e7e96f6c70bd971bc8a90b8463fda83e264fa9c5c1c98ceabd81/backports.weakref-1.0rc1.tar.gz",
        ],
    )

    filegroup_external(
        name = "org_python_license",
        licenses = ["notice"],  # Python 2.0
        sha256_urls = {
            "e76cacdf0bdd265ff074ccca03671c33126f597f39d0ed97bc3e5673d9170cf6": [
                "https://storage.googleapis.com/mirror.tensorflow.org/docs.python.org/2.7/_sources/license.rst.txt",
                "https://docs.python.org/2.7/_sources/license.rst.txt",
            ],
        },
    )

    # 310ba5ee72661c081129eb878c1bbcec936b20f0 is based on 3.8.0 with a fix for protobuf.bzl.
    PROTOBUF_URLS = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/310ba5ee72661c081129eb878c1bbcec936b20f0.tar.gz",
        "https://github.com/protocolbuffers/protobuf/archive/310ba5ee72661c081129eb878c1bbcec936b20f0.tar.gz",
    ]
    PROTOBUF_SHA256 = "b9e92f9af8819bbbc514e2902aec860415b70209f31dfc8c4fa72515a5df9d59"
    PROTOBUF_STRIP_PREFIX = "protobuf-310ba5ee72661c081129eb878c1bbcec936b20f0"

    # protobuf depends on @zlib, it has to be renamed to @zlib_archive because "zlib" is already
    # defined using bind for grpc.
    PROTOBUF_PATCH = "//third_party/protobuf:protobuf.patch"

    tf_http_archive(
        name = "com_google_protobuf",
        patch_file = clean_dep(PROTOBUF_PATCH),
        sha256 = PROTOBUF_SHA256,
        strip_prefix = PROTOBUF_STRIP_PREFIX,
        system_build_file = clean_dep("//third_party/systemlibs:protobuf.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:protobuf.bzl": "protobuf.bzl",
        },
        urls = PROTOBUF_URLS,
    )

    tf_http_archive(
        name = "nsync",
        sha256 = "caf32e6b3d478b78cff6c2ba009c3400f8251f646804bcb65465666a9cea93c4",
        strip_prefix = "nsync-1.22.0",
        system_build_file = clean_dep("//third_party/systemlibs:nsync.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/nsync/archive/1.22.0.tar.gz",
            "https://github.com/google/nsync/archive/1.22.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_google_googletest",
        sha256 = "ff7a82736e158c077e76188232eac77913a15dac0b22508c390ab3f88e6d6d86",
        strip_prefix = "googletest-b6cd405286ed8635ece71c72f118e659f4ade3fb",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
            "https://github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
        ],
    )

    tf_http_archive(
        name = "com_github_gflags_gflags",
        sha256 = "ae27cdbcd6a2f935baa78e4f21f675649271634c092b1be01469440495609d0e",
        strip_prefix = "gflags-2.2.1",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/gflags/gflags/archive/v2.2.1.tar.gz",
            "https://github.com/gflags/gflags/archive/v2.2.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "pcre",
        build_file = clean_dep("//third_party:pcre.BUILD"),
        sha256 = "aecafd4af3bd0f3935721af77b889d9024b2e01d96b58471bd91a3063fb47728",
        strip_prefix = "pcre-8.44",
        system_build_file = clean_dep("//third_party/systemlibs:pcre.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/ftp.exim.org/pub/pcre/pcre-8.44.tar.gz",
            "https://ftp.exim.org/pub/pcre/pcre-8.44.tar.gz",
        ],
    )

    tf_http_archive(
        name = "swig",
        build_file = clean_dep("//third_party:swig.BUILD"),
        sha256 = "58a475dbbd4a4d7075e5fe86d4e54c9edde39847cdb96a3053d87cb64a23a453",
        strip_prefix = "swig-3.0.8",
        system_build_file = clean_dep("//third_party/systemlibs:swig.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/ufpr.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
            "http://ufpr.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
            "http://pilotfiber.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
        ],
    )

    tf_http_archive(
        name = "curl",
        build_file = clean_dep("//third_party:curl.BUILD"),
	sha256 = "ed936c0b02c06d42cf84b39dd12bb14b62d77c7c4e875ade022280df5dcc81d7",
        strip_prefix = "curl-7.78.0",
        system_build_file = clean_dep("//third_party/systemlibs:curl.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/curl.haxx.se/download/curl-7.78.0.tar.gz",
            "https://curl.haxx.se/download/curl-7.78.0.tar.gz",
        ],
    )

    # WARNING: make sure ncteisen@ and vpai@ are cc-ed on any CL to change the below rule
    tf_http_archive(
        name = "grpc",
        patch_file = clean_dep("//third_party:grpc_gettid.patch"),
        sha256 = "67a6c26db56f345f7cee846e681db2c23f919eba46dd639b09462d1b6203d28c",
        strip_prefix = "grpc-4566c2a29ebec0835643b972eb99f4306c4234a3",
        system_build_file = clean_dep("//third_party/systemlibs:grpc.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/grpc/grpc/archive/4566c2a29ebec0835643b972eb99f4306c4234a3.tar.gz",
            "https://github.com/grpc/grpc/archive/4566c2a29ebec0835643b972eb99f4306c4234a3.tar.gz",
        ],
    )

    tf_http_archive(
        name = "com_github_nanopb_nanopb",
        sha256 = "8bbbb1e78d4ddb0a1919276924ab10d11b631df48b657d960e0c795a25515735",
        build_file = "@grpc//third_party:nanopb.BUILD",
        strip_prefix = "nanopb-f8ac463766281625ad710900479130c7fcb4d63b",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/nanopb/nanopb/archive/f8ac463766281625ad710900479130c7fcb4d63b.tar.gz",
            "https://github.com/nanopb/nanopb/archive/f8ac463766281625ad710900479130c7fcb4d63b.tar.gz",
        ],
    )

    tf_http_archive(
        name = "linenoise",
        build_file = clean_dep("//third_party:linenoise.BUILD"),
        sha256 = "7f51f45887a3d31b4ce4fa5965210a5e64637ceac12720cfce7954d6a2e812f7",
        strip_prefix = "linenoise-c894b9e59f02203dbe4e2be657572cf88c4230c3",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/antirez/linenoise/archive/c894b9e59f02203dbe4e2be657572cf88c4230c3.tar.gz",
            "https://github.com/antirez/linenoise/archive/c894b9e59f02203dbe4e2be657572cf88c4230c3.tar.gz",
        ],
    )

    # Check out LLVM and MLIR from llvm-project.
    LLVM_COMMIT = "387c3f74fd8efdc0be464b0e1a8033cc1eeb739c"
    LLVM_SHA256 = "5648d1bdd933f45aa6556cf104b32a0418121ce6961f5f47e8ef5bc6e428434f"
    LLVM_URLS = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
    ]
    tf_http_archive(
        name = "llvm-project",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        urls = LLVM_URLS,
        additional_build_files = {
            clean_dep("//third_party/llvm:llvm.autogenerated.BUILD"): "llvm/BUILD",
            "//third_party/mlir:BUILD": "mlir/BUILD",
            "//third_party/mlir:test.BUILD": "mlir/test/BUILD",
        },
    )

    tf_http_archive(
        name = "lmdb",
        build_file = clean_dep("//third_party:lmdb.BUILD"),
        sha256 = "f3927859882eb608868c8c31586bb7eb84562a40a6bf5cc3e13b6b564641ea28",
        strip_prefix = "lmdb-LMDB_0.9.22/libraries/liblmdb",
        system_build_file = clean_dep("//third_party/systemlibs:lmdb.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/LMDB/lmdb/archive/LMDB_0.9.22.tar.gz",
            "https://github.com/LMDB/lmdb/archive/LMDB_0.9.22.tar.gz",
        ],
    )

    tf_http_archive(
        name = "jsoncpp_git",
        build_file = clean_dep("//third_party:jsoncpp.BUILD"),
        sha256 = "c49deac9e0933bcb7044f08516861a2d560988540b23de2ac1ad443b219afdb6",
        strip_prefix = "jsoncpp-1.8.4",
        system_build_file = clean_dep("//third_party/systemlibs:jsoncpp.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz",
            "https://github.com/open-source-parsers/jsoncpp/archive/1.8.4.tar.gz",
        ],
    )

    tf_http_archive(
        name = "boringssl",
        sha256 = "1188e29000013ed6517168600fc35a010d58c5d321846d6a6dfee74e4c788b45",
        strip_prefix = "boringssl-7f634429a04abc48e2eb041c81c5235816c96514",
        system_build_file = clean_dep("//third_party/systemlibs:boringssl.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
            "https://github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
        ],
    )

    tf_http_archive(
        name = "zlib_archive",
        build_file = clean_dep("//third_party:zlib.BUILD"),
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
        strip_prefix = "zlib-1.2.11",
        system_build_file = clean_dep("//third_party/systemlibs:zlib.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/zlib.net/zlib-1.2.11.tar.gz",
            "https://zlib.net/zlib-1.2.11.tar.gz",
        ],
    )

    tf_http_archive(
        name = "fft2d",
        build_file = clean_dep("//third_party/fft2d:fft2d.BUILD"),
        sha256 = "ada7e99087c4ed477bfdf11413f2ba8db8a840ba9bbf8ac94f4f3972e2a7cec9",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/www.kurims.kyoto-u.ac.jp/~ooura/fft2d.tgz",
            "http://www.kurims.kyoto-u.ac.jp/~ooura/fft2d.tgz",
        ],
    )

    # Note: snappy is placed earlier as tensorflow's snappy does not include snappy-c
    http_archive(
        name = "snappy",
        build_file = clean_dep("//third_party:snappy.BUILD"),
        sha256 = "16b677f07832a612b0836178db7f374e414f94657c138e6993cbfc5dcc58651f",
        strip_prefix = "snappy-1.1.8",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/snappy/archive/1.1.8.tar.gz",
            "https://github.com/google/snappy/archive/1.1.8.tar.gz",
        ],
    )

    tf_http_archive(
        name = "nccl_archive",
        build_file = clean_dep("//third_party:nccl/archive.BUILD"),
        patch_file = clean_dep("//third_party/nccl:archive.patch"),
        sha256 = "9a7633e224982e2b60fa6b397d895d20d6b7498e3e02f46f98a5a4e187c5a44c",
        strip_prefix = "nccl-0ceaec9cee96ae7658aa45686853286651f36384",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/nvidia/nccl/archive/0ceaec9cee96ae7658aa45686853286651f36384.tar.gz",
            "https://github.com/nvidia/nccl/archive/0ceaec9cee96ae7658aa45686853286651f36384.tar.gz",
        ],
    )

    java_import_external(
        name = "junit",
        jar_sha256 = "59721f0805e223d84b90677887d9ff567dc534d7c502ca903c0c2b17f05c116a",
        jar_urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/repo1.maven.org/maven2/junit/junit/4.12/junit-4.12.jar",
            "http://repo1.maven.org/maven2/junit/junit/4.12/junit-4.12.jar",
            "http://maven.ibiblio.org/maven2/junit/junit/4.12/junit-4.12.jar",
        ],
        licenses = ["reciprocal"],  # Common Public License Version 1.0
        testonly_ = True,
        deps = ["@org_hamcrest_core"],
    )

    java_import_external(
        name = "org_hamcrest_core",
        jar_sha256 = "66fdef91e9739348df7a096aa384a5685f4e875584cce89386a7a47251c4d8e9",
        jar_urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/repo1.maven.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar",
            "http://repo1.maven.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar",
            "http://maven.ibiblio.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar",
        ],
        licenses = ["notice"],  # New BSD License
        testonly_ = True,
    )

    java_import_external(
        name = "com_google_testing_compile",
        jar_sha256 = "edc180fdcd9f740240da1a7a45673f46f59c5578d8cd3fbc912161f74b5aebb8",
        jar_urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/repo1.maven.org/maven2/com/google/testing/compile/compile-testing/0.11/compile-testing-0.11.jar",
            "http://repo1.maven.org/maven2/com/google/testing/compile/compile-testing/0.11/compile-testing-0.11.jar",
        ],
        licenses = ["notice"],  # New BSD License
        testonly_ = True,
        deps = ["@com_google_guava", "@com_google_truth"],
    )

    java_import_external(
        name = "com_google_truth",
        jar_sha256 = "032eddc69652b0a1f8d458f999b4a9534965c646b8b5de0eba48ee69407051df",
        jar_urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/repo1.maven.org/maven2/com/google/truth/truth/0.32/truth-0.32.jar",
            "http://repo1.maven.org/maven2/com/google/truth/truth/0.32/truth-0.32.jar",
        ],
        licenses = ["notice"],  # Apache 2.0
        testonly_ = True,
        deps = ["@com_google_guava"],
    )

    java_import_external(
        name = "org_checkerframework_qual",
        jar_sha256 = "a17501717ef7c8dda4dba73ded50c0d7cde440fd721acfeacbf19786ceac1ed6",
        jar_urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/repo1.maven.org/maven2/org/checkerframework/checker-qual/2.4.0/checker-qual-2.4.0.jar",
            "http://repo1.maven.org/maven2/org/checkerframework/checker-qual/2.4.0/checker-qual-2.4.0.jar",
        ],
        licenses = ["notice"],  # Apache 2.0
    )

    java_import_external(
        name = "com_squareup_javapoet",
        jar_sha256 = "5bb5abdfe4366c15c0da3332c57d484e238bd48260d6f9d6acf2b08fdde1efea",
        jar_urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/repo1.maven.org/maven2/com/squareup/javapoet/1.9.0/javapoet-1.9.0.jar",
            "http://repo1.maven.org/maven2/com/squareup/javapoet/1.9.0/javapoet-1.9.0.jar",
        ],
        licenses = ["notice"],  # Apache 2.0
    )

    tf_http_archive(
        name = "com_google_pprof",
        build_file = clean_dep("//third_party:pprof.BUILD"),
        sha256 = "e0928ca4aa10ea1e0551e2d7ce4d1d7ea2d84b2abbdef082b0da84268791d0c4",
        strip_prefix = "pprof-c0fb62ec88c411cc91194465e54db2632845b650",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/pprof/archive/c0fb62ec88c411cc91194465e54db2632845b650.tar.gz",
            "https://github.com/google/pprof/archive/c0fb62ec88c411cc91194465e54db2632845b650.tar.gz",
        ],
    )

    tf_http_archive(
        name = "cub_archive",
        build_file = clean_dep("//third_party:cub.BUILD"),
        sha256 = "6bfa06ab52a650ae7ee6963143a0bbc667d6504822cbd9670369b598f18c58c3",
        strip_prefix = "cub-1.8.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVlabs/cub/archive/1.8.0.zip",
            "https://github.com/NVlabs/cub/archive/1.8.0.zip",
        ],
    )

    tf_http_archive(
        name = "nvtx_archive",
        build_file = clean_dep("//third_party:nvtx.BUILD"),
        sha256 = "bb8d1536aad708ec807bc675e12e5838c2f84481dec4005cd7a9bbd49e326ba1",
        strip_prefix = "NVTX-3.0.1/c/include",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NVIDIA/NVTX/archive/v3.0.1.tar.gz",
            "https://github.com/NVIDIA/NVTX/archive/v3.0.1.tar.gz",
        ],
    )

    tf_http_archive(
        name = "rocprim_archive",
        build_file = clean_dep("//third_party:rocprim.BUILD"),
        sha256 = "3c178461ead70ce6adb60c836a35a52564968af31dfa81f4157ab72b5f14d31f",
        strip_prefix = "rocPRIM-4a33d328f8352df1654271939da66914f2334424",
        urls = [
            "https://mirror.bazel.build/github.com/ROCmSoftwarePlatform/rocPRIM/archive/4a33d328f8352df1654271939da66914f2334424.tar.gz",
            "https://github.com/ROCmSoftwarePlatform/rocPRIM/archive/4a33d328f8352df1654271939da66914f2334424.tar.gz",
        ],
    )

    tf_http_archive(
        name = "cython",
        build_file = clean_dep("//third_party:cython.BUILD"),
        delete = ["BUILD.bazel"],
        sha256 = "bccc9aa050ea02595b2440188813b936eaf345e85fb9692790cecfe095cf91aa",
        strip_prefix = "cython-0.28.4",
        system_build_file = clean_dep("//third_party/systemlibs:cython.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/cython/cython/archive/0.28.4.tar.gz",
            "https://github.com/cython/cython/archive/0.28.4.tar.gz",
        ],
    )

    tf_http_archive(
        name = "arm_neon_2_x86_sse",
        build_file = clean_dep("//third_party:arm_neon_2_x86_sse.BUILD"),
        sha256 = "213733991310b904b11b053ac224fee2d4e0179e46b52fe7f8735b8831e04dcc",
        strip_prefix = "ARM_NEON_2_x86_SSE-1200fe90bb174a6224a525ee60148671a786a71f",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/intel/ARM_NEON_2_x86_SSE/archive/1200fe90bb174a6224a525ee60148671a786a71f.tar.gz",
            "https://github.com/intel/ARM_NEON_2_x86_SSE/archive/1200fe90bb174a6224a525ee60148671a786a71f.tar.gz",
        ],
    )

    tf_http_archive(
        name = "double_conversion",
        build_file = clean_dep("//third_party:double_conversion.BUILD"),
        sha256 = "2f7fbffac0d98d201ad0586f686034371a6d152ca67508ab611adc2386ad30de",
        strip_prefix = "double-conversion-3992066a95b823efc8ccc1baf82a1cfc73f6e9b8",
        system_build_file = clean_dep("//third_party/systemlibs:double_conversion.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/double-conversion/archive/3992066a95b823efc8ccc1baf82a1cfc73f6e9b8.zip",
            "https://github.com/google/double-conversion/archive/3992066a95b823efc8ccc1baf82a1cfc73f6e9b8.zip",
        ],
    )

    tf_http_archive(
        name = "tflite_mobilenet_float",
        build_file = clean_dep("//third_party:tflite_mobilenet_float.BUILD"),
        sha256 = "2fadeabb9968ec6833bee903900dda6e61b3947200535874ce2fe42a8493abc0",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz",
            "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz",
        ],
    )

    tf_http_archive(
        name = "tflite_mobilenet_quant",
        build_file = clean_dep("//third_party:tflite_mobilenet_quant.BUILD"),
        sha256 = "d32432d28673a936b2d6281ab0600c71cf7226dfe4cdcef3012555f691744166",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz",
            "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz",
        ],
    )

    tf_http_archive(
        name = "tflite_mobilenet_ssd",
        build_file = str(Label("//third_party:tflite_mobilenet.BUILD")),
        sha256 = "767057f2837a46d97882734b03428e8dd640b93236052b312b2f0e45613c1cf0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_ssd_tflite_v1.zip",
            "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_ssd_tflite_v1.zip",
        ],
    )

    tf_http_archive(
        name = "tflite_mobilenet_ssd_quant",
        build_file = str(Label("//third_party:tflite_mobilenet.BUILD")),
        sha256 = "a809cd290b4d6a2e8a9d5dad076e0bd695b8091974e0eed1052b480b2f21b6dc",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_0.75_quant_2018_06_29.zip",
            "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_0.75_quant_2018_06_29.zip",
        ],
    )

    tf_http_archive(
        name = "tflite_mobilenet_ssd_quant_protobuf",
        build_file = str(Label("//third_party:tflite_mobilenet.BUILD")),
        sha256 = "09280972c5777f1aa775ef67cb4ac5d5ed21970acd8535aeca62450ef14f0d79",
        strip_prefix = "ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz",
            "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz",
        ],
    )

    tf_http_archive(
        name = "tflite_conv_actions_frozen",
        build_file = str(Label("//third_party:tflite_mobilenet.BUILD")),
        sha256 = "d947b38cba389b5e2d0bfc3ea6cc49c784e187b41a071387b3742d1acac7691e",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.googleapis.com/download.tensorflow.org/models/tflite/conv_actions_tflite.zip",
            "https://storage.googleapis.com/download.tensorflow.org/models/tflite/conv_actions_tflite.zip",
        ],
    )

    tf_http_archive(
        name = "tflite_smartreply",
        build_file = clean_dep("//third_party:tflite_smartreply.BUILD"),
        sha256 = "8980151b85a87a9c1a3bb1ed4748119e4a85abd3cb5744d83da4d4bd0fbeef7c",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.googleapis.com/download.tensorflow.org/models/tflite/smartreply_1.0_2017_11_01.zip",
            "https://storage.googleapis.com/download.tensorflow.org/models/tflite/smartreply_1.0_2017_11_01.zip",
        ],
    )

    tf_http_archive(
        name = "tflite_ovic_testdata",
        build_file = clean_dep("//third_party:tflite_ovic_testdata.BUILD"),
        sha256 = "033c941b7829b05ca55a124a26a6a0581b1ececc154a2153cafcfdb54f80dca2",
        strip_prefix = "ovic",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/storage.googleapis.com/download.tensorflow.org/data/ovic_2019_04_30.zip",
            "https://storage.googleapis.com/download.tensorflow.org/data/ovic_2019_04_30.zip",
        ],
    )

    tf_http_archive(
        name = "build_bazel_rules_android",
        sha256 = "cd06d15dd8bb59926e4d65f9003bfc20f9da4b2519985c27e190cddc8b7a7806",
        strip_prefix = "rules_android-0.1.1",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_android/archive/v0.1.1.zip",
            "https://github.com/bazelbuild/rules_android/archive/v0.1.1.zip",
        ],
    )

    tf_http_archive(
        name = "tbb",
        build_file = clean_dep("//third_party/ngraph:tbb.BUILD"),
        sha256 = "c3245012296f09f1418b78a8c2f17df5188b3bd0db620f7fd5fabe363320805a",
        strip_prefix = "tbb-2019_U1",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/01org/tbb/archive/2019_U1.zip",
            "https://github.com/01org/tbb/archive/2019_U1.zip",
        ],
    )

    tf_http_archive(
        name = "ngraph",
        build_file = clean_dep("//third_party/ngraph:ngraph.BUILD"),
        sha256 = "a1780f24a1381fc25e323b4b2d08b6ef5129f42e011305b2a34dcf43a48030d5",
        strip_prefix = "ngraph-0.11.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NervanaSystems/ngraph/archive/v0.11.0.tar.gz",
            "https://github.com/NervanaSystems/ngraph/archive/v0.11.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "nlohmann_json_lib",
        build_file = clean_dep("//third_party/ngraph:nlohmann_json.BUILD"),
        sha256 = "c377963a95989270c943d522bfefe7b889ef5ed0e1e15d535fd6f6f16ed70732",
        strip_prefix = "json-3.4.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/nlohmann/json/archive/v3.4.0.tar.gz",
            "https://github.com/nlohmann/json/archive/v3.4.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "ngraph_tf",
        build_file = clean_dep("//third_party/ngraph:ngraph_tf.BUILD"),
        sha256 = "742a642d2c6622277df4c902b6830d616d0539cc8cd843d6cdb899bb99e66e36",
        strip_prefix = "ngraph-tf-0.9.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/NervanaSystems/ngraph-tf/archive/v0.9.0.zip",
            "https://github.com/NervanaSystems/ngraph-tf/archive/v0.9.0.zip",
        ],
    )

    tf_http_archive(
        name = "pybind11",
        urls = [
            "https://mirror.bazel.build/github.com/pybind/pybind11/archive/v2.3.0.tar.gz",
            "https://github.com/pybind/pybind11/archive/v2.3.0.tar.gz",
        ],
        sha256 = "0f34838f2c8024a6765168227ba587b3687729ebf03dc912f88ff75c7aa9cfe8",
        strip_prefix = "pybind11-2.3.0",
        build_file = clean_dep("//third_party:pybind11.BUILD"),
    )

    tf_http_archive(
        name = "wrapt",
        build_file = clean_dep("//third_party:wrapt.BUILD"),
        sha256 = "8a6fb40e8f8b6a66b4ba81a4044c68e6a7b1782f21cfabc06fb765332b4c3e51",
        strip_prefix = "wrapt-1.11.1/src/wrapt",
        system_build_file = clean_dep("//third_party/systemlibs:wrapt.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/GrahamDumpleton/wrapt/archive/1.11.1.tar.gz",
            "https://github.com/GrahamDumpleton/wrapt/archive/1.11.1.tar.gz",
        ],
    )
    tf_http_archive(
        name = "cares",
        build_file = "//third_party/cares:cares.BUILD",
        sha256 = "45d3c1fd29263ceec2afc8ff9cd06d5f8f889636eb4e80ce3cc7f0eaf7aadc6e",
        strip_prefix = "c-ares-1.14.0",
        urls = ["https://storage.googleapis.com/mirror.tensorflow.org/github.com/c-ares.haxx.se/download/c-ares-1.14.0.tar.gz",
                "https://c-ares.haxx.se/download/c-ares-1.14.0.tar.gz"],
    )


    tf_http_archive(
        name = "libcuckoo",
        urls = [
            "https://github.com/efficient/libcuckoo/archive/8785773896d74f72b6224e59d37f5f8c3c1e022a.tar.gz",
            "https://github.com/efficient/libcuckoo/archive/8785773896d74f72b6224e59d37f5f8c3c1e022a.tar.gz",
        ],
        sha256 = "11cf338f342b3a12f946048deb8f96171b102450b8b73fefe9f1968b601bc00d",
        strip_prefix = "libcuckoo-8785773896d74f72b6224e59d37f5f8c3c1e022a",
        build_file = str(Label("//third_party:libcuckoo.BUILD")),
    )

    tf_http_archive(
        name = "com_github_google_leveldb",
        sha256 = "f99dc5dcb6f23e500b197db02e993ee0d3bafd1ac84b85ab50de9009b36fbf03",
        strip_prefix = "leveldb-5d94ad4d95c09d3ac203ddaf9922e55e730706a8",
        build_file = clean_dep("//third_party:leveldb.BUILD"),
        urls = [
            "https://github.com/google/leveldb/archive/5d94ad4d95c09d3ac203ddaf9922e55e730706a8.tar.gz",
            "https://github.com/google/leveldb/archive/5d94ad4d95c09d3ac203ddaf9922e55e730706a8.tar.gz"
        ],
    )

    tf_http_archive(
        name = "seastar_repo",
        sha256 = "bf97d343149f95983317888dd39a3231639f67c39d826413ea226f9c9c5267d4",
        strip_prefix = "seastar-b9357c525a552ad21b5609622c900e0649921712",
        build_file = str(Label("//third_party:seastar.BUILD")),
        urls = [
            "https://github.com/AlibabaPAI/seastar/archive/b9357c525a552ad21b5609622c900e0649921712.tar.gz",
            "https://github.com/AlibabaPAI/seastar/archive/b9357c525a552ad21b5609622c900e0649921712.tar.gz",
        ],
    )

    http_archive(
        name = "lz4",
        build_file = str(Label("//third_party:lz4.BUILD")),
        sha256 = "658ba6191fa44c92280d4aa2c271b0f4fbc0e34d249578dd05e50e76d0e5efcc",
        strip_prefix = "lz4-1.9.2/lib",
        urls = ["https://github.com/lz4/lz4/archive/v1.9.2.tar.gz"],
    )

    http_archive(
        name = "kafka",
        build_file = str(Label("//third_party:kafka.BUILD")),
        patch_cmds = [
            "rm -f src/win32_config.h",
            # TODO: Remove the fowllowing once librdkafka issue is resolved.
            """sed -i.bak '\\|rd_kafka_log(rk,|,/ exceeded);/ s/^/\\/\\//' src/rdkafka_cgrp.c""",
        ],
        sha256 = "f7fee59fdbf1286ec23ef0b35b2dfb41031c8727c90ced6435b8cf576f23a656",
        strip_prefix = "librdkafka-1.5.0",
        urls = [
            "https://mirror.tensorflow.org/github.com/edenhill/librdkafka/archive/v1.5.0.tar.gz",
            "https://github.com/edenhill/librdkafka/archive/v1.5.0.tar.gz",
        ],
    )

    http_archive(
        name = "arrow",
        build_file = clean_dep("//third_party:arrow.BUILD"),
        patch_cmds = [
            # TODO: Remove the fowllowing once arrow issue is resolved.
            """sed -i.bak 's/type_traits/std::max<int16_t>(sizeof(int16_t), type_traits/g' cpp/src/parquet/column_reader.cc""",
            """sed -i.bak 's/value_byte_size/value_byte_size)/g' cpp/src/parquet/column_reader.cc""",
        ],
        sha256 = "a27971e2a71c412ae43d998b7b6d06201c7a3da382c804dcdc4a8126ccbabe67",
        strip_prefix = "arrow-apache-arrow-4.0.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/apache/arrow/archive/apache-arrow-4.0.0.tar.gz",
            "https://github.com/apache/arrow/archive/apache-arrow-4.0.0.tar.gz",
        ],
    )

    http_archive(
        name = "brotli",
        build_file = clean_dep("//third_party:brotli.BUILD"),
        sha256 = "4c61bfb0faca87219ea587326c467b95acb25555b53d1a421ffa3c8a9296ee2c",
        strip_prefix = "brotli-1.0.7",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/brotli/archive/v1.0.7.tar.gz",
            "https://github.com/google/brotli/archive/v1.0.7.tar.gz",
        ],
    )

    http_archive(
        name = "bzip2",
        build_file = clean_dep("//third_party:bzip2.BUILD"),
        sha256 = "ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269",
        strip_prefix = "bzip2-1.0.8",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz",
            "https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz",
        ],
    )

    http_archive(
        name = "thrift",
        build_file = clean_dep("//third_party:thrift.BUILD"),
        sha256 = "5da60088e60984f4f0801deeea628d193c33cec621e78c8a43a5d8c4055f7ad9",
        strip_prefix = "thrift-0.13.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/apache/thrift/archive/v0.13.0.tar.gz",
            "https://github.com/apache/thrift/archive/v0.13.0.tar.gz",
        ],
    )

    http_archive(
        name = "xsimd",
        build_file = clean_dep("//third_party:xsimd.BUILD"),
        sha256 = "45337317c7f238fe0d64bb5d5418d264a427efc53400ddf8e6a964b6bcb31ce9",
        strip_prefix = "xsimd-7.5.0",
        urls = [
            "https://github.com/xtensor-stack/xsimd/archive/refs/tags/7.5.0.tar.gz",
        ],
    )

    http_archive(
        name = "zstd",
        build_file = clean_dep("//third_party:zstd.BUILD"),
        sha256 = "a364f5162c7d1a455cc915e8e3cf5f4bd8b75d09bc0f53965b0c9ca1383c52c8",
        strip_prefix = "zstd-1.4.4",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/facebook/zstd/archive/v1.4.4.tar.gz",
            "https://github.com/facebook/zstd/archive/v1.4.4.tar.gz",
        ],
    )

    http_archive(
        name = "rapidjson",
        build_file = clean_dep("//third_party:rapidjson.BUILD"),
        sha256 = "30bd2c428216e50400d493b38ca33a25efb1dd65f79dfc614ab0c957a3ac2c28",
        strip_prefix = "rapidjson-418331e99f859f00bdc8306f69eba67e8693c55e",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/miloyip/rapidjson/archive/418331e99f859f00bdc8306f69eba67e8693c55e.tar.gz",
            "https://github.com/miloyip/rapidjson/archive/418331e99f859f00bdc8306f69eba67e8693c55e.tar.gz",
        ],
    )

    http_archive(
        name = "aws_c_common",
        build_file = clean_dep("//third_party/aws_util:aws_c_common.BUILD"),
        sha256 = "e9462a141b5db30006704f537d19b92357a59be38d590272e6118976b0356ccd",
        strip_prefix = "aws-c-common-0.7.4",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-c-common/archive/refs/tags/v0.7.4.tar.gz",
            "https://github.com/awslabs/aws-c-common/archive/refs/tags/v0.7.4.tar.gz",
        ],
    )

    http_archive(
        name = "aws_c_io",
        build_file = clean_dep("//third_party/aws_util:aws_c_io.BUILD"),
        sha256 = "b60270d23b6e2f4a5d80e64ca6538ba114cd6044b53752964c940f87e59bf0d9",
        strip_prefix = "aws-c-io-0.11.2",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-c-io/archive/refs/tags/v0.11.2.tar.gz",
            "https://github.com/awslabs/aws-c-io/archive/refs/tags/v0.11.2.tar.gz",  
        ],
    )

    http_archive(
        name = "aws_c_event_stream",
        build_file = clean_dep("//third_party/aws_util:aws_c_event_stream.BUILD"),
        sha256 = "bae0c762b6a4b779a0db0f4730512da6cb500e76681ffdcb9f7286d8e26e547a",
        strip_prefix = "aws-c-event-stream-0.2.6",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-c-event-stream/archive/refs/tags/v0.2.6.tar.gz",
            "https://github.com/awslabs/aws-c-event-stream/archive/refs/tags/v0.2.6.tar.gz",  
        ],
    )

    http_archive(
        name = "aws_checksums",
        build_file = clean_dep("//third_party/aws_util:aws_checksums.BUILD"),
        sha256 = "394723034b81cc7cd528401775bc7aca2b12c7471c92350c80a0e2fb9d2909fe",
        strip_prefix = "aws-checksums-0.1.12",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-checksums/archive/refs/tags/v0.1.12.tar.gz",
            "https://github.com/awslabs/aws-checksums/archive/refs/tags/v0.1.12.tar.gz",  
        ],
    )

    http_archive(
        name = "aws_c_cal",
        build_file = clean_dep("//third_party/aws_util:aws_c_cal.BUILD"),
        sha256 = "40297da04443d4ee2988d1c5fb0dc4a156d0e4cfaf80e6a1df1867452566d540",
        strip_prefix = "aws-c-cal-0.5.17",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/awslabs/aws-c-cal/archive/refs/tags/v0.5.17.tar.gz",
            "https://github.com/awslabs/aws-c-cal/archive/refs/tags/v0.5.17.tar.gz",  
        ],
    )

def tf_bind():
    """Bind targets for some external repositories"""
    ##############################################################################
    # BIND DEFINITIONS
    #
    # Please do not add bind() definitions unless we have no other choice.
    # If that ends up being the case, please leave a comment explaining
    # why we can't depend on the canonical build target.

    # gRPC wants a cares dependency but its contents is not actually
    # important since we have set GRPC_ARES=0 in .bazelrc
    native.bind(
        name = "cares",
        actual = "@com_github_nanopb_nanopb//:nanopb",
    )

    # Needed by Protobuf
    native.bind(
        name = "grpc_cpp_plugin",
        actual = "@grpc//:grpc_cpp_plugin",
    )
    native.bind(
        name = "grpc_python_plugin",
        actual = "@grpc//:grpc_python_plugin",
    )

    native.bind(
        name = "grpc_lib",
        actual = "@grpc//:grpc++",
    )

    native.bind(
        name = "grpc_lib_unsecure",
        actual = "@grpc//:grpc++_unsecure",
    )

    # Needed by gRPC
    native.bind(
        name = "libssl",
        actual = "@boringssl//:ssl",
    )

    # Needed by gRPC
    native.bind(
        name = "nanopb",
        actual = "@com_github_nanopb_nanopb//:nanopb",
    )

    # Needed by gRPC
    native.bind(
        name = "protobuf",
        actual = "@com_google_protobuf//:protobuf",
    )

    # gRPC expects //external:protobuf_clib and //external:protobuf_compiler
    # to point to Protobuf's compiler library.
    native.bind(
        name = "protobuf_clib",
        actual = "@com_google_protobuf//:protoc_lib",
    )

    # Needed by gRPC
    native.bind(
        name = "protobuf_headers",
        actual = "@com_google_protobuf//:protobuf_headers",
    )

    # Needed by Protobuf
    native.bind(
        name = "python_headers",
        actual = clean_dep("//third_party/python_runtime:headers"),
    )

    # Needed by Protobuf
    native.bind(
        name = "six",
        actual = "@six_archive//:six",
    )

    # Needed by gRPC
    native.bind(
        name = "zlib",
        actual = "@zlib_archive//:zlib",
    )
