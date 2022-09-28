# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Config Utility to write .bazelrc based on tensorflow."""
import re
import sys

def write_config():
    try:
        with open(".bazelrc", "w") as bazel_rc:
            # static link here
            bazel_rc.write('build --define framework_shared_object=false\n')

            # grpc disable c-ares
            bazel_rc.write('build --define grpc_no_ares=true\n')

            bazel_rc.write('build --cxxopt="-std=c++14"\n')
            # skip the param 'python' or 'python3'
            for argv in sys.argv[2:]:
                if argv == "--cuda":
                    print("Bazel will build with --cuda")
                    bazel_rc.write('build --action_env TF_NEED_CUDA="1"\n')
                    bazel_rc.write(
                        'build --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda"\n'
                    )
                    bazel_rc.write(
                        'build --action_env CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"\n'
                    )
                    bazel_rc.write('build --action_env TF_CUDA_VERSION="10.1"\n')
                    bazel_rc.write('build --action_env TF_CUDNN_VERSION="7"\n')
           
                if argv == "--mkl":
                    print("Bazel will build with --mkl")
                    bazel_rc.write("build --define=build_with_mkl=true --define=enable_mkl=true\n")
                elif argv == "--mkl_open_source_only":
                    print("Bazel will build with --mkl_open_source_only\n")
                    bazel_rc.write("build --define=build_with_mkl_dnn_only=true\n")
                    bazel_rc.write("build --define=build_with_mkl=true --define=enable_mkl=true\n")
                elif argv == "--mkl_open_source_v1_only":
                    print("Bazel will build with --mkl_open_source_v1_only\n")
                    bazel_rc.write("build --define=build_with_mkl_dnn_v1_only=true\n")
                    bazel_rc.write("build --define=build_with_mkl=true --define=enable_mkl=true\n")
                elif argv == "--mkl_threadpool":
                    print("Bazel will build with --mkl_threadpool\n")
                    bazel_rc.write("build --define=build_with_mkldnn_threadpool=true\n")
                    bazel_rc.write("build --define=build_with_mkl=true --define=enable_mkl=true\n")
                elif argv == "--noaws":
                    print("Bazel will build with --noaws\n")
                    bazel_rc.write("build --define=no_aws_support=true\n")
                elif argv == "--nogcp":
                    print("Bazel will build with --nogcp\n")
                    bazel_rc.write("build --define=no_gcp_support=true\n")
                elif argv == "--nohdfs":
                    print("Bazel will build with --nohdfs\n")
                    bazel_rc.write("build --define=no_hdfs_support=true\n")
                elif argv == "--nokafka":
                    print("Bazel will build with --nokafka\n")
                    bazel_rc.write("build --define=no_kafka_support=true\n")
                elif argv == "--noignite":
                    print("Bazel will build with --noignite\n")
                    bazel_rc.write("build --define=no_ignite_support=true\n")
                elif argv == "--nonccl":
                    print("Bazel will build with --nonccl\n")
                    bazel_rc.write("build --define=no_nccl_support=true\n")
                else:
                    print("Bazel will build unknow args.\n")

            bazel_rc.write("build -c opt\n")

            # Enable platform specific config
            # bazel_rc.write('build --enable_platform_specific_config\n')
            # Use llvm toolchain
            bazel_rc.write('build:macos --crosstool_top=@llvm_toolchain//:toolchain"\n')
            # Needed for GRPC build
            bazel_rc.write('build:macos --copt="-DGRPC_BAZEL_BUILD"\n')
            # Stay with 10.13 for macOS
            bazel_rc.write('build:macos --copt="-mmacosx-version-min=10.13"\n')
            bazel_rc.write('build:macos --linkopt="-mmacosx-version-min=10.13"\n')
            bazel_rc.close()
    except OSError:
        print("ERROR: Writing .bazelrc")
        exit(1)


write_config()

