#!/bin/bash
# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
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

set -eo pipefail

export TF_NEED_TENSORRT=0
export TF_NEED_ROCM=0
export TF_NEED_COMPUTECPP=0
export TF_NEED_OPENCL=0
export TF_NEED_OPENCL_SYCL=0
export TF_ENABLE_XLA=1
export TF_NEED_MPI=0

DESTDIR=$1

cd $DESTDIR
yes "" | bash ./configure || true

set -x

TF_ALL_TARGETS='//tensorflow/examples/...'

# Disable failed UT cases temporarily.
export TF_BUILD_BAZEL_TARGET="$TF_ALL_TARGETS "\

for i in $(seq 1 3); do
    [ $i -gt 1 ] && echo "WARNING: cmd execution failed, will retry in $((i-1)) times later" && sleep 2
    ret=0
    bazel test -c opt --config=opt --verbose_failures --local_test_jobs=40 --test_output=errors -- $TF_BUILD_BAZEL_TARGET && break || ret=$?
done

exit $ret