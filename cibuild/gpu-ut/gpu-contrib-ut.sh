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

export TF_CUDA_COMPUTE_CAPABILITIES="7.5,8.0"
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

TF_ALL_TARGETS='//tensorflow/contrib/...'

# Disable failed UT cases temporarily.
export TF_BUILD_BAZEL_TARGET="$TF_ALL_TARGETS "\
"-//tensorflow/contrib/android/... "\
"-//tensorflow/contrib/compiler/tests:addsign_test_cpu "\
"-//tensorflow/contrib/compiler/tests:addsign_test_gpu "\
"-//tensorflow/contrib/cudnn_rnn:cudnn_rnn_ops_benchmark "\
"-//tensorflow/contrib/cudnn_rnn:cudnn_rnn_test "\
"-//tensorflow/contrib/distribute/python:parameter_server_strategy_test "\
"-//tensorflow/contrib/distributions:batch_normalization_test "\
"-//tensorflow/contrib/distributions:batch_normalization_test_gpu "\
"-//tensorflow/contrib/distributions:wishart_test "\
"-//tensorflow/contrib/distributions:wishart_test_gpu "\
"-//tensorflow/contrib/distribute/python:keras_backward_compat_test "\
"-//tensorflow/contrib/distribute/python:keras_backward_compat_test_gpu "\
"-//tensorflow/contrib/eager/python/examples/resnet50:resnet50_graph_test_gpu "\
"-//tensorflow/contrib/factorization:gmm_test "\
"-//tensorflow/contrib/factorization:wals_test "\
"-//tensorflow/contrib/layers:layers_test "\
"-//tensorflow/contrib/layers:layers_test_gpu "\
"-//tensorflow/contrib/layers:target_column_test "\
"-//tensorflow/contrib/learn:monitors_test "\
"-//tensorflow/contrib/quantize:quantize_parameterized_test "\
"-//tensorflow/contrib/quantize:fold_batch_norms_test "\
"-//tensorflow/contrib/rnn:gru_ops_test "\
"-//tensorflow/contrib/rpc/python/kernel_tests:rpc_op_test "\
"-//tensorflow/contrib/timeseries/python/timeseries/state_space_models:structural_ensemble_test "\
"-//tensorflow/contrib/eager/python:saver_test "\
"-//tensorflow/contrib/eager/python:saver_test_gpu "\
"-//tensorflow/contrib/compiler/tests:adamax_test_gpu "\
"-//tensorflow/contrib/cudnn_rnn:cudnn_rnn_ops_test_gpu "\
"-//tensorflow/contrib/compiler/tests:powersign_test_gpu "\

for i in $(seq 1 3); do
    [ $i -gt 1 ] && echo "WARNING: cmd execution failed, will retry in $((i-1)) times later" && sleep 2
    ret=0
    bazel test -c opt --config=cuda --verbose_failures \
    --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute  \
    --test_timeout="300,450,1200,3600" --local_test_jobs=1  \
    -- $TF_BUILD_BAZEL_TARGET && break || ret=$?
done

exit $ret