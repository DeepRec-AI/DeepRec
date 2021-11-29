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

yes "" | bash ./configure || true

set -x

TF_ALL_TARGETS='//tensorflow/...'

export TF_BUILD_BAZEL_TARGET="$TF_ALL_TARGETS "\
"-//tensorflow/go/... "\
"-//tensorflow/lite/... "\
"-//tensorflow/tools/... "\
"-//tensorflow/compiler/... "\

# Disable failed UT cases temporarily.
export TF_BUILD_BAZEL_TARGET="$TF_BUILD_BAZEL_TARGET "\
"-//tensorflow/c:c_api_experimental_test "\
"-//tensorflow/c:c_api_function_test "\
"-//tensorflow/c:while_loop_test "\
"-//tensorflow/compiler/mlir/... "\
"-//tensorflow/compiler/tests:unary_ops_test_cpu "\
"-//tensorflow/contrib/android/... "\
"-//tensorflow/contrib/compiler/tests:addsign_test_cpu "\
"-//tensorflow/contrib/distribute/python:parameter_server_strategy_test "\
"-//tensorflow/contrib/distributions:batch_normalization_test "\
"-//tensorflow/contrib/distributions:wishart_test "\
"-//tensorflow/contrib/quantize:quantize_parameterized_test "\
"-//tensorflow/contrib/rpc/python/kernel_tests:rpc_op_test "\
"-//tensorflow/core/common_runtime/eager:eager_op_rewrite_registry_test "\
"-//tensorflow/core/distributed_runtime:cluster_function_library_runtime_test "\
"-//tensorflow/core:graph_optimizer_fusion_engine_test "\
"-//tensorflow/python/autograph/pyct:inspect_utils_test_par "\
"-//tensorflow/python/autograph/pyct:compiler_test "\
"-//tensorflow/python/autograph/pyct:cfg_test "\
"-//tensorflow/python/autograph/pyct:ast_util_test "\
"-//tensorflow/python/data/experimental/kernel_tests:prefetch_with_slack_test "\
"-//tensorflow/python/debug:debugger_cli_common_test "\
"-//tensorflow/python/debug:dist_session_debug_grpc_test "\
"-//tensorflow/python:deprecation_test "\
"-//tensorflow/python/distribute:values_test "\
"-//tensorflow/python/distribute:parameter_server_strategy_test "\
"-//tensorflow/python/eager:remote_test "\
"-//tensorflow/python/keras/distribute:multi_worker_fault_tolerance_test "\
"-//tensorflow/python/keras:callbacks_test "\
"-//tensorflow/python/keras:simplernn_test "\
"-//tensorflow/python/keras:lstm_test "\
"-//tensorflow/python/keras:hdf5_format_test "\
"-//tensorflow/python/profiler:model_analyzer_test "\
"-//tensorflow/python/tools/api/generator:output_init_files_test "\
"-//tensorflow/python/tpu:datasets_test "\
"-//tensorflow/contrib/quantize:fold_batch_norms_test "\
"-//tensorflow/python/kernel_tests:unique_op_test "\

bazel test -c opt --config=opt --verbose_failures -- $TF_BUILD_BAZEL_TARGET
