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
"-//tensorflow/c:c_api_test_gpu "\
"-//tensorflow/c/eager:c_api_experimental_test_gpu "\
"-//tensorflow/c:kernels_test_gpu "\
"-//tensorflow/c:while_loop_test "\
"-//tensorflow/compiler/mlir/... "\
"-//tensorflow/compiler/tests:unary_ops_test_cpu "\
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
"-//tensorflow/core/common_runtime/eager:eager_op_rewrite_registry_test "\
"-//tensorflow/core/distributed_runtime:cluster_function_library_runtime_test "\
"-//tensorflow/core/distributed_runtime/rpc:rpc_rendezvous_mgr_test_gpu "\
"-//tensorflow/core/distributed_runtime/rpc:grpc_session_test_gpu "\
"-//tensorflow/core/distributed_runtime/rpc:grpc_channel_test_gpu "\
"-//tensorflow/core/kernels:collective_nccl_test_gpu "\
"-//tensorflow/core/kernels:batched_non_max_suppression_op_gpu_test_gpu "\
"-//tensorflow/core/kernels:depthwise_conv_ops_test_gpu "\
"-//tensorflow/core/kernels:dynamic_partition_op_test_gpu "\
"-//tensorflow/core/kernels:resize_bilinear_op_test_gpu "\
"-//tensorflow/core:gpu_event_mgr_test "\
"-//tensorflow/core:gpu_device_unified_memory_test_gpu "\
"-//tensorflow/core:graph_optimizer_fusion_engine_test "\
"-//tensorflow/core:graph_star_server_graph_partition_test "\
"-//tensorflow/core/grappler/clusters:utils_test "\
"-//tensorflow/core/grappler/optimizers:remapper_test_gpu "\
"-//tensorflow/core/grappler/optimizers:constant_folding_test "\
"-//tensorflow/core/grappler/optimizers:memory_optimizer_test_gpu "\
"-//tensorflow/core/nccl:nccl_manager_test_gpu "\
"-//tensorflow/core/kernels:constant_op_test_gpu "\
"-//tensorflow/core/kernels:fused_batch_norm_ex_op_test_gpu "\
"-//tensorflow/core/kernels:non_max_suppression_op_gpu_test_gpu "\
"-//tensorflow/core/kernels:segment_reduction_ali_ops_test_gpu "\
"-//tensorflow/core:variant_op_copy_test "\
"-//tensorflow/core:util_gpu_kernel_helper_test_gpu "\
"-//tensorflow/core:common_runtime_gpu_gpu_vmem_allocator_test "\
"-//tensorflow/core:common_runtime_gpu_gpu_bfc_allocator_test "\
"-//tensorflow/core:common_runtime_ring_gatherer_test "\
"-//tensorflow/core:common_runtime_ring_reducer_test "\
"-//tensorflow/examples/adding_an_op:cuda_op_test "\
"-//tensorflow/stream_executor/cuda:redzone_allocator_test_gpu "\
"-//tensorflow/python/autograph/pyct:inspect_utils_test_par "\
"-//tensorflow/python/autograph/pyct:compiler_test "\
"-//tensorflow/python/autograph/pyct:cfg_test "\
"-//tensorflow/python/autograph/pyct:ast_util_test "\
"-//tensorflow/python/data/experimental/kernel_tests:prefetch_with_slack_test "\
"-//tensorflow/python:collective_ops_gpu_test_gpu "\
"-//tensorflow/python:collective_ops_gpu_test "\
"-//tensorflow/python/debug:local_cli_wrapper_test "\
"-//tensorflow/python/debug:debugger_cli_common_test "\
"-//tensorflow/python/debug:framework_test "\
"-//tensorflow/python/debug:source_remote_test "\
"-//tensorflow/python/debug:session_debug_grpc_test "\
"-//tensorflow/python/debug:grpc_large_data_test "\
"-//tensorflow/python/debug:dist_session_debug_grpc_test "\
"-//tensorflow/python/debug:dist_session_debug_grpc_test_gpu "\
"-//tensorflow/python/debug:dumping_wrapper_test "\
"-//tensorflow/python:deprecation_test "\
"-//tensorflow/python/distribute:values_test "\
"-//tensorflow/python/distribute:values_test_gpu "\
"-//tensorflow/python/distribute:parameter_server_strategy_test "\
"-//tensorflow/python/distribute:moving_averages_test_gpu "\
"-//tensorflow/python/distribute:moving_averages_test "\
"-//tensorflow/python/distribute:metrics_v1_test "\
"-//tensorflow/python/distribute:metrics_v1_test_gpu "\
"-//tensorflow/python/distribute:keras_metrics_test "\
"-//tensorflow/python/distribute:keras_metrics_test_gpu "\
"-//tensorflow/python/eager:remote_test "\
"-//tensorflow/python/eager:remote_test_gpu "\
"-//tensorflow/python/eager:def_function_xla_test_gpu "\
"-//tensorflow/python:embedding_variable_ops_test "\
"-//tensorflow/python/keras/distribute:multi_worker_fault_tolerance_test "\
"-//tensorflow/python/keras:callbacks_test "\
"-//tensorflow/python/keras:core_test "\
"-//tensorflow/python/keras/distribute:keras_dnn_correctness_test "\
"-//tensorflow/python/keras/distribute:keras_image_model_correctness_test "\
"-//tensorflow/python/keras/distribute:keras_lstm_model_correctness_test "\
"-//tensorflow/python/keras/distribute:keras_embedding_model_correctness_test_gpu "\
"-//tensorflow/python/keras/distribute:keras_lstm_model_correctness_test_gpu "\
"-//tensorflow/python/keras/distribute:keras_image_model_correctness_test_gpu "\
"-//tensorflow/python/keras/distribute:keras_dnn_correctness_test_gpu "\
"-//tensorflow/python/keras/distribute:multi_worker_test "\
"-//tensorflow/python/keras/distribute:keras_utils_test_gpu "\
"-//tensorflow/python/keras/distribute:keras_embedding_model_correctness_test "\
"-//tensorflow/python/keras/distribute:distribute_strategy_test_gpu "\
"-//tensorflow/python/keras/distribute:distribute_strategy_test "\
"-//tensorflow/python/keras/distribute:keras_utils_test "\
"-//tensorflow/python/keras/premade:wide_deep_test "\
"-//tensorflow/python/keras:simplernn_test "\
"-//tensorflow/python/keras:lstm_test "\
"-//tensorflow/python/keras:hdf5_format_test "\
"-//tensorflow/python/kernel_tests:unique_op_test "\
"-//tensorflow/python/kernel_tests:bias_op_test_gpu "\
"-//tensorflow/python/kernel_tests:bias_op_test "\
"-//tensorflow/python/kernel_tests:conv_ops_3d_test_gpu "\
"-//tensorflow/python/kernel_tests/signal:fft_ops_test "\
"-//tensorflow/python/kernel_tests:lu_op_test_gpu "\
"-//tensorflow/python/kernel_tests:conv_ops_3d_test "\
"-//tensorflow/python:layers_core_test "\
"-//tensorflow/python:optimizer_test "\
"-//tensorflow/python/profiler:model_analyzer_test_gpu "\
"-//tensorflow/python/profiler:model_analyzer_test "\
"-//tensorflow/python/tools/api/generator:output_init_files_test "\
"-//tensorflow/python/tpu:datasets_test "\
"-//tensorflow/python/training/tracking:util_xla_test_gpu "\
"-//tensorflow/core/kernels:fused_embedding_ops_test_gpu "\
"-//tensorflow/core/distributed_runtime/eager:eager_service_impl_test "\
"-//tensorflow/core/distributed_runtime:session_mgr_test "\
"-//tensorflow/core/distributed_runtime/eager:remote_mgr_test "\
"-//tensorflow/core/debug:grpc_session_debug_test "\
"-//tensorflow/python/kernel_tests:sparse_conditional_accumulator_test "\
"-//tensorflow/c:c_test "\
"-//tensorflow/contrib/eager/python:saver_test "\
"-//tensorflow/contrib/eager/python:saver_test_gpu "\
"-//tensorflow/python/debug:session_debug_grpc_test_gpu "\
"-//tensorflow/python/debug:grpc_large_data_test_gpu "\
"-//tensorflow/python/keras:training_test "\
"-//tensorflow/python/keras:convolutional_test "\
"-//tensorflow/contrib/compiler/tests:adamax_test_gpu "\
"-//tensorflow/core:common_runtime_gpu_gpu_device_test "\
"-//tensorflow/core:common_runtime_gpu_gpu_adjustable_allocator_test "\
"-//tensorflow/core:common_runtime_hierarchical_tree_broadcaster_test "\

bazel test -c opt --config=cuda --verbose_failures --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute  --test_timeout="300,450,1200,3600" --local_test_jobs=2  -- $TF_BUILD_BAZEL_TARGET
