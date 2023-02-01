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
N_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

export TF_CUDA_COMPUTE_CAPABILITIES="7.5,8.0"
export TF_NEED_TENSORRT=0
export TF_NEED_ROCM=0
export TF_NEED_COMPUTECPP=0
export TF_NEED_OPENCL=0
export TF_NEED_OPENCL_SYCL=0
export TF_ENABLE_XLA=1
export TF_NEED_MPI=0
export TF_GPU_COUNT=${N_GPUS}

DESTDIR=$1

cd $DESTDIR
yes "" | bash ./configure || true

set -x

TF_ALL_TARGETS='//tensorflow/core/...'

# Disable failed UT cases temporarily.
export TF_BUILD_BAZEL_TARGET="$TF_ALL_TARGETS "\
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
"-//tensorflow/core/kernels:fused_embedding_ops_test_gpu "\
"-//tensorflow/core/distributed_runtime/eager:eager_service_impl_test "\
"-//tensorflow/core/distributed_runtime:session_mgr_test "\
"-//tensorflow/core/distributed_runtime/eager:remote_mgr_test "\
"-//tensorflow/core/debug:grpc_session_debug_test "\
"-//tensorflow/core:common_runtime_gpu_gpu_device_test "\
"-//tensorflow/core:common_runtime_gpu_gpu_adjustable_allocator_test "\
"-//tensorflow/core:common_runtime_hierarchical_tree_broadcaster_test "\
"-//tensorflow/core:gpu_debug_allocator_test "\
"-//tensorflow/core:common_runtime_gpu_pool_allocator_test "\
"-//tensorflow/core:ev_allocator_tests "\
"-//tensorflow/core/grappler/optimizers:concat_cast_fusing_test "

for i in $(seq 1 3); do
    [ $i -gt 1 ] && echo "WARNING: cmd execution failed, will retry in $((i-1)) times later" && sleep 2
    ret=0
    bazel test -c opt --config=cuda --verbose_failures --test_env='NVIDIA_TF32_OVERRIDE=0' \
    --test_env=TF_GPU_COUNT \
    --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute  \
    --test_timeout="300,450,1200,3600" --local_test_jobs=80 --test_output=errors \
    -- $TF_BUILD_BAZEL_TARGET && break || ret=$?
done

exit $ret
