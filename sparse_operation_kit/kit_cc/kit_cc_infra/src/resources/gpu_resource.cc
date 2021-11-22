/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "resources/gpu_resource.h"
#include "common.h"
#include <iostream>

namespace SparseOperationKit {

GpuResource::GpuResource(const size_t local_device_id, const size_t global_device_id, 
                        const uint64_t replica_uniform_seed,
                        const uint64_t replica_variant_seed,
                        const ncclComm_t& nccl_comm) 
: local_device_id_(local_device_id), global_device_id_(global_device_id), 
out_stream_(false), nccl_comm_(nccl_comm), sm_count_(0), cc_major_(0), cc_minor_(0),
max_shared_memory_size_per_sm_(0), warp_size_(0),
nccl_sync_data_(nullptr)
{
    CK_CUDA(cudaStreamCreateWithFlags(&computation_stream_, cudaStreamNonBlocking));
    CK_CUDA(cudaStreamCreateWithFlags(&memcpy_stream_, cudaStreamNonBlocking));
    CK_CUDA(cudaEventCreateWithFlags(&compute_wait_memcpy_event_, cudaEventDisableTiming));

    CK_CURAND(curandCreateGenerator(&replica_uniform_curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
    CK_CURAND(curandSetPseudoRandomGeneratorSeed(replica_uniform_curand_generator_, replica_uniform_seed));
    CK_CURAND(curandSetStream(replica_uniform_curand_generator_, computation_stream_));
    CK_CURAND(curandCreateGenerator(&replica_variant_curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
    CK_CURAND(curandSetPseudoRandomGeneratorSeed(replica_variant_curand_generator_, replica_variant_seed));
    CK_CURAND(curandSetStream(replica_variant_curand_generator_, computation_stream_));
    
    CK_CUSPARSE(cusparseCreate(&replica_cusparse_handle_));
    CK_CUSPARSE(cusparseSetStream(replica_cusparse_handle_, computation_stream_));

    CK_CUDA(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, local_device_id_));
    CK_CUDA(cudaDeviceGetAttribute(&cc_major_, cudaDevAttrComputeCapabilityMajor, local_device_id_));
    CK_CUDA(cudaDeviceGetAttribute(&cc_minor_, cudaDevAttrComputeCapabilityMinor, local_device_id_));
    CK_CUDA(cudaDeviceGetAttribute(&max_shared_memory_size_per_sm_, 
                                   cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
                                   local_device_id_));
    max_shared_memory_size_per_sm_ -= (4 * 1024ul); // FIXME: in case it allocates all shared memory.
    CK_CUDA(cudaDeviceGetAttribute(&warp_size_, cudaDevAttrWarpSize, local_device_id_));

    CK_CUDA(cudaMalloc(&nccl_sync_data_, sizeof(int32_t) * 1));
}

GpuResource::GpuResource(const size_t local_device_id, const size_t global_device_id, 
                        const uint64_t replica_uniform_seed,
                        const uint64_t replica_variant_seed,
                        const ncclComm_t& nccl_comm,
                        const cudaStream_t& cuda_stream) 
: local_device_id_(local_device_id), global_device_id_(global_device_id), 
computation_stream_(cuda_stream), out_stream_(true), 
nccl_comm_(nccl_comm), sm_count_(0), cc_major_(0), cc_minor_(0),
max_shared_memory_size_per_sm_(0), warp_size_(0),
nccl_sync_data_(nullptr)
{
    CK_CUDA(cudaStreamCreateWithFlags(&memcpy_stream_, cudaStreamNonBlocking));
    CK_CUDA(cudaEventCreateWithFlags(&compute_wait_memcpy_event_, cudaEventDisableTiming));

    CK_CURAND(curandCreateGenerator(&replica_uniform_curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
    CK_CURAND(curandSetPseudoRandomGeneratorSeed(replica_uniform_curand_generator_, replica_uniform_seed));
    CK_CURAND(curandSetStream(replica_uniform_curand_generator_, computation_stream_));
    CK_CURAND(curandCreateGenerator(&replica_variant_curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
    CK_CURAND(curandSetPseudoRandomGeneratorSeed(replica_variant_curand_generator_, replica_variant_seed));
    CK_CURAND(curandSetStream(replica_variant_curand_generator_, computation_stream_));

    CK_CUSPARSE(cusparseCreate(&replica_cusparse_handle_));
    CK_CUSPARSE(cusparseSetStream(replica_cusparse_handle_, computation_stream_));

    CK_CUDA(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, local_device_id_));
    CK_CUDA(cudaDeviceGetAttribute(&cc_major_, cudaDevAttrComputeCapabilityMajor, local_device_id_));
    CK_CUDA(cudaDeviceGetAttribute(&cc_minor_, cudaDevAttrComputeCapabilityMinor, local_device_id_));
    CK_CUDA(cudaDeviceGetAttribute(&max_shared_memory_size_per_sm_, 
                                   cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
                                   local_device_id_));
    max_shared_memory_size_per_sm_ -= (4 * 1024ul); // FIXME: in case it allocates all shared memory.
    CK_CUDA(cudaDeviceGetAttribute(&warp_size_, cudaDevAttrWarpSize, local_device_id_));

    CK_CUDA(cudaMalloc(&nccl_sync_data_, sizeof(int32_t) * 1));
}

GpuResource::~GpuResource() {
    try {
        CK_NCCL(ncclCommDestroy(nccl_comm_));
        CK_CURAND(curandDestroyGenerator(replica_uniform_curand_generator_));
        CK_CURAND(curandDestroyGenerator(replica_variant_curand_generator_));
        CK_CUSPARSE(cusparseDestroy(replica_cusparse_handle_));
        if (!out_stream_) CK_CUDA(cudaStreamDestroy(computation_stream_));
        CK_CUDA(cudaStreamDestroy(memcpy_stream_));
        CK_CUDA(cudaEventDestroy(compute_wait_memcpy_event_));
        if (nccl_sync_data_) CK_CUDA(cudaFree(nccl_sync_data_));
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
    }
}

std::shared_ptr<GpuResource> GpuResource::Create(
                const size_t local_device_id, const size_t global_device_id, 
                const uint64_t replica_uniform_seed,
                const uint64_t replica_variant_seed,
                const ncclComm_t& nccl_comm) {

    return std::shared_ptr<GpuResource>(new GpuResource(local_device_id, global_device_id,
                                                        replica_uniform_seed, replica_variant_seed,
                                                        nccl_comm));
}

std::shared_ptr<GpuResource> GpuResource::Create(
                const size_t local_device_id, const size_t global_device_id, 
                const uint64_t replica_uniform_seed,
                const uint64_t replica_variant_seed,
                const ncclComm_t& nccl_comm,
                const cudaStream_t& cuda_stream) {

    return std::shared_ptr<GpuResource>(new GpuResource(local_device_id, global_device_id,
                                                        replica_uniform_seed, replica_variant_seed,
                                                        nccl_comm, cuda_stream));
}

size_t GpuResource::get_local_device_id() const {
    return local_device_id_;
}

size_t GpuResource::get_global_device_id() const {
    return global_device_id_;
}

const cudaStream_t& GpuResource::get_stream() const {
    return computation_stream_;
}

const cudaStream_t& GpuResource::get_memcpy_stream() const {
    return memcpy_stream_;
}

size_t GpuResource::get_sm_count() const {
    return static_cast<size_t>(sm_count_);
}

size_t GpuResource::get_max_smem_size_per_sm() const {
    return static_cast<size_t>(max_shared_memory_size_per_sm_);
}

size_t GpuResource::get_warp_size() const {
    return static_cast<size_t>(warp_size_);
}

const curandGenerator_t& GpuResource::get_variant_curand_gen() const {
    return replica_variant_curand_generator_;
}

const curandGenerator_t& GpuResource::get_uniform_curand_gen() const {
    return replica_uniform_curand_generator_;
}

const ncclComm_t& GpuResource::get_nccl() const {
    return nccl_comm_;
}

const cusparseHandle_t& GpuResource::get_cusparse() const {
    return replica_cusparse_handle_;
}

void GpuResource::make_comput_wait_memcpy() const {
    CK_CUDA(cudaEventSynchronize(compute_wait_memcpy_event_)); // necessary??
    CK_CUDA(cudaEventRecord(compute_wait_memcpy_event_, memcpy_stream_));
    CK_CUDA(cudaStreamWaitEvent(computation_stream_, compute_wait_memcpy_event_, 0));
}

void GpuResource::sync_gpu_via_nccl(const cudaStream_t& stream) const {
    CK_NCCL(ncclAllReduce(/*sendbuff=*/nccl_sync_data_, 
                          /*recvbuff=*/nccl_sync_data_,
                          /*count=*/1ul,
                          /*datatype=*/ncclUint64,
                          /*op=*/ncclMax,
                          nccl_comm_,
                          stream));
}

} // namespace SparseOperationKit

