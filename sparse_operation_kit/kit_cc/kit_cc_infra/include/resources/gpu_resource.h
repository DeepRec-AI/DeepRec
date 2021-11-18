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

#ifndef GPU_RESOURCE_H
#define GPU_RESOURCE_H

#include <cuda_runtime.h>
#include <curand.h>
#include <cusparse.h>
#include <nccl.h>
#include <memory>

namespace SparseOperationKit {

class GpuResource {
private:
    const size_t local_device_id_;
    const size_t global_device_id_;
    cudaStream_t computation_stream_;
    cudaStream_t memcpy_stream_;
    cudaEvent_t compute_wait_memcpy_event_;
    const bool out_stream_; // true represent the computation stream is from outside.
    curandGenerator_t replica_uniform_curand_generator_;
    curandGenerator_t replica_variant_curand_generator_;
    cusparseHandle_t replica_cusparse_handle_;
    ncclComm_t nccl_comm_;
    int32_t sm_count_;
    int32_t cc_major_;
    int32_t cc_minor_;
    int32_t max_shared_memory_size_per_sm_;
    int32_t warp_size_;

    int32_t *nccl_sync_data_;

    GpuResource(const size_t local_device_id, const size_t global_device_id, 
                const uint64_t replica_uniform_seed,
                const uint64_t replica_variant_seed,
                const ncclComm_t& nccl_comm);
    GpuResource(const size_t local_device_id, const size_t global_device_id, 
                const uint64_t replica_uniform_seed,
                const uint64_t replica_variant_seed,
                const ncclComm_t& nccl_comm, 
                const cudaStream_t& cuda_stream);
public:
    GpuResource(const GpuResource&) = delete;
    GpuResource& operator=(const GpuResource&) = delete;
    ~GpuResource();
    
    static std::shared_ptr<GpuResource> Create(
                const size_t local_device_id, const size_t global_device_id, 
                const uint64_t replica_uniform_seed,
                const uint64_t replica_variant_seed,
                const ncclComm_t& nccl_comm);
    static std::shared_ptr<GpuResource> Create(
                const size_t local_device_id, const size_t global_device_id, 
                const uint64_t replica_uniform_seed,
                const uint64_t replica_variant_seed,
                const ncclComm_t& nccl_comm,
                const cudaStream_t& cuda_stream);

    size_t get_local_device_id() const;
    size_t get_global_device_id() const;
    const cudaStream_t& get_stream() const; 
    const cudaStream_t& get_memcpy_stream() const;
    size_t get_sm_count() const;
    size_t get_max_smem_size_per_sm() const;
    size_t get_warp_size() const;
    const curandGenerator_t& get_variant_curand_gen() const;
    const curandGenerator_t& get_uniform_curand_gen() const;
    const ncclComm_t& get_nccl() const;
    const cusparseHandle_t& get_cusparse() const;
    void make_comput_wait_memcpy() const;

    void sync_gpu_via_nccl(const cudaStream_t& stream) const;
};


} // namespace SparseOperationKit

#endif // GPU_RESOURCE_H