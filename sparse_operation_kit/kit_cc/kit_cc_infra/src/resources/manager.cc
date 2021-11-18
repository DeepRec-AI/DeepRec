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

#include "resources/manager.h"
#include "common.h"
#include <iostream>
#include <thread>
#include <random>

namespace SparseOperationKit {

ResourcesManager::ResourcesManager() 
: set_nccl_id_once_flag_(), cpu_resource_once_flag_(),
set_nccl_id_flag_(false), cpu_resource_flag_(false), 
global_gpu_count_(0), seed_(0)
{
    int32_t local_gpu_count = 0;
    CK_CUDA(cudaGetDeviceCount(&local_gpu_count));
    local_gpu_count_ = static_cast<size_t>(local_gpu_count);
    gpu_resources_.insert(gpu_resources_.begin(), local_gpu_count_, nullptr);
    
    for (size_t i = 0; i < local_gpu_count_; i++) 
        p2p_matrix_.push_back(std::vector<bool>(local_gpu_count_, false));
}

std::shared_ptr<ResourcesManager> ResourcesManager::Create() {
    return std::shared_ptr<ResourcesManager>(new ResourcesManager());
}

void ResourcesManager::get_nccl_unique_id(std::string& nccl_unique_id) {
    ncclUniqueId nid;
    CK_NCCL(ncclGetUniqueId(&nid));
    ncclUniqueId_to_string(nid, nccl_unique_id);
}

void ResourcesManager::get_nccl_unique_id(int32_t* nccl_unique_id) {
    ncclUniqueId nid;
    CK_NCCL(ncclGetUniqueId(&nid));
    ncclUniqueId_to_int(nid, nccl_unique_id);
}

void ResourcesManager::get_random_seed(uint64_t* seed) {
    if (0 == *seed) {
        std::random_device rd;
        *seed = rd();
    }
}

void ResourcesManager::set_nccl_unique_id(const int32_t* nccl_unique_id) {
    auto helper = [this, &nccl_unique_id]() {
        if (set_nccl_id_flag_) 
            throw std::runtime_error(ErrorBase + "ncclUniqueId is already set.");
        int_to_ncclUniqueId(nccl_unique_id, this->nid_);
        set_nccl_id_flag_ = true;
    };

    std::call_once(set_nccl_id_once_flag_, helper);
}

void ResourcesManager::create_cpu_resource(const uint64_t global_seed, const size_t num_replicas_in_sync) {
    auto helper = [this, &global_seed, &num_replicas_in_sync]() {
        if (cpu_resource_flag_)
            throw std::runtime_error(ErrorBase + "cpu_resource is already created.");
        this->seed_ = global_seed;
        cpu_resource_flag_ = true;
        global_gpu_count_ = num_replicas_in_sync;

        MESSAGE("Global seed is " + std::to_string(this->seed_));
        MESSAGE("Local GPU Count: " + std::to_string(this->local_gpu_count_));
        MESSAGE("Global GPU Count: " + std::to_string(this->global_gpu_count_));

        cpu_resource_ = CpuResource::Create(get_local_gpu_count());
    };

    std::call_once(cpu_resource_once_flag_, helper);
}

void ResourcesManager::create_gpu_resource(const size_t global_replica_id, 
            const size_t num_replicas_in_sync, const cudaStream_t& tf_stream) {

    const size_t local_replica_id = global_replica_id % local_gpu_count_;

    MESSAGE("Global Replica Id: " + std::to_string(global_replica_id) + "; " + 
            "Local Replica Id: " + std::to_string(local_replica_id));

    std::mt19937 gen(seed_);
    std::uniform_int_distribution<uint64_t> dis;
    const uint64_t replica_uniform_seed = dis(gen);
    const uint64_t replica_variant_seed = dis(gen);

    ncclComm_t nccl_comm;
    CK_NCCL(ncclCommInitRank(&nccl_comm, num_replicas_in_sync, nid_, global_replica_id));
    gpu_resources_[local_replica_id] = GpuResource::Create(local_replica_id,
                                                           global_replica_id,
                                                           replica_uniform_seed,
                                                           replica_variant_seed,
                                                           nccl_comm, tf_stream);
}

void ResourcesManager::init(const size_t global_replica_id, const size_t num_replicas_in_sync, 
                            const int32_t* nccl_unique_id, const uint64_t global_seed,
                            const cudaStream_t& tf_stream) {
    set_nccl_unique_id(nccl_unique_id);
    
    create_cpu_resource(global_seed, num_replicas_in_sync);

    while (!set_nccl_id_flag_ || !cpu_resource_flag_) { std::this_thread::yield(); }

    create_gpu_resource(global_replica_id, num_replicas_in_sync, tf_stream);

    enable_all_peer_access(global_replica_id);
}

void ResourcesManager::enable_all_peer_access(const size_t global_replica_id) {
    if (0 >= local_gpu_count_) throw std::runtime_error(ErrorBase + "There are no valid GPUs on this worker.");
    if (1 == local_gpu_count_) return; // only one GPU, no need to do enable peer access

    const size_t local_replica_id = global_replica_id % local_gpu_count_;
    std::shared_ptr<GpuResource> local_gpu = get_local_gpu(local_replica_id);

    for (size_t dev_id = 0; dev_id < local_gpu_count_; dev_id++) {
        if (dev_id == local_replica_id) continue; // itself

        // wait for all threads to finish create gpu resource.
        while (!get_local_gpu(dev_id)) std::this_thread::yield();

        int32_t can_access_peer;
        CK_CUDA(cudaDeviceCanAccessPeer(&can_access_peer, local_gpu->get_local_device_id(),
                                        get_local_gpu(dev_id)->get_local_device_id()));

        if (1 == can_access_peer) {
            p2p_matrix_[local_replica_id][dev_id] = true;
            cudaError_t ret = cudaDeviceEnablePeerAccess(get_local_gpu(dev_id)->get_local_device_id(), 0);
            if (ret != cudaSuccess && ret != cudaErrorPeerAccessAlreadyEnabled) {
                CK_CUDA(ret);
            } else {
                // cudaErrorPeerAccessAlreadyEnabled must not be handled as an error
                // so we reset it to cudaSuccess here
                cudaGetLastError();
            }
        } // if 1 == can_access_peer

    } // for dev_id

    auto helper = [this](){
        if (all_p2p_enabled()) MESSAGE("All peer to peer access enabled.");
        else MESSAGE("Not all peer to peer access enabled.");
    };
    blocking_call_once(helper);
}

bool ResourcesManager::p2p_enabled(const size_t src_dev, const size_t dst_dev) const {
    return p2p_matrix_[src_dev][dst_dev];
}

bool ResourcesManager::all_p2p_enabled() const {
    if (1 == local_gpu_count_) return false;
    for (size_t src_dev = 0; src_dev < local_gpu_count_; src_dev++) {
        size_t src_dev_id = get_local_gpu(src_dev)->get_local_device_id();
        for (size_t dst_dev = 0; dst_dev < local_gpu_count_; dst_dev++) {
            size_t dst_dev_id = get_local_gpu(dst_dev)->get_local_device_id();
            if (src_dev != dst_dev && !p2p_matrix_[src_dev_id][dst_dev_id]) return false;
        }
    }

    return true;
}

void ResourcesManager::sync_local_gpus() const {
    HugeCTR::CudaDeviceContext context;

    for (size_t dev_id = 0; dev_id < local_gpu_count_; ++dev_id) {
        const auto& local_gpu = get_local_gpu(dev_id);
        context.set_device(local_gpu->get_local_device_id());
        CK_CUDA(cudaStreamSynchronize(local_gpu->get_stream()));
    }
}

void ResourcesManager::sync_local_memcpys() const {
    HugeCTR::CudaDeviceContext context;
    for (size_t dev_id = 0; dev_id < local_gpu_count_; ++dev_id) {
        const auto& local_gpu = get_local_gpu(dev_id);
        context.set_device(local_gpu->get_local_device_id());
        CK_CUDA(cudaStreamSynchronize(local_gpu->get_memcpy_stream()));
    }
}

void ResourcesManager::sync_gpu(const size_t local_dev_id) const {
    // CK_CUDA(cudaStreamSynchronize(get_local_gpu(local_dev_id)->get_stream()));
    while (true) {
        auto error = cudaStreamQuery(get_local_gpu(local_dev_id)->get_stream());
        if (error == cudaErrorNotReady) std::this_thread::yield();
        else if (error == cudaSuccess) break;
        else CK_CUDA(error);
    }
}

void ResourcesManager::sync_all_workers() const {
    sync_local_gpus();

    CK_NCCL(ncclGroupStart());
    for (size_t dev_id = 0; dev_id < get_local_gpu_count(); dev_id++) {
        const auto &local_gpu = get_local_gpu(dev_id);
        local_gpu->sync_gpu_via_nccl(local_gpu->get_stream());
    } // for dev_id in local_gpu_count
    CK_NCCL(ncclGroupEnd());

    sync_local_gpus();
}

void ResourcesManager::sync_all_gpus(const size_t local_dev_id) const {
    sync_gpu(local_dev_id);

    const auto &local_gpu = get_local_gpu(local_dev_id);
    local_gpu->sync_gpu_via_nccl(local_gpu->get_stream());

    sync_gpu(local_dev_id);
}

void ResourcesManager::sync_cpu_threads() const {
    cpu_resource_->sync_cpu_threads();
}

void ResourcesManager::sync_threadpool() const {
    cpu_resource_->sync_threadpool();
}

size_t ResourcesManager::get_local_gpu_count() const {
    return local_gpu_count_;
}

size_t ResourcesManager::get_global_gpu_count() const {
    return global_gpu_count_;
}

const std::shared_ptr<GpuResource>& ResourcesManager::get_local_gpu(const size_t id) const {
    if (id >= local_gpu_count_) throw std::runtime_error(ErrorBase + "id is out of the range of local_gpu_count.");
    if (id >= gpu_resources_.size()) 
        throw std::runtime_error(ErrorBase + "id is out of the range of valid gpu_resources.");
    return gpu_resources_[id];
}

void ResourcesManager::allocate_memory(const size_t global_replica_id) const {
    /*There is no buffer in ResourcesManager, so that this function did nothing.*/
}

size_t ResourcesManager::get_worker_id() const {
    return get_local_gpu(0)->get_global_device_id() / get_local_gpu_count();
}

size_t ResourcesManager::get_workers_num() const {
    return get_global_gpu_count() / get_local_gpu_count();
}


size_t ResourcesManager::cal_local_id_from_global_id(const size_t global_replica_id) const {
    return global_replica_id % get_local_gpu_count();
}

size_t ResourcesManager::cal_global_id_from_local_id(const size_t local_replica_id) const {
    return local_replica_id + get_worker_id() * get_local_gpu_count();
}

size_t ResourcesManager::cal_worker_id_from_global_id(const size_t global_replica_id) const {
    return global_replica_id / get_local_gpu_count();
}


} // namespace SparseOperationKit