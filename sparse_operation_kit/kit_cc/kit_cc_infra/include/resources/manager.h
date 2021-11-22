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

#ifndef RESOURCES_MANAGER_H
#define RESOURCES_MANAGER_H

#include "resources/gpu_resource.h"
#include "resources/cpu_resource.h"

#include <nccl.h>
#include <cuda_runtime.h>
#include <string>
#include <memory>
#include <mutex>
#include <vector>
#include <atomic>

namespace SparseOperationKit {

class ResourcesManager final {
public:
    ~ResourcesManager() = default;
    ResourcesManager(const ResourcesManager&) = delete;
    ResourcesManager& operator=(const ResourcesManager&) = delete;

    static std::shared_ptr<ResourcesManager> Create();
    static void get_nccl_unique_id(std::string& nccl_unique_id);
    static void get_nccl_unique_id(int32_t* nccl_unique_id);
    static void get_random_seed(uint64_t* seed);
    
    void init(const size_t global_replica_id, const size_t num_replicas_in_sync, 
              const int32_t* nccl_unique_id, const uint64_t global_seed,
              const cudaStream_t& tf_stream);

    bool p2p_enabled(const size_t src_dev, const size_t dst_dev) const;
    bool all_p2p_enabled() const;

    // synchonrize CPU with GPU stream
    void sync_local_gpus() const;
    void sync_local_memcpys() const;
    void sync_gpu(const size_t local_dev_id) const;
    void sync_all_workers() const; // synchronize each CPU-process
    void sync_all_gpus(const size_t local_dev_id) const; // synchronize each CPU-thread

    // synchronize CPU threads
    void sync_cpu_threads() const;

    template <typename Callable, typename... Args>
    void blocking_call_once(Callable&& func, Args&&... args) {
        cpu_resource_->blocking_call_once(std::forward<Callable>(func), std::forward<Args>(args)...);
    }

    template <typename Callable, typename... Args>
    void one_at_a_time(Callable&& func, Args&&... args) {
        cpu_resource_->one_at_a_time(std::forward<Callable>(func), std::forward<Args>(args)...);
    }

    template <typename Callable, typename... Args>
    void push_to_threadpool(Callable&& func, Args&&... args) {
        cpu_resource_->push_to_threadpool(std::forward<Callable>(func), std::forward<Args>(args)...);
    }

    void sync_threadpool() const;

    size_t get_local_gpu_count() const;
    size_t get_global_gpu_count() const;
    const std::shared_ptr<GpuResource>& get_local_gpu(const size_t id) const;

    void allocate_memory(const size_t global_replica_id) const;

    size_t get_worker_id() const;
    size_t get_workers_num() const;

    size_t cal_local_id_from_global_id(const size_t global_replica_id) const;
    size_t cal_global_id_from_local_id(const size_t local_replica_id) const;
    size_t cal_worker_id_from_global_id(const size_t global_replica_id) const;

private:
    ResourcesManager();
    void set_nccl_unique_id(const int32_t* nccl_unique_id);
    void create_cpu_resource(const uint64_t global_seed, const size_t num_replicas_in_sync);
    void create_gpu_resource(const size_t global_replica_id, const size_t num_replicas_in_sync,
                             const cudaStream_t& tf_stream);
    void enable_all_peer_access(const size_t global_replica_id);

    ncclUniqueId nid_;
    std::once_flag set_nccl_id_once_flag_;
    std::once_flag cpu_resource_once_flag_;
    std::atomic<bool> set_nccl_id_flag_;
    std::atomic<bool> cpu_resource_flag_;
    size_t local_gpu_count_;
    size_t global_gpu_count_;
    uint64_t seed_;
    std::vector<std::vector<bool>> p2p_matrix_;

    std::vector<std::shared_ptr<GpuResource>> gpu_resources_;
    std::shared_ptr<CpuResource> cpu_resource_;
};

} // namespace SparseOperationKit
#endif