/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_GRAPH_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_GRAPH_UTIL_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/gpu/gpu_driver.h"

namespace xla {
namespace gpu {

struct MutexedGraphExecCache {
  tensorflow::mutex exec_graph_cache_mu_;
  int64 cache_size_ GUARDED_BY(exec_graph_cache_mu) = 0;
  stream_executor::gpu::GpuContext* GUARDED_BY(exec_graph_cache_mu)
      gpu_context_ = nullptr;
  std::list<void*> gpu_exec_graphs_ GUARDED_BY(exec_graph_cache_mu);
  std::unordered_map<BufferAllocations::KeyType, std::list<void*>::iterator>
      gpu_key_to_exec_graphs_map_ GUARDED_BY(exec_graph_cache_mu);

  // Pushing in a new pair of key and exec graph.
  void update_cache(BufferAllocations::KeyType key, void* gpu_exec_graph) {
    tensorflow::mutex_lock lock(exec_graph_cache_mu_);
    gpu_exec_graphs_.push_front(gpu_exec_graph);
    if (gpu_exec_graphs_.size() > cache_size_) {
      auto& graph_exec = gpu_exec_graphs_.back();
      auto* exec_graph =
          reinterpret_cast<stream_executor::gpu::GpuGraphExecHandle*>(
              &gpu_exec_graphs_.back());
      using stream_executor::gpu::GpuDriver;
      GpuDriver::DestroyExecutableGraph(gpu_context_, exec_graph);
      gpu_exec_graphs_.pop_back();
    }
    gpu_key_to_exec_graphs_map_[key] = gpu_exec_graphs_.begin();
  }

  void* get_exec_graph(BufferAllocations::KeyType key) {
    tensorflow::mutex_lock lock(exec_graph_cache_mu_);
    if (gpu_key_to_exec_graphs_map_.find(key) !=
        gpu_key_to_exec_graphs_map_.end()) {
      auto it = std::find(gpu_exec_graphs_.begin(), gpu_exec_graphs_.end(),
                          *(gpu_key_to_exec_graphs_map_[key]));
      if (it == gpu_exec_graphs_.end()) {
        gpu_key_to_exec_graphs_map_.erase(key);
        return nullptr;
      }
      auto gpu_exec_graph = *(gpu_key_to_exec_graphs_map_[key]);
      gpu_exec_graphs_.remove(gpu_exec_graph);
      gpu_exec_graphs_.push_front(gpu_exec_graph);
      gpu_key_to_exec_graphs_map_[key] = gpu_exec_graphs_.begin();
      return gpu_exec_graph;
    }
    return nullptr;
  }

  void set_cache_size(int64 cache_size) {
    tensorflow::mutex_lock lock(exec_graph_cache_mu_);
    cache_size_ = cache_size;
  }

  void set_gpu_context(stream_executor::gpu::GpuContext* gpu_context) {
    tensorflow::mutex_lock lock(exec_graph_cache_mu_);
    gpu_context_ = gpu_context;
  }

  size_t get_current_cache_size() {
    tensorflow::mutex_lock lock(exec_graph_cache_mu_);
    return gpu_exec_graphs_.size();
  }
};

struct MutexedGraphCacheStats {
  tensorflow::mutex graph_cache_mu;
  uint64 cache_hits GUARDED_BY(graph_cache_mu) = 0;
  uint64 temp_buffer_cache_hits GUARDED_BY(graph_cache_mu) = 0;
  uint64 cache_miss GUARDED_BY(graph_cache_mu) = 0;
  uint64 times_called GUARDED_BY(graph_cache_mu) = 0;
  size_t last_buf_key_hash GUARDED_BY(graph_cache_mu) = 0;
  uint64 last_buf_key_hits GUARDED_BY(graph_cache_mu) = 0;
};
}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_GRAPH_UTIL_H_