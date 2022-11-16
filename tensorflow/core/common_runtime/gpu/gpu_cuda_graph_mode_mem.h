/* Copyright 2022 The DeepRec Authors. All Rights Reserved.

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
#if GOOGLE_CUDA
#ifndef TENSORFLOW_CORE_FRAMEWORK_GPU_CUDA_GRAPH_MODE_MEM_H_
#define TENSORFLOW_CORE_FRAMEWORK_GPU_CUDA_GRAPH_MODE_MEM_H_

#include <cuda_runtime.h>
#include <atomic>
#include <functional>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tensorflow {

class CudaGraphModeMem {
 public:
  CudaGraphModeMem();
  CudaGraphModeMem(const CudaGraphModeMem& cuda_graph_mode_mem);
  ~CudaGraphModeMem();

  void Add(void* mem, size_t size, bool mem_reuse = true);
  void MarkDeallocated(void* mem);
  bool ContainMem(void* mem);
  void* GetReuseMem(size_t size);
  size_t GetAllocatedNum();
  size_t GetDeallocatedNum();
  size_t GetAllocatedSize();
  std::unordered_set<void*> GetDeallocatedMems();

 private:
  std::mutex mtx_;
  std::unordered_set<void*> mems_;

  std::unordered_map<void*, bool> deallocated_states_;
  std::unordered_map<void*, bool> reuse_states_;
  std::set<std::pair<size_t, void*>> mem_sets_;
  size_t n;
  size_t deallocated_n;
};
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_FRAMEWORK_GPU_CUDA_GRAPH_MODE_MEM_H_
#endif // GOOGLE_CUDA
