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
#include "tensorflow/core/common_runtime/gpu/gpu_cuda_graph_mode_mem.h"

#include <iostream>
#include <mutex>
#include <vector>

namespace tensorflow {

CudaGraphModeMem::CudaGraphModeMem() : n(0) {}

CudaGraphModeMem::CudaGraphModeMem(const CudaGraphModeMem& cuda_graph_mode_mem)
    : deallocated_states_(cuda_graph_mode_mem.deallocated_states_),
      mem_sets_(cuda_graph_mode_mem.mem_sets_),
      n(cuda_graph_mode_mem.n) {}

CudaGraphModeMem::~CudaGraphModeMem() {}

void CudaGraphModeMem::Add(void* mem, size_t size, bool mem_reuse) {
  std::lock_guard<std::mutex> lck(mtx_);
  if (ContainMem(mem)) return;
  mem_sets_.emplace(size, mem);
  deallocated_states_[mem] = false;
  reuse_states_[mem] = mem_reuse;
  ++n;
}

void CudaGraphModeMem::MarkDeallocated(void* mem) {
  std::lock_guard<std::mutex> lck(mtx_);
  deallocated_states_[mem] = true;
}

bool CudaGraphModeMem::ContainMem(void* mem) {
  if (deallocated_states_.size() > 0) {
    return deallocated_states_.find(mem) != deallocated_states_.end();
  } else {
    return false;
  }
}

void* CudaGraphModeMem::GetReuseMem(size_t size) {
  std::lock_guard<std::mutex> lck(mtx_);
  for (auto& x : mem_sets_) {
    if (x.first >= size && deallocated_states_[x.second] &&
        reuse_states_[x.second]) {
      deallocated_states_[x.second] = false;
      return x.second;
    }
  }
  return nullptr;
}

size_t CudaGraphModeMem::GetAllocatedNum() { return n; }

size_t CudaGraphModeMem::GetAllocatedSize() {
  size_t ret = 0;
  for (auto& x : mem_sets_) {
    ret += x.first;
  }
  return ret;
}

std::unordered_set<void*> CudaGraphModeMem::GetDeallocatedMems() {
  std::unordered_set<void*> ret;
  for (auto& x : deallocated_states_) {
    if (x.second == true) {
      ret.insert(x.first);
    }
  }
  return ret;
}

}  // namespace tensorflow
#endif  // GOOGLE_CUDA
