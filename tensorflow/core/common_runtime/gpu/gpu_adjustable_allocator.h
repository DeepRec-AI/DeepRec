/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_ADJUSTABLE_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_ADJUSTABLE_ALLOCATOR_H_

#include <algorithm>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

// A GPU BFC memory allocator which can support dynamic
// resource adjustment.
// This class is a friend of class BFCAllocator.
class GPUAdjustableAllocator final {
 public:
  // Adjust the memory_limit_ to allow memory grow/shrink at runtime
  // Returns adjusted memory_limit_. If the return value is less than
  // the new_memory_limit, the adjustment failed.
  size_t AdjustMemoryLimit(size_t new_memory_limit,
                           BFCAllocator* bfc_allocator);

  // Get the memory pool size and in used memory size of the bfc_allocator.
  void GetMemPoolStats(BFCAllocator* bfc_allocator,
                      int64_t* deviceMemPoolSize, int64_t* deviceMemStable);

 private:
  // Free the memory regions that are not in use
  size_t FreeEmptyMemory(size_t target_memory_bytes,
                         BFCAllocator* bfc_allocator)
      EXCLUSIVE_LOCKS_REQUIRED(lock_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_ADJUSTABLE_ALLOCATOR_H_
