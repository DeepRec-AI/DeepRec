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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_VMEM_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_VMEM_ALLOCATOR_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

// GPUVMemAllocator is an encapsulation of
// GPUBFCAllocator & GPUHostBFCAllocator
class GPUVMemAllocator : public Allocator {
 public:
  GPUVMemAllocator(Allocator* device_allocator,
                   Allocator* host_allocator,
                   int device_id,
                   se::StreamExecutor* stream_exec) {
    name_ = strings::StrCat("GPUVMem_", device_id, "_bfc");
    device_allocator_ = device_allocator;
    host_allocator_ = host_allocator;
    int64 total_memory = 0;
    int64 available_memory = 0;
    if (!stream_exec->DeviceMemoryUsage(&available_memory, &total_memory)) {
      LOG(ERROR) << "Failed to query available memory for GPU " << device_id;
    }
    int64 ro_planned_gpu = 100;
    ReadInt64FromEnvVar("TF_RO_PLANNED_GPU", 100,
                        &ro_planned_gpu);
    memory_planned_ = int64(total_memory * (ro_planned_gpu/100.0));
  }

  virtual ~GPUVMemAllocator() {
    delete device_allocator_;
    delete host_allocator_;
  }

  string Name() override { return name_; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override;

  void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr) override;

  void DeallocateRaw(void* ptr) override;

  bool TracksAllocationSizes() const override {
    return device_allocator_->TracksAllocationSizes();
  }

  size_t RequestedSize(const void* ptr) const override;

  size_t AllocatedSize(const void* ptr) const override;

  int64 AllocationId(const void* ptr) const override;

  absl::optional<AllocatorStats> GetStats() override;

  void ClearStats() override;

  Allocator* DeviceAllocator() const { return device_allocator_; }

  Allocator* HostAllocator() const { return host_allocator_; }

  TF_DISALLOW_COPY_AND_ASSIGN(GPUVMemAllocator);

 private:
  string name_;
  Allocator* device_allocator_;
  Allocator* host_allocator_;
  mutable mutex lock_;
  std::set<const void*> device_ptrs_ GUARDED_BY(lock_);
  int64 memory_planned_;
};

Allocator* maybe_create_gpu_vmem_allocator(Allocator* gpu_allocator,
                                           int bus_id,
                                           PlatformGpuId platform_gpu_id,
                                           int tf_gpu_id,
                                           se::StreamExecutor* stream_exec);
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_VMEM_ALLOCATOR_H_
