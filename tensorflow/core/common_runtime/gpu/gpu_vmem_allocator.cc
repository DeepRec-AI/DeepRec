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

#include "tensorflow/core/common_runtime/gpu/gpu_vmem_allocator.h"
#include "tensorflow/core/common_runtime/allocator_retry.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_host_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

void* GPUVMemAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
    mutex_lock l(lock_);
    AllocationAttributes new_attr;
    // Tell the device_allocator_ not to retry
    // since we can alloc host memory as backup
    new_attr.no_retry_on_failure = true;
    void* ret = device_allocator_->AllocateRaw(alignment, num_bytes, new_attr);
    if (ret != nullptr) {
      device_ptrs_.insert(ret);
      return ret;
    }
    ret = host_allocator_->AllocateRaw(alignment, num_bytes);
    VLOG(3) << "host_allocator_ allocates " << (num_bytes/1024.0/1024) << " MiB";
    return ret;
}

void* GPUVMemAllocator::AllocateRaw(size_t alignment, size_t num_bytes,
    const AllocationAttributes& allocation_attr) {
    mutex_lock l(lock_);
    AllocationAttributes new_attr;
    // Tell the device_allocator_ not to retry
    // since we can alloc host memory as backup
    new_attr.no_retry_on_failure = true;
    void* ret = device_allocator_->AllocateRaw(alignment, num_bytes, new_attr);
    if (ret != nullptr) {
      device_ptrs_.insert(ret);
      return ret;
    }
    absl::optional<AllocatorStats> stats;
    stats = this->GetStats();
    if (stats->peak_bytes_in_use > memory_planned_) {
      LOG(ERROR) << "Host memory allocation failed: this job has already used"
                 << (stats->peak_bytes_in_use/1024.0/1024)
                 << " MiB memory which is beyound the upper limit of "
                 << "memory allocation ("
                 << (memory_planned_/1024.0/1024) << "MiB).";
      return ret;
    }
    ret = host_allocator_->AllocateRaw(alignment, num_bytes, allocation_attr);
    VLOG(3) << "host_allocator_ allocates " << (num_bytes/1024.0/1024) << " MiB";
    return ret;
}

void GPUVMemAllocator::DeallocateRaw(void* ptr) {
    mutex_lock l(lock_);
    if (device_ptrs_.count(ptr) > 0) {
      device_allocator_->DeallocateRaw(ptr);
      device_ptrs_.erase(ptr);
      return;
    } else {
      host_allocator_->DeallocateRaw(ptr);
    }
}

size_t GPUVMemAllocator::RequestedSize(const void* ptr) const {
    mutex_lock l(lock_);
    if (device_ptrs_.count(ptr) > 0) {
      return device_allocator_->RequestedSize(ptr);
    } else {
      return host_allocator_->RequestedSize(ptr);
    }
}

size_t GPUVMemAllocator::AllocatedSize(const void* ptr) const {
    mutex_lock l(lock_);
    if (device_ptrs_.count(ptr) > 0) {
      return device_allocator_->AllocatedSize(ptr);
    } else {
      return host_allocator_->AllocatedSize(ptr);
    }
}

int64 GPUVMemAllocator::AllocationId(const void* ptr) const {
    mutex_lock l(lock_);
    if (device_ptrs_.count(ptr) > 0) {
      return device_allocator_->AllocationId(ptr);
    } else {
      return host_allocator_->AllocationId(ptr);
    }
}

absl::optional<AllocatorStats> GPUVMemAllocator::GetStats() {
    absl::optional<AllocatorStats> stats = device_allocator_->GetStats();
    absl::optional<AllocatorStats> allocator_stats = host_allocator_->GetStats();
    stats->peak_bytes_in_use += (allocator_stats ? allocator_stats->peak_bytes_in_use : 0);
    return stats;
}

void GPUVMemAllocator::ClearStats() {
    device_allocator_->ClearStats();
    host_allocator_->ClearStats();
}

Allocator* maybe_create_gpu_vmem_allocator(Allocator* gpu_allocator,
                                           int bus_id,
                                           PlatformGpuId platform_gpu_id,
                                           int tf_gpu_id,
                                           se::StreamExecutor* stream_exec) {
  bool gpu_vmem = false;
  Status status = ReadBoolFromEnvVar("TF_GPU_VMEM",
                                     true/*enabled by default*/,
                                     &gpu_vmem);
  if (!status.ok()) {
    LOG(ERROR) << "GetGPUAllocator: " << status.error_message();
  }
  if (!gpu_vmem) {
    return gpu_allocator;
  }
  SubAllocator* sub_allocator = new GpuHostAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      bus_id, {}, {});
  int64 cuda_host_mem_limit_in_mb = -1;
  status = ReadInt64FromEnvVar("TF_CUDA_HOST_MEM_LIMIT_IN_MB",
                               1LL << 16 /*64GB max by default*/,
                               &cuda_host_mem_limit_in_mb);
  if (!status.ok()) {
    LOG(ERROR) << "GetGpuHostAllocator: " << status.error_message();
  }
  int64 cuda_host_mem_limit = cuda_host_mem_limit_in_mb * (1LL << 20);
  Allocator* host_allocator =
      new BFCAllocator(sub_allocator, cuda_host_mem_limit,
                       true /*allow_growth*/,
                       strings::StrCat("GPUHost_", tf_gpu_id, "_bfc"));
  Allocator* gpu_vmem_allocator = new GPUVMemAllocator(gpu_allocator,
                                                       host_allocator,
                                                       tf_gpu_id,
                                                       stream_exec);
  return gpu_vmem_allocator;
}

}  // namespace tensorflow
