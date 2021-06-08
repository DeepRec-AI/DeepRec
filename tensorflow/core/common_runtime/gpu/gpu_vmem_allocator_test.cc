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

#if GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_host_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_vmem_allocator.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

static void CheckStats(Allocator* a, int64 num_allocs, int64 bytes_in_use,
                       int64 peak_bytes_in_use, int64 largest_alloc_size) {
  absl::optional<AllocatorStats> stats;
  stats = a->GetStats();
  LOG(INFO) << "Alloc stats: " << std::endl << stats->DebugString();
  EXPECT_EQ(stats->bytes_in_use, bytes_in_use);
  EXPECT_EQ(stats->peak_bytes_in_use, peak_bytes_in_use);
  EXPECT_EQ(stats->num_allocs, num_allocs);
  EXPECT_EQ(stats->largest_alloc_size, largest_alloc_size);
}

TEST(GPUVMemAllocatorTest, VMemCreate) {
  PlatformGpuId platform_gpu_id(0);
  se::StreamExecutor* se =
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      se, platform_gpu_id, false /*use_unified_memory*/, {}, {});
  Allocator* device_allocator = new GPUBFCAllocator(sub_allocator, 1 << 30, "GPU_0_bfc");

  // VMEM disabled, expect OOM
  setenv("TF_GPU_VMEM", "false", 1);
  Allocator* allocator = maybe_create_gpu_vmem_allocator(device_allocator,
                                                         0, platform_gpu_id, 0, se);
  void* raw_ptr = allocator->AllocateRaw(1, (1 << 30) + 1024);
  CHECK_EQ(raw_ptr, nullptr);

  // VMEM enabled, expect success
  setenv("TF_GPU_VMEM", "true", 1);
  setenv("TF_CUDA_HOST_MEM_LIMIT_IN_MB", "1025", 1);
  allocator = maybe_create_gpu_vmem_allocator(device_allocator,
                                              0, platform_gpu_id, 0, se);
  raw_ptr = allocator->AllocateRaw(1, (1 << 30) + 1024);
  CHECK_NE(raw_ptr, nullptr);

  // VMEM enbaled, but exceeds host memory size, expect OOM
  setenv("TF_GPU_VMEM", "true", 1);
  setenv("TF_CUDA_HOST_MEM_LIMIT_IN_MB", "1024", 1);
  allocator = maybe_create_gpu_vmem_allocator(device_allocator,
                                              0, platform_gpu_id, 0, se);
  raw_ptr = allocator->AllocateRaw(1, (1 << 30) + 1024);
  CHECK_EQ(raw_ptr, nullptr);
}

TEST(GPUVMemAllocatorTest, VMemNoDups) {
  PlatformGpuId platform_gpu_id(0);
  se::StreamExecutor* se =
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      se, platform_gpu_id, false /*use_unified_memory*/, {}, {});
  Allocator* device_allocator = new GPUBFCAllocator(sub_allocator, 1 << 30, "GPU_0_bfc");

  SubAllocator* host_sub_allocator = new GpuHostAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      0, {}, {});
  Allocator* host_allocator = new BFCAllocator(host_sub_allocator, 1 << 30,
      true /*allow_growth*/,
      "GPUHost_0_bfc");

  GPUVMemAllocator a(device_allocator, host_allocator, 0, se);
  CheckStats(&a, 0, 0, 0, 0);

  // Allocate a lot of raw pointers
  std::vector<void*> ptrs;
  for (int s = 1; s < 1024; s++) {
    void* raw = a.AllocateRaw(1, s);
    ptrs.push_back(raw);
  }
  CheckStats(&a, 1023, 654336, 654336, 1024);

  std::sort(ptrs.begin(), ptrs.end());

  // Make sure none of them are equal, and that none of them overlap.
  for (size_t i = 1; i < ptrs.size(); i++) {
    ASSERT_NE(ptrs[i], ptrs[i - 1]);  // No dups
    size_t req_size = a.RequestedSize(ptrs[i - 1]);
    ASSERT_GT(req_size, 0);
    ASSERT_GE(static_cast<char*>(ptrs[i]) - static_cast<char*>(ptrs[i - 1]),
              req_size);
  }

  for (size_t i = 0; i < ptrs.size(); i++) {
    a.DeallocateRaw(ptrs[i]);
  }
  CheckStats(device_allocator, 1023, 0, 654336, 1024);
  CheckStats(&a, 1023, 0, 654336, 1024);
}

TEST(GPUVMemAllocatorTest, VMemOOM) {
  PlatformGpuId platform_gpu_id(0);
  se::StreamExecutor* se =
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      se, platform_gpu_id, false /*use_unified_memory*/, {}, {});
  Allocator* device_allocator = new GPUBFCAllocator(sub_allocator, 1 << 30, "GPU_0_bfc");

  SubAllocator* host_sub_allocator = new GpuHostAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      0, {}, {});
  Allocator* host_allocator = new BFCAllocator(host_sub_allocator, 1 << 30,
      true /*allow_growth*/,
      "GPUHost_0_bfc");

  GPUVMemAllocator a(device_allocator, host_allocator, 0, se);
  CheckStats(&a, 0, 0, 0, 0);

  void* raw_ptr  = a.AllocateRaw(1, 1 << 29);
  CHECK_NE(raw_ptr, nullptr);

  raw_ptr = device_allocator->AllocateRaw(1, (1 << 29) + 1024);
  CHECK_EQ(raw_ptr, nullptr);

  raw_ptr = a.AllocateRaw(1, (1 << 29) + 1024);
  CHECK_NE(raw_ptr, nullptr);

  CheckStats(device_allocator, 1, 1 << 29, 1 << 29, 1 << 29);
  CheckStats(host_allocator, 1, (1 << 29) + 1024, (1 << 29) + 1024, (1 << 29) + 1024);
  CheckStats(&a, 1, 1 << 29, (1 << 30) + 1024, 1 << 29);

  a.ClearStats();
  CheckStats(device_allocator, 0, 1 << 29, 1 << 29, 0);
  CheckStats(host_allocator, 0, (1 << 29) + 1024, (1 << 29) + 1024, 0);
  CheckStats(&a, 0, 1 << 29, (1 << 30) + 1024, 0);
}

}
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
