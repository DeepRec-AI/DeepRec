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

#include <algorithm>
#include <vector>

#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_adjustable_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_host_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/gpu_vmem_allocator.h"
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

TEST(GPUAdjustableAllocatorTest, ShrinkMemoryLimitAndAllocateToHost) {
  // Enable VMEM.
  setenv("TF_GPU_VMEM", "true", 1);
  // setenv("TF_CUDA_HOST_MEM_LIMIT_IN_MB", "16384", 1);

  PlatformGpuId platform_gpu_id(0);
  GPUOptions options;
  options.set_allow_growth(true);
  se::StreamExecutor* se =
    GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      se, platform_gpu_id, false /*use_unified_memory*/, {}, {});
  Allocator* device_allocator = new GPUBFCAllocator(
      sub_allocator, 1UL << 30, options, "GPU_0_bfc");

  SubAllocator* host_sub_allocator = new GpuHostAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      0, {}, {});
  Allocator* host_allocator = new BFCAllocator(host_sub_allocator, 1UL << 30,
      true /*allow_growth*/,
      "GPUHost_0_bfc");

  GPUVMemAllocator a(device_allocator, host_allocator, 0, se);
  CheckStats(&a, 0, 0, 0, 0);
  AllocatorStats stats;

  // allocate 700KB
  void *p1 = a.AllocateRaw(1, 700 * 1024);

  // allocate 2MB
  void *p2 = a.AllocateRaw(1, 2 * 1024 * 1024);

  // allocate 400KB, split chunk
  void *p3 = a.AllocateRaw(1, 400 * 1024);

  // allocate 400KB
  void *p4 = a.AllocateRaw(1, 400 * 1024);

  // free 400KB
  a.DeallocateRaw(p4);

  // free 400KB
  a.DeallocateRaw(p3);

  // free 2MB
  a.DeallocateRaw(p2);

  // limit to 2MB
  GPUAdjustableAllocator* adj = new GPUAdjustableAllocator();
  size_t new_limit =  adj->AdjustMemoryLimit(
      2UL * 1024 * 1024, dynamic_cast<GPUBFCAllocator *>(device_allocator));
  EXPECT_EQ(new_limit, 2UL * 1024 * 1024);
  // expect to release two regions

  // allocate 2MB, expect to allocate on host
  void *p5 = a.AllocateRaw(1, 2 * 1024 * 1024);
  CHECK_NE(p5, nullptr);
  CheckStats(device_allocator, 4, 1 * 1024 * 1024,
      (1 + 2) * 1024 * 1024 + (400 + 400) * 1024, 2 * 1024 * 1024);
  CheckStats(host_allocator, 1, 2 * 1024 * 1024,
      2 * 1024 * 1024, 2 * 1024 * 1024);

  // allocate 8MB, expect to allocate on host
  void *p6 = a.AllocateRaw(1, 8 * 1024 * 1024);
  CHECK_NE(p6, nullptr);
  CheckStats(device_allocator, 4, 1 * 1024 * 1024,
      (1 + 2) * 1024 * 1024 + (400 + 400) * 1024, 2 * 1024 * 1024);
  CheckStats(host_allocator, 2, 10 * 1024 * 1024,
      10 * 1024 * 1024, 8 * 1024 * 1024);

  // free
  a.DeallocateRaw(p1);
  a.DeallocateRaw(p5);
  a.DeallocateRaw(p6);
}

TEST(GPUAdjustableAllocatorTest, GrowthMemoryLimitAndAllocateToDevice) {
  // Enable VMEM.
  setenv("TF_GPU_VMEM", "true", 1);
  // setenv("TF_CUDA_HOST_MEM_LIMIT_IN_MB", "16384", 1);

  PlatformGpuId platform_gpu_id(0);
  GPUOptions options;
  options.set_allow_growth(true);
  se::StreamExecutor* se =
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      se, platform_gpu_id, false /*use_unified_memory*/, {}, {});
  Allocator* device_allocator = new GPUBFCAllocator(
      sub_allocator, 1UL << 30, options, "GPU_0_bfc");

  SubAllocator* host_sub_allocator = new GpuHostAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      0, {}, {});
  Allocator* host_allocator = new BFCAllocator(host_sub_allocator, 1UL << 30,
      true /*allow_growth*/,
      "GPUHost_0_bfc");

  GPUVMemAllocator a(device_allocator, host_allocator, 0, se);
  CheckStats(&a, 0, 0, 0, 0);
  AllocatorStats stats;

  // allocate 700KB
  void *p1 = a.AllocateRaw(1, 700 * 1024);
  CheckStats(device_allocator, 1, 1 * 1024 * 1024,
      1 * 1024 * 1024, 1 * 1024 * 1024);

  // allocate 20MB
  void *p2 = a.AllocateRaw(1, 20 * 1024 * 1024);
  CheckStats(device_allocator, 2, 33 * 1024 * 1024,
      33 * 1024 * 1024, 32 * 1024 * 1024);

  // allocate 400KB, split chunk
  void *p3 = a.AllocateRaw(1, 400 * 1024);
  CheckStats(device_allocator, 3, 33 * 1024 * 1024 + 400 * 1024,
            33 * 1024 * 1024 + 400 * 1024, 32 * 1024 * 1024);

  // allocate 400KB
  void *p4 = a.AllocateRaw(1, 400 * 1024);
  CheckStats(device_allocator, 4, 33 * 1024 * 1024 + 400 * 1024 * 2,
            33 * 1024 * 1024 + 400 * 1024 * 2, 32 * 1024 * 1024);

  // free 400KB
  a.DeallocateRaw(p4);

  // free 400KB
  a.DeallocateRaw(p3);

  // free 700KB
  a.DeallocateRaw(p1);

  // limit to 32MB
  GPUAdjustableAllocator* adj = new GPUAdjustableAllocator();
  size_t new_limit =  adj->AdjustMemoryLimit(
      32UL * 1024 * 1024, dynamic_cast<GPUBFCAllocator *>(device_allocator));
  EXPECT_EQ(new_limit, 32UL * 1024 * 1024);
  // expect to release two regions

  // allocate 20MB, expect to allocate on host
  void *p5 = a.AllocateRaw(1, 20 * 1024 * 1024);
  CHECK_NE(p5, nullptr);
  CheckStats(device_allocator, 4, 32 * 1024 * 1024,
            33 * 1024 * 1024 + 400 * 1024 * 2, 32 * 1024 * 1024);
  CheckStats(host_allocator, 1, 32 * 1024 * 1024,
            32 * 1024 * 1024, 32 * 1024 * 1024);

  // allocate 32MB, expect to allocate on host
  void *p6 = a.AllocateRaw(1, 32 * 1024 * 1024);
  CHECK_NE(p6, nullptr);
  CheckStats(device_allocator, 4, 32 * 1024 * 1024,
            33 * 1024 * 1024 + 400 * 1024 * 2, 32 * 1024 * 1024);
  CheckStats(host_allocator, 2, 32 * 2 * 1024 * 1024,
            32 * 2 * 1024 * 1024, 32 * 1024 * 1024);

  // growth the memory limit of device to 2GB
  new_limit =  adj->AdjustMemoryLimit(
      2UL * 1024 * 1024 * 1024,
      dynamic_cast<GPUBFCAllocator *>(device_allocator));
  EXPECT_EQ(new_limit, 2UL * 1024 * 1024 * 1024);

  // allocate 20MB, expect to allocate on device
  void *p7 = a.AllocateRaw(1, 20 * 1024 * 1024);

  CHECK_NE(p7, nullptr);
  CheckStats(device_allocator, 5, 32 * 2 * 1024 * 1024,
      32 * 2 * 1024 * 1024, 32 * 1024 * 1024);
  CheckStats(host_allocator, 2, 32 * 2 * 1024 * 1024,
      32 * 2 * 1024 * 1024, 32 * 1024 * 1024);

  // free 20MB from device
  a.DeallocateRaw(p2);
  // free 32MB from host
  a.DeallocateRaw(p6);
  // free 20MB from device
  a.DeallocateRaw(p7);
  // free 20MB from device
  a.DeallocateRaw(p5);
}

TEST(GPUAdjustableAllocatorTest, RepeatedlyShrinkAndGrowthMemoryLimit) {
  // Enable VMEM.
  setenv("TF_GPU_VMEM", "true", 1);
  // setenv("TF_CUDA_HOST_MEM_LIMIT_IN_MB", "16384", 1);

  PlatformGpuId platform_gpu_id(0);
  GPUOptions options;
  options.set_allow_growth(true);
  se::StreamExecutor* se =
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      se, platform_gpu_id, false /*use_unified_memory*/, {}, {});
  Allocator* device_allocator = new GPUBFCAllocator(
      sub_allocator, 1UL << 32, options, "GPU_0_bfc");

  SubAllocator* host_sub_allocator = new GpuHostAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      0, {}, {});
  Allocator* host_allocator = new BFCAllocator(host_sub_allocator, 1UL << 30,
      true /*allow_growth*/,
      "GPUHost_0_bfc");

  GPUVMemAllocator a(device_allocator, host_allocator, 0, se);
  CheckStats(&a, 0, 0, 0, 0);
  absl::optional<AllocatorStats> stats;

  // allocate 700KB
  void *p1 = a.AllocateRaw(1, 700 * 1024);
  CheckStats(device_allocator, 1, 1 * 1024 * 1024,
      1 * 1024 * 1024, 1 * 1024 * 1024);

  // allocate 20MB
  void *p2 = a.AllocateRaw(1, 20 * 1024 * 1024);
  CheckStats(device_allocator, 2, 33 * 1024 * 1024,
      33 * 1024 * 1024, 32 * 1024 * 1024);

  // allocate 400KB, split chunk
  void *p3 = a.AllocateRaw(1, 400 * 1024);
  CheckStats(device_allocator, 3, 33 * 1024 * 1024 + 400 * 1024,
      33 * 1024 * 1024 + 400 * 1024, 32 * 1024 * 1024);

  // allocate 400KB
  void *p4 = a.AllocateRaw(1, 400 * 1024);
  CheckStats(device_allocator, 4, 33 * 1024 * 1024 + 400 * 1024 * 2,
      33 * 1024 * 1024 + 400 * 1024 * 2, 32 * 1024 * 1024);

  // free 20MB
  a.DeallocateRaw(p2);

  // free 700KB
  a.DeallocateRaw(p1);

  // limit to 32MB
  GPUAdjustableAllocator* adj = new GPUAdjustableAllocator();
  size_t new_limit =  adj->AdjustMemoryLimit(
      32UL * 1024 * 1024, dynamic_cast<GPUBFCAllocator *>(device_allocator));
  EXPECT_EQ(new_limit, 32UL * 1024 * 1024);
  CheckStats(device_allocator, 4, 400 * 1024 * 2,
      33 * 1024 * 1024 + 400 * 1024 * 2, 32 * 1024 * 1024);

  // allocate 32MB, expect to allocate on host
  void *p5 = a.AllocateRaw(1, 32 * 1024 * 1024);
  CHECK_NE(p5, nullptr);
  CheckStats(device_allocator, 4, 400 * 1024 * 2,
      33 * 1024 * 1024 + 400 * 1024 * 2, 32 * 1024 * 1024);
  CheckStats(host_allocator, 1, 32 * 1024 * 1024,
      32 * 1024 * 1024, 32 * 1024 * 1024);

  // allocate 12MB, expect to allocate on device
  void *p6 = a.AllocateRaw(1, 12 * 1024 * 1024);
  CHECK_NE(p6, nullptr);
  CheckStats(device_allocator, 5, 400 * 1024 * 2 + 12 * 1024 * 1024,
      33 * 1024 * 1024 + 400 * 1024 * 2, 32 * 1024 * 1024);
  CheckStats(host_allocator, 1, 32 * 1024 * 1024,
      32 * 1024 * 1024, 32 * 1024 * 1024);

  // allocate 2MB, expect to allocate on device
  void *p7 = a.AllocateRaw(1, 2 * 1024 * 1024);
  CHECK_NE(p7, nullptr);
  CheckStats(device_allocator, 6, 400 * 1024 * 2 + 14 * 1024 * 1024,
      33 * 1024 * 1024 + 400 * 1024 * 2, 32 * 1024 * 1024);
  CheckStats(host_allocator, 1, 32 * 1024 * 1024,
      32 * 1024 * 1024, 32 * 1024 * 1024);

  // allocate 32MB, expect to allocate on host
  void *p8 = a.AllocateRaw(1, 32 * 1024 * 1024);
  CHECK_NE(p8, nullptr);
  CheckStats(device_allocator, 6, 400 * 1024 * 2 + 14 * 1024 * 1024,
      33 * 1024 * 1024 + 400 * 1024 * 2, 32 * 1024 * 1024);
  CheckStats(host_allocator, 2, 64 * 1024 * 1024,
      64 * 1024 * 1024, 32 * 1024 * 1024);

  // growth the memory limit of device to 256MB
  new_limit =  adj->AdjustMemoryLimit(
      256UL * 1024 * 1024, dynamic_cast<GPUBFCAllocator *>(device_allocator));
  EXPECT_EQ(new_limit, 256UL * 1024 * 1024);
  CheckStats(device_allocator, 6, 400 * 1024 * 2 + 14 * 1024 * 1024,
            33 * 1024 * 1024 + 400 * 1024 * 2, 32 * 1024 * 1024);

  // allocate 128MB, expect to allocate on device
  void *p9 = a.AllocateRaw(1, 128 * 1024 * 1024);
  CHECK_NE(p9, nullptr);
  CheckStats(device_allocator, 7, 400 * 1024 * 2 + 142 * 1024 * 1024,
      400 * 1024 * 2 + 142 * 1024 * 1024, 128 * 1024 * 1024);
  CheckStats(host_allocator, 2, 64 * 1024 * 1024,
      64 * 1024 * 1024, 32 * 1024 * 1024);

  // allocate 2MB, expect to allocate on device
  void *p10 = a.AllocateRaw(1, 2 * 1024 * 1024);
  CHECK_NE(p10, nullptr);
  CheckStats(device_allocator, 8, 400 * 1024 * 2 + 144 * 1024 * 1024,
      400 * 1024 * 2 + 144 * 1024 * 1024, 128 * 1024 * 1024);
  CheckStats(host_allocator, 2, 64 * 1024 * 1024,
      64 * 1024 * 1024, 32 * 1024 * 1024);

  // allocate 512MB, expect to allocate on host
  void *p11 = a.AllocateRaw(1, 512 * 1024 * 1024);
  CHECK_NE(p11, nullptr);
  CheckStats(device_allocator, 8, 400 * 1024 * 2 + 144 * 1024 * 1024,
      400 * 1024 * 2 + 144 * 1024 * 1024, 128 * 1024 * 1024);
  CheckStats(host_allocator, 3, (512 + 64) * 1024 * 1024,
      (512 + 64) * 1024 * 1024, 512 * 1024 * 1024);

  // free 2MB
  a.DeallocateRaw(p10);

  // free 2MB
  a.DeallocateRaw(p7);

  // free 12MB
  a.DeallocateRaw(p6);

  // free 400KB
  a.DeallocateRaw(p3);

  // free 400KB, merge
  a.DeallocateRaw(p4);
  CheckStats(device_allocator, 8, 128 * 1024 * 1024,
      400 * 1024 * 2 + 144 * 1024 * 1024, 128 * 1024 * 1024);
  CheckStats(host_allocator, 3, (512 + 64) * 1024 * 1024,
      (512 + 64) * 1024 * 1024, 512 * 1024 * 1024);

  // allocate 2MB, expect to allocate on device
  void *p12 = a.AllocateRaw(1, 2 * 1024 * 1024);
  CHECK_NE(p12, nullptr);
  CheckStats(device_allocator, 9, (128 + 2) * 1024 * 1024,
      400 * 1024 * 2 + 144 * 1024 * 1024, 128 * 1024 * 1024);
  CheckStats(host_allocator, 3, (512 + 64) * 1024 * 1024,
      (512 + 64) * 1024 * 1024, 512 * 1024 * 1024);

  // // limit to 128MB, expect a failure
  // succ = dynamic_cast<GPUAdjustableAllocator *>
  //     (device_allocator)->AdjustMemoryLimit(
  //         128UL * 1024 * 1024);
  // EXPECT_EQ(succ, false);
  // CheckStats(device_allocator, 9, (128 + 2) * 1024 * 1024,
  //     400 * 1024 * 2 + 144 * 1024 * 1024, 128 * 1024 * 1024);

  // limit to 160MB
  new_limit =  adj->AdjustMemoryLimit(
      160UL * 1024 * 1024, dynamic_cast<GPUBFCAllocator *>(device_allocator));
  EXPECT_EQ(new_limit, 160UL * 1024 * 1024);
  CheckStats(device_allocator, 9, (128 + 2) * 1024 * 1024,
      400 * 1024 * 2 + 144 * 1024 * 1024, 128 * 1024 * 1024);

  // allocate 32MB, expect to allocate on host
  void *p13 = a.AllocateRaw(1, 32 * 1024 * 1024);
  CHECK_NE(p13, nullptr);
  CheckStats(device_allocator, 9, (128 + 2) * 1024 * 1024,
      400 * 1024 * 2 + 144 * 1024 * 1024, 128 * 1024 * 1024);
  CheckStats(host_allocator, 4, (512 + 64 + 32) * 1024 * 1024,
      (512 + 64 + 32) * 1024 * 1024, 512 * 1024 * 1024);

  // growth the memory limit of device to 2GB
  new_limit =  adj->AdjustMemoryLimit(
      2UL * 1024 * 1024 * 1024,
      dynamic_cast<GPUBFCAllocator *>(device_allocator));
  EXPECT_EQ(new_limit, 2UL * 1024 * 1024 * 1024);
  CheckStats(device_allocator, 9, (128 + 2) * 1024 * 1024,
      400 * 1024 * 2 + 144 * 1024 * 1024, 128 * 1024 * 1024);

  // allocate 2MB, expect to allocate on device
  void *p14 = a.AllocateRaw(1, 2 * 1024 * 1024);
  CHECK_NE(p14, nullptr);
  CheckStats(device_allocator, 10, (128 + 2 + 2) * 1024 * 1024,
      400 * 1024 * 2 + 144 * 1024 * 1024, 128 * 1024 * 1024);
  CheckStats(host_allocator, 4, (512 + 64 + 32) * 1024 * 1024,
      (512 + 64 + 32) * 1024 * 1024, 512 * 1024 * 1024);

  // allocate 16MB, expect to allocate on device
  void *p15 = a.AllocateRaw(1, 16 * 1024 * 1024);
  CHECK_NE(p15, nullptr);
  CheckStats(device_allocator, 11, (32 + 128) * 1024 * 1024,
      (32 + 128) * 1024 * 1024, 128 * 1024 * 1024);
  CheckStats(host_allocator, 4, (512 + 64 + 32) * 1024 * 1024,
      (512 + 64 + 32) * 1024 * 1024, 512 * 1024 * 1024);

  // free 16MB
  a.DeallocateRaw(p15);

  // limit to 128MB, expect a large limit
  new_limit =  adj->AdjustMemoryLimit(
      128UL * 1024 * 1024,
      dynamic_cast<GPUBFCAllocator *>(device_allocator));
  EXPECT_EQ(new_limit, 160UL * 1024 * 1024);
  CheckStats(device_allocator, 11, (128 + 2 + 2) * 1024 * 1024,
      (32 + 128) * 1024 * 1024, 128 * 1024 * 1024);

  // free all
  a.DeallocateRaw(p14);
  a.DeallocateRaw(p13);
  a.DeallocateRaw(p8);
  a.DeallocateRaw(p9);
  a.DeallocateRaw(p5);
  a.DeallocateRaw(p11);
  a.DeallocateRaw(p12);
  stats = a.GetStats();
  // LOG(INFO) << "vmem alloc stats: " << std::endl << stats.DebugString();
  // device_allocator->GetStats(&stats);
  // LOG(INFO) << "GPU alloc stats: " << std::endl << stats.DebugString();
  // host_allocator->GetStats(&stats);
  // LOG(INFO) << "CPU alloc stats: " << std::endl << stats.DebugString();

  // Unset flag TF_GPU_VMEM.
  unsetenv("TF_GPU_VMEM");
}

}  // namespace

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
