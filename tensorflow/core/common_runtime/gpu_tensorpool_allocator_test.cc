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
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#include <thread>
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/gpu_mem_allocator.h"
#include "tensorflow/core/common_runtime/gpu_memory_planner.h"
#include "tensorflow/core/common_runtime/gpu_tensorpool_allocator.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

TEST(GPUTensorPoolAllocatorTest, SmallAllocationWithAlignment32B) {
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  void* p = allocator.AllocateRaw(32, 100);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(GPUTensorPoolAllocatorTest, SmallAllocationWithAlignment8B) {
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  void* p = allocator.AllocateRaw(8, 100);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(GPUTensorPoolAllocatorTest, SmallAllocationWithAlignment16B) {
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  void* p = allocator.AllocateRaw(16, 100);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(GPUTensorPoolAllocatorTest, AlignedSmallAllocationWithAlignment16B) {
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  void* p = allocator.AllocateRaw(16, 128);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(GPUTensorPoolAllocatorTest, SmallAllocationWithoutAlignment) {
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  void* p = allocator.AllocateRaw(64, 100);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(GPUTensorPoolAllocatorTest, BigAllocation100KBWithoutAlignment) {
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  void* p = allocator.AllocateRaw(64, 100000);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(GPUTensorPoolAllocatorTest, BigAllocation100KBWithAlignment128B) {
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  void* p = allocator.AllocateRaw(64, 100000);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(GPUTensorPoolAllocatorTest, BigAllocation1MBWithoutAlignment32B) {
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  void* p = allocator.AllocateRaw(64, 1024*1024);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(GPUTensorPoolAllocatorTest, BigAllocation128KBWithoutAlignment16B) {
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  void* p = allocator.AllocateRaw(64, 128 * 1024);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(TensorPoolAllcatorTest, MixedAllocationLoops) {
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  std::vector<int> alignments = {8, 16, 32, 64, 128};
  std::vector<int> sizes = {100, 512, 2048, 128*1024, 100 * 1024};
  std::vector<void*> vec;
  for (int i = 0; i < 100; ++i) {
    for (auto alignment : alignments) {
      for (auto size : sizes) {
        void* p = allocator.AllocateRaw(alignment, size);
        EXPECT_TRUE(p != nullptr);
        vec.emplace_back(p);
      }
    }
  }
  for (auto p : vec) {
    allocator.DeallocateRaw(p);
  }
}

TEST(GPUTensorPoolAllocatorTest, MultipleThreadMixedAllocationLoops) {
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  auto func = [&allocator] {
    std::vector<int> alignments = {8, 16, 32, 64, 128};
    std::vector<int> sizes = {100, 512, 2048, 128*1024, 100 * 1024};
    std::vector<void*> vec;
    for (int i = 0; i < 100; ++i) {
      for (auto alignment : alignments) {
        for (auto size : sizes) {
          void* p = allocator.AllocateRaw(alignment, size);
          EXPECT_TRUE(p != nullptr);
          vec.emplace_back(p);
        }
      }
    }
    for (auto p : vec) {
      allocator.DeallocateRaw(p);
    }
  };

  std::vector<std::thread*> ths;
  for (int i = 0; i < 3; ++i) {
    auto th = new std::thread(func);
    ths.emplace_back(th);
  }
  for (auto th : ths) {
    th->join();
  }
}

TEST(GPUTensorPoolAllocatorTest, MemoryPlannerBasic) {
  GPUMemoryPlannerFactory::GetMemoryPlanner()->Reset();
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  for (int i = 0; i < 2000; ++i) {
    GPUScopedMemoryCollector c;
    std::vector<int> alignments = {8, 16, 64, 128};
    std::vector<int> sizes = {100, 128, 1024, 128*1024};
    std::vector<void*> vec;
    for (int i = 0; i < 2; ++i) {
      for (auto alignment : alignments) {
        for (auto size : sizes) {
          void* p = allocator.AllocateRaw(alignment, size);
          EXPECT_TRUE(p != nullptr);
          vec.emplace_back(p);
        }
      }
    }
    for (auto p : vec) {
      allocator.DeallocateRaw(p);
    }
  }
  std::vector<int> alignments = {8, 16, 32, 64, 128};
  std::vector<int> sizes = {100, 512, 2048, 128*1024, 100 * 1024};
  std::vector<void*> vec;
  for (int i = 0; i < 100; ++i) {
    for (auto alignment : alignments) {
      for (auto size : sizes) {
        void* p = allocator.AllocateRaw(alignment, size);
        EXPECT_TRUE(p != nullptr);
        vec.emplace_back(p);
      }
    }
  }
  for (auto p : vec) {
    allocator.DeallocateRaw(p);
  }
  sleep(1);
}

TEST(GPUTensorPoolAllocatorTest, MemoryPlannerSingletonTest) {
  GPUMemoryPlannerFactory::GetMemoryPlanner()->Reset();
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  for (int i = 0; i < 2000; ++i) {
    GPUScopedMemoryCollector c;
    std::vector<int> alignments = {8, 16, 64, 128};
    std::vector<int> sizes = {100, 128, 1024, 128*1024};
    std::vector<void*> vec;
    for (int i = 0; i < 2; ++i) {
      for (auto alignment : alignments) {
        for (auto size : sizes) {
          void* p = allocator.AllocateRaw(alignment, size);
          EXPECT_TRUE(p != nullptr);
          vec.emplace_back(p);
        }
      }
    }
    for (auto p : vec) {
      allocator.DeallocateRaw(p);
    }
  }
  sleep(1);
}

TEST(GPUTensorPoolAllocatorTest, MemoryPlannerMemoryConsumption) {
  GPUMemoryPlannerFactory::GetMemoryPlanner()->Reset();
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  for (int i = 0; i < 2000; ++i) {
    GPUScopedMemoryCollector c;
    std::vector<int> alignments = {64};
    std::vector<int> sizes = {128*1024, 64*1024};
    std::vector<void*> vec;
    for (int i = 0; i < 2; ++i) {
      for (auto alignment : alignments) {
        for (auto size : sizes) {
          void* p = allocator.AllocateRaw(alignment, size);
          EXPECT_TRUE(p != nullptr);
          vec.emplace_back(p);
        }
      }
    }
    for (auto p : vec) {
      allocator.DeallocateRaw(p);
    }
  }
  sleep(1);
}

TEST(GPUTensorPoolAllocatorTest, HugeMemoryAllocation) {
  GPUMemoryPlannerFactory::GetMemoryPlanner()->Reset();
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  for (int i = 0; i < 2000; ++i) {
    GPUScopedMemoryCollector c;
    std::vector<int> alignments = {64};
    std::vector<int> sizes = {16*1024*1024, 8*1024*1024};
    std::vector<void*> vec;
    for (int i = 0; i < 2; ++i) {
      for (auto alignment : alignments) {
        for (auto size : sizes) {
          void* p = allocator.AllocateRaw(alignment, size);
          EXPECT_TRUE(p != nullptr);
          vec.emplace_back(p);
        }
      }
    }
    for (auto p : vec) {
      allocator.DeallocateRaw(p);
    }
  }
  sleep(1);
}

static void BM_GPUTensorPoolAllocator_SmallAllocation(int iters) {
  GPUMemoryPlannerFactory::GetMemoryPlanner()->Reset();
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  std::vector<int> sizes = {64, 128, 144, 256, 400, 512};

  int size_index = 0;
  while (--iters > 0) {
    GPUScopedMemoryCollector c;
    int bytes = sizes[size_index++ % sizes.size()];
    void* p = allocator.AllocateRaw(1, bytes);
    allocator.DeallocateRaw(p);
  }
}

BENCHMARK(BM_GPUTensorPoolAllocator_SmallAllocation);

static void BM_GPUTensorPoolAllocator_BigAllocation(int iters) {
  GPUMemoryPlannerFactory::GetMemoryPlanner()->Reset();
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUTensorPoolAllocator allocator(sub_allocator, "GPU_0_tensorpool", 1 << 30);
  std::vector<int> sizes = {64 * 1024, 128 * 1024,
    144 * 1024, 256 * 2048, 400 * 4096, 512 * 4096};

  int size_index = 0;
  while (--iters > 0) {
    GPUScopedMemoryCollector c;
    int bytes = sizes[size_index++ % sizes.size()];
    void* p = allocator.AllocateRaw(1, bytes);
    allocator.DeallocateRaw(p);
  }
}

BENCHMARK(BM_GPUTensorPoolAllocator_BigAllocation);

static void BM_BFCAllocatorGPU_SmallAllocation(int iters) {
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUOptions options;
  options.set_allow_growth(true);
  GPUBFCAllocator allocator(sub_allocator, 1LL << 30, options, "GPU_0_bfc");
  std::vector<int> sizes = {64, 128, 144, 256, 400, 512};

  int size_index = 0;
  while (--iters > 0) {
    int bytes = sizes[size_index++ % sizes.size()];
    void* p = allocator.AllocateRaw(1, bytes);
    allocator.DeallocateRaw(p);
  }
}

BENCHMARK(BM_BFCAllocatorGPU_SmallAllocation);

static void BM_BFCAllocatorGPU_BigAllocation(int iters) {
  PlatformGpuId platform_gpu_id(0);
  GPUMemAllocator* sub_allocator = new GPUMemAllocator(
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
      platform_gpu_id, false /*use_unified_memory*/, {}, {});
  GPUOptions options;
  options.set_allow_growth(true);
  GPUBFCAllocator allocator(sub_allocator, 1LL << 30, options, "GPU_0_bfc");
  std::vector<int> sizes = {64 * 1024, 128 * 1024,
    144 * 1024, 256 * 2048, 400 * 4096, 512 * 4096};

  int size_index = 0;
  while (--iters > 0) {
    int bytes = sizes[size_index++ % sizes.size()];
    void* p = allocator.AllocateRaw(1, bytes);
    allocator.DeallocateRaw(p);
  }
}

BENCHMARK(BM_BFCAllocatorGPU_BigAllocation);

}
}  // namespace tensorflow

#include <unistd.h>
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
