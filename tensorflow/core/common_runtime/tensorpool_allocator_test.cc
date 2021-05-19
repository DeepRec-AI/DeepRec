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
#include <thread>
#include "tensorflow/core/common_runtime/memory_planner.h"
#include "tensorflow/core/common_runtime/tensorpool_allocator.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include <unistd.h>

namespace tensorflow {
namespace {

TEST(TensorPoolAllocatorTest, SmallAllocationWithAlignment32B) {
  TensorPoolAllocator allocator;
  void* p = allocator.AllocateRaw(32, 100);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(TensorPoolAllocatorTest, SmallAllocationWithAlignment8B) {
  TensorPoolAllocator allocator;
  void* p = allocator.AllocateRaw(8, 100);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(TensorPoolAllocatorTest, SmallAllocationWithAlignment16B) {
  TensorPoolAllocator allocator;
  void* p = allocator.AllocateRaw(16, 100);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(TensorPoolAllocatorTest, AlignedSmallAllocationWithAlignment16B) {
  TensorPoolAllocator allocator;
  void* p = allocator.AllocateRaw(16, 128);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(TensorPoolAllocatorTest, SmallAllocationWithoutAlignment) {
  TensorPoolAllocator allocator;
  void* p = allocator.AllocateRaw(64, 100);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(TensorPoolAllocatorTest, SmallAllocation100BWithHeaderCheck) {
  typedef LightHeader LiteHeader;
  TensorPoolAllocator allocator;
  void* p = allocator.AllocateRaw(16, 100);
  LiteHeader* header = (LiteHeader*)((char*)p - sizeof(LiteHeader));
  
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(TensorPoolAllocatorTest, BigAllocation100KBWithoutAlignment) {
  TensorPoolAllocator allocator;
  void* p = allocator.AllocateRaw(64, 100000);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(TensorPoolAllocatorTest, BigAllocation100KBWithAlignment128B) {
  TensorPoolAllocator allocator;
  void* p = allocator.AllocateRaw(64, 100000);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(TensorPoolAllocatorTest, BigAllocation1MBWithoutAlignment32B) {
  TensorPoolAllocator allocator;
  void* p = allocator.AllocateRaw(64, 1024*1024);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(TensorPoolAllocatorTest, BigAllocation128KBWithoutAlignment16B) {
  TensorPoolAllocator allocator;
  void* p = allocator.AllocateRaw(64, 128 * 1024);
  EXPECT_TRUE(p != nullptr);
  allocator.DeallocateRaw(p);
}

TEST(TensorPoolAllocatorTest, BigAllocation100KBWithHeaderCheck) { 
  TensorPoolAllocator allocator;
  void* p = allocator.AllocateRaw(16, 1000000);
  Header* header = (Header*)((char*)p - sizeof(Header));
  auto header_size = header->total_size - header->user_size;
  EXPECT_EQ((char*)header->raw_ptr + header_size,
            (char*)header->user_ptr);
  EXPECT_EQ(header->user_ptr, p);
  allocator.DeallocateRaw(p);
}

TEST(TensorPoolAllcatorTest, MixedAllocationLoops) {
  TensorPoolAllocator allocator;
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

TEST(TensorPoolAllocatorTest, MultipleThreadMixedAllocationLoops) {
  TensorPoolAllocator allocator;
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

TEST(TensorPoolAllocatorTest, MemoryPlannerBasic) {
  thread::ThreadPool* threads = new thread::ThreadPool(Env::Default(), "test", 2);
  MemoryPlannerFactory::GetMemoryPlanner()->SetThreadPool(threads);
  TensorPoolAllocator allocator;
  for (int i = 0; i < 2000; ++i) {
    ScopedMemoryCollector c;
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

TEST(TensorPoolAllocatorTest, MemoryPlannerSingletonTest) {
  thread::ThreadPool* threads = new thread::ThreadPool(Env::Default(), "test", 2);
  MemoryPlannerFactory::GetMemoryPlanner()->Reset();
  MemoryPlannerFactory::GetMemoryPlanner()->SetThreadPool(threads);
  TensorPoolAllocator allocator;
  for (int i = 0; i < 2000; ++i) {
    ScopedMemoryCollector c;
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

TEST(TensorPoolAllocatorTest, MemoryPlannerMemoryConsumption) {
  thread::ThreadPool* threads = new thread::ThreadPool(Env::Default(), "test", 2);
  MemoryPlannerFactory::GetMemoryPlanner()->Reset();
  MemoryPlannerFactory::GetMemoryPlanner()->SetThreadPool(threads);
  TensorPoolAllocator allocator;
  for (int i = 0; i < 2000; ++i) {
    ScopedMemoryCollector c;
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

TEST(TensorPoolAllocatorTest, HugeMemoryAllocation) {
  thread::ThreadPool* threads = new thread::ThreadPool(Env::Default(), "test", 2);
  MemoryPlannerFactory::GetMemoryPlanner()->Reset();
  MemoryPlannerFactory::GetMemoryPlanner()->SetThreadPool(threads);
  TensorPoolAllocator allocator;
  for (int i = 0; i < 2000; ++i) {
    ScopedMemoryCollector c;
    std::vector<int> alignments = {64};
    std::vector<int> sizes = {512*1024*1024, 64*1024*1024};
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

}
}  // namespace tensorflow
