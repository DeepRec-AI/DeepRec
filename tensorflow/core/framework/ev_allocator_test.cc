#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool_interface.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

struct ProcMemory {
  long size;      // total program size
  long resident;  // resident set size
  long share;     // shared pages
  long trs;       // text (code)
  long lrs;       // library
  long drs;       // data/stack
  long dt;        // dirty pages

  ProcMemory() : size(0), resident(0), share(0),
                 trs(0), lrs(0), drs(0), dt(0) {}
};

ProcMemory getProcMemory() {
  ProcMemory m;
  FILE* fp = fopen("/proc/self/statm", "r");
  if (fp == NULL) {
    LOG(ERROR) << "Fail to open /proc/self/statm.";
    return m;
  }

  if (fscanf(fp, "%ld %ld %ld %ld %ld %ld %ld",
             &m.size, &m.resident, &m.share,
             &m.trs, &m.lrs, &m.drs, &m.dt) != 7) {
    fclose(fp);
    LOG(ERROR) << "Fail to fscanf /proc/self/statm.";
    return m;
  }
  fclose(fp);

  return m;
}

double getSize() {
  ProcMemory m = getProcMemory();
  return m.size;
}

double getResident() {
  ProcMemory m = getProcMemory();
  return m.resident;
}

TEST(EVAllocator, TestSmallAllocation) {
  auto allocator = ev_allocator();
  constexpr int size = 10;
  void** ptrs = new void*[size];
  for (int i = 0; i < size; ++i) {
    ptrs[i] = allocator->AllocateRaw(4, 4);
  }

  for (int i = 0; i < size; ++i) {
    allocator->DeallocateRaw(ptrs[i]);
  }
}

TEST(EVAllocator, TestBatchSmallAllocation) {
  auto allocator = ev_allocator();
  constexpr int size = 100000000;
  void** ptrs = new void*[size];

  size_t allocated_size = allocator->BatchAllocateRaw(
      size, 4, 4, ptrs);

  for (int i = 0; i < allocated_size; ++i) {
    allocator->DeallocateRaw(ptrs[i]);
  }
}

TEST(EVAllocator, TestUnAlignedAllocation) {
  auto allocator = ev_allocator();
  constexpr int size = 1000;
  void** ptrs = new void*[size];
  memset(ptrs, 0, size * sizeof(void*));

  for (int i = 0; i < size; ++i) {
    ptrs[i] = allocator->AllocateRaw(4, 8);
  }

  for (int i = 0; i < size; ++i) {
    allocator->DeallocateRaw(ptrs[i]);
  }
}

TEST(EVAllocator, TestBatchUnAlignedAllocation) {
  auto allocator = ev_allocator();
  constexpr int size = 1000;
  void** ptrs = new void*[size];
  memset(ptrs, 0, size * sizeof(void*));

  size_t allocated_size = allocator->BatchAllocateRaw(
      size, 4, 8, ptrs);

  for (int i = 0; i < allocated_size; ++i) {
    allocator->DeallocateRaw(ptrs[i]);
  }
}

TEST(EVAllocator, TestBigAllocation) {
  auto allocator = ev_allocator();
  constexpr int size = 1000;
  void** ptrs = new void*[size];

  for (int i = 0; i < size; ++i) {
    ptrs[i] = allocator->AllocateRaw(4, 4096);
  }

  for (int i = 0; i < size; ++i) {
    allocator->DeallocateRaw(ptrs[i]);
  }
}

TEST(EVAllocator, TestBatchBigAllocation) {
  auto allocator = ev_allocator();
  constexpr int size = 1000;
  void** ptrs = new void*[size];

  size_t allocated_size = allocator->BatchAllocateRaw(
      size, 4, 4096, ptrs);

  for (int i = 0; i < size; ++i) {
    allocator->DeallocateRaw(ptrs[i]);
  }
}

TEST(EVAllocator, TestAllocateAndDeallocate) {
  auto allocator = ev_allocator();
  constexpr int size = 1000;

  for (int i = 0; i < size; ++i) {
    auto ptr = allocator->AllocateRaw(4, 128);
    allocator->DeallocateRaw(ptr);
  }
}

TEST(EVAllocator, TestBatchAllocateAndDeallocate) {
  auto allocator = ev_allocator();
  constexpr int size = 1000;

  void** ptrs = new void*[size];
  auto alloc_num = allocator->BatchAllocateRaw(size, 4, 128, ptrs);
  for (int i = 0; i < alloc_num; ++i) {
    allocator->DeallocateRaw(ptrs[i]);
  }
}

TEST(EVAllocator, TestMultiThreadAllocate40B) {
  auto allocator = ev_allocator();
  
  double t0 = getResident()*getpagesize()/1024.0/1024.0;

  constexpr int loop_size = 100000;
  constexpr int allocate_size = 40;
  auto func = [allocator, loop_size, allocate_size]() {
    constexpr int alignment = 4;
    for (int i = 0; i < loop_size; ++i) {
      void* tensor_val = allocator->AllocateRaw(alignment, allocate_size);
      memset(tensor_val, 0, allocate_size);
    }
  };

  constexpr int THREAD_NUM = 40;
  std::vector<std::unique_ptr<Thread>> threads(THREAD_NUM);

  size_t start = Env::Default()->NowMicros();
  for (int i = 0 ; i < THREAD_NUM; i++) {
    threads[i].reset(Env::Default()->StartThread(ThreadOptions(), "", func));
  }

  for (auto &t : threads) {
    t.reset(nullptr);
  }
  size_t stop = Env::Default()->NowMicros();
  
  size_t diff = stop - start;
  LOG(INFO) << "timespan: " << diff / 1000 << "ms";

  double t1 = getResident() * getpagesize() / 1024.0 / 1024.0;
  size_t real = t1 - t0;
  size_t compare =
    (size_t)THREAD_NUM * allocate_size * loop_size / 1024.0 / 1024.0;
  LOG(INFO) << "memory real: " << real << "MB";
  LOG(INFO) << "memory theoretic: " << compare << "MB";
  LOG(INFO) << "fragment ratio:"
            << (double) (real - compare) * 100.0 / compare << "%";
}

TEST(EVAllocator, TestMultiThreadBatchAllocate40B) {
  auto allocator = ev_allocator();
  double t0 = getResident() * getpagesize() / 1024.0 / 1024.0;

  constexpr int loop_size = 1000;
  constexpr int batch_size = 1000;
  constexpr int allocate_size = 40;
  auto func = [allocator, loop_size, allocate_size, batch_size]() {
    constexpr int alignment = 4;
    void** ptrs = new void*[batch_size];
    size_t total = 0;
    for (int i = 0; i < loop_size; ++i) {
      auto alloc_num = allocator->BatchAllocateRaw(
          batch_size, alignment, allocate_size, ptrs);
      total += alloc_num;
      for (int j = 0; j < alloc_num; ++j) {
        memset(ptrs[j], 0, allocate_size);
      }
    }
  };

  constexpr int THREAD_NUM = 40;
  std::vector<std::unique_ptr<Thread>> threads(THREAD_NUM);

  size_t start = Env::Default()->NowMicros();
  for (int i = 0 ; i < THREAD_NUM; i++) {
    threads[i].reset(Env::Default()->StartThread(ThreadOptions(), "", func));
  }

  for (auto &t : threads) {
    t.reset(nullptr);
  }
  size_t stop = Env::Default()->NowMicros();
  size_t diff = stop - start;
  LOG(INFO) << "timespan: " << diff / 1000 << "ms";

  double t1 = getResident() * getpagesize() / 1024.0 / 1024.0;
  size_t real = t1 - t0;
  size_t compare =
    (size_t)THREAD_NUM * allocate_size * loop_size * batch_size / 1024.0 / 1024.0;
  LOG(INFO) << "memory real: " << real << "MB";
  LOG(INFO) << "memory theoretic: " << compare << "MB";
  LOG(INFO) << "fragment ratio:"
            << (double) (real - compare) * 100.0 / compare << "%";
}

TEST(EVAllocator, TestMultiThreadAllocate4B) {
  auto allocator = ev_allocator();
  
  double t0 = getResident()*getpagesize()/1024.0/1024.0;

  constexpr int loop_size = 10000;
  constexpr int allocate_size = 4;
  auto func = [allocator, loop_size, allocate_size]() {
    constexpr int alignment = 4;
    for (int i = 0; i < loop_size; ++i) {
      void* tensor_val = allocator->AllocateRaw(alignment, allocate_size);
      memset(tensor_val, 0, allocate_size);
    }
  };

  size_t start = Env::Default()->NowMicros();
  constexpr int THREAD_NUM = 40;
  std::vector<std::unique_ptr<Thread>> threads(THREAD_NUM);
  for (int i = 0 ; i < THREAD_NUM; i++) {
    threads[i].reset(Env::Default()->StartThread(ThreadOptions(), "", func));
  }

  for (auto &t : threads) {
    t.reset(nullptr);
  }
  
  size_t stop = Env::Default()->NowMicros();
  size_t diff = stop - start;
  LOG(INFO) << "allocation time consumption: " << diff / 1000 << "ms";

  double t1 = getResident() * getpagesize() / 1024.0 / 1024.0;
  size_t real = t1 - t0;
  size_t compare =
    (size_t)THREAD_NUM * allocate_size * loop_size / 1024.0 / 1024.0;
  LOG(INFO) << "memory real: " << real << "MB";
  LOG(INFO) << "memory theoretic: " << compare << "MB";
  LOG(INFO) << "fragment ratio:"
            << (double) (real - compare) * 100.0 / compare << "%";
}

TEST(EVAllocator, TestMultiThreadAllocate80B) {
  auto allocator = ev_allocator();
  
  double t0 = getResident()*getpagesize()/1024.0/1024.0;

  constexpr int loop_size = 10000;
  constexpr int allocate_size = 80;
  auto func = [allocator, loop_size, allocate_size]() {
    constexpr int alignment = 8;
    for (int i = 0; i < loop_size; ++i) {
      void* tensor_val = allocator->AllocateRaw(alignment, allocate_size);
      memset(tensor_val, 0, allocate_size);
    }
  };

  size_t start = Env::Default()->NowMicros();
  constexpr int THREAD_NUM = 40;
  std::vector<std::unique_ptr<Thread>> threads(THREAD_NUM);
  for (int i = 0 ; i < THREAD_NUM; i++) {
    threads[i].reset(Env::Default()->StartThread(ThreadOptions(), "", func));
  }

  for (auto &t : threads) {
    t.reset(nullptr);
  }
  
  size_t stop = Env::Default()->NowMicros();
  size_t diff = stop - start;
  LOG(INFO) << "allocation time consumption: " << diff / 1000 << "ms";

  double t1 = getResident() * getpagesize() / 1024.0 / 1024.0;
  size_t real = t1 - t0;
  size_t compare =
    (size_t)THREAD_NUM * allocate_size * loop_size / 1024.0 / 1024.0;
  LOG(INFO) << "memory real: " << real << "MB";
  LOG(INFO) << "memory theoretic: " << compare << "MB";
  LOG(INFO) << "fragment ratio:"
            << (double) (real - compare) * 100.0 / compare << "%";
}

TEST(EVAllocator, TestMultiThreadAllocate256B) {
  auto allocator = ev_allocator();
  
  double t0 = getResident()*getpagesize()/1024.0/1024.0;

  constexpr int loop_size = 10000;
  constexpr int allocate_size = 256;
  auto func = [allocator, loop_size, allocate_size]() {
    constexpr int alignment = 8;
    for (int i = 0; i < loop_size; ++i) {
      void* tensor_val = allocator->AllocateRaw(alignment, allocate_size);
      memset(tensor_val, 0, allocate_size);
    }
  };

  size_t start = Env::Default()->NowMicros();
  constexpr int THREAD_NUM = 40;
  std::vector<std::unique_ptr<Thread>> threads(THREAD_NUM);
  for (int i = 0 ; i < THREAD_NUM; i++) {
    threads[i].reset(Env::Default()->StartThread(ThreadOptions(), "", func));
  }

  for (auto &t : threads) {
    t.reset(nullptr);
  }
  
  size_t stop = Env::Default()->NowMicros();
  size_t diff = stop - start;
  LOG(INFO) << "allocation time consumption: " << diff / 1000 << "ms";

  double t1 = getResident() * getpagesize() / 1024.0 / 1024.0;
  size_t real = t1 - t0;
  size_t compare =
    (size_t)THREAD_NUM * allocate_size * loop_size / 1024.0 / 1024.0;
  LOG(INFO) << "memory real: " << real << "MB";
  LOG(INFO) << "memory theoretic: " << compare << "MB";
  LOG(INFO) << "fragment ratio:"
            << (double) (real - compare) * 100.0 / compare << "%";
}

TEST(EVAllocator, TestMultiThreadDeallocateCrossThread) {
  auto allocator = ev_allocator();

  constexpr int loop_size = 1000;
  constexpr int allocate_size = 40;

  void** ptrs = new void*[loop_size];
  auto alloc_func = [allocator, loop_size, allocate_size, ptrs]() {
    constexpr int alignment = 4;
    for (int i = 0; i < loop_size; ++i) {
      void* tensor_val = allocator->AllocateRaw(alignment, allocate_size);
      memset(tensor_val, 0, allocate_size);
      ptrs[i] = tensor_val;
    }
  };

  // Allocation Thread
  Thread* alloc_th =
    Env::Default()->StartThread(ThreadOptions(), "", alloc_func);
  delete alloc_th;

  auto dealloc_func = [allocator, loop_size, ptrs]() {
    for (int i = 0; i < loop_size; ++i) {
      allocator->DeallocateRaw(ptrs[i]);
    }
  };

  // Deallocation Thread
  Thread* dealloc_th =
    Env::Default()->StartThread(ThreadOptions(), "", dealloc_func);
  delete dealloc_th;
}

TEST(EVAllocator, TestMultiThreadAllocateDeallocateLongRun) {
  auto allocator = ev_allocator();

  auto func = [allocator]() {
    constexpr int array_size = 1000;
    constexpr int loop_size = 10000;
    constexpr int allocate_size = 40;
    constexpr int alignment = 8;
    void** ptr = new void*[array_size];
    memset(ptr, 0, array_size * sizeof(void*));
    for (int64 i = 0; i < loop_size; ++i) {
      void* tensor_val = allocator->AllocateRaw(alignment, allocate_size);
      memset(tensor_val, 0, allocate_size);
      ptr[i % array_size] = tensor_val;

      if (i % array_size == 0 && i > 0) {
        for (int64 i = 0; i < array_size; ++i) {
          allocator->DeallocateRaw(ptr[i]);
        }
      }
    }
  };

  constexpr int THREAD_NUM = 40;
  std::vector<std::thread> threads(THREAD_NUM);
  for (size_t i = 0 ; i < THREAD_NUM; i++) {
    threads[i] = std::thread(func);
  }

  for (auto &t : threads) {
    t.join();
  }
}

}
} // namespace tensorflow
