/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <atomic>
#include <list>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/core/spin_lock.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// If true, ev allocator collects more stats.
static bool ev_allocator_collect_stats = false;

static const int kMaxTotalAllocationWarnings = 1;

static const int kMaxSingleAllocationWarnings = 5;

// If ev_allocator_collect_stats is true, warn when the total allocated memory
// exceeds this threshold.
static const double kTotalAllocationWarningThreshold = 0.5;

// Individual allocations large than this amount will trigger a warning.
static const double kLargeAllocationWarningThreshold = 0.1;

// Cache first invocation to port::AvailableRam, as it can be expensive.
static int64_t LargeAllocationWarningBytes() {
  static int64_t value = static_cast<int64>(port::AvailableRam() *
                                            kLargeAllocationWarningThreshold);
  return value;
}

static int64_t TotalAllocationWarningBytes() {
  static int64_t value = static_cast<int64>(port::AvailableRam() *
                                            kTotalAllocationWarningThreshold);
  return value;
}

namespace {
constexpr size_t kChunkSize = ( 1 << 22);  // 4MB chunk size
constexpr size_t kPageSize = (1 << 12);    // 4KB page by default
constexpr size_t kPageShift = 12;
constexpr size_t kPageCount = kChunkSize / kPageSize;

#if defined __x86_64__
constexpr int kAddressBits =
  (sizeof(void*) < 8 ? (8 * sizeof(void*)) : 48);
#else
constexpr int kAddressBits = 8 * sizeof(void*);
#endif

class Bin;
class PageMap {
 public:
  PageMap() : root_{}, bytes_used_(0) {}

  Bin* GetBin(const void* ptr) const {
    const auto k =
      reinterpret_cast<std::uintptr_t>(ptr) >> kPageShift;
    const auto i1 = k >> kLeafBits;
    const auto i2 = k & (kLeafLength - 1);
    if ((k >> kBits) > 0 || root_[i1] == nullptr) {
      return nullptr;
    }
    return root_[i1]->bin[i2];
  }

  void SetBin(const void* ptr, size_t npages, Bin* b) {
    const auto start =
      reinterpret_cast<std::uintptr_t>(ptr) >> kPageShift;
    std::lock_guard<spin_lock> l(lock_);
    for (auto key = start; key < start + npages; ++key) {
      const auto i1 = key >> kLeafBits;
      const auto i2 = key & (kLeafLength - 1);

      CHECK(i1 < kRootLength);
      if (root_[i1] == nullptr) {
        Leaf* leaf = new Leaf;
        CHECK(leaf != nullptr);
        memset(leaf, 0, sizeof(*leaf));
        bytes_used_ += sizeof(Leaf);
        root_[i1] = leaf;
      }
      root_[i1]->bin[i2] = b;
    }
  }
 
 private:
  static constexpr int kBits = kAddressBits - kPageShift;
  // The leaf node (regardless of pointer size) always maps 2^15 entries;
  // with 8K pages, this gives us 256MB mapped per leaf node.
  static constexpr int kLeafBits = 15;
  static constexpr int kLeafLength = 1 << kLeafBits;
  static constexpr int kRootBits =
    (kBits >= kLeafBits) ? (kBits - kLeafBits) : 0;
  // (1<<kRootBits) must not overflow an "int"
  static_assert(kRootBits < sizeof(int) * 8 - 1, "kRootBits is too large");
  static constexpr int kRootLength = 1 << kRootBits;

  struct Leaf {
    Bin* bin[kLeafLength];
  };

  mutable spin_lock lock_;
  Leaf* root_[kRootLength];  // Top-level node
  size_t bytes_used_;
};

class FreeList {
 public:
  // Return current length of list
  size_t length() const { return list_.size(); }

  // Is list empty?
  bool empty() const { return list_.empty(); }

  void Push(void* ptr) {
    list_.push_front(ptr);
  }

  bool TryPop(void** ret) {
    if (list_.empty()) {
      return false;
    }

    *ret = list_.back();
    list_.pop_back();
    return true;
  }

  // PushBatch and PopBatch do not guarantee an ordering.
  void PushBatch(int N, void** ptrs) {
    for (int i = 0; i < N; ++i) {
      list_.push_front(ptrs[i]);
    }
  }

  int PopBatch(int N, void** ret) {
    if (list_.size() >= N) {
      for (int i = 0; i < N; ++i) {
        ret[i] = list_.back();
        list_.pop_back();
      }
      return N;
    } else {
      auto loop = list_.size();
      for (int i = 0; i < loop; ++i) {
        ret[i] = list_.back();
        list_.pop_back();
      }
      return loop;
    }
  }

 private:
  std::list<void*> list_;
};

class Chunk {
 public:
  Chunk(size_t chunk_size, size_t slot_size, Bin* bin, PageMap* pm) :
      chunk_size_(chunk_size), slot_size_(slot_size) {
    slot_count_ = chunk_size_ / slot_size_;
    start_ = (char*)port::AlignedMalloc(chunk_size_, kPageSize);
    if (start_ == nullptr) {
      LOG(FATAL) << "OOM, can't create new Chunk for EVAllocator,"
                 << "please check free memory.";
    }
    pm->SetBin(start_, kPageCount, bin);
    current_ = start_;
    end_ = start_ + chunk_size_;
  }

  ~Chunk() {
    delete start_;
  }

  void* Allocate() {
    if (current_ + slot_size_ <= end_) {
      auto ret = current_;
      current_ += slot_size_;
      return ret;
    }
    return nullptr;
  }

  size_t BatchAllocate(size_t num, void** ret) {
    for (int i = 0; i < num; ++i) {
      if (current_ + slot_size_ > end_) {
        return i;
      }
      ret[i] = current_;
      current_ += slot_size_;
    }
    return num;
  }

  size_t FullAllocate(void** ret) {
    for (int i = 0; i < slot_count_; ++i) {
      ret[i] = current_;
      current_ += slot_size_;
    }
    return slot_count_;
  }

  size_t Count() {
    return slot_count_;
  }

 private:
  char* start_ = nullptr;
  char* current_ = nullptr;
  char* end_ = nullptr;
  size_t chunk_size_;
  size_t slot_size_;
  size_t slot_count_;
};

class Bin {
 public:
  Bin(size_t s, PageMap* pm) : bin_size_(s), page_map_(pm) {
    current_chunk_ = CreateChunk();
  }

  ~Bin() {
    for (auto it : chunks_) {
      delete it;
    }
  }

  void* Allocate() {
    void* ptr = nullptr;
    if (free_list_.TryPop(&ptr)) {
      return ptr;
    }

    ptr = current_chunk_->Allocate();
    if (ptr == nullptr) {
      current_chunk_ = CreateChunk();
      ptr = current_chunk_->Allocate();
    }
    return ptr;
  }

  size_t BatchAllocate(size_t num, void** ret) {
    auto allocated = free_list_.PopBatch(num, ret);
    auto remains = num - allocated;
    if (remains == 0) {
      return num;
    }

    void** cur = ret + allocated;
    allocated = current_chunk_->BatchAllocate(remains, cur);
    remains -= allocated;
    if (remains == 0) {
      return num;
    }

    cur += allocated;
    if (remains < current_chunk_->Count()) {
      current_chunk_ = CreateChunk();

      allocated = current_chunk_->BatchAllocate(remains, cur);
      return num - (remains - allocated);
    }

    // Allocate in multiple chunks.
    auto chunk_num = remains / current_chunk_->Count();
    for (int i = 0; i < chunk_num; ++i) {
      current_chunk_ = CreateChunk();
      allocated = current_chunk_->FullAllocate(cur);

      cur += allocated;
      remains -= allocated;
    }

    current_chunk_ = CreateChunk();
    allocated = current_chunk_->BatchAllocate(remains, cur);
    return num - (remains - allocated);
  }

  void Deallocate(void* ptr) {
    free_list_.Push(ptr);
  }

  size_t BinSize() const {
    return bin_size_;
  }

 private:
  Chunk* CreateChunk() {
    auto c = new Chunk(kChunkSize, bin_size_, this, page_map_);
    chunks_.emplace_back(c);
    return c;
  }

 private:
  size_t bin_size_;
  PageMap* page_map_ = nullptr;
  Chunk* current_chunk_ = nullptr;

  FreeList free_list_;
  std::vector<Chunk*> chunks_;
};

// Thread local arena
class ThreadLocalArena {
 public:
  ThreadLocalArena(PageMap* pm) : page_map_(pm) {}

  ~ThreadLocalArena() {
    for (auto it = bins_.begin(); it != bins_.end(); ++it) {
      delete it->second;
    }
    bins_.clear();
  }

  void* Allocate(size_t num_bytes) {
    auto it = bins_.find(num_bytes);
    if (it != bins_.end()) {
      return it->second->Allocate();
    }
    auto b = new Bin(num_bytes, page_map_);
    bins_.emplace(num_bytes, b);
    return b->Allocate();
  }

  size_t BatchAllocate(size_t num, size_t num_bytes, void** ret) {
    auto it = bins_.find(num_bytes);
    if (it != bins_.end()) {
      return it->second->BatchAllocate(num, ret);
    }
    auto b = new Bin(num_bytes, page_map_);
    bins_.emplace(num_bytes, b);
    return b->BatchAllocate(num, ret);
  }

  void Deallocate(size_t num_bytes, void* ptr) {
    auto it = bins_.find(num_bytes);
    if (it != bins_.end()) {
      return it->second->Deallocate(ptr);
    }
    auto b = new Bin(num_bytes, page_map_);
    bins_.emplace(num_bytes, b);
    return b->Deallocate(ptr);
  }

 private:
  std::unordered_map<size_t, Bin*> bins_;
  PageMap* page_map_ = nullptr;
};

class EVAllocatorImpl {
 public:
  EVAllocatorImpl() {
    pthread_key_create(&key_, nullptr);
    page_map_ = new PageMap();
  }

  ~EVAllocatorImpl() {
    pthread_key_delete(key_);
  }

  void* Allocate(size_t num_bytes) {
    return GetArena()->Allocate(num_bytes);
  }

  size_t BatchAllocate(size_t num, size_t num_bytes, void** ret) {
    return GetArena()->BatchAllocate(num, num_bytes, ret);
  }

  void Deallocate(void* ptr) {
    GetArena()->Deallocate(AllocatedSize(ptr), ptr);
  }

  size_t AllocatedSize(const void* ptr) const {
    auto bin = page_map_->GetBin(ptr);
    if (bin != nullptr) {
      return bin->BinSize();
    }
    return 0;
  }

 private:
  ThreadLocalArena* GetArena() {
    ThreadLocalArena* arena =
      static_cast<ThreadLocalArena*>(pthread_getspecific(key_));
    if (arena == nullptr) {
      arena = new ThreadLocalArena(page_map_);
      pthread_setspecific(key_, arena);
    }
    return arena;
  }

 private:
  pthread_key_t key_;
  PageMap* page_map_ = nullptr;
};

class EVAllocator : public Allocator {
 public:
  EVAllocator()
      : single_allocation_warning_count_(0),
        total_allocation_warning_count_(0) {}

  ~EVAllocator() override {}

  string Name() override { return "ev_allocator"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    if (num_bytes > LargeAllocationWarningBytes() &&
        single_allocation_warning_count_ < kMaxSingleAllocationWarnings) {
      ++single_allocation_warning_count_;
      LOG(WARNING) << "Allocation of " << num_bytes << " exceeds "
                   << 100 * kLargeAllocationWarningThreshold
                   << "% of system memory.";
    }

    // support 4B no fragment allocation.
    alignment = (num_bytes <= 4) ? 4 : 8;
    void* p = impl_.Allocate(num_bytes);
    if (ev_allocator_collect_stats) {
      const std::size_t alloc_size = impl_.AllocatedSize(p);
      mutex_lock l(mu_);
      ++stats_.num_allocs;
      stats_.bytes_in_use += alloc_size;
      stats_.peak_bytes_in_use =
          std::max<int64>(stats_.peak_bytes_in_use, stats_.bytes_in_use);
      stats_.largest_alloc_size =
          std::max<int64>(stats_.largest_alloc_size, alloc_size);

      if (stats_.bytes_in_use > TotalAllocationWarningBytes() &&
          total_allocation_warning_count_ < kMaxTotalAllocationWarnings) {
        ++total_allocation_warning_count_;
        LOG(WARNING) << "Total allocated memory " << stats_.bytes_in_use
                     << "exceeds " << 100 * kTotalAllocationWarningThreshold
                     << "% of system memory";
      }
    }
    return p;
  }

  size_t BatchAllocateRaw(size_t num, size_t alignment,
      size_t num_bytes, void** ret) override {
    if (num_bytes > LargeAllocationWarningBytes() &&
        single_allocation_warning_count_ < kMaxSingleAllocationWarnings) {
      ++single_allocation_warning_count_;
      LOG(WARNING) << "Allocation of " << num_bytes << " exceeds "
                   << 100 * kLargeAllocationWarningThreshold
                   << "% of system memory.";
    }

    auto allocated_num = impl_.BatchAllocate(num, num_bytes, ret);
    if (allocated_num == 0) {
      LOG(WARNING) << "Can't allocate num:"
                   << num << ", num_bytes:" << num_bytes;
      return 0;
    }

    if (ev_allocator_collect_stats) {
      auto p = ret[0];
      const std::size_t alloc_size = impl_.AllocatedSize(p);
      mutex_lock l(mu_);
      stats_.num_allocs += allocated_num;
      stats_.bytes_in_use += alloc_size * allocated_num;
      stats_.peak_bytes_in_use =
          std::max<int64>(stats_.peak_bytes_in_use, stats_.bytes_in_use);
      stats_.largest_alloc_size =
          std::max<int64>(stats_.largest_alloc_size, alloc_size);

      if (stats_.bytes_in_use > TotalAllocationWarningBytes() &&
          total_allocation_warning_count_ < kMaxTotalAllocationWarnings) {
        ++total_allocation_warning_count_;
        LOG(WARNING) << "Total allocated memory " << stats_.bytes_in_use
                     << "exceeds " << 100 * kTotalAllocationWarningThreshold
                     << "% of system memory";
      }
    }
    return allocated_num;
  }

  void DeallocateRaw(void* ptr) override {
    if (ev_allocator_collect_stats) {
      const std::size_t alloc_size = impl_.AllocatedSize(ptr);
      
      mutex_lock l(mu_);
      stats_.bytes_in_use -= alloc_size;
    }

    impl_.Deallocate(ptr);
  }

  absl::optional<AllocatorStats> GetStats() override {
    mutex_lock l(mu_);
    return stats_;
  }

  void ClearStats() override {
    mutex_lock l(mu_);
    stats_.num_allocs = 0;
    stats_.peak_bytes_in_use = stats_.bytes_in_use;
    stats_.largest_alloc_size = 0;
  }

  size_t AllocatedSizeSlow(const void* ptr) const override {
    return impl_.AllocatedSize(ptr);
  }

 private:
  mutex mu_;
  AllocatorStats stats_ GUARDED_BY(mu_);

  // Use <atomic> for single allocations to avoid mutex contention when
  // statistics are disabled.
  std::atomic<int> single_allocation_warning_count_;
  int total_allocation_warning_count_ GUARDED_BY(mu_);

  EVAllocatorImpl impl_;

  TF_DISALLOW_COPY_AND_ASSIGN(EVAllocator);
};

class EVAllocatorFactory : public AllocatorFactory {
 public:
  Allocator* CreateAllocator() override { return CreateEVAllocator(); }

  Allocator* CreateEVAllocator() override {
    return new EVAllocator;
  }

  SubAllocator* CreateSubAllocator(int numa_node) override {
    return new EVSubAllocator(new EVAllocator);
  }

 private:
  class EVSubAllocator : public SubAllocator {
   public:
    explicit EVSubAllocator(EVAllocator* ev_allocator)
        : SubAllocator({}, {}), ev_allocator_(ev_allocator) {}

    void* Alloc(size_t alignment, size_t num_bytes) override {
      return ev_allocator_->AllocateRaw(alignment, num_bytes);
    }

    void Free(void* ptr, size_t num_bytes) override {
      ev_allocator_->DeallocateRaw(ptr);
    }
   private:
    EVAllocator* ev_allocator_;
  };
};

REGISTER_MEM_ALLOCATOR("EVAllocator", 20, EVAllocatorFactory);
}  // namespace

}  // namespace tensorflow
