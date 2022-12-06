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
#include <vector>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/lib/core/spin_lock.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define ARENA_ARRAY_SIZE 1024

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

// The max num of ptr that ThreadLocalBin can cache.
static const int kThreadLocalBinMaxPtrNum = 16;

static const int kThreadLocalBinExchangeMaxPtrNum = \
  kThreadLocalBinMaxPtrNum >> 1;

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

constexpr int kAddressBits =
  (sizeof(void*) < 8 ? (8 * sizeof(void*)) : 48);

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

  size_t BatchAllocate(size_t num, void** ret) {
    mutex_lock l(mu_);
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

  void BatchDeallocate(std::vector<void *> &ptrs) {
    mutex_lock l(mu_);
    free_list_.PushBatch(ptrs.size(), ptrs.data());
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
  mutex mu_;  
  size_t bin_size_;
  PageMap* page_map_ = nullptr GUARDED_BY(mu_);
  Chunk* current_chunk_ = nullptr GUARDED_BY(mu_);

  FreeList free_list_ GUARDED_BY(mu_);
  std::vector<Chunk*> chunks_ GUARDED_BY(mu_);
};

class Arena {
 public:
  Arena(PageMap* pm) : page_map_(pm) {}

  ~Arena() {
    for (auto it = bins_.begin(); it != bins_.end(); ++it) {
      delete it->second;
    }
    bins_.clear();
  }

  size_t BatchAllocate(size_t num, size_t bin_size, void** ret) {
    Bin *bin = nullptr;
    {
      mutex_lock l(mu_);
      auto it = bins_.find(bin_size);
      if (it == bins_.end()) {
	bin = new Bin(bin_size, page_map_);
	bins_.emplace(bin_size, bin);
      } else {
	bin = it->second;
      }
    }
    
    return bin->BatchAllocate(num, ret);
  }

  void BatchDeallocate(size_t bin_size, std::vector<void*>& ptrs) {
    Bin *bin = nullptr;
    {
      mutex_lock l(mu_);
      auto it = bins_.find(bin_size);
      if (it == bins_.end()) {
	bin = new Bin(bin_size, page_map_);
	bins_.emplace(bin_size, bin);
      } else {
	bin = it->second;
      }
    }
    
    return bin->BatchDeallocate(ptrs);
  }

 private:
  mutex mu_;
  std::unordered_map<size_t, Bin*> bins_ GUARDED_BY(mu_);
  PageMap* page_map_ = nullptr;
};

class ThreadLocalBin {
 public:
  ThreadLocalBin(size_t t_bin_size, PageMap *pm, Arena *arena) :
    t_bin_size_(t_bin_size), page_map_(pm), arena_(arena) {}

  ~ThreadLocalBin() {
    FlushBackToArena(list_.size());
  }

  void *Allocate() {
    void *ret = nullptr;

    if (list_.empty()) {
      std::vector<void *> ptrs(kThreadLocalBinExchangeMaxPtrNum, nullptr);
      int ptrs_num = arena_->BatchAllocate(kThreadLocalBinExchangeMaxPtrNum,
                                           t_bin_size_, ptrs.data());
      for (int i = 0; i < ptrs_num; i++) {
	list_.push_front(ptrs[i]);
      }
    }

    if (likely(!list_.empty())) {
      ret = list_.back();
      list_.pop_back();
    }
    
    return ret;
  }

  size_t BatchAllocate(size_t num, void **ret) {
    if (list_.size() >= num) {
      for (int i = 0; i < num; i++) {
	ret[i] = list_.back();
	list_.pop_back();
      }
      return num;
    }

    return arena_->BatchAllocate(num, t_bin_size_, ret);
  }

  void Deallocate(void *ptr) {
    list_.push_front(ptr);

    if (unlikely(list_.size() > kThreadLocalBinMaxPtrNum)) {
      FlushBackToArena(kThreadLocalBinExchangeMaxPtrNum);
    }
  }

private:
  void FlushBackToArena(int num) {
    std::unordered_map<Bin*, std::vector<void *>> bin_ptr_map;
    for (int i = 0; i < num; i++) {
      void *ptr = list_.back();
      list_.pop_back();
      Bin *bin = page_map_->GetBin(ptr);
      bin_ptr_map[bin].push_back(ptr);
    }

    for (auto iter = bin_ptr_map.begin(); iter != bin_ptr_map.end(); ++iter) {
      (iter->first)->BatchDeallocate(iter->second);
    }
  }

private:
  size_t t_bin_size_;
  PageMap *page_map_ = nullptr; // not owned
  Arena *arena_ = nullptr; // not owned
  std::list<void *> list_;
};

class ThreadLocalCache {
 public:
  ThreadLocalCache(PageMap* pm, Arena* arena) : page_map_(pm), arena_(arena) {}
  
  ~ThreadLocalCache() {
    for (auto it = t_bins_.begin(); it != t_bins_.end(); ++it) {
      delete it->second;
    }
    t_bins_.clear();
  }

  void *Allocate(size_t num_bytes) {
    auto it = t_bins_.find(num_bytes);
    if (it != t_bins_.end()) {
      return it->second->Allocate();
    }
    auto b = new ThreadLocalBin(num_bytes, page_map_, arena_);
    t_bins_.emplace(num_bytes, b);
    return b->Allocate();
  }

  size_t BatchAllocate(size_t num, size_t num_bytes, void **ret) {
    auto it = t_bins_.find(num_bytes);
    if (it != t_bins_.end()) {
      return it->second->BatchAllocate(num, ret);
    }
    auto b = new ThreadLocalBin(num_bytes, page_map_, arena_);
    t_bins_.emplace(num_bytes, b);
    return b->BatchAllocate(num, ret);
  }

  void Deallocate(size_t num_bytes, void *ptr) {
    auto it = t_bins_.find(num_bytes);
    if (it != t_bins_.end()) {
      return it->second->Deallocate(ptr);
    }
    auto b = new ThreadLocalBin(num_bytes, page_map_, arena_);
    t_bins_.emplace(num_bytes, b);
    b->Deallocate(ptr);
  }

 private:
  PageMap * page_map_ = nullptr; // not owned
  Arena * arena_ = nullptr; // not owned
  std::unordered_map<size_t, ThreadLocalBin*> t_bins_;
};

class EVAllocatorImpl {
 public:
  EVAllocatorImpl() {
    pthread_key_create(&key_, ThreadLocalCacheCleanup);
    page_map_ = new PageMap();
    arenas_ = new std::vector<Arena>(ARENA_ARRAY_SIZE, page_map_);
    arena_cur_index = 0;
    
  }

  ~EVAllocatorImpl() {
    pthread_key_delete(key_);
    delete arenas_;
    delete page_map_;
  }

  void* Allocate(size_t num_bytes) {
    
    return GetThreadLocalCache()->Allocate(num_bytes);
  }

  size_t BatchAllocate(size_t num, size_t num_bytes, void** ret) {
    return GetThreadLocalCache()->BatchAllocate(num, num_bytes, ret);
  }

  void Deallocate(void* ptr) {
    GetThreadLocalCache()->Deallocate(AllocatedSize(ptr), ptr);
  }

  size_t AllocatedSize(const void* ptr) const {
    auto bin = page_map_->GetBin(ptr);
    if (bin != nullptr) {
      return bin->BinSize();
    }
    return 0;
  }

 private:
  ThreadLocalCache* GetThreadLocalCache() {
    ThreadLocalCache* tCache =
      static_cast<ThreadLocalCache*>(pthread_getspecific(key_));
    if (tCache == nullptr) {
      Arena *arena = GetNewArena();
      tCache = new ThreadLocalCache(page_map_, arena);
      pthread_setspecific(key_, tCache);
    }
    return tCache;
  }

  Arena *GetNewArena() {
    Arena *ret = nullptr;
    {
      mutex_lock l(mu_arena_index_);
      ret = &((*arenas_)[arena_cur_index]);
      arena_cur_index = (arena_cur_index + 1) % ARENA_ARRAY_SIZE;
    }

    return ret;
  }

  static void ThreadLocalCacheCleanup(void *ptr) {
    auto t_ptr = (ThreadLocalCache *)ptr;
    delete t_ptr;
  }

 private:
  pthread_key_t key_;
  mutex mu_arena_index_;
  PageMap* page_map_ = nullptr;
  std::vector<Arena> *arenas_ = nullptr;
  int arena_cur_index GUARDED_BY(mu_arena_index_);
};

class EVAllocator : public Allocator {
 public:
  EVAllocator()
      : single_allocation_warning_count_(0),
        total_allocation_warning_count_(0) {}

  ~EVAllocator() override {}

  string Name() override { return "ev_allocator"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    num_bytes = AlignedSize(num_bytes);

    if (num_bytes > kChunkSize) {
      LOG(FATAL) << "Allocation of " << num_bytes << " exceeds "
                 << kChunkSize << " in EVAllocator.";
    }

    if (num_bytes > LargeAllocationWarningBytes() &&
        single_allocation_warning_count_ < kMaxSingleAllocationWarnings) {
      ++single_allocation_warning_count_;
      LOG(WARNING) << "Allocation of " << num_bytes << " exceeds "
                   << 100 * kLargeAllocationWarningThreshold
                   << "% of system memory.";
    }

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
    num_bytes = AlignedSize(num_bytes);

    if (num_bytes > kChunkSize) {
      LOG(FATAL) << "Allocation of " << num_bytes << " exceeds "
                 << kChunkSize << " in EVAllocator.";
    }

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
  // Return the smallest alignment multiple that is >= s.
  #define ALIGNMENT_CEILING(s, alignment)         \
    (((s) + (alignment - 1)) & ((~(alignment)) + 1))

  size_t AlignedSize(size_t num_bytes) {
    // small allocation no need alignment here.
    if (num_bytes <= 4 * sizeof(float)) {
      return num_bytes;
    }

    // Use _mm_load_ps instructions need aligned address.
    return ALIGNMENT_CEILING(num_bytes, 4 * sizeof(float));
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
