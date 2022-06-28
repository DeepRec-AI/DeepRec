#ifndef TENSORFLOW_COMMON_RUNTIME_MEMORYPLANNER_H_
#define TENSORFLOW_COMMON_RUNTIME_MEMORYPLANNER_H_

#include "tensorflow/core/lib/core/spin_lock.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include <atomic>
#include <map>
#include <vector>

namespace tensorflow {
namespace {
constexpr size_t _128B = (1 << 7);
constexpr size_t _32MB = (1 << 25);
constexpr size_t _64MB = (1 << 26);
constexpr size_t _128MB = (1 << 27);
constexpr size_t _32KB = (1 << 15);
constexpr size_t _4KB = (1 << 12);
constexpr size_t _4KB_OFFSET = 12;
constexpr size_t _8KB = (1 << 13);
constexpr size_t _8KB_OFFSET = 13;
constexpr size_t _16KB = (1 << 14);
constexpr size_t _16KB_OFFSET = 14;
constexpr size_t _32KB_OFFSET = 15;

inline bool SmallAlloc(size_t s) {
  return s <= _32KB;
}

inline size_t Index(size_t s, size_t alignment, size_t alignment_offset) {
  if (SmallAlloc(s)) {
    return -1;
  }

  int64_t aligned = alignment *
    ((s + alignment - 1) / alignment);
  return ((aligned - _32KB) >> alignment_offset) - 1;
}

inline double Timeval2Double(const timeval& tv) {
  return tv.tv_sec * 1000 * 1000 + tv.tv_usec;
}
}
  
struct AllocStats {
  double begin;
  double end;
  size_t size;
  bool IsOverlap(const AllocStats* other);
};

class AllocBlock {
 public:
  AllocBlock(size_t size, size_t bin_index);
  virtual ~AllocBlock();

  void Insert(AllocStats* alloc_stats);
  bool CanInsert(AllocStats* alloc_stats);
  size_t BinIndex() const { return bin_index_; }

 private: 
  std::vector<AllocStats*> stats_;
  size_t size_;
  size_t bin_index_;
};

class VirtualAllocBlock {
 public:
  VirtualAllocBlock(AllocBlock* block, size_t s) :
    internal_block_(block), size_(s) {
    }; 

  size_t BinIndex() const {
    return internal_block_->BinIndex();
  }

 private:
  AllocBlock* internal_block_;
  size_t size_;
};

class LifetimePolicy;
class LifetimeBin {
 public:
  LifetimeBin(size_t bin_index, size_t chunk_size);
  virtual ~LifetimeBin();

  void TrackAllocate(size_t alignment);
  void TrackDeallocate(AllocStats* stats);
  size_t TotalMem() const;
  void Dump() const;
  bool BestFit(LifetimePolicy* policy);
  void Cleanup();

  AllocBlock* FindBlock(AllocStats* stats);

  size_t BlockSize() const;
  size_t ChunkSize() const;
  size_t Alignment() const;
  size_t BinIndex() const { return bin_index_; }
  std::vector<VirtualAllocBlock*>& VBlocks() {
    return virtual_blocks_;
  }

 private:
  mutable spin_lock stats_lock_;
  std::vector<AllocStats*> stats_;
  std::vector<AllocBlock*> blocks_;
  std::vector<VirtualAllocBlock*> virtual_blocks_;
  size_t bin_index_;
  size_t chunk_size_;
  int64_t max_alignment_;
};

class Header;
class LifetimePolicy {
 public:
  LifetimePolicy(size_t interval, size_t interval_offset, size_t start);
  virtual ~LifetimePolicy() {};

  void TrackAllocate(size_t alignment, size_t num_bytes);
  void TrackDeallocate(Header* header);
  size_t TotalMem() const;

  void Dump() const;
  void Cleanup();

  AllocBlock* FindBlock(AllocStats* stats, size_t bin_index);

  bool BestFit();
  size_t Interval();

  std::vector<LifetimeBin*>& GetBins();
  std::map<size_t, LifetimeBin*>& GetLargeBins();

  size_t Alignment() const;
  size_t AlignmentOffset() const;

 private:
  LifetimeBin* GetBin(size_t index);

 private:
  std::vector<LifetimeBin*> bins_;
  std::map<size_t, LifetimeBin*> large_bins_;
  mutable spin_lock large_bin_lock_;
  const size_t interval_;
  const size_t interval_offset_;
  const size_t start_;
  const size_t large_bin_index_;
};

class TensorPoolAllocator;
class MemoryPlannerBase {
 public:
  virtual void SetAllocator(TensorPoolAllocator* allocator) = 0;
  virtual void SetThreadPool(thread::ThreadPool* thread_pool) = 0;
  virtual void StartCollect() = 0;
  virtual void StopCollect() = 0;
  virtual void TrackAllocate(size_t alignment, size_t num_bytes) = 0;
  virtual void TrackDeallocate(Header* header) = 0;
  virtual LifetimePolicy* BestLifetimePolicy() = 0;

  virtual void Reset() = 0;
};

class NullableMemoryPlanner : public MemoryPlannerBase {
  void SetAllocator(TensorPoolAllocator* allocator) override {}
  void SetThreadPool(thread::ThreadPool* thread_pool) override {}
  void StartCollect() override {}
  void StopCollect() override {}
  void TrackAllocate(size_t alignment, size_t num_bytes) override {}
  void TrackDeallocate(Header* header) override {}

  LifetimePolicy* BestLifetimePolicy() override {
    LOG(ERROR) << "Memory Optimization is disable, shouldn't be here";
    return nullptr;
  }
  void Reset() override {}
};

class MemoryPlanner : public MemoryPlannerBase {
 public:
  MemoryPlanner();
  virtual ~MemoryPlanner();

  void SetAllocator(TensorPoolAllocator* allocator) override;
  void SetThreadPool(thread::ThreadPool* thread_pool) override;

  void StartCollect() override;
  void StopCollect() override;
  void TrackAllocate(size_t alignment, size_t num_bytes) override;
  void TrackDeallocate(Header* header) override;

  LifetimePolicy* BestLifetimePolicy() override;
  void Reset() override;

 private:
  void Schedule(std::function<void()> f);
  void InitPolicy();
  void InitStepInfo();
  void CollectDone();
  void Cleanup();

 private:
  // statistics
  std::atomic_bool is_stats_;
  std::vector<LifetimePolicy*> lifetime_stats_polices_;

  TensorPoolAllocator* allocator_;
  thread::ThreadPool* thread_pool_;

  // step information
  std::atomic<int64_t> counter_;
  int64 start_step_;
  int64 stable_step_;
  int64 max_stat_step_;
  int64 current_stable_step_;
  int64 current_stat_step_;
};

class MemoryPlannerFactory {
 public:
  static MemoryPlannerBase* GetMemoryPlanner() {
    static MemoryPlannerFactory factory;
    return factory.memory_planner_;
  }
 private:
  MemoryPlannerFactory();
 private:
  bool enable_memory_opt_;
  MemoryPlannerBase* memory_planner_;
};

class ScopedMemoryCollector {
 public:
  ScopedMemoryCollector() {
    MemoryPlannerFactory::GetMemoryPlanner()->StartCollect();
  }
  ~ScopedMemoryCollector() {
    MemoryPlannerFactory::GetMemoryPlanner()->StopCollect();
  }
};

}

#endif // TENSORFLOW_COMMON_RUNTIME_MEMORYPLANNER_H_
