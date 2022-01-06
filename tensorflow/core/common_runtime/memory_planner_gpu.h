#ifndef TENSORFLOW_COMMON_RUNTIME_MEMORYPLANNER_GPU_H_
#define TENSORFLOW_COMMON_RUNTIME_MEMORYPLANNER_GPU_H_

#include "tensorflow/core/lib/core/spin_lock.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/common_runtime/memory_planner.h"
#include "tensorflow/core/common_runtime/size_class.h"
#include <atomic>
#include <map>
#include <vector>
#include <unordered_map>

namespace tensorflow {
namespace {

const SizeMap kSmallSizeMap;

inline size_t RoundedBytes(size_t bytes, size_t alignment) {
  return alignment * ((bytes + alignment - 1) / alignment);
}

}

class AllocBlockGPU {
 public:
  AllocBlockGPU(size_t size, size_t bin_index);
  virtual ~AllocBlockGPU() {}

  void Insert(AllocStats* alloc_stats);
  bool CanInsert(AllocStats* alloc_stats);
  size_t BinIndex() const { return bin_index_; }
  void ResetStats();

 private: 
  std::vector<AllocStats*> stats_;  // not owned
  size_t size_;
  size_t bin_index_;
};

class VirtualAllocBlockGPU {
 public:
  VirtualAllocBlockGPU(AllocBlockGPU* block, size_t s) :
    internal_block_(block), size_(s) {
    }; 

  size_t BinIndex() const {
    return internal_block_->BinIndex();
  }

 private:
  AllocBlockGPU* internal_block_;
  size_t size_;
};

class LifetimePolicyGPU;
class LifetimeBinGPU {
 public:
  LifetimeBinGPU(size_t bin_index, size_t chunk_size);
  virtual ~LifetimeBinGPU();

  void TrackAllocate(size_t alignment);
  void TrackDeallocate(AllocStats* stats);
  void BeginStep();
  size_t TotalMem() const;
  void Dump() const;
  void BestFit(LifetimePolicyGPU* policy);
  void SmallFit();
  void Cleanup();

  AllocBlockGPU* FindBlock(AllocStats* stats);

  size_t BlockSize() const;
  size_t ChunkSize() const;
  size_t Alignment() const;
  size_t BinIndex() const { return bin_index_; }
  std::vector<VirtualAllocBlockGPU*>& VBlocks() {
    return virtual_blocks_;
  }

  void ResetStats();

 private:
  mutable spin_lock stats_lock_;
  std::vector<AllocStats*> stats_;  // not owned
  std::vector<AllocBlockGPU*> blocks_;
  std::vector<VirtualAllocBlockGPU*> virtual_blocks_;
  size_t bin_index_;
  size_t chunk_size_;
  int64_t max_alignment_;
};

class LifetimePolicyGPU {
 public:
  LifetimePolicyGPU(size_t interval, size_t interval_offset, size_t start);
  virtual ~LifetimePolicyGPU() {};

  void TrackAllocate(size_t alignment, size_t num_bytes);
  void TrackDeallocate(AllocStats* stats);
  size_t TotalMem() const;

  void Dump() const;
  void Cleanup();

  AllocBlockGPU* FindBlock(AllocStats* stats, size_t bin_index);

  void BestFit();
  size_t Interval();

  std::vector<LifetimeBinGPU*>& GetBins();
  std::map<size_t, LifetimeBinGPU*>& GetLargeBins();

  size_t Alignment() const;
  size_t AlignmentOffset() const;

  void ResetStats();

 private:
  LifetimeBinGPU* GetBin(size_t index);

 private:
  std::vector<LifetimeBinGPU*> bins_;
  std::map<size_t, LifetimeBinGPU*> large_bins_;
  mutable spin_lock large_bin_lock_;
  const size_t interval_;
  const size_t interval_offset_;
  const size_t start_;
  const size_t large_bin_index_;
};

class TensorPoolAllocatorGPU;
class MemoryPlannerBaseGPU {
 public:
  virtual void SetAllocator(TensorPoolAllocatorGPU* allocator) = 0;
  virtual void SetThreadPool(thread::ThreadPool* thread_pool) = 0;
  virtual void StartCollect() = 0;
  virtual void StopCollect() = 0;
  virtual void TrackAllocate(size_t alignment, size_t num_bytes, void* ptr) = 0;
  virtual void TrackDeallocate(void* ptr) = 0;
  virtual LifetimePolicyGPU* BestLifetimePolicy() = 0;
  virtual std::vector<LifetimeBinGPU*>& GetSmallBins() = 0;

  virtual void Reset() = 0;
};

class NullableMemoryPlannerGPU : public MemoryPlannerBaseGPU {
  void SetAllocator(TensorPoolAllocatorGPU* allocator) override {}
  void SetThreadPool(thread::ThreadPool* thread_pool) override {}
  void StartCollect() override {}
  void StopCollect() override {}
  void TrackAllocate(size_t alignment, size_t num_bytes, void* ptr) override {}
  void TrackDeallocate(void* ptr) override {}

  LifetimePolicyGPU* BestLifetimePolicy() override {
    LOG(ERROR) << "Memory Optimization is disable, shouldn't be here";
    return nullptr;
  }

  std::vector<LifetimeBinGPU*>& GetSmallBins() override {
    std::vector<LifetimeBinGPU*> tmp;
    LOG(ERROR) << "Memory Optimization is disable, shouldn't be here";
    return tmp;
  }

  void Reset() override {}
};

class MemoryPlannerGPU : public MemoryPlannerBaseGPU {
 public:
  MemoryPlannerGPU();
  virtual ~MemoryPlannerGPU();

  void SetAllocator(TensorPoolAllocatorGPU* allocator) override;
  void SetThreadPool(thread::ThreadPool* thread_pool) override;

  void StartCollect() override;
  void StopCollect() override;
  void TrackAllocate(size_t alignment, size_t num_bytes, void* ptr) override;
  void TrackDeallocate(void* ptr) override;

  LifetimePolicyGPU* BestLifetimePolicy() override;
  std::vector<LifetimeBinGPU*>& GetSmallBins() override;
  void Reset() override;

 private:
  void Schedule(std::function<void()> f);
  void InitPolicy();
  void InitStepInfo();
  void CollectDone();
  void Cleanup();
  void ResetStats();
  void BestFit();

  LifetimeBinGPU* GetSmallBin(size_t size);

 private:
  // statistics
  std::atomic_bool is_stats_;
  std::vector<LifetimePolicyGPU*> lifetime_stats_polices_;
  std::vector<LifetimeBinGPU*> small_bins_;

  TensorPoolAllocatorGPU* allocator_;
  thread::ThreadPool* thread_pool_;

  mutable spin_lock stats_lock_;
  std::unordered_map<void*, AllocStats*> ptr_stats_;
  std::vector<AllocStats*> alloc_stats_;

  // step information
  std::atomic<int64_t> counter_;
  int64 start_step_;
  int64 stop_step_;
  std::atomic_bool inited_;
};

class MemoryPlannerFactoryGPU {
 public:
  static MemoryPlannerBaseGPU* GetMemoryPlanner() {
    static MemoryPlannerFactoryGPU factory;
    return factory.memory_planner_;
  }
 private:
  MemoryPlannerFactoryGPU();
 private:
  bool enable_memory_opt_;
  MemoryPlannerBaseGPU* memory_planner_;
};

class ScopedMemoryCollectorGPU {
 public:
  ScopedMemoryCollectorGPU() {
    MemoryPlannerFactoryGPU::GetMemoryPlanner()->StartCollect();
  }
  ~ScopedMemoryCollectorGPU() {
    MemoryPlannerFactoryGPU::GetMemoryPlanner()->StopCollect();
  }
};

}

#endif // TENSORFLOW_COMMON_RUNTIME_MEMORYPLANNER_GPU_H_
