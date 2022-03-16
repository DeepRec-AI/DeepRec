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

class GPUAllocBlock {
 public:
  GPUAllocBlock(size_t size, size_t bin_index);
  virtual ~GPUAllocBlock() {}

  void Insert(AllocStats* alloc_stats);
  bool CanInsert(AllocStats* alloc_stats);
  size_t BinIndex() const { return bin_index_; }
  void ResetStats();

 private:
  std::vector<AllocStats*> stats_;  // not owned
  size_t size_;
  size_t bin_index_;
};

class VirtualGPUAllocBlock {
 public:
  VirtualGPUAllocBlock(GPUAllocBlock* block, size_t s) :
    internal_block_(block), size_(s) {
    };

  size_t BinIndex() const {
    return internal_block_->BinIndex();
  }

 private:
  GPUAllocBlock* internal_block_;
  size_t size_;
};

class GPULifetimePolicy;
class GPULifetimeBin {
 public:
  GPULifetimeBin(size_t bin_index, size_t chunk_size);
  virtual ~GPULifetimeBin();

  void TrackAllocate(size_t alignment);
  void TrackDeallocate(AllocStats* stats);
  void BeginStep();
  size_t TotalMem() const;
  void Dump() const;
  void BestFit(GPULifetimePolicy* policy);
  void SmallFit();
  void Cleanup();

  GPUAllocBlock* FindBlock(AllocStats* stats);

  size_t BlockSize() const;
  size_t ChunkSize() const;
  size_t Alignment() const;
  size_t BinIndex() const { return bin_index_; }
  std::vector<VirtualGPUAllocBlock*>& VBlocks() {
    return virtual_blocks_;
  }

  void ResetStats();

 private:
  mutable spin_lock stats_lock_;
  std::vector<AllocStats*> stats_;  // not owned
  std::vector<GPUAllocBlock*> blocks_;
  std::vector<VirtualGPUAllocBlock*> virtual_blocks_;
  size_t bin_index_;
  size_t chunk_size_;
  int64_t max_alignment_;
};

class GPULifetimePolicy {
 public:
  GPULifetimePolicy(size_t interval, size_t interval_offset, size_t start);
  virtual ~GPULifetimePolicy() {};

  void TrackAllocate(size_t alignment, size_t num_bytes);
  void TrackDeallocate(AllocStats* stats);
  size_t TotalMem() const;

  void Dump() const;
  void Cleanup();

  GPUAllocBlock* FindBlock(AllocStats* stats, size_t bin_index);

  void BestFit();
  size_t Interval();

  std::vector<GPULifetimeBin*>& GetBins();
  std::map<size_t, GPULifetimeBin*>& GetLargeBins();

  size_t Alignment() const;
  size_t AlignmentOffset() const;

  void ResetStats();

 private:
  GPULifetimeBin* GetBin(size_t index);

 private:
  std::vector<GPULifetimeBin*> bins_;
  std::map<size_t, GPULifetimeBin*> large_bins_;
  mutable spin_lock large_bin_lock_;
  const size_t interval_;
  const size_t interval_offset_;
  const size_t start_;
  const size_t large_bin_index_;
};

class GPUTensorPoolAllocator;
class GPUMemoryPlannerBase {
 public:
  virtual void SetAllocator(GPUTensorPoolAllocator* allocator) = 0;
  virtual void SetThreadPool(thread::ThreadPool* thread_pool) = 0;
  virtual void StartCollect() = 0;
  virtual void StopCollect() = 0;
  virtual void TrackAllocate(size_t alignment, size_t num_bytes, void* ptr) = 0;
  virtual void TrackDeallocate(void* ptr) = 0;
  virtual GPULifetimePolicy* BestLifetimePolicy() = 0;
  virtual std::vector<GPULifetimeBin*>& GetSmallBins() = 0;

  virtual void Reset() = 0;
};

class NullableGPUMemoryPlanner : public GPUMemoryPlannerBase {
  void SetAllocator(GPUTensorPoolAllocator* allocator) override {}
  void SetThreadPool(thread::ThreadPool* thread_pool) override {}
  void StartCollect() override {}
  void StopCollect() override {}
  void TrackAllocate(size_t alignment, size_t num_bytes, void* ptr) override {}
  void TrackDeallocate(void* ptr) override {}

  GPULifetimePolicy* BestLifetimePolicy() override {
    LOG(ERROR) << "Memory Optimization is disable, shouldn't be here";
    return nullptr;
  }

  std::vector<GPULifetimeBin*>& GetSmallBins() override {
    std::vector<GPULifetimeBin*> tmp;
    LOG(ERROR) << "Memory Optimization is disable, shouldn't be here";
    return tmp;
  }

  void Reset() override {}
};

class GPUMemoryPlanner : public GPUMemoryPlannerBase {
 public:
  GPUMemoryPlanner();
  virtual ~GPUMemoryPlanner();

  void SetAllocator(GPUTensorPoolAllocator* allocator) override;
  void SetThreadPool(thread::ThreadPool* thread_pool) override;

  void StartCollect() override;
  void StopCollect() override;
  void TrackAllocate(size_t alignment, size_t num_bytes, void* ptr) override;
  void TrackDeallocate(void* ptr) override;

  GPULifetimePolicy* BestLifetimePolicy() override;
  std::vector<GPULifetimeBin*>& GetSmallBins() override;
  void Reset() override;

 private:
  void Schedule(std::function<void()> f);
  void InitPolicy();
  void InitStepInfo();
  void CollectDone();
  void Cleanup();
  void ResetStats();
  void BestFit();

  GPULifetimeBin* GetSmallBin(size_t size);

 private:
  // statistics
  std::atomic_bool is_stats_;
  std::vector<GPULifetimePolicy*> lifetime_stats_polices_;
  std::vector<GPULifetimeBin*> small_bins_;

  GPUTensorPoolAllocator* allocator_;
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

class GPUMemoryPlannerFactory {
 public:
  static GPUMemoryPlannerBase* GetMemoryPlanner() {
    static GPUMemoryPlannerFactory factory;
    return factory.memory_planner_;
  }

 private:
  GPUMemoryPlannerFactory();

 private:
  bool enable_memory_opt_;
  GPUMemoryPlannerBase* memory_planner_;
};

class GPUScopedMemoryCollector {
 public:
  GPUScopedMemoryCollector() {
    GPUMemoryPlannerFactory::GetMemoryPlanner()->StartCollect();
  }
  ~GPUScopedMemoryCollector() {
    GPUMemoryPlannerFactory::GetMemoryPlanner()->StopCollect();
  }
};

}

#endif // TENSORFLOW_COMMON_RUNTIME_MEMORYPLANNER_GPU_H_
