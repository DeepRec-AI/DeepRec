#ifndef TENSORFLOW_COMMON_RUNTIME_TENSORPOOL_ALLOCATOR_GPU_H_
#define TENSORFLOW_COMMON_RUNTIME_TENSORPOOL_ALLOCATOR_GPU_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/spin_lock.h"

#include <atomic>
#include <map>
#include <stack>
#include <vector>

namespace tensorflow {

class GPUMemoryPlannerBase;
class VirtualGPUAllocBlock;

class GPUTensorPoolAllocator : public Allocator {
 public:
  GPUTensorPoolAllocator(SubAllocator* sub_allocator, string name,
                      size_t total_memory);
  ~GPUTensorPoolAllocator() override;

  GPUTensorPoolAllocator(const GPUTensorPoolAllocator&) = delete;
  GPUTensorPoolAllocator& operator=(const GPUTensorPoolAllocator&) = delete;

  void Init();
  void BeginStep();

  string Name() override { return name_; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;
  void DeallocateRawAsync(void* ptr) override;

  absl::optional<AllocatorStats> GetStats() override;

  void DumpStats();

  class Bin;
  Bin* GetBin(size_t bin_index);

  class VirtualBuffer {
   public:
    VirtualBuffer(std::vector<VirtualGPUAllocBlock*>& vblocks,
        GPUTensorPoolAllocator* tp);
    virtual ~VirtualBuffer() {}

    void* Allocate();
    void BeginStep();

   private:
    std::vector<Bin*> internal_bins_;
    std::atomic<size_t> curr_index_;
  };

  class Buffer {
   public:
    Buffer(size_t len, size_t chunk_size,
        size_t alignment, void* begin);

    void* Allocate();
    void Deallocate(void* p);

   private:
    mutable spin_lock lock_;
    std::stack<void*> buffer_;
    void* begin_;
    void* end_;
  };

  class Bin {
   public:
    Bin(size_t len, size_t chunk_size, size_t alignment,
        std::vector<VirtualGPUAllocBlock*>& vblocks,
        GPUTensorPoolAllocator* tp, void* begin);
    virtual ~Bin(){}

    Bin(const Bin&) = delete;
    Bin& operator=(const Bin&) = delete;

    void* Allocate();
    void* AllocateRaw();
    void DeallocateRaw(void* p);

    void BeginStep();

   private:
    Buffer buffer_;
    VirtualBuffer virtual_buffer_;
  };

  class SmallBin {
   public:
    SmallBin(size_t len, size_t chunk_size, size_t alignment, void* begin);
    virtual ~SmallBin(){}

    SmallBin(const SmallBin&) = delete;
    Bin& operator=(const Bin&) = delete;

    void* AllocateRaw();
    void DeallocateRaw(void* p);

   private:
    mutable spin_lock lock_;
    std::stack<void*> buffer_;
    void* begin_;
    void* end_;
  };

 private:
  bool IsBigOwned(void *ptr);
  bool IsSmallOwned(void *ptr);
  void* BigAllocate(size_t alignment, size_t num_bytes);
  void* BigAllocateStatistic(size_t alignment, size_t num_bytes);
  void BigDeallocate(void* ptr);

  SmallBin* GetSmallBin(size_t size);
  void* SmallAllocate(size_t alignment, size_t num_bytes);
  void SmallDeallocate(void* ptr);

 private:
  mutable spin_lock free_lock_;
  std::vector<void*> async_free_list_;
  string name_;
  AllocatorStats alloc_stats_;

  bool stats_;
  std::atomic_bool inited_;
  std::atomic_bool initing_;

  std::unique_ptr<SubAllocator> sub_allocator_;
  GPUMemoryPlannerBase* mem_planner_;

  size_t large_bin_index_;
  std::vector<Bin*> lifetime_bins_;
  std::vector<SmallBin*> small_bins_;
  std::map<size_t, Bin*> large_lifetime_bins_;

  size_t alignment_;
  size_t alignment_offset_;
  size_t big_bytes_;
  void *big_mem_begin_;
  void *big_mem_end_;
  std::map<size_t, Bin*> offset_to_bin_;

  size_t small_bytes_;
  void *small_mem_begin_;
  void *small_mem_end_;
  std::map<size_t, SmallBin*> offset_to_small_bin_;

  // Statistic
  std::atomic<int64_t> null_bin_counter_;
  std::atomic<int64_t> hit_counter_;
  std::atomic<int64_t> missed_counter_;
};

}

#endif // TENSORFLOW_COMMON_RUNTIME_TENSORPOOL_ALLOCATOR_GPU_H_
