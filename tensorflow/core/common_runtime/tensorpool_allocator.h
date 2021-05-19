#ifndef TENSORFLOW_COMMON_RUNTIME_TENSORPOOL_ALLOCATOR_H_
#define TENSORFLOW_COMMON_RUNTIME_TENSORPOOL_ALLOCATOR_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/spin_lock.h"

#include <atomic>
#include <map>
#include <stack>
#include <vector>

namespace tensorflow {
// >32KB's allocation header
struct Header {
  double begin;
  double end; 
  void* bin;
  void* internal_bin;
  void* raw_ptr;
  void* user_ptr;
  size_t total_size;
  size_t user_size;

  Header() : begin(0), end(0), bin(nullptr), internal_bin(nullptr),
      raw_ptr(nullptr), user_ptr(nullptr), total_size(0), user_size(0) {
  }
};
  
// <32KB's allocation header
const static std::string CHECK_SUM("AAA"); 
struct LightHeader {
  char checksum[4];
  int32_t header_size;

  explicit LightHeader(size_t hs) : header_size(hs) {
    memcpy(checksum, CHECK_SUM.c_str(), 4);
  }
};

class MemoryPlannerBase;
class VirtualAllocBlock;

class TensorPoolAllocator : public Allocator {
 public:
  TensorPoolAllocator();
  ~TensorPoolAllocator() override {}

  TensorPoolAllocator(const TensorPoolAllocator&) = delete;
  TensorPoolAllocator& operator=(const TensorPoolAllocator&) = delete;

  void Init();

  string Name() override { return "tensorpool_cpu"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;

  void DumpStats();

  class Bin;
  Bin* GetBin(size_t bin_index);

  class VirtualBuffer {
   public:
    VirtualBuffer(std::vector<VirtualAllocBlock*>& vblocks,
        TensorPoolAllocator* tp);
    virtual ~VirtualBuffer() {}

    std::pair<void*, Bin*> Allocate();
    void Deallocate(void* p, Bin* bin);

   private:
    mutable spin_lock lock_;
    std::stack<Bin*> internal_bins_;
  };

  class Buffer {
   public:
    Buffer(size_t len, size_t chunk_size,
        size_t alignment, SubAllocator* sub_allocator);

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
        std::vector<VirtualAllocBlock*>& vblocks,
        SubAllocator* sub_allocator, TensorPoolAllocator* tp);
    virtual ~Bin(){}

    Bin(const Bin&) = delete;
    Bin& operator=(const Bin&) = delete;

    void* Allocate(size_t total, size_t header_size);
    void Deallocate(Header* header);

    void* AllocateRaw();
    void DeallocateRaw(void* p);
   
   private:
    Buffer buffer_;
    VirtualBuffer virtual_buffer_;
    SubAllocator* sub_allocator_;
  };

 private:
  void* BigAllocate(size_t alignment, size_t num_bytes);
  void* BigAllocateStatistic(size_t alignment, size_t num_bytes);
  void BigDeallocate(Header* header);
  
 private:
  bool stats_;
  std::atomic_bool inited_;
  std::atomic_bool initing_;

  std::unique_ptr<SubAllocator> sub_allocator_;
  MemoryPlannerBase* mem_planner_;
 
  size_t large_bin_index_;
  std::vector<Bin*> lifetime_bins_;
  std::map<size_t, Bin*> large_lifetime_bins_;

  size_t alignment_;
  size_t alignment_offset_;
 
  // Statistic
  std::atomic<int64_t> null_bin_counter_;
  std::atomic<int64_t> hit_counter_;
  std::atomic<int64_t> missed_counter_;
};

}

#endif // TENSORFLOW_COMMON_RUNTIME_TENSORPOOL_ALLOCATOR_H_
