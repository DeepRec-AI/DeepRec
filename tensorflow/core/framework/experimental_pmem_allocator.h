#ifndef TENSORFLOW_CORE_FRAMEWORK_EXPERIMENTAL_PMEM_ALLOCATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EXPERIMENTAL_PMEM_ALLOCATOR_H_

#include <fcntl.h>
#include <sys/mman.h>

#include <atomic>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/framework/experimental_pmem_allocator_utils.h"
#include "tensorflow/core/lib/core/spin_lock.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
const uint64_t kPMemNull = UINT64_MAX;
const uint64_t kMaxInstance = 1024;
const uint64_t kMinMovableListSize = 16;

// Following configs can be configured by linux env variable
// ENV: LIBPMEM_MAX_ALLOCATION_SIZE
const uint64_t DEFAULT_LIBPMEM_MAX_ALLOCATION_SIZE = 16384;

// These can be configurable, as this is a experimental edition, hardcode
// them for fast development
const uint64_t kMaxAccessThreads = 512;
const uint64_t kAllocationUnit = 64;
const uint64_t kSegmentSize = 1 << 20;
const float kBGThreadInterval = 1;

// bg_thread_interval: interval to call bg thread to balance freed space among
// access threads
// allocation_unit: minimal allocation unit, shoud be 2^n and no
// less than 8 bytes
// max_allocation_size: max allocation size of the allocator, recommand no
// larger than allocation_unit * 1024
// segment_size: It should be equal or larger than max(1MB,max_allocation_size),
// recommand larger than 128 * max_allocation_size, it should be devidable by
// allocation_unit
//
// See doc/pmem_allocator.md for more details
struct ExperimentalPMemAllocatorConfig {
  ExperimentalPMemAllocatorConfig() = default;

  ExperimentalPMemAllocatorConfig(uint64_t _segment_size,
                                  uint32_t _allocation_unit,
                                  uint32_t _bg_thread_interval,
                                  uint64_t _max_allocation_size)
      : segment_size(_segment_size),
        allocation_unit(_allocation_unit),
        bg_thread_interval(_bg_thread_interval),
        max_allocation_size(_max_allocation_size) {}

  uint64_t segment_size = kSegmentSize;
  uint32_t allocation_unit = kAllocationUnit;
  float bg_thread_interval = kBGThreadInterval;
  uint64_t max_allocation_size = DEFAULT_LIBPMEM_MAX_ALLOCATION_SIZE;
};

// Manage allocation/de-allocation of PMem space at block unit
//
// PMem space consists of several segment, and a segment is consists of
// several blocks, a block is the minimal allocation unit of PMem space
class ExperimentalPMemAllocator : public Allocator {
 public:
  // Create a new PMem allocator instance, map space at pmem file
  // pmem_file: the file on DAX file system or devdax device for mapping PMem
  // space
  // pmem_size: max usable space max_access_threads: max concurrent
  // threads to access this allocator, resource of a access thread is release
  // only if the thread exit or call allocator->Release()
  // config: allocator internal configs
  //
  // See doc/pmem_allocator.md for more details
  static ExperimentalPMemAllocator* NewExperimentalPMemAllocator(
      const std::string& pmem_file, uint64_t pmem_size,
      uint32_t max_access_threads,
      const ExperimentalPMemAllocatorConfig& config);

  ExperimentalPMemAllocator(char* pmem, const std::string& pmem_file_name,
                            uint64_t pmem_size, uint32_t max_access_threads,
                            const ExperimentalPMemAllocatorConfig& config);

  ExperimentalPMemAllocator(const ExperimentalPMemAllocator&) = delete;

  ~ExperimentalPMemAllocator();

  string Name() override { return "pmem_experimental"; }

  // Allocate a PMem space, return address and actually allocated space in bytes
  void* AllocateRaw(size_t alignment, size_t size) override;

  // Free a PMem space entry. The entry should be allocated by this allocator
  void DeallocateRaw(void* addr) override;

  // Return allocated buffer size at ptr
  size_t AllocatedSize(const void* ptr) const {
    auto segment = Addr2Segment(ptr);
    if (segment < segment_record_size_.size()) {
      return segment_record_size_[segment];
    }
    return 0;
  }

  // Release this access thread from the allocator, this will be auto-called
  // while the thread exit
  void Release() {
    if (access_threads_.size() < instance_id_) {
      return;
    }
    access_threads_[instance_id_].Release();
  }

  // Regularly execute by background thread, move freelist of thread caches to
  // pool
  void BackgroundWork();

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
    return AllocatedSize(ptr);
  }

  static bool ValidateConfig(const ExperimentalPMemAllocatorConfig& config) {
    auto is_2pown = [](uint64_t n) { return (n > 0) && (n & (n - 1)) == 0; };

    if (config.allocation_unit < 8) {
      LOG(FATAL) << "Experimental PMem Allocator: Validate config error, "
                    "allocation unit should > 8";
      return false;
    }

    if (!is_2pown(config.allocation_unit)) {
      LOG(FATAL) << "Experimental PMem Allocator: Validate config error, "
                    "allocation unit should be 2^n";
      return false;
    }

    if (config.max_allocation_size > config.allocation_unit * 1024) {
      LOG(FATAL) << "Experimental PMem Allocator: Validate config error, max "
                    "allocation size should <= allocation_unit * 1024";
      return false;
    }

    if (config.segment_size < 1 << 20) {
      LOG(FATAL)
          << "Experimental PMem Allocator: Validate config error, segment_size "
             "should larger than 1MB and max_allocation_size ( "
             "recommand > 128 * max_allocation_size) for performance";
      return false;
    }

    return true;
  }

 private:
  using FreeList = std::vector<void*>;

  struct Segment {
    Segment() : addr(nullptr), size(0) {}

    Segment(void* _addr, uint64_t _size) : addr(_addr), size(_size) {}

    void* addr;
    uint64_t size;
  };

  // free entry pool consists of three level vectors, the first level
  // indicates different block size, each block size consists of several free
  // space entry lists (the second level), and each list consists of several
  // free space entries (the third level).
  //
  // For a specific block size, a write thread will move a entry list from the
  // pool to its thread cache while no usable free space in the cache, or move a
  // entry list to the pool while too many entries cached.
  //
  // Organization of the three level vectors:
  //
  // block size (1st level)   entry lists (2nd level)   entries (3th level)
  //     1   -----------------   list1    ------------   entry1
  //                    |                         |---   entry2
  //                    |-----   list2    ------------   entry1
  //                                              |---   entry2
  //                                              |---   entry3
  //                              ...
  //     2   -----------------   list1    ------------   entry1
  //                    |                         |---   entry2
  //                    |                         |---   entry3
  //                    |-----   list2
  //                              ...
  //    ...
  // max_block_size   --------   list1
  //                    |-----   list2
  class SpaceEntryPool {
   public:
    SpaceEntryPool(uint32_t max_classified_b_size)
        : pool_(max_classified_b_size + 1), spins_(max_classified_b_size + 1) {}

    // move a entry list of b_size free space entries to pool, "src" will be
    // empty after move
    void MoveEntryList(std::vector<void*>& src, uint32_t b_size);

    // try to fetch b_size free space entries from a entry list of pool to dst
    bool FetchEntryList(std::vector<void*>& dst, uint32_t b_size);

   private:
    NoCopyArray<std::vector<FreeList>> pool_;
    // Entry lists of a same block size guarded by a spin lock
    NoCopyArray<spin_lock> spins_;
  };

  // Populate PMem space on init a new instance, so the following access can be
  // faster This will zero the entire PMem space
  void PopulateSpace();

  inline int MaybeInitAccessThread() {
    if (access_threads_.size() <= instance_id_) {
      access_threads_.resize(instance_id_ + 1);
    }
    return thread_manager_->MaybeInitThread(access_threads_[instance_id_]);
  }

  inline void* Offset2Addr(uint64_t offset) const {
    if (ValidateOffset(offset)) {
      return pmem_ + offset;
    }
    return nullptr;
  }

  inline uint64_t Addr2Offset(const void* addr) const {
    if (addr) {
      uint64_t offset = (char*)addr - pmem_;
      if (ValidateOffset(offset)) {
        return offset;
      }
    }
    return kPMemNull;
  }

  inline void* Segment2Addr(uint64_t segment) const {
    return Offset2Addr(segment * segment_size_);
  }

  inline uint64_t Addr2Segment(const void* addr) const {
    uint64_t offset = Addr2Offset(addr);
    return offset == kPMemNull ? kPMemNull : offset / segment_size_;
  }

  inline bool ValidateOffset(uint64_t offset) const {
    return offset < pmem_size_ && offset != kPMemNull;
  }

  // Write threads cache a list of dedicated PMem segments and free lists to
  // avoid contention
  struct alignas(64) ThreadCache {
    ThreadCache(uint32_t max_classified_block_size)
        : freelists(max_classified_block_size + 1),
          segments(max_classified_block_size + 1),
          locks(max_classified_block_size + 1) {}

    ThreadCache(const ThreadCache&) = delete;

    // A array of array to store freed space, the space size is aligned to
    // block_size_, each array corresponding to a dedicated block size which is
    // equal to its index
    NoCopyArray<FreeList> freelists;
    // AllocatorThread own segments, each segment corresponding to a dedicated
    // block size which is equal to its index
    NoCopyArray<Segment> segments;
    // Protect freelists;
    NoCopyArray<spin_lock> locks;

    char padding[64 - sizeof(freelists) - sizeof(segments) - sizeof(locks)];
  };

  static_assert(sizeof(ThreadCache) % 64 == 0);

  bool AllocateSegmentSpace(Segment* segment, uint32_t record_size);

  void init_data_size_2_block_size() {
    data_size_2_block_size_.resize(max_allocation_size_);
    for (size_t i = 0; i < data_size_2_block_size_.size(); i++) {
      data_size_2_block_size_[i] =
          (i / block_size_) + (i % block_size_ == 0 ? 0 : 1);
    }
  }

  inline uint32_t Size2BlockSize(uint32_t data_size) {
    if (data_size < data_size_2_block_size_.size()) {
      return data_size_2_block_size_[data_size];
    }
    return CalculateBlockSize(data_size);
  }

  inline uint32_t CalculateBlockSize(uint32_t data_size) {
    return data_size / block_size_ + (data_size % block_size_ == 0 ? 0 : 1);
  }

  inline int64 TotalAllocationWarningBytes();

  char* pmem_;
  const std::string pmem_file_;
  const uint64_t pmem_size_;
  const uint64_t segment_size_;
  const uint32_t block_size_;
  const uint32_t max_classified_record_block_size_;
  const uint32_t bg_thread_interval_;
  const uint64_t max_allocation_size_;

  SpaceEntryPool pool_;
  std::atomic<uint64_t> segment_head_;
  std::vector<uint32_t> segment_record_size_;

  NoCopyArray<ThreadCache> thread_cache_;
  std::shared_ptr<ThreadManager> thread_manager_;
  std::vector<tensorflow::Thread*> bg_threads_;
  // For quickly get corresponding block size of a requested data size
  std::vector<uint16_t> data_size_2_block_size_;

  bool closing_;

  uint64_t instance_id_;
  static std::atomic<uint64_t> next_instance_;
  static thread_local std::vector<AllocatorThread> access_threads_;

  // Stats
  mutex mu_;
  int total_allocation_warning_count_{0} GUARDED_BY(mu_);
  AllocatorStats stats_ GUARDED_BY(mu_);
};

class ExperimentalPMEMAllocatorFactory : public AllocatorFactory {
 public:
  Allocator* CreateAllocator() override {
    if (pmem_path_.empty()) {
      LOG(FATAL)
          << "Experimental PMem Allocator: PMem path is not initialized, "
             "please set pmem path and allocator size by calling Init()";
      return nullptr;
    }
    int res = create_dir_if_missing(pmem_path_);

    if (res != 0) {
      LOG(FATAL) << "Experimental PMem Allocator: Create pmem allocator path "
                 << pmem_path_ << " error";
      return nullptr;
    }

    int64 max_allocation_size;
    Status s = ReadInt64FromEnvVar("LIBPMEM_MAX_ALLOCATION_SIZE",
                                   DEFAULT_LIBPMEM_MAX_ALLOCATION_SIZE,
                                   &max_allocation_size);
    if (!s.ok()) {
      LOG(FATAL) << "Experimental PMem Allocator: Read max allocation size "
                    "from env variable LIBPMEM_MAX_ALLOCATION_SIZE error";
      return nullptr;
    }

    std::string allocator_file(pmem_path_ +
                               std::to_string(allocator_cnt_.fetch_add(1)));
    return ExperimentalPMemAllocator::NewExperimentalPMemAllocator(
        allocator_file, allocator_size_, kMaxAccessThreads,
        ExperimentalPMemAllocatorConfig(kSegmentSize, kAllocationUnit,
                                        kBGThreadInterval,
                                        max_allocation_size));
  }

  SubAllocator* CreateSubAllocator(int numa_node) override {
    Allocator* pmem_allocator = CreateAllocator();
    if (pmem_allocator != nullptr) {
      return new ExperimentalPMEMSubAllocator(pmem_allocator);
    }
    return nullptr;
  }

  void Init(const std::string& path, size_t size) {
    std::lock_guard<std::mutex> lg(mu_);
    pmem_path_ = format_dir_path(path);
    allocator_size_ = size;
  }

 private:
  inline std::string format_dir_path(const std::string& dir) {
    return dir.back() == '/' ? dir : dir + "/";
  }

  class ExperimentalPMEMSubAllocator : public SubAllocator {
   public:
    explicit ExperimentalPMEMSubAllocator(Allocator* pmem_allocator)
        : SubAllocator({}, {}), pmem_allocator_(pmem_allocator) {}

    void* Alloc(size_t alignment, size_t num_bytes) override {
      return pmem_allocator_->AllocateRaw(alignment, num_bytes);
    }

    void Free(void* ptr, size_t num_bytes) override {
      pmem_allocator_->DeallocateRaw(ptr);
    }

    ~ExperimentalPMEMSubAllocator() {
      if (pmem_allocator_ != nullptr) {
        delete pmem_allocator_;
      }
    }

   private:
    Allocator* pmem_allocator_;
  };
  std::atomic<uint64_t> allocator_cnt_{0};
  std::string pmem_path_;
  size_t allocator_size_;
  std::mutex mu_;
};
}  // namespace tensorflow

#endif
