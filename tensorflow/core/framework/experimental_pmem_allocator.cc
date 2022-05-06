#include "tensorflow/core/framework/experimental_pmem_allocator.h"

#include <string.h>
#include <unistd.h>

#include "libpmem.h"

namespace tensorflow {
// If true, pmem allocator collects more stats.
static bool pmem_allocator_collect_stats = false;
static const int kMaxTotalAllocationWarnings = 1;
// If ev_allocator_collect_stats is true, warn when the total allocated memory
// exceeds this threshold.
static const double kTotalAllocationWarningThreshold = 0.5;

std::atomic<uint64_t> ExperimentalPMemAllocator::next_instance_(0);
thread_local std::vector<AllocatorThread>
    ExperimentalPMemAllocator::access_threads_(0);

ExperimentalPMemAllocator*
ExperimentalPMemAllocator::NewExperimentalPMemAllocator(
    const std::string& pmem_file, uint64_t pmem_size,
    uint32_t max_access_threads,
    const ExperimentalPMemAllocatorConfig& config) {
  if (!ExperimentalPMemAllocator::ValidateConfig(config)) {
    return nullptr;
  }
  int is_pmem;
  uint64_t mapped_size;
  char* pmem;
  if ((pmem =
           (char*)pmem_map_file(pmem_file.c_str(), pmem_size, PMEM_FILE_CREATE,
                                0666, &mapped_size, &is_pmem)) == nullptr) {
    LOG(FATAL) << "Experimental PMem Allocator: PMem map file " << pmem_file
               << " failed: " << strerror(errno);
    return nullptr;
  }

  if (!is_pmem) {
    LOG(FATAL) << "Experimental PMem Allocator: " << pmem_file
               << " is not a valid pmem path";
    return nullptr;
  }

  if (mapped_size != pmem_size) {
    LOG(FATAL) << "Experimental PMem Allocator: PMem map file " << pmem_file
               << " size " << mapped_size << " is not same as expected "
               << pmem_size;
    return nullptr;
  }

  ExperimentalPMemAllocator* allocator = nullptr;
  allocator = new ExperimentalPMemAllocator(pmem, pmem_file, pmem_size,
                                            max_access_threads, config);
  return allocator;
}

void ExperimentalPMemAllocator::SpaceEntryPool::MoveEntryList(
    std::vector<void*>& src, uint32_t b_size) {
  std::lock_guard<spin_lock> lg(spins_[b_size]);
  assert(b_size < pool_.size());
  pool_[b_size].emplace_back();
  pool_[b_size].back().swap(src);
}

bool ExperimentalPMemAllocator::SpaceEntryPool::FetchEntryList(
    std::vector<void*>& dst, uint32_t b_size) {
  if (pool_[b_size].size() != 0) {
    std::lock_guard<spin_lock> lg(spins_[b_size]);
    if (pool_[b_size].size() != 0) {
      dst.swap(pool_[b_size].back());
      pool_[b_size].pop_back();
      return true;
    }
  }
  return false;
}

void ExperimentalPMemAllocator::BackgroundWork() {
  while (1) {
    if (closing_) return;
    usleep(bg_thread_interval_ * 1000000);
    // Move cached list to pool
    std::vector<void*> moving_list;
    for (size_t i = 0; i < thread_cache_.size(); i++) {
      auto& tc = thread_cache_[i];
      moving_list.clear();
      for (size_t b_size = 1; b_size < tc.freelists.size(); b_size++) {
        moving_list.clear();
        std::lock_guard<spin_lock> lg(tc.locks[b_size]);

        if (tc.freelists[b_size].size() >= kMinMovableListSize) {
          if (tc.freelists[b_size].size() >= kMinMovableListSize) {
            moving_list.swap(tc.freelists[b_size]);
          }
        }
        if (moving_list.size() > 0) {
          pool_.MoveEntryList(moving_list, b_size);
        }
      }
    }
  }
}

ExperimentalPMemAllocator::ExperimentalPMemAllocator(
    char* pmem, const std::string& pmem_file_name, uint64_t pmem_size,
    uint32_t max_access_threads, const ExperimentalPMemAllocatorConfig& config)
    : pmem_(pmem),
      pmem_file_(pmem_file_name),
      pmem_size_(pmem_size),
      segment_size_(config.segment_size),
      block_size_(config.allocation_unit),
      max_classified_record_block_size_(
          CalculateBlockSize(config.max_allocation_size)),
      bg_thread_interval_(config.bg_thread_interval),
      max_allocation_size_(config.max_allocation_size),
      pool_(max_classified_record_block_size_),
      segment_head_(0),
      segment_record_size_(pmem_size / segment_size_, 0),
      thread_cache_(max_access_threads, max_classified_record_block_size_),
      thread_manager_(std::make_shared<ThreadManager>(max_access_threads)),
      closing_(false),
      instance_id_(next_instance_.fetch_add(1, std::memory_order_relaxed)) {
  if (instance_id_ > next_instance_) {
    LOG(FATAL) << "Experimental PMem Allocator: Too many instance created (>"
               << kMaxInstance << "), abort";
  }
  init_data_size_2_block_size();
  if (bg_thread_interval_ > 0) {
    bg_threads_.emplace_back(Env::Default()->StartThread(
        tensorflow::ThreadOptions(), "execute_thread",
        [&] { this->BackgroundWork(); }));
  }
}

void ExperimentalPMemAllocator::DeallocateRaw(void* addr) {
  if (addr == nullptr) {
    return;
  }

  int t_id = MaybeInitAccessThread();

  if (t_id < 0) {
    LOG(FATAL) << "Experimental PMem Allocator: Too many thread access "
                  "allocator! max threads: "
               << kMaxAccessThreads;
    std::abort();
  }

  uint64_t segment = Addr2Segment(addr);
  if (segment == kPMemNull) return;

  if (pmem_allocator_collect_stats) {
    const std::size_t alloc_size = AllocatedSize(addr);

    mutex_lock l(mu_);
    stats_.bytes_in_use -= alloc_size;
  }

  uint32_t b_size = segment_record_size_[segment];
  assert(b_size > 0);

  if (b_size > 0) {
    auto& thread_cache = thread_cache_[t_id];
    // Conflict with bg thread happens only if free entries more than
    // kMinMovableListSize
    std::lock_guard<spin_lock> lg(thread_cache.locks[b_size]);
    assert(b_size < thread_cache.freelists.size());
    thread_cache.freelists[b_size].emplace_back(addr);
  }
}

void ExperimentalPMemAllocator::PopulateSpace() {
  LOG(WARNING) << "Experimental PMem Allocator: Polulating PMem space ...";
  for (size_t i = 0; i < pmem_size_ / 32; i++) {
    _mm256_stream_si256(reinterpret_cast<__m256i*>(pmem_) + i,
                        _mm256_set1_epi32(0ULL));
  }
  memset(pmem_ + pmem_size_ - (pmem_size_ % 32), 0, pmem_size_ % 32);
  _mm_mfence();
  LOG(WARNING) << "Experimental PMem Allocator:  Populating done";
}

ExperimentalPMemAllocator::~ExperimentalPMemAllocator() {
  closing_ = true;
  for (tensorflow::Thread* t : bg_threads_) {
    delete t;
  }
  pmem_unmap(pmem_, pmem_size_);
  remove(pmem_file_.c_str());
}

bool ExperimentalPMemAllocator::AllocateSegmentSpace(Segment* segment,
                                                     uint32_t record_size) {
  while (1) {
    uint64_t new_segment = segment_head_.load(std::memory_order_relaxed);
    if (new_segment * segment_size_ + segment_size_ < pmem_size_) {
      if (segment_head_.compare_exchange_strong(new_segment, new_segment + 1)) {
        *segment = Segment{Segment2Addr(new_segment), segment_size_};
        segment_record_size_[new_segment] = record_size;
        return true;
      }
      continue;
    }
    return false;
  }
}

void* ExperimentalPMemAllocator::AllocateRaw(size_t alignment, size_t size) {
  void* ret = nullptr;
  int t_id = MaybeInitAccessThread();
  if (t_id < 0) {
    LOG(FATAL) << "Experimental PMem Allocator: Too many thread access "
                  "allocator! max threads: "
               << kMaxAccessThreads;
    return nullptr;
  }
  uint32_t b_size = Size2BlockSize(size);
  uint32_t aligned_size = b_size * block_size_;
  if (aligned_size > max_allocation_size_ || aligned_size == 0) {
    LOG(FATAL) << "Experimental PMem Allocator: Allocating size: " << size
               << ", is 0 or larger than PMem allocator max allocation size "
               << max_allocation_size_;
    return nullptr;
  }
  auto& thread_cache = thread_cache_[t_id];
  for (auto i = b_size; i < thread_cache.freelists.size(); i++) {
    if (thread_cache.segments[i].size < aligned_size) {
      // Fetch free list from pool
      {
        std::lock_guard<spin_lock> lg(thread_cache.locks[i]);
        if (thread_cache.freelists[i].empty()) {
          pool_.FetchEntryList(thread_cache.freelists[i], i);
        }
        // Get space from free list
        if (thread_cache.freelists[i].size() > 0) {
          ret = thread_cache.freelists[i].back();
          thread_cache.freelists[i].pop_back();
          break;
        }
      }
      // Allocate a new segment for requesting block size
      if (!AllocateSegmentSpace(&thread_cache.segments[b_size], b_size)) {
        continue;
      } else {
        i = b_size;
      }
    }
    assert(thread_cache.segments[i].size >= aligned_size);
    ret = thread_cache.segments[i].addr;
    thread_cache.segments[i].size -= aligned_size;
    thread_cache.segments[i].addr =
        (char*)thread_cache.segments[i].addr + aligned_size;
    break;
  }

  if (pmem_allocator_collect_stats) {
    const std::size_t alloc_size = AllocatedSize(ret);
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
      LOG(WARNING) << "Total allocated pmem " << stats_.bytes_in_use
                   << "exceeds " << 100 * kTotalAllocationWarningThreshold
                   << "% of usable pmem";
    }
  }

  return ret;
}

int64 ExperimentalPMemAllocator::TotalAllocationWarningBytes() {
  return static_cast<int64>(pmem_size_ * kTotalAllocationWarningThreshold);
}

REGISTER_MEM_ALLOCATOR("ExperimentalPMEMAllocator", 30,
                       ExperimentalPMEMAllocatorFactory);
}  // namespace tensorflow
