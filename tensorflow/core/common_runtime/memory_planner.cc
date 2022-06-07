#include "tensorflow/core/common_runtime/tensorpool_allocator.h"
#include "tensorflow/core/common_runtime/memory_planner.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"
#include <algorithm>
#include <limits>
#include <signal.h>
#include <sys/time.h>

namespace tensorflow {

namespace {
constexpr int64 DEFAULT_START_STATISTIC_STEP = 100;
constexpr int64 DEFAULT_STABLE_STATISTIC_STEP = 10;
constexpr int64 DEFAULT_MAX_STATISTIC_STEP = 100;
}

MemoryPlanner::MemoryPlanner() :
    is_stats_(false),
    allocator_(nullptr),
    thread_pool_(nullptr),
    counter_(0),
    start_step_(DEFAULT_START_STATISTIC_STEP),
    stable_step_(DEFAULT_STABLE_STATISTIC_STEP),
    max_stat_step_(DEFAULT_MAX_STATISTIC_STEP),
    current_stable_step_(0),
    current_stat_step_(0) {
  InitPolicy();
  InitStepInfo();
}

MemoryPlanner::~MemoryPlanner() {
}

void MemoryPlanner::InitPolicy() {
  lifetime_stats_polices_.emplace_back(
      new LifetimePolicy(_4KB, _4KB_OFFSET, _32KB));
  lifetime_stats_polices_.emplace_back(
      new LifetimePolicy(_8KB, _8KB_OFFSET, _32KB));
  lifetime_stats_polices_.emplace_back(
      new LifetimePolicy(_16KB, _16KB_OFFSET, _32KB));
}

void MemoryPlanner::InitStepInfo() {
  Status s = ReadInt64FromEnvVar("START_STATISTIC_STEP",
      DEFAULT_START_STATISTIC_STEP,
      &start_step_);
  if (!s.ok()) {
    LOG(FATAL) << "Read START_STATISTIC_STEP envrionment error. "
                << s.error_message();
  }
  s = ReadInt64FromEnvVar("STABLE_STATISTIC_STEP",
      DEFAULT_STABLE_STATISTIC_STEP,
      &stable_step_);
  if (!s.ok()) {
    LOG(FATAL) << "Read STABLE_STATISTIC_STEP envrionment error. "
                << s.error_message();
  }
  s = ReadInt64FromEnvVar("MAX_STATISTIC_STEP",
      DEFAULT_MAX_STATISTIC_STEP,
      &max_stat_step_);
  if (!s.ok()) {
    LOG(FATAL) << "Read MAX_STATISTIC_STEP envrionment error. "
                << s.error_message();
  }
}

// lifetime policy
LifetimePolicy* MemoryPlanner::BestLifetimePolicy() {
  LifetimePolicy* best_policy = nullptr;
  auto total_mem = std::numeric_limits<size_t>::max();
  for (auto policy : lifetime_stats_polices_) {
    auto policy_mem = policy->TotalMem();
    if (policy_mem < total_mem) {
      best_policy = policy;
      total_mem = policy_mem;
    }
  }
  // LOG(INFO_DEV) << "MemoryPlanner's best lifetime policy consume Memory:"
  //          << total_mem;
  return best_policy;
}

void MemoryPlanner::Reset() {
  counter_ = 0;
  Cleanup();
}

void MemoryPlanner::StartCollect() {
  auto current = counter_.fetch_add(1);
  if (current == start_step_) {
    is_stats_ = true;
  }
}

void MemoryPlanner::StopCollect() {
  if (is_stats_) {
    Schedule([this]() {
      // stop collecting stat when generating policy
      is_stats_ = false;
      ++current_stat_step_;
      bool stable = true;
      for (auto policy : lifetime_stats_polices_) {
        if (!policy->BestFit()) {
          stable = false;
        }
      }
      if (stable) {
        ++current_stable_step_;
      } else {
        current_stable_step_ = 0;
      }
      if (current_stable_step_ > stable_step_
          || current_stat_step_ > max_stat_step_) {
        VLOG(2) << "end planner: " << current_stat_step_;
        CollectDone();
      } else {
        is_stats_ = true;
      }
    });
  }
}

void MemoryPlanner::CollectDone() {
  Schedule([this]() {
    if (allocator_ != nullptr) {
      allocator_->Init();
    }
    Cleanup();
  });
}

void MemoryPlanner::Cleanup() {
  for (auto policy : lifetime_stats_polices_) {
    policy->Cleanup();
  }
}

void MemoryPlanner::SetAllocator(TensorPoolAllocator* allocator) {
  allocator_ = allocator;
}

void MemoryPlanner::SetThreadPool(thread::ThreadPool* thread_pool) {
  if (thread_pool_ == nullptr) {
    thread_pool_ = thread_pool;
  }
}
void MemoryPlanner::Schedule(std::function<void()> f) {
  if (thread_pool_ == nullptr) {
    f();
  } else {
    thread_pool_->Schedule(std::move(f));
  }
}

void MemoryPlanner::TrackAllocate(size_t alignment, size_t num_bytes) {
  if (!is_stats_.load()) {
    return;
  }
  for (auto lifetime_policy : lifetime_stats_polices_) {
    lifetime_policy->TrackAllocate(alignment, num_bytes);
  }
}

void MemoryPlanner::TrackDeallocate(Header* header) {
  if (!is_stats_.load()) {
    return;
  }
  for (auto lifetime_policy : lifetime_stats_polices_) {
    lifetime_policy->TrackDeallocate(header);
  }
}

LifetimePolicy::LifetimePolicy(size_t interval,
    size_t interval_offset, size_t start) :
    interval_(interval), interval_offset_(interval_offset), start_(start),
    large_bin_index_(Index(_32MB, interval_, interval_offset_) + 1) {
  auto cur = start_ + interval_;
  bins_.resize(large_bin_index_);
  for (auto i = 0; i < large_bin_index_; ++i) {
    bins_[i] = new LifetimeBin(i, cur);
    cur += interval_;
  }
}

void LifetimePolicy::TrackAllocate(size_t alignment, size_t num_bytes) {
  auto index = Index(num_bytes, interval_, interval_offset_);
  if (index < 0) {
    LOG(ERROR) << "TensorPoolAllocator Invalid Index:" << index
               << ", size:" << num_bytes;
    return;
  }
  GetBin(index)->TrackAllocate(alignment);
}

LifetimeBin* LifetimePolicy::GetBin(size_t index) {
  if (index >= large_bin_index_) {
    std::lock_guard<spin_lock> l(large_bin_lock_);
    auto bin = large_bins_.find(index);
    if (bin == large_bins_.end()) {
      auto chunk_size = start_ + interval_ * (index + 1);
      bin = large_bins_.emplace(index, new LifetimeBin(index, chunk_size)).first;
    }
    return bin->second;
  } else {
    return bins_[index];
  }
}

void LifetimePolicy::TrackDeallocate(Header* header) {
  timeval tmp;
  gettimeofday(&tmp, nullptr);
  header->end = Timeval2Double(tmp);

  auto alloc_stats = new AllocStats;
  alloc_stats->begin = header->begin;
  alloc_stats->end = header->end;
  alloc_stats->size = header->total_size;
  
  auto index = Index(alloc_stats->size, interval_, interval_offset_);
  if (index < 0) {
    LOG(ERROR) << "TensorPoolAllocator Invalid Index:" << index
               << ", size:" << alloc_stats->size;
    return;
  }
  GetBin(index)->TrackDeallocate(alloc_stats);
}

size_t LifetimePolicy::TotalMem() const {
  size_t total_mem = 0;
  for (auto bin : bins_) {
    total_mem += bin->TotalMem();
  }
  {
    std::lock_guard<spin_lock> l(large_bin_lock_);
    for (auto large_bin : large_bins_) {
      auto bin_info = large_bin.second;
      total_mem += bin_info->TotalMem();
    }
  }
  return total_mem;
}

void LifetimePolicy::Dump() const {
  // LOG(INFO_DEV) << "LifetimePolicy, start:" << start_
  //          << ", interval:" << interval_
  //          << ", Detail:";
  for (auto& b : bins_) {
    b->Dump();
  }
  {
    std::lock_guard<spin_lock> l(large_bin_lock_);
    for (auto& large_bin : large_bins_) {
      auto bin_info = large_bin.second;
      bin_info->Dump();
    }
  }
}

void LifetimePolicy::Cleanup() {
  for (auto bin : bins_) {
    bin->Cleanup();
  }
  {
    std::lock_guard<spin_lock> l(large_bin_lock_);
    for (auto bin : large_bins_) {
      auto bin_info = bin.second;
      bin_info->Cleanup();
    }
  }
}

LifetimeBin::LifetimeBin(size_t bin_index, size_t chunk_size)
    : bin_index_(bin_index),
      chunk_size_(chunk_size),
      max_alignment_(Allocator::kAllocatorAlignment) {
}

LifetimeBin::~LifetimeBin() {
}

void LifetimeBin::TrackAllocate(size_t alignment) {
  max_alignment_ = std::max<int64_t>(max_alignment_, alignment);
}

void LifetimeBin::TrackDeallocate(AllocStats* stats) {
  // multiple thread enter
  std::lock_guard<spin_lock> l(stats_lock_);
  stats_.emplace_back(stats);
}

bool LifetimePolicy::BestFit() {
  bool stable = true;
  std::lock_guard<spin_lock> l(large_bin_lock_);
  for (auto it = large_bins_.rbegin();
      it != large_bins_.rend(); ++it) {
    auto bin_info = it->second;
    bool ret = bin_info->BestFit(this);
    if (!ret) {
      stable = false;
    }
  }
  for (auto it = bins_.rbegin(); it != bins_.rend(); ++it) {
    bool ret = (*it)->BestFit(this);
    if (!ret) {
      stable = false;
    }
  }
  return stable;
}

std::vector<LifetimeBin*>& LifetimePolicy::GetBins() {
  return bins_;
}

std::map<size_t, LifetimeBin*>& LifetimePolicy::GetLargeBins() {
  return large_bins_;
}

size_t LifetimePolicy::Alignment() const {
  return interval_;
}

size_t LifetimePolicy::AlignmentOffset() const {
  return interval_offset_;
}

void LifetimeBin::Cleanup() {
  for (auto block : blocks_) {
    delete block;
  }
  blocks_.clear();

  for (auto vblock : virtual_blocks_) {
    delete vblock;
  }
  virtual_blocks_.clear();

  // stats_ pointer's memory would be clear by blocks.
  // protect stats_ could be touched by other thread.
  std::lock_guard<spin_lock> l(stats_lock_);
  stats_.clear();
}

size_t LifetimePolicy::Interval() {
  return interval_;
}

bool LifetimeBin::BestFit(LifetimePolicy* policy) {
  std::lock_guard<spin_lock> l(stats_lock_);
  if (stats_.empty()) {
    return true;
  }
  bool stable = true;
  for (auto s : stats_) {
    auto block = FindBlock(s);
    if (block != nullptr) {
      block->Insert(s);
      continue;
    }

    block = policy->FindBlock(s, bin_index_+1);
    if (block != nullptr) {
      block->Insert(s);
      auto vblock = new VirtualAllocBlock(block, chunk_size_); 
      virtual_blocks_.emplace_back(vblock);
      continue;
    }
    block = new AllocBlock(chunk_size_, bin_index_);
    block->Insert(s);
    blocks_.emplace_back(block);
    stable = false;
  }
  stats_.clear();
  return stable;
}

AllocBlock* LifetimeBin::FindBlock(AllocStats* stats) {
  for (auto block : blocks_) {
    if (block->CanInsert(stats)) {
      return block;
    }
  }
  return nullptr;
}

size_t LifetimeBin::BlockSize() const {
  return blocks_.size();
}

size_t LifetimeBin::ChunkSize() const {
  return chunk_size_;
}

size_t LifetimeBin::Alignment() const {
  return max_alignment_;
}

AllocBlock* LifetimePolicy::FindBlock(
    AllocStats* stats, size_t bindex) {
  for ( ; bindex < large_bin_index_; ++bindex) {
    auto block = bins_[bindex]->FindBlock(stats);
    if (block != nullptr) {
      return block;
    }
  }
  for (auto it = large_bins_.lower_bound(bindex);
      it != large_bins_.end(); ++it) {
    auto block = (it->second)->FindBlock(stats);
    if (block != nullptr) {
      return block;
    }
  }
  return nullptr;
}

AllocBlock::AllocBlock(size_t size, size_t bin_index)
    : size_(size), bin_index_(bin_index) {
}

AllocBlock::~AllocBlock() {
  for (auto stats : stats_) {
    delete stats;
  }
  stats_.clear();
}

bool AllocStats::IsOverlap(const AllocStats* other) {
  // We counter micro seconds, if equal mean probabaly is overlap.
  if (begin == other->begin || begin == other->end
      || end == other->begin || end == other->end) {
    return true;
  }

  return ((begin > other->begin &&
           begin < other->end) ||
          (begin < other->begin &&
           end > other->begin));
}

bool AllocBlock::CanInsert(AllocStats* alloc_stats) {
  for (auto s : stats_) {
    if (s->IsOverlap(alloc_stats)) {
      return false;
    }
  }
  return true;
}

void AllocBlock::Insert(AllocStats* alloc_stats) {
  // single thread enter
  stats_.emplace_back(alloc_stats);
}

size_t LifetimeBin::TotalMem() const {
  return blocks_.size() * chunk_size_;
}

void LifetimeBin::Dump() const {
  size_t stats_size = 0;
  {
    std::lock_guard<spin_lock> l(stats_lock_);
    if (stats_.empty()) {
      return;
    }
    stats_size = stats_.size();
  }
  // LOG(INFO_DEV) << "Bin index:" << bin_index_
  //          << ", chunk size:" << chunk_size_
  //          << ", stats counter:" << stats_size
  //          << ", blocks counter:" << blocks_.size()
  //          << ", vblocks counter:" << virtual_blocks_.size()
  //          << ", realsize:" << blocks_.size() * chunk_size_;
}

MemoryPlannerFactory::MemoryPlannerFactory() {
  // Enable Memory Optimization by default
  Status s = ReadBoolFromEnvVar("ENABLE_MEMORY_OPTIMIZATION",
      true,
      &enable_memory_opt_);
  if (enable_memory_opt_) {
    //LOG(INFO_DEV) << "Enable Memory Optimization!";
    memory_planner_ = new MemoryPlanner();
  } else {
    memory_planner_ = new NullableMemoryPlanner();
  }
}

} // tensorflow
