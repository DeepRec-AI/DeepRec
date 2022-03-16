#include "tensorflow/core/common_runtime/gpu_tensorpool_allocator.h"
#include "tensorflow/core/common_runtime/gpu_memory_planner.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"
#include <algorithm>
#include <limits>
#include <signal.h>
#include <sys/time.h>

namespace tensorflow {

namespace {
constexpr int64 DEFAULT_START_STATISTIC_STEP = 10;
constexpr int64 DEFAULT_STOP_STATISTIC_STEP = 110;

bool AllocTimeCompare(AllocStats* s1, AllocStats*s2) {
  return s1->begin < s2->begin;
}
}

GPUMemoryPlanner::GPUMemoryPlanner() :
    is_stats_(false),
    inited_(false),
    allocator_(nullptr),
    thread_pool_(nullptr),
    counter_(0),
    start_step_(DEFAULT_START_STATISTIC_STEP),
    stop_step_(DEFAULT_STOP_STATISTIC_STEP) {
  InitStepInfo();
  InitPolicy();
}

GPUMemoryPlanner::~GPUMemoryPlanner() {
}

void GPUMemoryPlanner::InitPolicy() {
  lifetime_stats_polices_.emplace_back(
      new GPULifetimePolicy(_4KB, _4KB_OFFSET, _32KB));
  lifetime_stats_polices_.emplace_back(
      new GPULifetimePolicy(_8KB, _8KB_OFFSET, _32KB));
  lifetime_stats_polices_.emplace_back(
      new GPULifetimePolicy(_16KB, _16KB_OFFSET, _32KB));
  small_bins_.resize(kClassNum);
  for (int i = 0; i < kClassNum; ++i) {
    small_bins_[i] = new GPULifetimeBin(i, kSizeClass[i]);
  }
}

void GPUMemoryPlanner::InitStepInfo() {
  Status s = ReadInt64FromEnvVar("START_STATISTIC_STEP",
      DEFAULT_START_STATISTIC_STEP,
      &start_step_);
  s = ReadInt64FromEnvVar("STOP_STATISTIC_STEP",
      DEFAULT_STOP_STATISTIC_STEP,
      &stop_step_);
}

// lifetime policy
GPULifetimePolicy* GPUMemoryPlanner::BestLifetimePolicy() {
  GPULifetimePolicy* best_policy = nullptr;
  auto total_mem = std::numeric_limits<size_t>::max();
  for (auto policy : lifetime_stats_polices_) {
    auto policy_mem = policy->TotalMem();
    if (policy_mem < total_mem) {
      best_policy = policy;
      total_mem = policy_mem;
    }
  }
  return best_policy;
}

std::vector<GPULifetimeBin*>& GPUMemoryPlanner::GetSmallBins() {
  return small_bins_;
}

GPULifetimeBin* GPUMemoryPlanner::GetSmallBin(size_t size) {
  return small_bins_[kSmallSizeMap.GetClass(size)];
}

void GPUMemoryPlanner::Reset() {
  counter_ = 0;
  Cleanup();
}

void GPUMemoryPlanner::StartCollect() {
  if (is_stats_.load()) {
    BestFit();
    ResetStats();
  }

  auto current = counter_.fetch_add(1);
  if (current == start_step_) {
    is_stats_ = true;
  } else if (current == stop_step_) {
    is_stats_ = false;
    CollectDone();
  }
  if (allocator_ != nullptr) {
    allocator_->BeginStep();
  }
}

void GPUMemoryPlanner::BestFit() {
  for (auto policy : lifetime_stats_polices_) {
    policy->BestFit();
  }
  for (auto bin : small_bins_) {
    bin->SmallFit();
  }
}

void GPUMemoryPlanner::ResetStats() {
  for (auto policy : lifetime_stats_polices_) {
    policy->ResetStats();
  }
  for (auto bin : small_bins_) {
    bin->ResetStats();
  }
  std::lock_guard<spin_lock> l(stats_lock_);
  for (auto s : alloc_stats_) {
    delete s;
  }
  alloc_stats_.clear();
}

void GPUMemoryPlanner::StopCollect() {
  // Make sure counter_ load is atomic.
}

void GPUMemoryPlanner::CollectDone() {
  Schedule([this]() {
    if (allocator_ != nullptr) {
      allocator_->Init();
    }
    Cleanup();
    inited_ = true;
  });
}

void GPUMemoryPlanner::Cleanup() {
  for (auto policy : lifetime_stats_polices_) {
    policy->Cleanup();
  }
  for (auto bin : small_bins_) {
    bin->Cleanup();
  }
  std::lock_guard<spin_lock> l(stats_lock_);
  for (auto it : ptr_stats_) {
    delete it.second;
  }
  ptr_stats_.clear();
  for (auto s : alloc_stats_) {
    delete s;
  }
  alloc_stats_.clear();
}

void GPUMemoryPlanner::SetAllocator(GPUTensorPoolAllocator* allocator) {
  allocator_ = allocator;
}

void GPUMemoryPlanner::SetThreadPool(thread::ThreadPool* thread_pool) {
  if (thread_pool_ == nullptr) {
    thread_pool_ = thread_pool;
  }
}
void GPUMemoryPlanner::Schedule(std::function<void()> f) {
  if (thread_pool_ == nullptr) {
    f();
  } else {
    thread_pool_->Schedule(std::move(f));
  }
}

void GPUMemoryPlanner::TrackAllocate(size_t alignment, size_t num_bytes, void* ptr) {
  if (!is_stats_.load()) {
    return;
  }

  timeval tmp;
  gettimeofday(&tmp, nullptr);

  auto alloc_stats = new AllocStats;
  alloc_stats->begin = Timeval2Double(tmp);
  alloc_stats->size = num_bytes;
  {
    std::lock_guard<spin_lock> l(stats_lock_);
    ptr_stats_[ptr] = alloc_stats;
  }

  if (SmallAlloc(num_bytes)) {
    GetSmallBin(num_bytes)->TrackAllocate(alignment);
    return;
  }

  for (auto lifetime_policy : lifetime_stats_polices_) {
    lifetime_policy->TrackAllocate(alignment, num_bytes);
  }
}

void GPUMemoryPlanner::TrackDeallocate(void* ptr) {
  if (!is_stats_.load()) {
    return;
  }
  timeval tmp;
  gettimeofday(&tmp, nullptr);

  AllocStats* alloc_stats;
  {
    std::lock_guard<spin_lock> l(stats_lock_);
    auto iter = ptr_stats_.find(ptr);
    if (iter == ptr_stats_.end()) {
      return;
    }
    alloc_stats = iter->second;
    ptr_stats_.erase(iter);
    alloc_stats_.emplace_back(alloc_stats);
  }
  alloc_stats->end = Timeval2Double(tmp);

  if (SmallAlloc(alloc_stats->size)) {
    GetSmallBin(alloc_stats->size)->TrackDeallocate(alloc_stats);
    return;
  }

  for (auto lifetime_policy : lifetime_stats_polices_) {
    lifetime_policy->TrackDeallocate(alloc_stats);
  }
}

GPULifetimePolicy::GPULifetimePolicy(size_t interval,
    size_t interval_offset, size_t start) :
    interval_(interval), interval_offset_(interval_offset), start_(start),
    large_bin_index_(Index(_32MB, interval_, interval_offset_) + 1) {
  auto cur = start_ + interval_;
  bins_.resize(large_bin_index_);
  for (auto i = 0; i < large_bin_index_; ++i) {
    bins_[i] = new GPULifetimeBin(i, cur);
    cur += interval_;
  }
}

void GPULifetimePolicy::TrackAllocate(size_t alignment, size_t num_bytes) {
  auto index = Index(num_bytes, interval_, interval_offset_);
  if (index < 0) {
    LOG(ERROR) << "GPUTensorPoolAllocator Invalid Index:" << index
               << ", size:" << num_bytes;
    return;
  }
  GetBin(index)->TrackAllocate(alignment);
}

GPULifetimeBin* GPULifetimePolicy::GetBin(size_t index) {
  if (index >= large_bin_index_) {
    std::lock_guard<spin_lock> l(large_bin_lock_);
    auto bin = large_bins_.find(index);
    if (bin == large_bins_.end()) {
      auto chunk_size = start_ + interval_ * (index + 1);
      bin = large_bins_.emplace(index, new GPULifetimeBin(index, chunk_size)).first;
    }
    return bin->second;
  } else {
    return bins_[index];
  }
}

void GPULifetimePolicy::TrackDeallocate(AllocStats* alloc_stats) {
  auto index = Index(alloc_stats->size, interval_, interval_offset_);
  if (index < 0) {
    LOG(ERROR) << "GPUTensorPoolAllocator Invalid Index:" << index
               << ", size:" << alloc_stats->size;
    return;
  }
  GetBin(index)->TrackDeallocate(alloc_stats);
}

size_t GPULifetimePolicy::TotalMem() const {
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

void GPULifetimePolicy::Dump() const {
  // LOG(INFO_DEV) << "GPULifetimePolicy, start:" << start_
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

void GPULifetimePolicy::Cleanup() {
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

GPULifetimeBin::GPULifetimeBin(size_t bin_index, size_t chunk_size)
    : bin_index_(bin_index),
      chunk_size_(chunk_size),
      max_alignment_(Allocator::kAllocatorAlignment) {
}

GPULifetimeBin::~GPULifetimeBin() {
}

void GPULifetimeBin::TrackAllocate(size_t alignment) {
  std::lock_guard<spin_lock> l(stats_lock_);
  max_alignment_ = std::max<int64_t>(max_alignment_, alignment);
}

void GPULifetimeBin::TrackDeallocate(AllocStats* stats) {
  // multiple thread enter
  std::lock_guard<spin_lock> l(stats_lock_);
  stats_.emplace_back(stats);
}

void GPULifetimePolicy::BestFit() {
  std::lock_guard<spin_lock> l(large_bin_lock_);
  for (auto it = large_bins_.rbegin();
      it != large_bins_.rend(); ++it) {
    auto bin_info = it->second;
    bin_info->BestFit(this);
  }
  for (auto it = bins_.rbegin(); it != bins_.rend(); ++it) {
    (*it)->BestFit(this);
  }
}

std::vector<GPULifetimeBin*>& GPULifetimePolicy::GetBins() {
  return bins_;
}

std::map<size_t, GPULifetimeBin*>& GPULifetimePolicy::GetLargeBins() {
  return large_bins_;
}

size_t GPULifetimePolicy::Alignment() const {
  return interval_;
}

size_t GPULifetimePolicy::AlignmentOffset() const {
  return interval_offset_;
}

size_t GPULifetimePolicy::Interval() {
  return interval_;
}

void GPULifetimePolicy::ResetStats() {
  {
    std::lock_guard<spin_lock> l(large_bin_lock_);
    for (auto it : large_bins_) {
      auto bin_info = it.second;
      bin_info->ResetStats();
    }
  }
  for (auto bin : bins_) {
    bin->ResetStats();
  }
}

void GPULifetimeBin::Cleanup() {
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

void GPULifetimeBin::BestFit(GPULifetimePolicy* policy) {
  std::lock_guard<spin_lock> l(stats_lock_);
  for (auto vb : virtual_blocks_) {
    delete vb;
  }
  virtual_blocks_.clear();
  if (stats_.empty()) {
    return;
  }
  // sort by alloc time
  std::sort(stats_.begin(), stats_.end(), AllocTimeCompare);
  for (auto s : stats_) {
    auto block = FindBlock(s);
    if (block != nullptr) {
      block->Insert(s);
      continue;
    }
    block = policy->FindBlock(s, bin_index_+1);
    if (block != nullptr) {
      block->Insert(s);
      auto vblock = new VirtualGPUAllocBlock(block, chunk_size_);
      virtual_blocks_.emplace_back(vblock);
      continue;
    }
    block = new GPUAllocBlock(chunk_size_, bin_index_);
    block->Insert(s);
    blocks_.emplace_back(block);
  }
}

void GPULifetimeBin::SmallFit() {
  std::lock_guard<spin_lock> l(stats_lock_);
  if (stats_.empty()) {
    return;
  }
  for (auto s : stats_) {
    auto block = FindBlock(s);
    if (block != nullptr) {
      block->Insert(s);
      continue;
    }
    block = new GPUAllocBlock(chunk_size_, bin_index_);
    block->Insert(s);
    blocks_.emplace_back(block);
  }
}

void GPULifetimeBin::ResetStats() {
  std::lock_guard<spin_lock> l(stats_lock_);
  for (auto b : blocks_) {
    b->ResetStats();
  }
  stats_.clear();
}

GPUAllocBlock* GPULifetimeBin::FindBlock(AllocStats* stats) {
  for (auto block : blocks_) {
    if (block->CanInsert(stats)) {
      return block;
    }
  }
  return nullptr;
}

size_t GPULifetimeBin::BlockSize() const {
  return blocks_.size();
}

size_t GPULifetimeBin::ChunkSize() const {
  return chunk_size_;
}

size_t GPULifetimeBin::Alignment() const {
  return max_alignment_;
}

GPUAllocBlock* GPULifetimePolicy::FindBlock(
    AllocStats* stats, size_t bindex) {
  for ( ; bindex < large_bin_index_; ++bindex) {
    auto block = bins_[bindex]->FindBlock(stats);
    if (block != nullptr) {
      return block;
    }
  }
  // no need to lock, BestFit already hold large_bin_lock_ firstly
  for (auto it = large_bins_.lower_bound(bindex);
      it != large_bins_.end(); ++it) {
    auto block = (it->second)->FindBlock(stats);
    if (block != nullptr) {
      return block;
    }
  }
  return nullptr;
}

size_t GPULifetimeBin::TotalMem() const {
  return blocks_.size() * RoundedBytes(chunk_size_, max_alignment_);
}

void GPULifetimeBin::Dump() const {
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

GPUAllocBlock::GPUAllocBlock(size_t size, size_t bin_index)
    : size_(size), bin_index_(bin_index) {
}

bool GPUAllocBlock::CanInsert(AllocStats* alloc_stats) {
  for (auto s : stats_) {
    if (s->IsOverlap(alloc_stats)) {
      return false;
    }
  }
  return true;
}

void GPUAllocBlock::Insert(AllocStats* alloc_stats) {
  // single thread enter
  stats_.emplace_back(alloc_stats);
}

void GPUAllocBlock::ResetStats() {
  stats_.clear();
}

GPUMemoryPlannerFactory::GPUMemoryPlannerFactory() {
  // Enable Memory Optimization by default
  Status s = ReadBoolFromEnvVar("ENABLE_MEMORY_OPTIMIZATION",
      true,
      &enable_memory_opt_);
  if (enable_memory_opt_) {
    //LOG(INFO_DEV) << "Enable Memory Optimization!";
    memory_planner_ = new GPUMemoryPlanner();
  } else {
    memory_planner_ = new NullableGPUMemoryPlanner();
  }
}

} // tensorflow
