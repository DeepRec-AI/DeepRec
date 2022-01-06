#include "tensorflow/core/common_runtime/memory_planner_gpu.h"
#include "tensorflow/core/common_runtime/tensorpool_allocator_gpu.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/platform/mem.h"
#include <sys/time.h>
#include <iostream>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace tensorflow {

TensorPoolAllocatorGPU::TensorPoolAllocatorGPU(SubAllocator* sub_allocator, string name, 
                                         size_t total_memory) :
    name_(name),
    stats_(false),
    inited_(false),
    initing_(false),
    sub_allocator_(sub_allocator),
    mem_planner_(MemoryPlannerFactoryGPU::GetMemoryPlanner()),
    large_bin_index_(0),
    null_bin_counter_(0),
    hit_counter_(0),
    missed_counter_(0),
    big_mem_begin_(nullptr),
    big_mem_end_(nullptr),
    small_mem_begin_(nullptr),
    small_mem_end_(nullptr) {
  mem_planner_->SetAllocator(this);
  alloc_stats_.bytes_limit = static_cast<int64>(total_memory);
}

TensorPoolAllocatorGPU::~TensorPoolAllocatorGPU() {
  if (big_mem_begin_ != nullptr) {
    sub_allocator_->Free(big_mem_begin_, big_bytes_);
  }
  if (small_mem_begin_ != nullptr) {
    sub_allocator_->Free(small_mem_begin_, small_bytes_);
  }
  for (auto bin : lifetime_bins_) {
    if (bin != nullptr) {
      delete bin;
    }
  }
  for (auto it : large_lifetime_bins_) {
    delete it.second;
  }
  for (auto bin : small_bins_) {
    if (bin != nullptr) {
      delete bin;
    }
  }
}

void TensorPoolAllocatorGPU::Init() {
  bool tmp = false;
  if (initing_.compare_exchange_strong(tmp, true)) {
    auto lifetime_policy = mem_planner_->BestLifetimePolicy();

    alignment_ = lifetime_policy->Alignment();
    alignment_offset_ = lifetime_policy->AlignmentOffset();

    big_bytes_ = 0;
    std::map<size_t, size_t> bin_to_offset;

    auto policy_bins = lifetime_policy->GetBins();
    large_bin_index_ = policy_bins.size();
    lifetime_bins_.resize(large_bin_index_);

    size_t max_alignment = 0;

    for (auto it = policy_bins.begin(); it != policy_bins.end();
        ++it) {
      if ((*it)->BlockSize() > 0) {
        // add padding between two bins
        big_bytes_ = RoundedBytes(big_bytes_, (*it)->Alignment());
        bin_to_offset[(*it)->BinIndex()] = big_bytes_;
        big_bytes_ += (*it)->TotalMem();
        max_alignment = std::max<size_t>(max_alignment, (*it)->Alignment());
      }
    }

    auto policy_large_bins = lifetime_policy->GetLargeBins();
    for (auto it = policy_large_bins.begin();
        it != policy_large_bins.end(); ++it) {
      auto bin_info = it->second;
      if (bin_info->BlockSize() > 0) {
        // add padding between two bins
        big_bytes_ = RoundedBytes(big_bytes_, bin_info->Alignment());
        bin_to_offset[bin_info->BinIndex()] = big_bytes_;
        big_bytes_ += bin_info->TotalMem();
        max_alignment = std::max<size_t>(max_alignment, bin_info->Alignment());
      }
    }

    big_mem_begin_ = sub_allocator_->Alloc(max_alignment, big_bytes_);
    if (big_bytes_ > 0 && big_mem_begin_ == nullptr) {
      LOG(FATAL) << "OOM!!! Try to alloc(" 
                 << max_alignment << ", " << big_bytes_ << ")";
    }
    if (big_bytes_ > 0) {
      big_mem_end_ = big_mem_begin_ + big_bytes_;
    } else {
      big_mem_end_ = nullptr;
    }

    // create bigger bin first
    for (auto rit = policy_large_bins.rbegin();
        rit != policy_large_bins.rend(); ++rit) {
      auto bin_info = rit->second;
      Bin* bin = nullptr;
      if (bin_info->BlockSize() > 0) {
        auto offset = bin_to_offset[rit->first];
        bin = new Bin(bin_info->BlockSize(), bin_info->ChunkSize(),
            bin_info->Alignment(), bin_info->VBlocks(), 
            this, big_mem_begin_ + offset);
        offset_to_bin_[offset] = bin;
      } else if (bin_info->VBlocks().size() > 0) {
        bin = new Bin(bin_info->BlockSize(), bin_info->ChunkSize(),
            bin_info->Alignment(), bin_info->VBlocks(), 
            this, nullptr);
      }
      if (bin != nullptr) {
        large_lifetime_bins_.emplace(rit->first, bin);
      }
    }

    for (auto it = policy_bins.rbegin(); it != policy_bins.rend();
        ++it) {
      Bin* bin = nullptr;
      if ((*it)->BlockSize() > 0) {
        auto offset = bin_to_offset[(*it)->BinIndex()];
        bin = new Bin((*it)->BlockSize(), (*it)->ChunkSize(),
            (*it)->Alignment(), (*it)->VBlocks(), 
            this, big_mem_begin_ + offset);
        offset_to_bin_[offset] = bin;
      } else if ((*it)->VBlocks().size() > 0) {
        bin = new Bin((*it)->BlockSize(), (*it)->ChunkSize(),
            (*it)->Alignment(), (*it)->VBlocks(), 
            this, nullptr);
      }
      lifetime_bins_[(*it)->BinIndex()] = bin;
    }

    auto small_bins = mem_planner_->GetSmallBins();
    small_bins_.resize(small_bins.size());
    small_bytes_ = 0;
    max_alignment = 0;
    bin_to_offset.clear();
    for (auto b : small_bins) {
      if (b->BlockSize() > 0) {
        small_bytes_ = RoundedBytes(small_bytes_, b->Alignment());
        bin_to_offset[b->BinIndex()] = small_bytes_;
        small_bytes_ +=  b->TotalMem();
        max_alignment = std::max<size_t>(max_alignment, b->Alignment());
      }
    }

    small_mem_begin_ = sub_allocator_->Alloc(max_alignment, small_bytes_);
    if (small_bytes_ > 0 && small_mem_begin_ == nullptr) {
      LOG(FATAL) << "OOM!!! Try to alloc(" 
                 << max_alignment << ", " << small_bytes_ << ")";
    }
    if (small_bytes_ > 0) {
      small_mem_end_ = small_mem_begin_ + small_bytes_;
    } else {
      small_mem_end_ = nullptr;
    }

    for (auto b : small_bins) {
      SmallBin* bin = nullptr;
      if (b->BlockSize() > 0) {
        auto offset = bin_to_offset[b->BinIndex()];
        bin = new SmallBin(b->BlockSize(), b->ChunkSize(),
                           b->Alignment(), small_mem_begin_ + offset);
        offset_to_small_bin_[offset] = bin;
      }
      small_bins_[b->BinIndex()] = bin;
    }

    inited_ = true;
  }
}

void TensorPoolAllocatorGPU::BeginStep() {
  if (inited_.load()) {
    for (auto b : lifetime_bins_) {
      if (b != nullptr) {
        b->BeginStep();
      }
    }
    for (auto it : large_lifetime_bins_) {
      it.second->BeginStep();
    }
  }
  std::lock_guard<spin_lock> l(free_lock_);
  for (auto ptr : async_free_list_) {
    sub_allocator_->Free(ptr, 0);
  }
  async_free_list_.clear();
}

void* TensorPoolAllocatorGPU::AllocateRaw(size_t alignment,
    size_t num_bytes) {
  if (!inited_.load()) {
    auto ptr = sub_allocator_->Alloc(alignment, num_bytes);
    mem_planner_->TrackAllocate(alignment, num_bytes, ptr);
    return ptr;
  }

  if (SmallAlloc(num_bytes)) {
    return SmallAllocate(alignment, num_bytes);
  }
  if (unlikely(stats_)) {
    return BigAllocateStatistic(alignment, num_bytes);
  } else {
    return BigAllocate(alignment, num_bytes);
  }
}

void TensorPoolAllocatorGPU::DeallocateRaw(void* ptr) {
  if (!inited_.load()) {
    mem_planner_->TrackDeallocate(ptr);
    sub_allocator_->Free(ptr, 0);
  } else if (IsBigOwned(ptr)) {
    BigDeallocate(ptr);
  } else if (IsSmallOwned(ptr)) {
    SmallDeallocate(ptr);
  } else {
    sub_allocator_->Free(ptr, 0);
  }
}

void TensorPoolAllocatorGPU::DeallocateRawAsync(void* ptr) {
  if (!inited_.load()) {
    mem_planner_->TrackDeallocate(ptr);
    {
      std::lock_guard<spin_lock> l(free_lock_);
      async_free_list_.push_back(ptr);
    }
  } else if (IsBigOwned(ptr)) {
    BigDeallocate(ptr);
  } else if (IsSmallOwned(ptr)) {
    SmallDeallocate(ptr);
  } else {
    std::lock_guard<spin_lock> l(free_lock_);
    async_free_list_.push_back(ptr);
  }
}

absl::optional<AllocatorStats> TensorPoolAllocatorGPU::GetStats() {
  return alloc_stats_;
}

TensorPoolAllocatorGPU::Bin* TensorPoolAllocatorGPU::GetBin(
    size_t bin_index) {
  if (unlikely(bin_index < 0)) {
    return nullptr;
  }

  if (unlikely(bin_index >= large_bin_index_)) {
    auto it = large_lifetime_bins_.find(bin_index);
    if (it == large_lifetime_bins_.end()) {
      return nullptr;
    } else {
      return it->second;
    }
  }
  return lifetime_bins_[bin_index];
}

TensorPoolAllocatorGPU::SmallBin* TensorPoolAllocatorGPU::GetSmallBin(
    size_t size) {
  auto id = kSmallSizeMap.GetClass(size);
  if (unlikely(id >= small_bins_.size())) {
    LOG(FATAL) << "logic error";
    return nullptr;
  }
  return small_bins_[id];
}

TensorPoolAllocatorGPU::SmallBin::SmallBin(size_t len,
    size_t chunk_size, size_t alignment, void* begin) {
  auto rounded_bytes = RoundedBytes(chunk_size, alignment);
  auto buffer_size = rounded_bytes * len;
  begin_ = begin;
  if (begin != nullptr) {
    end_ = begin + buffer_size;
  } else {
    end_ = nullptr;
  }

  for (auto i = 0; i < len; ++i) {
    buffer_.emplace(begin + rounded_bytes *i);
  }
}

void* TensorPoolAllocatorGPU::SmallBin::AllocateRaw() {
  std::lock_guard<spin_lock> l(lock_);
  if (unlikely(buffer_.empty())) {
    return nullptr;
  }
  auto ptr = buffer_.top();
  buffer_.pop();
  return ptr;
}

void TensorPoolAllocatorGPU::SmallBin::DeallocateRaw(void* p) {
  if (unlikely(begin_ == nullptr || p < begin_ || p > end_)) {
    LOG(WARNING) << "probabaly memory corruption!! begin_: " << begin_ 
                 << " end_: " << end_ << " p: " << p;
  }
  std::lock_guard<spin_lock> l(lock_);
  buffer_.emplace(p);
}

TensorPoolAllocatorGPU::Bin::Bin(size_t len,
    size_t chunk_size, size_t alignment,
    std::vector<VirtualAllocBlockGPU*>& vblocks,
    TensorPoolAllocatorGPU* tp, void* begin) :
  buffer_(len, chunk_size, alignment, begin),
  virtual_buffer_(vblocks, tp) {
}

void* TensorPoolAllocatorGPU::Bin::Allocate() {
  auto ptr = buffer_.Allocate();
  if (ptr != nullptr) {
    return ptr;
  } 
  return virtual_buffer_.Allocate();
}

void* TensorPoolAllocatorGPU::Bin::AllocateRaw() {
  return buffer_.Allocate();
}

void TensorPoolAllocatorGPU::Bin::DeallocateRaw(void* p) {
  buffer_.Deallocate(p);
}

void TensorPoolAllocatorGPU::Bin::BeginStep() {
  return virtual_buffer_.BeginStep();
}

TensorPoolAllocatorGPU::Buffer::Buffer(size_t len, size_t chunk_size,
    size_t alignment, void* begin) {
  auto rounded_bytes = RoundedBytes(chunk_size, alignment);
  auto buffer_size = rounded_bytes * len;
  begin_ = begin;
  if (begin != nullptr) {
    end_ = begin + buffer_size;
  } else {
    end_ = nullptr;
  }

  for (auto i = 0; i < len; ++i) {
    buffer_.emplace(begin + rounded_bytes *i);
  }
}

void* TensorPoolAllocatorGPU::Buffer::Allocate() {
  std::lock_guard<spin_lock> l(lock_);
  if (unlikely(buffer_.empty())) {
    return nullptr;
  }
  auto ptr = buffer_.top();
  buffer_.pop();
  return ptr;
}

void TensorPoolAllocatorGPU::Buffer::Deallocate(void* p) {
  if (unlikely(begin_ == nullptr || p < begin_ || p > end_)) {
    LOG(WARNING) << "probabaly memory corruption!! begin_: " << begin_ 
                 << " end_: " << end_ << " p: " << p;
  }
  std::lock_guard<spin_lock> l(lock_);
  buffer_.emplace(p);
}

TensorPoolAllocatorGPU::VirtualBuffer::VirtualBuffer(
    std::vector<VirtualAllocBlockGPU*>& vblocks,
    TensorPoolAllocatorGPU* tp) {
  for (auto vblock : vblocks) {
    auto bin_index = vblock->BinIndex();
    auto internal_bin = tp->GetBin(bin_index);
    if (internal_bin == nullptr) {
      LOG(WARNING) << "logic error or not allocate correctly";
    }
    internal_bins_.emplace_back(internal_bin);
  }
  curr_index_ = 0;
}

void* TensorPoolAllocatorGPU::VirtualBuffer::Allocate() {
  if (unlikely(internal_bins_.empty())) {
    return nullptr;
  }
  auto index = curr_index_.fetch_add(1) % internal_bins_.size();
  auto bin = internal_bins_[index];
  return bin->AllocateRaw();
}

void TensorPoolAllocatorGPU::VirtualBuffer::BeginStep() {
  curr_index_ = 0;
}

void TensorPoolAllocatorGPU::DumpStats() {
  if (stats_) {
    double hit_rate = (double)hit_counter_ /
      (hit_counter_ + missed_counter_ + null_bin_counter_);
    LOG(INFO) << "If you're TensorFlow user, "
      << "please ignore following debugging statistic."
      << "TensorPoolAllocatorGPU Statistic:"
      << " hit_counter[" << hit_counter_
      << "], missed_counter[" << missed_counter_
      << "], null_bin_counter[" << null_bin_counter_
      << "], hit_rate[" << hit_rate
      << "]";

    stats_ = false;
    hit_counter_ = 0;
    missed_counter_ = 0;
    null_bin_counter_ = 0;
  } else {
    stats_ = true;
    LOG(INFO) << "Start counting TensorPoolAllocatorGPU";
  }
}

bool TensorPoolAllocatorGPU::IsBigOwned(void *ptr) {
  return (ptr >= big_mem_begin_ && ptr <= big_mem_end_);
}

bool TensorPoolAllocatorGPU::IsSmallOwned(void *ptr) {
  return (ptr >= small_mem_begin_ && ptr <= small_mem_end_);
}

void* TensorPoolAllocatorGPU::SmallAllocate(size_t alignment, size_t num_bytes) {
  auto bin = GetSmallBin(num_bytes);
  if (unlikely(bin == nullptr)) {
    return sub_allocator_->Alloc(alignment, num_bytes);
  }
  auto ptr = bin->AllocateRaw();
  if (likely(ptr != nullptr)) {
    return ptr;
  }
  return sub_allocator_->Alloc(alignment, num_bytes);
}

void* TensorPoolAllocatorGPU::BigAllocate(size_t alignment,
    size_t num_bytes) {
  auto id = Index(num_bytes, alignment_, alignment_offset_);
  if (unlikely(id < 0)) {
    return sub_allocator_->Alloc(alignment, num_bytes);
  }

  auto b = GetBin(id);
  if (unlikely(b == nullptr)) {
    return sub_allocator_->Alloc(alignment, num_bytes);
  }

  auto ptr = b->Allocate();
  if (likely(ptr != nullptr)) {
    return ptr;
  }

  return sub_allocator_->Alloc(alignment, num_bytes);
}

// unlikely execute this path which do some atomic operations
void* TensorPoolAllocatorGPU::BigAllocateStatistic(size_t alignment,
    size_t num_bytes) {
  auto id = Index(num_bytes, alignment_, alignment_offset_);
  if (unlikely(id < 0)) {
    return sub_allocator_->Alloc(alignment, num_bytes);
  }

  auto b = GetBin(id);
  if (unlikely(b == nullptr)) {
    ++null_bin_counter_;
    return sub_allocator_->Alloc(alignment, num_bytes);
  }

  auto ptr = b->Allocate();
  if (likely(ptr != nullptr)) {
    ++hit_counter_;
    return ptr;
  }

  ++missed_counter_;
  return sub_allocator_->Alloc(alignment, num_bytes);
}

void TensorPoolAllocatorGPU::SmallDeallocate(void* ptr) {
  size_t offset = reinterpret_cast<uint8_t *>(ptr) - 
                  reinterpret_cast<uint8_t *>(small_mem_begin_);
  auto it = offset_to_small_bin_.upper_bound(offset);
  it = std::prev(it);
  it->second->DeallocateRaw(ptr);
}

void TensorPoolAllocatorGPU::BigDeallocate(void* ptr) {
  size_t offset = reinterpret_cast<uint8_t *>(ptr) - 
                  reinterpret_cast<uint8_t *>(big_mem_begin_);
  auto it = offset_to_bin_.upper_bound(offset);
  it = std::prev(it);
  it->second->DeallocateRaw(ptr);
}

} // tensorflow
