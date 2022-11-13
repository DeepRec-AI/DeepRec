#include "tensorflow/core/common_runtime/memory_planner.h"
#include "tensorflow/core/common_runtime/tensorpool_allocator.h"
#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/platform/mem.h"
#include <sys/time.h>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace tensorflow {
namespace {
void signal_handler(int sig_num) {
    // kill -10 would output TensorPoolAllocator's statistic information
  if (sig_num == SIGUSR1) {
    TensorPoolAllocator* p =
      dynamic_cast<TensorPoolAllocator*>(cpu_allocator());
    p->DumpStats();
  }
}

size_t RoundedBytes(size_t bytes, size_t alignment) {
  return alignment * ((bytes + alignment - 1) / alignment);
}

void* SetLightHeader(void* p, size_t total_bytes, size_t header_size) {
  // LightHeader *KB max(sizeof(LightHeader)=8B, alignment)
  //   { | .....| checksum (4B) | header_size (4B)}
  auto user_ptr = (char*)p + header_size;
  auto header = new((char*)user_ptr - sizeof(LightHeader))
                    LightHeader(header_size);
  return user_ptr;
}

LightHeader* GetLightHeader(void* p) {
  auto light_header = (LightHeader*)((char*)p - sizeof(LightHeader));
  return (strcmp(light_header->checksum, CHECK_SUM.c_str()) == 0)
           ? light_header
           : nullptr;
}

Header* GetHeader(void* p) {
  auto header = (Header*)((char*) p - sizeof(Header));
  if (header->user_ptr != p) {
    auto light_header = GetLightHeader(p);
    LOG(FATAL) << "Memory corruption!"
               << ", p:" << p
               << ", p->header_size:" << light_header->header_size
               << ", p->checksum:" << light_header->checksum;
  }
  return header;
}

void* SetDefaultHeader(bool stats_time, void* p,
    size_t total_bytes, size_t header_size) {
  auto h = new((char*)p + header_size - sizeof(Header))Header();
  h->user_size = total_bytes - header_size;
  h->user_ptr = (char*)p + header_size;
  h->raw_ptr = p;
  h->total_size = total_bytes;

  if (stats_time) {
    timeval tmp;
    gettimeofday(&tmp, nullptr);
    h->begin = Timeval2Double(tmp);
  }
  return h->user_ptr;
}

void* SetHeader(void* p, size_t total_bytes, size_t header_size,
    void* bin, void* internal_bin) {
  // Header *KB max(sizeof(Header)=64B, alignment)
  // { |........| begin_time 16B | end_time 16B | total_size 8B
  //            |  user_size 8B | raw_ptr 8B | user_ptr 8B }
  auto h = new((char*)p + header_size - sizeof(Header))Header();
  h->user_size = total_bytes - header_size;
  h->user_ptr = (char*)p + header_size;
  h->raw_ptr = p;
  h->total_size = total_bytes;
  h->bin = bin;
  h->internal_bin = internal_bin;
  return h->user_ptr;
}

class DefaultCPUSubAllocator : public SubAllocator {
 public:
  DefaultCPUSubAllocator() : SubAllocator({}, {}) {}
  ~DefaultCPUSubAllocator() override {}

  void* Alloc(size_t alignment, size_t num_bytes) override {
    return port::AlignedMalloc(num_bytes, alignment);
  }

  void Free(void* ptr, size_t num_bytes) override {
    port::AlignedFree(ptr);
  }
};
}

TensorPoolAllocator::TensorPoolAllocator() :
    stats_(false),
    inited_(false),
    initing_(false),
    sub_allocator_(new DefaultCPUSubAllocator),
    mem_planner_(MemoryPlannerFactory::GetMemoryPlanner()),
    large_bin_index_(0),
    null_bin_counter_(0),
    hit_counter_(0),
    missed_counter_(0) {
  mem_planner_->SetAllocator(this);
}

void TensorPoolAllocator::Init() {
  bool tmp = false;
  if (initing_.compare_exchange_strong(tmp, true)) {
    signal(SIGUSR1, signal_handler);
    auto lifetime_policy = mem_planner_->BestLifetimePolicy();
    lifetime_policy->Dump();

    alignment_ = lifetime_policy->Alignment();
    alignment_offset_ = lifetime_policy->AlignmentOffset();

    auto policy_large_bins = lifetime_policy->GetLargeBins();
    for (auto rit = policy_large_bins.rbegin();
        rit != policy_large_bins.rend(); ++rit) {
      auto bin_info = rit->second;
      auto bin = new Bin(bin_info->BlockSize(), bin_info->ChunkSize(),
          bin_info->Alignment(), bin_info->VBlocks(), 
          sub_allocator_.get(), this);
      large_lifetime_bins_.emplace(rit->first, bin);
    }
    auto policy_bins = lifetime_policy->GetBins();
    large_bin_index_ = policy_bins.size();
    lifetime_bins_.resize(large_bin_index_);

    // create bigger bin first
    for (auto it = policy_bins.rbegin(); it != policy_bins.rend();
        ++it) {
      Bin* bin = nullptr;
      if ((*it)->BlockSize() > 0 || (*it)->VBlocks().size() > 0) {
        bin = new Bin((*it)->BlockSize(), (*it)->ChunkSize(),
            (*it)->Alignment(), (*it)->VBlocks(), 
            sub_allocator_.get(), this);
      }
      lifetime_bins_[(*it)->BinIndex()] = bin;
    }
    LOG(INFO) << "TensorPoolAllocator enabled";
    inited_ = true;
  }
}

void* TensorPoolAllocator::AllocateRaw(size_t alignment,
    size_t num_bytes) {
  if (SmallAlloc(num_bytes)) {
    auto header_size = std::max(sizeof(LightHeader), alignment);
    auto total = num_bytes + header_size;
    auto ptr = sub_allocator_->Alloc(alignment, total);
    return SetLightHeader(ptr, total, header_size);
  }

  if (unlikely(stats_)) {
    return BigAllocateStatistic(alignment, num_bytes);
  } else {
    return BigAllocate(alignment, num_bytes);
  }
}

void TensorPoolAllocator::DeallocateRaw(void* ptr) {
  auto light_header = GetLightHeader(ptr);
  if (light_header != nullptr) {
    auto header_size = light_header->header_size;
    auto raw_ptr = static_cast<char*>(ptr) - header_size;
    // LightHeader not record allocation size
    // Free interface ignore the freed num_bytes
    sub_allocator_->Free(raw_ptr, 0);
    return;
  }

  auto header = GetHeader(ptr);
  BigDeallocate(header);
}

TensorPoolAllocator::Bin* TensorPoolAllocator::GetBin(
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

TensorPoolAllocator::Bin::Bin(size_t len,
    size_t chunk_size, size_t alignment,
    std::vector<VirtualAllocBlock*>& vblocks,
    SubAllocator* sub_allocator, TensorPoolAllocator* tp) :
  buffer_(len, chunk_size, alignment, sub_allocator),
  virtual_buffer_(vblocks, tp), sub_allocator_(sub_allocator) {
}

void* TensorPoolAllocator::Bin::Allocate(size_t total, size_t header_size) {
  auto ptr = buffer_.Allocate();
  if (ptr != nullptr) {
    return SetHeader(ptr, total, header_size, (void*)this, nullptr);
  } 
  auto info = virtual_buffer_.Allocate();
  if (info.first != nullptr) {
    return SetHeader(info.first, total, header_size, (void*)this,
        info.second);
  }
  return nullptr;
}

void TensorPoolAllocator::Bin::Deallocate(Header* header) {
  if (header->internal_bin == nullptr) {
    buffer_.Deallocate(header->raw_ptr);
  } else {
    virtual_buffer_.Deallocate(header->raw_ptr,
        (TensorPoolAllocator::Bin*)(header->internal_bin));
  }
}

void* TensorPoolAllocator::Bin::AllocateRaw() {
  return buffer_.Allocate();
}

void TensorPoolAllocator::Bin::DeallocateRaw(void* p) {
  buffer_.Deallocate(p);
}

TensorPoolAllocator::Buffer::Buffer(size_t len, size_t chunk_size,
    size_t alignment, SubAllocator* sub_allocator) {
  auto rounded_bytes = RoundedBytes(chunk_size, alignment);
  auto buffer_size = rounded_bytes * len;
  auto p = static_cast<char*>(sub_allocator->Alloc(alignment, buffer_size));
  begin_ = p;
  end_ = p + buffer_size;

  for (auto i = 0; i < len; ++i) {
    buffer_.emplace(p + rounded_bytes *i);
  }
}

void* TensorPoolAllocator::Buffer::Allocate() {
  std::lock_guard<spin_lock> l(lock_);
  if (unlikely(buffer_.empty())) {
    return nullptr;
  }
  auto ptr = buffer_.top();
  buffer_.pop();
  return ptr;
}

void TensorPoolAllocator::Buffer::Deallocate(void* p) {
  if (unlikely(p < begin_ || p > end_)) {
    LOG(WARNING) << "probabaly memory corruption!!";
  }
  std::lock_guard<spin_lock> l(lock_);
  buffer_.emplace(p);
}

TensorPoolAllocator::VirtualBuffer::VirtualBuffer(
    std::vector<VirtualAllocBlock*>& vblocks,
    TensorPoolAllocator* tp) {
  for (auto vblock : vblocks) {
    auto bin_index = vblock->BinIndex();
    auto internal_bin = tp->GetBin(bin_index);
    if (internal_bin == nullptr) {
      LOG(WARNING) << "logic error or not allocate correctly";
    }
    internal_bins_.emplace(internal_bin);
  }
}

std::pair<void*, TensorPoolAllocator::Bin*>
TensorPoolAllocator::VirtualBuffer::Allocate() {
  std::lock_guard<spin_lock> l(lock_);
  if (unlikely(internal_bins_.empty())) {
    return std::make_pair(nullptr, nullptr);
  }
  auto internal_bin = internal_bins_.top();
  auto ptr = internal_bin->AllocateRaw();
  if (ptr == nullptr) {
    // todo: need to loop as much as internal_bins
    // not only just one
    return std::make_pair(nullptr, nullptr);
  }
  internal_bins_.pop();
  return std::make_pair(ptr, internal_bin);
}

void TensorPoolAllocator::VirtualBuffer::Deallocate(
    void* p, TensorPoolAllocator::Bin* bin) {
  std::lock_guard<spin_lock> l(lock_);
  bin->DeallocateRaw(p);
  internal_bins_.emplace(bin);
}

void TensorPoolAllocator::DumpStats() {
  if (stats_) {
    double hit_rate = (double)hit_counter_ /
      (hit_counter_ + missed_counter_ + null_bin_counter_);
    LOG(INFO) << "If you're TensorFlow user, "
      << "please ignore following debugging statistic."
      << "TensorPoolAllocator Statistic:"
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
    LOG(INFO) << "Start counting TensorPoolAllocator";
  }
}

void* TensorPoolAllocator::BigAllocate(size_t alignment,
    size_t num_bytes) {
  auto header_size = std::max(sizeof(Header), alignment);
  auto total = num_bytes + header_size; 
  if (!inited_.load()) {
    mem_planner_->TrackAllocate(alignment, total);
    auto ptr = sub_allocator_->Alloc(alignment, total);
    return SetDefaultHeader(!inited_.load(), ptr, total, header_size);
  }

  auto id = Index(total, alignment_, alignment_offset_);
  if (unlikely(id < 0)) {
    auto ptr = sub_allocator_->Alloc(alignment, total);
    return SetDefaultHeader(!inited_.load(), ptr, total, header_size);
  }

  auto b = GetBin(id);
  if (unlikely(b == nullptr)) {
    auto ptr = sub_allocator_->Alloc(alignment, total);
    return SetDefaultHeader(!inited_.load(), ptr, total, header_size);
  }

  auto ptr = b->Allocate(total, header_size);
  if (likely(ptr != nullptr)) {
    return ptr;
  }
  ptr = sub_allocator_->Alloc(alignment, total);
  return SetDefaultHeader(!inited_.load(), ptr, total, header_size);
}

// unlikely execute this path which do some atomic operations
void* TensorPoolAllocator::BigAllocateStatistic(size_t alignment,
    size_t num_bytes) {
  auto header_size = std::max(sizeof(Header), alignment);
  auto total = num_bytes + header_size; 
  if (!inited_.load()) {
    mem_planner_->TrackAllocate(alignment, total);
    auto ptr = sub_allocator_->Alloc(alignment, total);
    return SetDefaultHeader(!inited_.load(), ptr, total, header_size);
  }

  auto id = Index(total, alignment_, alignment_offset_);
  if (unlikely(id < 0)) {
    auto ptr = sub_allocator_->Alloc(alignment, total);
    return SetDefaultHeader(!inited_.load(), ptr, total, header_size);
  }

  auto b = GetBin(id);
  if (unlikely(b == nullptr)) {
    ++null_bin_counter_;
    auto ptr = sub_allocator_->Alloc(alignment, total);
    return SetDefaultHeader(!inited_.load(), ptr, total, header_size);
  }

  auto ptr = b->Allocate(total, header_size);
  if (likely(ptr != nullptr)) {
    ++hit_counter_;
    return ptr;
  }
  ++missed_counter_;
  ptr = sub_allocator_->Alloc(alignment, total);
  return SetDefaultHeader(!inited_.load(), ptr, total, header_size);
}

void TensorPoolAllocator::BigDeallocate(Header* header) {
  auto ptr = header->raw_ptr;
  auto num_bytes = header->total_size;
  if (!inited_.load()) {
    mem_planner_->TrackDeallocate(header);
    sub_allocator_->Free(ptr, num_bytes);
    return;
  }
  
  if (header->bin == nullptr) {
    sub_allocator_->Free(ptr, num_bytes);
    return;
  }

  auto bin = (Bin*) (header->bin);
  bin->Deallocate(header);
}

class TensorPoolAllocatorFactory : public AllocatorFactory {
 public:
  Allocator* CreateAllocator() override { return new TensorPoolAllocator; }
  
  SubAllocator* CreateSubAllocator(int numa_node) override {
    return new TensorPoolSubAllocator(new TensorPoolAllocator);
  }

 private:
  class TensorPoolSubAllocator : public SubAllocator {
   public:
    explicit TensorPoolSubAllocator(TensorPoolAllocator* allocator)
      : SubAllocator({}, {}), allocator_(allocator) {}

    void* Alloc(size_t alignment, size_t num_bytes) override {
      return allocator_->AllocateRaw(alignment, num_bytes);
    }
    
    void Free(void* ptr, size_t num_bytes) override {
      allocator_->DeallocateRaw(ptr);
    }

   private:
    TensorPoolAllocator* allocator_;
  };
};

REGISTER_MEM_ALLOCATOR("TensorPoolAllocator", 300, TensorPoolAllocatorFactory);
} // tensorflow
