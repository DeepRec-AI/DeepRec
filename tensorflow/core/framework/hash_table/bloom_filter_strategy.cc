/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/hash_table/bloom_filter_strategy.h"

namespace tensorflow {

namespace {

const static std::vector<int64> kHashSeeds = {
  2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
  31, 37, 41, 43, 47, 53, 59, 61, 67
};

#define mix(h) ({                \
  (h) ^= (h) >> 23;              \
  (h) *= 0x2127599bf4325c37ULL;  \
  (h) ^= (h) >> 47;              \
})

uint64_t FastHash64(const void* buf, size_t len, uint64_t seed) {
  const uint64_t    m = 0x880355f21e6d1965ULL;
  const uint64_t *pos = (const uint64_t *)buf;
  const uint64_t *end = pos + (len / 8);
  const unsigned char *pos2;
  uint64_t h = seed ^ (len * m);
  uint64_t v;
  
  while (pos != end) {
    v  = *pos++;
    h ^= mix(v);
    h *= m;
  }
  
  pos2 = (const unsigned char*)pos;
  v = 0;
  
  switch (len & 7) {
  case 7: v ^= (uint64_t)pos2[6] << 48;
  case 6: v ^= (uint64_t)pos2[5] << 40;
  case 5: v ^= (uint64_t)pos2[4] << 32;
  case 4: v ^= (uint64_t)pos2[3] << 24;
  case 3: v ^= (uint64_t)pos2[2] << 16;
  case 2: v ^= (uint64_t)pos2[1] << 8;
  case 1: v ^= (uint64_t)pos2[0];
    h ^= mix(v);
    h *= m;
  }
  
  return mix(h);
}

class ScopedTimer {
 public:
  ScopedTimer(const std::string& name) : name_(name) {
    start_ = std::chrono::high_resolution_clock::now();
  }
  ~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> diff = end - start_;
    LOG(INFO) << "time of " << name_ << " : " << diff.count() << "ns";
  }
 private:
  std::string name_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

}  // namespace

BloomFilterAdmitStrategy::BloomFilterAdmitStrategy(int64 minimum_frequency,
                                                   int64 num_hash_func,
                                                   DataType dtype,
                                                   const TensorShape& shape,
                                                   int64 slice_offset,
                                                   int64 max_slice_size)
  : minimum_frequency_(minimum_frequency), 
    num_hash_func_(num_hash_func),
    dtype_(dtype),
    shape_(shape),
    slice_offset_(slice_offset),
    max_slice_size_(max_slice_size),
    bucket_(nullptr) {
  segment_size_ = shape_.dim_size(1);
  bucket_ = new int8[shape.num_elements() * DataTypeSize(dtype)] {};
  LOG(INFO) << "segment size: " << segment_size_
      << ", slice: " << shape_.dim_size(0)
      << ", slice_offset: " << slice_offset_
      << ", max_slice_size: " << max_slice_size_;
  GenerateSeeds();
  switch (dtype) {
    case DT_UINT8:
      max_freq_ = static_cast<int64>(std::numeric_limits<uint8>::max());
      break;
    case DT_UINT16:
      max_freq_ = static_cast<int64>(std::numeric_limits<uint16>::max());
      break;
    case DT_UINT32:
      max_freq_ = static_cast<int64>(std::numeric_limits<uint32>::max());
      break;
    default:
      LOG(FATAL) << "Not support data type " << dtype_;
      break;
  }
}

BloomFilterAdmitStrategy::~BloomFilterAdmitStrategy() {
  if (bucket_) {
    delete[] bucket_;
  }
}

bool BloomFilterAdmitStrategy::Admit(int64 key) {
  return AdmitInternal(key, kDefaultFrequency);
}

bool BloomFilterAdmitStrategy::Admit(int64 key, int64 freq) {
  return AdmitInternal(key, freq);
}

bool BloomFilterAdmitStrategy::AdmitInternal(int64 key, int64 counting) {
  //auto t = ScopedTimer("AdmitInternal");
  CHECK(seeds_.size() > 0) << "BloomFilter not initialized";
  CHECK(counting > 0) << "counting should be larger than zero";
  int64 id = (uint64_t)key % max_slice_size_ - slice_offset_;
  CHECK(id >= 0) << "invalid key slice: key=" << key <<  ",max_slice_size="
      << max_slice_size_ << ",slice_offset=" << slice_offset_;
  std::vector<uint64_t> hash_vals;
  hash_vals.reserve(seeds_.size());
  std::transform(seeds_.begin(), seeds_.end(), std::back_inserter(hash_vals),
                 [this, key](int64 seed) {
    return DefaultHashFunc(key, seed) % segment_size_;
  });
  mutex_lock lock(mu_);
  bool result = true;
  for (auto& index : hash_vals) {
    int64 k = id*segment_size_+index;
    CHECK(k < shape_.num_elements()) << "invalid k=" << k;
    int64 v= counting + Read(k);
    int64 update =  std::min(max_freq_, v);
    if (update < minimum_frequency_) {
      result = false;
    }
    Write(update, k);
  }
  return result;
}

void BloomFilterAdmitStrategy::GenerateSeeds() {
  seeds_.clear();
  seeds_.reserve(num_hash_func_);
  int64 reuse_size = std::min(num_hash_func_, static_cast<int64>(kHashSeeds.size()));
  std::copy(kHashSeeds.begin(), kHashSeeds.begin() + reuse_size, std::back_inserter(seeds_));
  int64 next_seed = seeds_.back();
  for (int64 i = reuse_size; i < num_hash_func_; ++i) {
    while (true) {
      ++next_seed;
      if (next_seed % 2 == 0) {
        continue;
      }
      bool is_prime = true;
      for (int64 j = 3; j <= std::sqrt(next_seed) + 1; j += 2) {
        if (next_seed % j == 0) {
          is_prime = false;
          break;
        }
      }
      if (is_prime) {
        break;
      }
    }
    seeds_.push_back(next_seed);
  }
}

uint64_t BloomFilterAdmitStrategy::DefaultHashFunc(int64 key, int64 seed) {
  return FastHash64(&key, sizeof(key), seed);
}

std::vector<int8> BloomFilterAdmitStrategy::Snapshot() {
  mutex_lock lock(mu_);
  std::vector<int8> ret(shape_.num_elements() * DataTypeSize(dtype_));
  int8* ptr = reinterpret_cast<int8*>(ret.data());
  std::memcpy(ptr, bucket_, sizeof(int8) * ret.size());
  return ret;
}

void BloomFilterAdmitStrategy::Restore(int64 src_beg, int64 src_length,
                                       int64 dst_beg, int64 dst_length,
                                       const std::vector<int8>& src) {
  CHECK(!(src_beg >= dst_beg + dst_length || dst_beg >= src_beg + src_length))
      << "Cannot restore from this slice: src_beg=" << src_beg
      << ", src_length=" << src_length << ", dst_beg=" << dst_beg
      << ", dst_length=" << dst_length;
  mutex_lock lock(mu_);
  int64 beg = std::max(src_beg, dst_beg);
  int64 len = std::min(src_beg + src_length, dst_beg + dst_length) - beg;
  int64 src_start = beg - src_beg;
  CHECK(src_start < src_length);
  int64 dst_start = beg - dst_beg;
  CHECK(dst_start < dst_length);
  std::memcpy(bucket_ + dst_start * segment_size_ * DataTypeSize(dtype_),
              src.data() + src_start * segment_size_ * DataTypeSize(dtype_),
              len * segment_size_ * DataTypeSize(dtype_));
}

}  // namespace tensorflow
