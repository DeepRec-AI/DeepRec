/* Copyright 2022 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
======================================================================*/

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_BLOOM_FILTER_POLICY_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_BLOOM_FILTER_POLICY_H_

#include "tensorflow/core/framework/embedding/embedding_config.h"
#include "tensorflow/core/framework/embedding/filter_policy.h"

namespace tensorflow {
namespace embedding{
template <class K, class V>
class StorageManager;
}

namespace {
const static std::vector<int64> default_seeds = {
 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
};
}

template<typename K, typename V, typename EV>
class BloomFilterPolicy : public FilterPolicy<K, V, EV> {
 public:
  BloomFilterPolicy(const EmbeddingConfig& config, EV* ev,
      embedding::StorageManager<K, V>* storage_manager) :
      config_(config), ev_(ev), storage_manager_(storage_manager) {
    switch (config_.counter_type){
      case DT_UINT64:
        VLOG(2) << "The type of bloom counter is uint64";
        bloom_counter_ = (void *)calloc(config_.num_counter, sizeof(long));
        break;
      case DT_UINT32:
        VLOG(2) << "The type of bloom counter is uint32";
        bloom_counter_ = (void *)calloc(config_.num_counter, sizeof(int));
        break;
      case DT_UINT16:
        VLOG(2) << "The type of bloom counter is uint16";
        bloom_counter_ = (void *)calloc(config_.num_counter, sizeof(int16));
        break;
      case DT_UINT8:
        VLOG(2) << "The type of bloom counter is uint8";
        bloom_counter_ = (void *)calloc(config_.num_counter, sizeof(bool));
        break;
      default:
        VLOG(2) << "defualt type of counter is uint64";
        bloom_counter_ = (void *)calloc(config_.num_counter, sizeof(long));
    }
    GenerateSeed(config.kHashFunc);
  }

  Status Lookup(EV* ev, K key, V* val, const V* default_value_ptr,
      const V* default_value_no_permission) override {
    return errors::Unimplemented("Can't use CBF filter in EV for inference.");
  }

  void LookupOrCreate(K key, V* val, const V* default_value_ptr,
                      ValuePtr<V>** value_ptr, int count,
                      const V* default_value_no_permission) override {
    if (GetBloomFreq(key) >= config_.filter_freq) {
      TF_CHECK_OK(ev_->LookupOrCreateKey(key, value_ptr));
      V* mem_val = ev_->LookupOrCreateEmb(*value_ptr, default_value_ptr);
      memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
    } else {
      AddFreq(key, count);
      memcpy(val, default_value_no_permission, sizeof(V) * ev_->ValueLen());
    }
  }

  void CopyEmbeddingsToBuffer(
      V* val_base, int64 size,
      int64 slice_elems, int64 value_len,
      V** memcpy_address) {
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** val,
      bool* is_filter) override {
    if (GetFreq(key, *val) >= config_.filter_freq) {
      *is_filter = true;
      return ev_->LookupOrCreateKey(key, val);
    }
    *is_filter = false;
    return Status::OK();
  }

  int64 GetFreq(K key, ValuePtr<V>*) override {
    return GetBloomFreq(key);
  }

  int64 GetFreq(K key) override {
    return GetBloomFreq(key);
  }

  void* GetBloomCounter() const {
    return bloom_counter_;
  }

 private:
  int64 GetBloomFreq(K key) {
    std::vector<int64> hash_val;
    for (int64 i = 0; i < config_.kHashFunc; i++) {
      hash_val.emplace_back(
          FastHash64(key, seeds_[i]) % config_.num_counter);
    }
    int64 min_freq;
    switch (config_.counter_type){
      case DT_UINT64:
        min_freq = GetMinFreq<uint64>(hash_val);
        break;
      case DT_UINT32:
        min_freq = GetMinFreq<uint32>(hash_val);
        break;
      case DT_UINT16:
        min_freq = GetMinFreq<uint16>(hash_val);
        break;
      case DT_UINT8:
        min_freq = GetMinFreq<uint8>(hash_val);
        break;
      default:
        min_freq = GetMinFreq<uint64>(hash_val);
    }
    return min_freq;
  }

#define mix(h) ({                                 \
                   (h) ^= (h) >> 23;              \
                   (h) *= 0x2127599bf4325c37ULL;  \
                   (h) ^= (h) >> 47;              \
                })

  uint64_t FastHash64(K key, uint64_t seed) {
    const uint64_t    m = 0x880355f21e6d1965ULL;

    uint64_t h = seed ^ (8 * m);
    uint64_t v;
    v = key;
    h ^= mix(v);
    h *= m;

    v = 0;
    h ^= mix(v);
    h *= m;

    return mix(h);
  }

  template<typename VBloom>
  int64 GetMinFreq(std::vector<int64> hash_val) {
    VBloom min_freq = *((VBloom*)bloom_counter_ + hash_val[0]);
    for (auto it : hash_val) {
      min_freq = std::min(*((VBloom*)bloom_counter_ + it), min_freq);
    }
    return min_freq;
  }

  template<typename VBloom>
  void SetMinFreq(std::vector<int64> hash_val, int64 freq) {
    for (auto it : hash_val) {
      *((VBloom*)bloom_counter_ + it) = freq;
    }
  }

  void SetBloomFreq(K key, int64 freq) {
    std::vector<int64> hash_val;
    for (int64 i = 0; i < config_.kHashFunc; i++) {
      hash_val.emplace_back(
          FastHash64(key, seeds_[i]) % config_.num_counter);
    }
    switch (config_.counter_type){
      case DT_UINT64:
        SetMinFreq<uint64>(hash_val, freq);
        break;
      case DT_UINT32:
        SetMinFreq<uint32>(hash_val, freq);
        break;
      case DT_UINT16:
        SetMinFreq<uint16>(hash_val, freq);
        break;
      case DT_UINT8:
        SetMinFreq<uint8>(hash_val, freq);
        break;
      default:
        SetMinFreq<uint64>(hash_val, freq);
    }
  }

  Status Import(RestoreBuffer& restore_buff,
                int64 key_num,
                int bucket_num,
                int64 partition_id,
                int64 partition_num,
                bool is_filter) override {
    K* key_buff = (K*)restore_buff.key_buffer;
    V* value_buff = (V*)restore_buff.value_buffer;
    int64* version_buff = (int64*)restore_buff.version_buffer;
    int64* freq_buff = (int64*)restore_buff.freq_buffer;
    for (auto i = 0; i < key_num; ++i) {
      // this can describe by graph(Mod + DynamicPartition),
      // but memory waste and slow
      if (*(key_buff + i) % bucket_num % partition_num != partition_id) {
        LOG(INFO) << "skip EV key:" << *(key_buff + i);
        continue;
      }
      ValuePtr<V>* value_ptr = nullptr;
      int64 new_freq = freq_buff[i];
      if (!is_filter) {
        if (freq_buff[i] >= config_.filter_freq) {
          SetBloomFreq(key_buff[i], freq_buff[i]);
        } else {
          SetBloomFreq(key_buff[i], config_.filter_freq);
          new_freq = config_.filter_freq;
        }
      } else {
        SetBloomFreq(key_buff[i], freq_buff[i]);
      }
      if (new_freq >= config_.filter_freq){
        ev_->CreateKey(key_buff[i], &value_ptr);
        if (config_.steps_to_live != 0 || config_.record_version) {
          value_ptr->SetStep(version_buff[i]);
        }
        if (!is_filter){
          V* v = ev_->LookupOrCreateEmb(value_ptr,
              value_buff + i * ev_->ValueLen());
        } else {
          V* v = ev_->LookupOrCreateEmb(value_ptr,
              ev_->GetDefaultValue(key_buff[i]));
        }
      }
    }
    if (ev_->IsMultiLevel() && !ev_->IsUseHbm() && config_.is_primary()) {
      ev_->UpdateCache(key_buff, key_num, version_buff, freq_buff);
    }
    return Status::OK();
  }

  Status ImportToDram(RestoreBuffer& restore_buff,
                int64 key_num,
                int bucket_num,
                int64 partition_id,
                int64 partition_num,
                bool is_filter,
                V* default_values) override {
    LOG(FATAL)<<"BloomFilter dosen't support ImportToDRAM";
    return Status::OK();
  }

  void AddFreq(K key) {
    std::vector<int64> hash_val;
    for (int64 i = 0; i < config_.kHashFunc; i++) {
      hash_val.emplace_back(
          FastHash64(key, seeds_[i]) % config_.num_counter);
    }

    for (auto it : hash_val){
      switch (config_.counter_type){
        case DT_UINT64:
          if (*((uint64*)bloom_counter_ + it) < config_.filter_freq)
            __sync_fetch_and_add((uint64*)bloom_counter_ + it, 1);
          break;
        case DT_UINT32:
          if (*((uint32*)bloom_counter_ +it) < config_.filter_freq)
            __sync_fetch_and_add((uint32*)bloom_counter_ + it, 1);
          break;
        case DT_UINT16:
          if (*((uint16*)bloom_counter_ +it) < config_.filter_freq)
            __sync_fetch_and_add((uint16*)bloom_counter_ + it, 1);
          break;
        case DT_UINT8:
          if (*((uint8*)bloom_counter_ + it) < config_.filter_freq)
            __sync_fetch_and_add((uint8*)bloom_counter_ + it, 1);
          break;
        default:
          if (*((uint64*)bloom_counter_ + it) < config_.filter_freq)
            __sync_fetch_and_add((uint64*)bloom_counter_ + it, 1);
      }
    }
  }

  void AddFreq(K key, int64 count) {
    std::vector<int64> hash_val;
    for (int64 i = 0; i < config_.kHashFunc; i++) {
      hash_val.emplace_back(
          FastHash64(key, seeds_[i]) % config_.num_counter);
    }

    for (auto it : hash_val){
      switch (config_.counter_type){
        case DT_UINT64:
          if (*((uint64*)bloom_counter_ + it) < config_.filter_freq)
            __sync_fetch_and_add((uint64*)bloom_counter_ + it, count);
          break;
        case DT_UINT32:
          if (*((uint32*)bloom_counter_ +it) < config_.filter_freq)
            __sync_fetch_and_add((uint32*)bloom_counter_ + it, count);
          break;
        case DT_UINT16:
          if (*((uint16*)bloom_counter_ +it) < config_.filter_freq)
            __sync_fetch_and_add((uint16*)bloom_counter_ + it, count);
          break;
        case DT_UINT8:
          if (*((uint8*)bloom_counter_ + it) < config_.filter_freq)
            __sync_fetch_and_add((uint8*)bloom_counter_ + it, count);
          break;
        default:
          if (*((uint64*)bloom_counter_ + it) < config_.filter_freq)
            __sync_fetch_and_add((uint64*)bloom_counter_ + it, count);
      }
    }
  }

  void GenerateSeed(int64 kHashFunc) {
    if (kHashFunc < default_seeds.size()) {
      for (int64 i = 0; i < kHashFunc; i++) {
        seeds_.emplace_back(default_seeds[i]);
      }
    } else {
      for (int64 i = 0; i < default_seeds.size(); i++) {
        seeds_.emplace_back(default_seeds[i]);
      }
      int64 last_seed = 98;
      for (int64 i = default_seeds.size(); i < kHashFunc; i++) {
        for (int64 j = last_seed; ; j++) {
          if (j % 2 == 0)
            continue;
          bool is_prime = true;
          for (int64 k = 0; k <= std::sqrt(j) + 1; k++) {
            if (j % k == 0)
              is_prime = false;
          }
          if (is_prime) {
            seeds_.emplace_back(j);
            last_seed = j;
            break;
          }
        }
      }
    }
  }

 private:
  void* bloom_counter_;
  EmbeddingConfig config_;
  EV* ev_;
  std::vector<int64> seeds_;
  embedding::StorageManager<K, V>* storage_manager_;
};
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_BLOOM_FILTER_POLICY_H_

