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
#include "tensorflow/core/framework/embedding/intra_thread_copy_id_allocator.h"

namespace tensorflow {

namespace {
const static std::vector<int64> default_seeds = {
 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
};
}

template<typename K, typename V, typename EV>
class BloomFilterPolicy : public FilterPolicy<K, V, EV> {
 using FilterPolicy<K, V, EV>::ev_;
 using FilterPolicy<K, V, EV>::config_;

 public:
  BloomFilterPolicy(const EmbeddingConfig& config, EV* ev,
                    embedding::FeatureDescriptor<V>* feat_desc)
      : feat_desc_(feat_desc),
        FilterPolicy<K, V, EV>(config, ev) {
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

  Status Lookup(K key, V* val, const V* default_value_ptr,
      const V* default_value_no_permission) override {
    void* value_ptr = nullptr;
    Status s = ev_->LookupKey(key, &value_ptr);
    if (s.ok()) {
      V* mem_val = feat_desc_->GetEmbedding(value_ptr, config_.emb_index);
      memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
    } else {
      memcpy(val, default_value_no_permission, sizeof(V) * ev_->ValueLen());
    }
    return Status::OK();
  }

#if GOOGLE_CUDA
  void BatchLookup(const EmbeddingVarContext<GPUDevice>& ctx,
                   const K* keys, V* output,
                   int64 num_of_keys,
                   V* default_value_ptr,
                   V* default_value_no_permission) override {
    std::vector<void*> value_ptr_list(num_of_keys, nullptr);
    ev_->BatchLookupKey(ctx, keys, value_ptr_list.data(), num_of_keys);
    std::vector<V*> embedding_ptr(num_of_keys, nullptr);
    auto do_work = [this, value_ptr_list, &embedding_ptr,
                    default_value_ptr, default_value_no_permission]
        (int64 start, int64 limit) {
      for (int i = start; i < limit; i++) {
        void* value_ptr = value_ptr_list[i];
        if (value_ptr != nullptr) {
          embedding_ptr[i] =
              feat_desc_->GetEmbedding(value_ptr, config_.emb_index);
        } else {
          embedding_ptr[i] = default_value_no_permission;
        }
      }
    };
    auto worker_threads = ctx.worker_threads;
    Shard(worker_threads->num_threads,
          worker_threads->workers, num_of_keys,
          1000, do_work);
    auto stream = ctx.compute_stream;
    auto event_mgr = ctx.event_mgr;
    ev_->CopyEmbeddingsToBuffer(
        output, num_of_keys, embedding_ptr.data(),
        stream, event_mgr, ctx.gpu_device);
  }

  void BatchLookupOrCreateKey(const EmbeddingVarContext<GPUDevice>& ctx,
                              const K* keys, void** value_ptrs_list,
                              int64 num_of_keys) {
    int num_worker_threads = ctx.worker_threads->num_threads;
    std::vector<std::vector<K>> lookup_or_create_ids(num_worker_threads);
    std::vector<std::vector<int>>
        lookup_or_create_cursor(num_worker_threads);
    std::vector<std::vector<void*>>
        lookup_or_create_ptrs(num_worker_threads);
    IntraThreadCopyIdAllocator thread_copy_id_alloc(num_worker_threads);
    std::vector<std::list<int64>>
        not_found_cursor_list(num_worker_threads + 1);
    uint64 main_thread_id = Env::Default()->GetCurrentThreadId();

    auto do_work = [this, keys, value_ptrs_list,
                    &lookup_or_create_ids,
                    &lookup_or_create_ptrs,
                    &lookup_or_create_cursor,
                    main_thread_id,
                    &thread_copy_id_alloc]
         (int64 start, int64 limit) {
      int copy_id =
          thread_copy_id_alloc.GetCopyIdOfThread(main_thread_id);
      for (int i = start; i < limit; i++) {
        if (GetBloomFreq(keys[i]) >= config_.filter_freq) {
          lookup_or_create_ids[copy_id].emplace_back(keys[i]);
          lookup_or_create_ptrs[copy_id].emplace_back(value_ptrs_list[i]);
          lookup_or_create_cursor[copy_id].emplace_back(i);
        } else {
          AddFreq(keys[i], 1);
        }
      }
    };
    auto worker_threads = ctx.worker_threads;
    Shard(worker_threads->num_threads,
          worker_threads->workers, num_of_keys,
          1000, do_work);

    std::vector<K> total_ids(num_of_keys);
    std::vector<void*> total_ptrs(num_of_keys);
    std::vector<int> total_cursors(num_of_keys);
    int num_of_admit_id = 0;
    for (int i = 0; i < num_worker_threads; i++) {
      if (lookup_or_create_ids[i].size() > 0) {
        memcpy(total_ids.data() + num_of_admit_id,
               lookup_or_create_ids[i].data(),
               sizeof(K) * lookup_or_create_ids[i].size());
        memcpy(total_ptrs.data() + num_of_admit_id,
               lookup_or_create_ptrs[i].data(),
               sizeof(void*) * lookup_or_create_ptrs[i].size());
        memcpy(total_cursors.data() + num_of_admit_id,
               lookup_or_create_cursor[i].data(),
               sizeof(int) * lookup_or_create_cursor[i].size());
        num_of_admit_id += lookup_or_create_ids[i].size();
      }
    }

    ev_->BatchLookupOrCreateKey(ctx, total_ids.data(), total_ptrs.data(),
                                num_of_keys, not_found_cursor_list);
    for (int i = 0; i < total_ptrs.size(); i++) {
      value_ptrs_list[total_cursors[i]] = total_ptrs[i];
    }
  }
#endif //GOOGLE_CUDA

  void LookupOrCreate(K key, V* val, const V* default_value_ptr,
                      void** value_ptr, int count,
                      const V* default_value_no_permission) override {
    if (GetBloomFreq(key) >= config_.filter_freq) {
      bool is_filter = true;
      TF_CHECK_OK(LookupOrCreateKey(key, value_ptr, &is_filter, count));
      V* mem_val = feat_desc_->GetEmbedding(*value_ptr, config_.emb_index);
      memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
    } else {
      AddFreq(key, count);
      memcpy(val, default_value_no_permission, sizeof(V) * ev_->ValueLen());
    }
  }

  Status LookupOrCreateKey(K key, void** value_ptr,
      bool* is_filter, int64 count) override {
    *value_ptr = nullptr;
    if ((GetFreq(key, *value_ptr) + count) >= config_.filter_freq) {
      Status s = ev_->LookupKey(key, value_ptr);
      if (!s.ok()) {
        *value_ptr = feat_desc_->Allocate();
        feat_desc_->SetDefaultValue(*value_ptr, key);
        ev_->storage()->Insert(key, value_ptr);
        s = Status::OK();
      }
      *is_filter = true;
      feat_desc_->AddFreq(*value_ptr, count);
    } else {
      *is_filter = false;
      AddFreq(key, count);
    }
    return Status::OK();
  }

  int64 GetFreq(K key, void* val) override {
    return GetBloomFreq(key);
  }

  int64 GetFreq(K key) override {
    return GetBloomFreq(key);
  }

  void* GetBloomCounter() const {
    return bloom_counter_;
  }

  bool is_admit(K key, void* value_ptr) override {
    if (value_ptr == nullptr) {
      return false;
    } else {
      return GetFreq(key, value_ptr) >= config_.filter_freq;
    }
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

  Status Restore(int64 key_num, int bucket_num, int64 partition_id,
                 int64 partition_num, int64 value_len, bool is_filter,
                 bool to_dram, bool is_incr, RestoreBuffer& restore_buff) override {
    K* key_buff = (K*)restore_buff.key_buffer;
    V* value_buff = (V*)restore_buff.value_buffer;
    int64* version_buff = (int64*)restore_buff.version_buffer;
    int64* freq_buff = (int64*)restore_buff.freq_buffer;
    if (to_dram) {
      LOG(FATAL)<<"BloomFilter dosen't support ImportToDRAM";
      return Status::OK();
    }

    for (auto i = 0; i < key_num; ++i) {
      // this can describe by graph(Mod + DynamicPartition),
      // but memory waste and slow
      if (*(key_buff + i) % bucket_num % partition_num != partition_id) {
        LOG(INFO) << "skip EV key:" << *(key_buff + i);
        continue;
      }
      void* value_ptr = nullptr;
      int64 new_freq = freq_buff[i];
      int64 import_version = -1;
      if (config_.steps_to_live != 0 || config_.record_version) {
        import_version = version_buff[i];
      }
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
        ev_->storage()->Import(key_buff[i],
            value_buff + i * ev_->ValueLen(),
            new_freq, import_version, config_.emb_index);
      }
    }
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
  embedding::FeatureDescriptor<V>* feat_desc_;
  std::vector<int64> seeds_;
};
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_BLOOM_FILTER_POLICY_H_

