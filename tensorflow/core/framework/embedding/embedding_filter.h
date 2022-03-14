#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_FILTER_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_FILTER_H_

//#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "tensorflow/core/framework/embedding/embedding_config.h"

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
class EmbeddingFilter {
 public:
  virtual void LookupOrCreate(K key, V* val, const V* default_value_ptr) = 0;
  virtual void LookupOrCreateWithFreq(K key, V* val, const V* default_value_ptr) = 0;
  virtual void LookupOrCreate(K key, V* val, const V* default_value_ptr, int64 count) = 0;
  virtual Status LookupOrCreateKey(K key, ValuePtr<V>** val, bool* is_filter,
      int update_version = -1) = 0;

  virtual int64 GetFreq(K key, ValuePtr<V>* value_ptr) = 0;
  virtual int64 GetFreq(K key) = 0;
};

template<typename K, typename V, typename EV>
class BloomFilter : public EmbeddingFilter<K, V, EV> {
 public:
  BloomFilter(const EmbeddingConfig& config, EV* ev, embedding::StorageManager<K, V>* storage_manager) :
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

  void LookupOrCreate(K key, V* val, const V* default_value_ptr) override {
    ValuePtr<V>* value_ptr = nullptr;
    if (GetBloomFreq(key) >= config_.filter_freq) {
      TF_CHECK_OK(ev_->LookupOrCreateKey(key, &value_ptr));
      V* mem_val = ev_->LookupOrCreateEmb(value_ptr, default_value_ptr);
      memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
      value_ptr->Free(mem_val);
    } else {
      AddFreq(key);
      int64 default_value_dim = ev_->GetDefaultValueDim();
      V* default_value = ev_->GetDefaultValuePtr();
      if (default_value == default_value_ptr)
        memcpy(val, default_value_ptr + (key % default_value_dim) * ev_->ValueLen(), sizeof(V) * ev_->ValueLen());
      else
        memcpy(val, default_value_ptr, sizeof(V) * ev_->ValueLen());
    }
  }

  void LookupOrCreateWithFreq(K key, V* val, const V* default_value_ptr) override {
    ValuePtr<V>* value_ptr = nullptr;
    if (GetBloomFreq(key) >= config_.filter_freq) {
      TF_CHECK_OK(ev_->LookupOrCreateKey(key, &value_ptr));
      V* mem_val = ev_->LookupOrCreateEmb(value_ptr, default_value_ptr);
      memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
      value_ptr->Free(mem_val);
    } else {
      int64 default_value_dim = ev_->GetDefaultValueDim();
      V* default_value = ev_->GetDefaultValuePtr();
      if (default_value == default_value_ptr)
        memcpy(val, default_value_ptr + (key % default_value_dim) * ev_->ValueLen(), sizeof(V) * ev_->ValueLen());
      else
        memcpy(val, default_value_ptr, sizeof(V) * ev_->ValueLen());
    }
    AddFreq(key);
  }

  void LookupOrCreate(K key, V* val, const V* default_value_ptr, int64 count) override {
    ValuePtr<V>* value_ptr = nullptr;
    if (GetBloomFreq(key) >= config_.filter_freq) {
      TF_CHECK_OK(ev_->LookupOrCreateKey(key, &value_ptr));
      V* mem_val = ev_->LookupOrCreateEmb(value_ptr, default_value_ptr);
      memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
      value_ptr->Free(mem_val);
    } else {
      AddFreq(key, count);
      memcpy(val, default_value_ptr, sizeof(V) * ev_->ValueLen());
    }
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** val, bool* is_filter,
        int update_version = -1) override {
    if (GetFreq(key, *val) >= config_.filter_freq) {
      *is_filter = true;
      return ev_->LookupOrCreateKey(key, val, update_version);
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
      hash_val.emplace_back(FastHash64(key, seeds_[i]) % config_.num_counter);
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
      hash_val.push_back(FastHash64(key, seeds_[i]) % config_.num_counter);
    }
   int64 min_freq;
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

  void AddFreq(K key) {
    std::vector<int64> hash_val;
    for (int64 i = 0; i < config_.kHashFunc; i++) {
      hash_val.push_back(FastHash64(key, seeds_[i]) % config_.num_counter);
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
      hash_val.push_back(FastHash64(key, seeds_[i]) % config_.num_counter);
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
        seeds_.push_back(default_seeds[i]);
      }
    }else{
      for (int64 i = 0; i < default_seeds.size(); i++) {
        seeds_.push_back(default_seeds[i]);
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
            seeds_.push_back(j);
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

template<typename K, typename V, typename EV>
class CounterFilter : public EmbeddingFilter<K, V, EV> {
 public:
  CounterFilter(const EmbeddingConfig& config,
      EV* ev, embedding::StorageManager<K, V>* storage_manager)
       : config_(config), ev_(ev), storage_manager_(storage_manager) {
  }

  void LookupOrCreate(K key, V* val, const V* default_value_ptr) override {
    ValuePtr<V>* value_ptr = nullptr;
    TF_CHECK_OK(ev_->LookupOrCreateKey(key, &value_ptr));
    if (GetFreq(key, value_ptr) >= config_.filter_freq) {
      V* mem_val = ev_->LookupOrCreateEmb(value_ptr, default_value_ptr);
      memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
      value_ptr->Free(mem_val);
    } else {
      value_ptr->AddFreq();
      int64 default_value_dim= ev_->GetDefaultValueDim();
      V* default_value = ev_->GetDefaultValuePtr();
      if (default_value == default_value_ptr)
        memcpy(val, default_value_ptr + (key % default_value_dim) * ev_->ValueLen(), sizeof(V) * ev_->ValueLen());
      else
        memcpy(val, default_value_ptr, sizeof(V) * ev_->ValueLen());
    }
  }

  void LookupOrCreateWithFreq(K key, V* val, const V* default_value_ptr) override {
    ValuePtr<V>* value_ptr = nullptr;
    TF_CHECK_OK(ev_->LookupOrCreateKey(key, &value_ptr));
    if (GetFreq(key, value_ptr) >= config_.filter_freq) {
      V* mem_val = ev_->LookupOrCreateEmb(value_ptr, default_value_ptr);
      memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
      value_ptr->Free(mem_val);
    } else {
      int64 default_value_dim= ev_->GetDefaultValueDim();
      V* default_value = ev_->GetDefaultValuePtr();
      if (default_value == default_value_ptr)
        memcpy(val, default_value_ptr + (key % default_value_dim) * ev_->ValueLen(), sizeof(V) * ev_->ValueLen());
      else
        memcpy(val, default_value_ptr, sizeof(V) * ev_->ValueLen());
    }
    value_ptr->AddFreq();
  }

  void LookupOrCreate(K key, V* val, const V* default_value_ptr, int64 count) override {
    ValuePtr<V>* value_ptr = nullptr;
    TF_CHECK_OK(ev_->LookupOrCreateKey(key, &value_ptr));
    if (GetFreq(key, value_ptr) >= config_.filter_freq) {
      V* mem_val = ev_->LookupOrCreateEmb(value_ptr, default_value_ptr);
      memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
      value_ptr->Free(mem_val);
    } else {
      value_ptr->AddFreq(count);
      memcpy(val, default_value_ptr, sizeof(V) * ev_->ValueLen());
    }
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** val, bool* is_filter,
      int update_version = -1) override {
    Status s = ev_->LookupOrCreateKey(key, val, update_version);
    *is_filter = GetFreq(key, *val) >= config_.filter_freq;
    return s;
  }

  int64 GetFreq(K key, ValuePtr<V>* value_ptr) override {
    return value_ptr->GetFreq();
  }

  int64 GetFreq(K key) override {
    ValuePtr<V>* value_ptr = nullptr;
    TF_CHECK_OK(ev_->LookupOrCreateKey(key, &value_ptr));
    return value_ptr->GetFreq();
  }

 private:
  EmbeddingConfig config_;
  embedding::StorageManager<K, V>* storage_manager_;
  EV* ev_;
};

template<typename K, typename V, typename EV>
class NullableFilter : public EmbeddingFilter<K, V, EV> {
 public:
  NullableFilter(const EmbeddingConfig& config,
      EV* ev, embedding::StorageManager<K, V>* storage_manager)
       : config_(config), ev_(ev), storage_manager_(storage_manager) {
  }

  void LookupOrCreate(K key, V* val, const V* default_value_ptr) override {
    ValuePtr<V>* value_ptr = nullptr;
    TF_CHECK_OK(ev_->LookupOrCreateKey(key, &value_ptr));
    V* mem_val = ev_->LookupOrCreateEmb(value_ptr, default_value_ptr);
    memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
    value_ptr->Free(mem_val);
  }

  void LookupOrCreateWithFreq(K key, V* val, const V* default_value_ptr) override {
    ValuePtr<V>* value_ptr = nullptr;
    TF_CHECK_OK(ev_->LookupOrCreateKey(key, &value_ptr));
    V* mem_val = ev_->LookupOrCreateEmb(value_ptr, default_value_ptr);
    memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
    value_ptr->AddFreq();
    value_ptr->Free(mem_val);
  }

  void LookupOrCreate(K key, V* val, const V* default_value_ptr, int64 count) override {
    ValuePtr<V>* value_ptr = nullptr;
    TF_CHECK_OK(ev_->LookupOrCreateKey(key, &value_ptr));
    V* mem_val = ev_->LookupOrCreateEmb(value_ptr, default_value_ptr);
    memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
    value_ptr->Free(mem_val);
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** val, bool* is_filter,
      int update_version = -1) override {
    *is_filter = true;
    return ev_->LookupOrCreateKey(key, val, update_version);
  }

  int64 GetFreq(K key, ValuePtr<V>* value_ptr) override {
    if (storage_manager_->GetLayoutType() != LayoutType::LIGHT) {
      return value_ptr->GetFreq();
    }else {
      return 0;
    }
  }

  int64 GetFreq(K key) override {
    if (storage_manager_->GetLayoutType() != LayoutType::LIGHT) {
      ValuePtr<V>* value_ptr = nullptr;
      TF_CHECK_OK(ev_->LookupOrCreateKey(key, &value_ptr));
      return value_ptr->GetFreq();
    }else {
      return 0;
    }
  }

 private:
  EmbeddingConfig config_;
  embedding::StorageManager<K, V>* storage_manager_;
  EV* ev_;
};

class FilterFactory {
 public:
  template<typename K, typename V, typename EV>
  static EmbeddingFilter<K, V, EV>* CreateFilter(const EmbeddingConfig& config,
      EV* ev, embedding::StorageManager<K, V>* storage_manager) {
    if (config.filter_freq > 0) {
      if (config.kHashFunc != 0) {
        return new BloomFilter<K, V, EV>(config, ev, storage_manager);
      } else {
        return new CounterFilter<K, V, EV>(config, ev, storage_manager);
      }
    } else {
      return new NullableFilter<K, V, EV>(config, ev, storage_manager);
    }
  }
};

} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_FILTER_H_

