#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_SSD_STORAGE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_SSD_STORAGE_H_

#if GOOGLE_CUDA
#if !TENSORFLOW_USE_GPU_EV
#include "tensorflow/core/framework/embedding/lockless_hash_map_cpu.h"
#include "tensorflow/core/framework/embedding/multi_tier_storage.h"

namespace tensorflow {
template <class V>
class ValuePtr;

namespace embedding {
template<typename K, typename V>
class HbmDramSsdStorage : public MultiTierStorage<K, V> {
 public:
  HbmDramSsdStorage(const StorageConfig& sc, Allocator* gpu_alloc,
      Allocator* cpu_alloc, LayoutCreator<V>* lc, const std::string& name)
      : cpu_alloc_(cpu_alloc), gpu_alloc_(gpu_alloc),
        layout_creator_(lc), MultiTierStorage<K, V>(sc, name),
        dram_capacity_(-1) {
    hbm_kv_ = new LocklessHashMap<K, V>();
    dram_kv_ = new LocklessHashMapCPU<K, V>(gpu_alloc);
    ssd_kv_ = new SSDHashKV<K, V>(sc.path, cpu_alloc);

    MultiTierStorage<K, V>::kvs_.emplace_back(
        KVInterfaceDescriptor<K, V>(hbm_kv_, gpu_alloc_, hbm_mu_));
    MultiTierStorage<K, V>::kvs_.emplace_back(
        KVInterfaceDescriptor<K, V>(dram_kv_, cpu_alloc_, dram_mu_));
     MultiTierStorage<K, V>::kvs_.emplace_back(
        KVInterfaceDescriptor<K, V>(ssd_kv_, cpu_alloc_, ssd_mu_));
  }

  ~HbmDramSsdStorage() override {
    MultiTierStorage<K, V>::ReleaseValues(
        {std::make_pair(hbm_kv_, gpu_alloc_),
         std::make_pair(dram_kv_, cpu_alloc_)});
    delete dram_cache_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(HbmDramSsdStorage);

  void SetAllocLen(int64 value_len, int slot_num) override {
    while (Storage<K, V>::flag_.test_and_set(std::memory_order_acquire));
    // The start address of every slot should be aligned to 16 bytes,
    // otherwise a coredump will happen in the ApplyOp.
    Storage<K, V>::alloc_len_ = Storage<K, V>::ComputeAllocLen(value_len);

    int64 temp = Storage<K, V>::alloc_len_ * slot_num;
    if (temp > Storage<K, V>::total_dims_) {
      Storage<K, V>::total_dims_ = temp;
      SetTotalDims(Storage<K, V>::total_dims_);

      MultiTierStorage<K, V>::cache_capacity_ =
          Storage<K, V>::storage_config_.size[0]
          / (Storage<K, V>::total_dims_ * sizeof(V));
          
      dram_capacity_ = Storage<K, V>::storage_config_.size[1]
          / (Storage<K, V>::total_dims_ * sizeof(V));
      MultiTierStorage<K, V>::ready_eviction_ = true;
    }
    Storage<K, V>::flag_.clear(std::memory_order_release);
  }

  Status Get(K key, ValuePtr<V>** value_ptr) override {
    Status s = hbm_kv_->Lookup(key, value_ptr);
    if (!s.ok()) {
      s = dram_kv_->Lookup(key, value_ptr);
    }
    if (!s.ok()) {
      s = ssd_kv_->Lookup(key, value_ptr);
    }
    return s;
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size) override {
    Status s = hbm_kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      return s;
    }
    // Not found in HBM, Lookup in DRAM
    s = dram_kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      // copy dram value to hbm
      *value_ptr = CopyToGpuValuePtr(*value_ptr, size);
      return s;
    }
    // Not found in DRAM, Lookup in SSD
    s = ssd_kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      auto temp = CopyToGpuValuePtr(*value_ptr, size);
      *value_ptr = temp;
      delete *value_ptr;
      return s;  
    }

    *value_ptr = layout_creator_->Create(gpu_alloc_, size);
    s = hbm_kv_->Insert(key, *value_ptr);
    if (s.ok()) {
      return s;
    }
    // Insert Failed, key already exist
    (*value_ptr)->Destroy(gpu_alloc_);
    delete *value_ptr;
    return hbm_kv_->Lookup(key, value_ptr);
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size, CopyBackFlag &need_copyback) override {
    need_copyback = NOT_COPYBACK;
    Status s = hbm_kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      return s;
    }
    s = dram_kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      need_copyback = COPYBACK;
      return s;
    }
    s = ssd_kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      need_copyback = COPYBACK_AND_DESTROY;
      return s;
    }

    *value_ptr = layout_creator_->Create(gpu_alloc_, size);
    s = hbm_kv_->Insert(key, *value_ptr);
    if (s.ok()) {
      return s;
    }
    // Insert Failed, key already exist
    (*value_ptr)->Destroy(gpu_alloc_);
    delete *value_ptr;
    return hbm_kv_->Lookup(key, value_ptr);
  }

  void InitCache(embedding::CacheStrategy cache_strategy) override {
    MultiTierStorage<K, V>::InitCache(cache_strategy);
    dram_cache_ = new LRUCache<K>();
  }

  void CopyBackToGPU(int total, K* keys, int64 size, CopyBackFlag* copyback_flags,
      V** memcpy_address, size_t value_len, int *copyback_cursor,
      ValuePtr<V> **gpu_value_ptrs, V* memcpy_buffer_gpu) override {
    auto memcpy_buffer_cpu = (V*)malloc(total * value_len * sizeof(V));
    int j = 0;
    for (int i = 0; i < size;i++) {
      if (copyback_flags[i]) {
        ValuePtr<V>* gpu_value_ptr = layout_creator_->Create(gpu_alloc_, size);
        //Copy Header Info
        memcpy((char *)gpu_value_ptr->GetPtr(),
               (char *)memcpy_address[i] - sizeof(FixedLengthHeader),
               sizeof(FixedLengthHeader));
        V* cpu_data_address = memcpy_address[i];
        V* gpu_data_address = gpu_value_ptr->GetValue(0, 0);
        memcpy(memcpy_buffer_cpu + j * value_len,
            cpu_data_address, value_len * sizeof(V));
        if (copyback_flags[i] == COPYBACK_AND_DESTROY) {
          cpu_alloc_->DeallocateRaw((char *)memcpy_address[i]
                                        - sizeof(FixedLengthHeader));
        }
        copyback_cursor[j] = i;
        gpu_value_ptrs[j] = gpu_value_ptr;
        j++;
        hbm_kv_->Insert(keys[i], gpu_value_ptr);
      }
    }

    cudaMemcpy(memcpy_buffer_gpu, memcpy_buffer_cpu,
        total * value_len * sizeof(V), cudaMemcpyHostToDevice);
  }

  Status Remove(K key) override {
    hbm_kv_->Remove(key);
    dram_kv_->Remove(key);
    ssd_kv_->Remove(key);
    return Status::OK();
  }

  int64 Size() const override {
    int64 total_size = hbm_kv_->Size();
    total_size += dram_kv_->Size();
    total_size += ssd_kv_->Size();
    return total_size;
  }

  bool IsUseHbm() override {
    return true;
  }

  void iterator_mutex_lock() override {
    ssd_mu_.lock();
  }

  void iterator_mutex_unlock() override {
    ssd_mu_.unlock();
  }

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<ValuePtr<V>* >* value_ptr_list) override {
    {
      mutex_lock l(hbm_mu_);
      TF_CHECK_OK(hbm_kv_->GetSnapshot(key_list, value_ptr_list));
    }
    {
      mutex_lock l(dram_mu_);
      TF_CHECK_OK(dram_kv_->GetSnapshot(key_list, value_ptr_list));
    }
    {
      mutex_lock l(ssd_mu_);
      TF_CHECK_OK(ssd_kv_->GetSnapshot(key_list, value_ptr_list));
    }
    return Status::OK();
  }

   Status BatchCommit(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) override {
    MultiTierStorage<K, V>::ReleaseValuePtrs(dram_value_ptr_out_of_date_,
                                             cpu_alloc_);
    mutex_lock l(ssd_mu_);
    mutex_lock l1(dram_mu_);
    dram_kv_->BatchCommit(keys, value_ptrs);
    dram_cache_->add_to_rank(keys.data(), keys.size());
    int64 dram_count = dram_cache_->size();
    constexpr int DramEvictionSize = 10000;
    K dram_evic_ids[DramEvictionSize];
    if (dram_count > dram_capacity_) {
      int k_size = dram_count - dram_capacity_;
      k_size = std::min(k_size, DramEvictionSize);
      size_t true_size = dram_cache_->get_evic_ids(dram_evic_ids, k_size);
      ValuePtr<V>* value_ptr;
      for (int64 i = 0; i < true_size; ++i) {
        if (dram_kv_->Lookup(dram_evic_ids[i], &value_ptr).ok()) {
            V* tmp = value_ptr->GetValue(0, 0);
            TF_CHECK_OK(ssd_kv_->Commit(dram_evic_ids[i], value_ptr));
            TF_CHECK_OK(dram_kv_->Remove(dram_evic_ids[i]));
            dram_value_ptr_out_of_date_.emplace_back(value_ptr);
        }
      }
    }
    return Status::OK();
  }

  void BatchEviction() override{
    constexpr int EvictionSize = 10000;
    K evic_ids[EvictionSize];
    if (!MultiTierStorage<K, V>::ready_eviction_)
      return;
    mutex_lock l(hbm_mu_);
    mutex_lock l1(dram_mu_);
    //Release the memory of invlid valuetprs
    MultiTierStorage<K, V>::ReleaseInvalidValuePtr();

    int64 cache_count = MultiTierStorage<K, V>::cache_->size();
    if (cache_count > MultiTierStorage<K, V>::cache_capacity_) {
      // eviction
      int k_size = cache_count - MultiTierStorage<K, V>::cache_capacity_;
      k_size = std::min(k_size, EvictionSize);
      size_t true_size =
          MultiTierStorage<K, V>::cache_->get_evic_ids(evic_ids, k_size);
      ValuePtr<V>* value_ptr;
      std::vector<K> keys;
      std::vector<ValuePtr<V>*> value_ptrs;

      for (int64 i = 0; i < true_size; ++i) {
        if (hbm_kv_->Lookup(evic_ids[i], &value_ptr).ok()) {
          TF_CHECK_OK(hbm_kv_->Remove(evic_ids[i]));
          keys.emplace_back(evic_ids[i]);
          value_ptrs.emplace_back(value_ptr);
        }
      }
      MultiTierStorage<K, V>::eviction_manager_->Schedule(
        [this, keys, value_ptrs]() {
          BatchCommit(keys, value_ptrs);
        }
      );    
    }
  }

 protected:
  void SetTotalDims(int64 total_dims) override {
    dram_kv_->SetTotalDims(total_dims);
    ssd_kv_->SetTotalDims(total_dims);
  }

  ValuePtr<V>* CopyToGpuValuePtr(ValuePtr<V>* cpu_ptr, int64 size) {
    auto gpu_value_ptr = layout_creator_->Create(gpu_alloc_, size);
    V* cpu_data_address = cpu_ptr->GetValue(0, 0);
    V* gpu_data_address = gpu_value_ptr->GetValue(0, 0);
    cudaMemcpy(gpu_data_address, cpu_data_address,
        size * sizeof(V), cudaMemcpyHostToDevice);
    return gpu_value_ptr;
  }

 private:
  KVInterface<K, V>* hbm_kv_;
  KVInterface<K, V>* dram_kv_;
  KVInterface<K, V>* ssd_kv_;
  Allocator* gpu_alloc_;
  Allocator* cpu_alloc_;
  LayoutCreator<V>* layout_creator_;
  BatchCache<K>* dram_cache_;
  int64 dram_capacity_;
  std::vector<ValuePtr<V>*> dram_value_ptr_out_of_date_;
  mutex hbm_mu_; //must be locked before dram_mu_ and ssd_mu_ are locked;
  mutex dram_mu_; //must be locked after ssd_mu_ is locked;
  mutex ssd_mu_;
};
} // embedding
} // tensorflow

#endif  // TENSORFLOW_USE_GPU_EV
#endif  // GOOGLE_CUDA
#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_SSD_STORAGE_H_
