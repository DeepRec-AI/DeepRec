#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_SSD_STORAGE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_SSD_STORAGE_H_

#if GOOGLE_CUDA
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
    ReleaseValues({std::make_pair(hbm_kv_, gpu_alloc_),
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

  void Insert(const std::vector<K>& keys,
              ValuePtr<V>** value_ptrs) override{
    for (size_t i = 0; i < keys.size(); i++) {
      do {
        Status s = hbm_kv_->Insert(keys[i], value_ptrs[i]);
        if (s.ok()) {
          break;
        } else {
          (value_ptrs[i])->Destroy(gpu_alloc_);
          delete value_ptrs[i];
        }
      } while (!(hbm_kv_->Lookup(keys[i], &value_ptrs[i])).ok());
    }
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
    (*value_ptr)->SetPtr(mem_pool_->Allocate());
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

  void CopyBackToGPU(int total, const K* keys,
      const std::list<int64>& copyback_cursor,
      V** memcpy_address, size_t value_len,
      ValuePtr<V> **gpu_value_ptrs, V* memcpy_buffer_gpu) override {
    auto memcpy_buffer_cpu = (V*)malloc(total * value_len * sizeof(V));
    V* dev_value_address = (V*)gpu_alloc_->AllocateRaw(
                               Allocator::kAllocatorAlignment,
                               total * value_len
                                   * sizeof(V));
    int64 i = 0;
    auto it = copyback_cursor.cbegin();
    for ( ; it != copyback_cursor.cend(); ++it, ++i) {
      ValuePtr<V>* gpu_value_ptr = layout_creator_->Create(gpu_alloc_, value_len);
      gpu_value_ptr->SetPtr(dev_value_address + i * value_len);
      //Get cursor and destroy flag
      int64 j = *it & 0x0fffffffffffffff;
      bool destroy_flag = (*it >> 63) & 0x1;
      //Copy Header Info
      memcpy((char *)gpu_value_ptr->GetPtr(),
             (char *)memcpy_address[j] - sizeof(FixedLengthHeader),
             sizeof(FixedLengthHeader));
      V* cpu_data_address = memcpy_address[j];
      V* gpu_data_address = gpu_value_ptr->GetValue(0, 0);
      memcpy(memcpy_buffer_cpu + i * value_len,
            cpu_data_address, value_len * sizeof(V));
      if (destroy_flag) {
        cpu_alloc_->DeallocateRaw((char *)memcpy_address[j]
                                   - sizeof(FixedLengthHeader));
      }
      gpu_value_ptrs[i] = gpu_value_ptr;
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

  void ReleaseValues(
      const std::vector<std::pair<KVInterface<K, V>*,
                        Allocator*>>& kvs) {
    for (int i = 0; i < kvs.size(); i++) {
      std::vector<K> key_list;
      std::vector<ValuePtr<V>*> value_ptr_list;
      kvs[i].first->GetSnapshot(&key_list, &value_ptr_list);
      if (i == 0) {
        delete mem_pool_;
      }
      for (auto value_ptr : value_ptr_list) {
        if (i != 0)
          value_ptr->Destroy(kvs[i].second);
        delete value_ptr;
      }
    }
  }

  Status DramToSsdBatchCommit(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) {
    MultiTierStorage<K, V>::ReleaseValuePtrs(dram_value_ptr_out_of_date_,
                                             cpu_alloc_);
    mutex_lock l(ssd_mu_);
    mutex_lock l1(dram_mu_);

    dram_cache_->add_to_rank(keys.data(), keys.size());
    int64 dram_count = dram_cache_->size();
    if (dram_count > dram_capacity_) {
      int k_size = dram_count - dram_capacity_;
      constexpr int DramEvictionSize = 10000;
      k_size = std::min(k_size, DramEvictionSize);
      K dram_evic_ids[DramEvictionSize];
      size_t true_size = dram_cache_->get_evic_ids(dram_evic_ids, k_size);
      ValuePtr<V>* value_ptr;
      for (int64 i = 0; i < true_size; ++i) {
        if (dram_kv_->Lookup(dram_evic_ids[i], &value_ptr).ok()) {
          TF_CHECK_OK(ssd_kv_->Commit(dram_evic_ids[i], value_ptr));
          TF_CHECK_OK(dram_kv_->Remove(dram_evic_ids[i]));
          dram_value_ptr_out_of_date_.emplace_back(value_ptr);
        }
      }
    }
    return Status::OK();
  }

  void BatchEviction() override {
    constexpr int EvictionSize = 10000;
    K evic_ids[EvictionSize];
    if (!MultiTierStorage<K, V>::ready_eviction_) {
      return;
    }
    mutex_lock l(hbm_mu_);
    mutex_lock l1(dram_mu_);

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
          keys.emplace_back(evic_ids[i]);
          value_ptrs.emplace_back(value_ptr);
        }
      }
      dram_kv_->BatchCommit(keys, value_ptrs);
      mem_pool_->Deallocate(value_ptrs);
      for (auto it : keys) {
        TF_CHECK_OK(hbm_kv_->Remove(it));
      }
      MultiTierStorage<K, V>::eviction_manager_->Schedule(
        [this, keys, value_ptrs]() {
          DramToSsdBatchCommit(keys, value_ptrs);
        }
      );
    }
  }

  void CreateMemoryPool(Allocator* alloc,
                        int64 value_len,
                        int64 block_size) override {
    mem_pool_ = new EmbeddingMemoryPool<V>(alloc, value_len, block_size);
  }

  void AllocateMemory(
      const std::vector<ValuePtr<V>*>& value_ptr_list) override {
    for (auto it : value_ptr_list) {
      it->SetPtr(mem_pool_->Allocate());
    }
  }

 protected:
  void SetTotalDims(int64 total_dims) override {
    dram_kv_->SetTotalDims(total_dims);
    ssd_kv_->SetTotalDims(total_dims);
  }

  ValuePtr<V>* CopyToGpuValuePtr(ValuePtr<V>* cpu_ptr, int64 size) {
    auto gpu_value_ptr = layout_creator_->Create(gpu_alloc_, size);
    gpu_value_ptr->SetPtr(mem_pool_->Allocate());
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
  EmbeddingMemoryPool<V>* mem_pool_;
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

#endif  // GOOGLE_CUDA
#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_SSD_STORAGE_H_
