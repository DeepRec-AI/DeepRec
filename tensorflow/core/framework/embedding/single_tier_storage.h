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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SINGLE_TIER_STORAGE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SINGLE_TIER_STORAGE_H_

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/cpu_hash_map_kv.h"
#include "tensorflow/core/framework/embedding/globalstep_shrink_policy.h"
#if GOOGLE_CUDA
#include "tensorflow/core/framework/embedding/gpu_hash_map_kv.h"
#endif // GOOGLE_CUDA
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/l2weight_shrink_policy.h"
#include "tensorflow/core/framework/embedding/leveldb_kv.h"
#include "tensorflow/core/framework/embedding/ssd_hash_kv.h"
#include "tensorflow/core/framework/embedding/storage_config.h"
#include "tensorflow/core/framework/embedding/storage.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
template <class K, class V>
class EmbeddingVar;

template <class K>
struct SsdRecordDescriptor;

namespace embedding {
template<class K, class V>
class DramSsdHashStorage;

template<class K, class V>
class DramPmemStorage;

template<class K, class V>
class DramLevelDBStore;

#if GOOGLE_CUDA
template<class K, class V>
class HbmDramStorage;

template<class K, class V>
class HbmDramSsdStorage;
#endif

template<typename K, typename V>
class SingleTierStorage : public Storage<K, V> {
 public:
  SingleTierStorage(const StorageConfig& sc,
      KVInterface<K, V>* kv, FeatureDescriptor<V>* feat_desc)
      : kv_(kv), feat_desc_(feat_desc),
        Storage<K, V>(sc) {
    if (sc.embedding_config.steps_to_live != 0) {
      shrink_policy_ =
          new GlobalStepShrinkPolicy<K, V>(
              sc.embedding_config.steps_to_live,
              feat_desc_,
              kv_);
    } else if (sc.embedding_config.l2_weight_threshold != -1.0) {
      shrink_policy_ =
          new L2WeightShrinkPolicy<K, V>(
              sc.embedding_config.l2_weight_threshold,
              sc.embedding_config.primary_emb_index,
              feat_desc_,
              kv_);
    } else {
      shrink_policy_ = new NonShrinkPolicy<K, V>();
    }
  }
  
  ~SingleTierStorage() override {
    mutex_lock l(Storage<K, V>::mu_);
    std::vector<K> key_list;
    std::vector<void*> value_ptr_list;
    kv_->GetSnapshot(&key_list, &value_ptr_list);
    for (auto value_ptr : value_ptr_list) {
      feat_desc_->Deallocate(value_ptr);
    }
    delete kv_;
    delete shrink_policy_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(SingleTierStorage);

  Status Get(K key, void** value_ptr) override {
    return kv_->Lookup(key, value_ptr);
  }

  Status Contains(K key) override {
    return kv_->Contains(key);
  }

  virtual void CreateAndInsert(K key, void** value_ptr,
      bool to_dram=false) override {
    do {
      *value_ptr = feat_desc_->Allocate();
      Status s = kv_->Insert(key, *value_ptr);
      if (s.ok()) {
        break;
      } else {
        feat_desc_->Deallocate(*value_ptr);
      }
    } while (!(kv_->Lookup(key, value_ptr)).ok());
  }

  virtual void Insert(K key, void** value_ptr) override {
    do {
      Status s = kv_->Insert(key, *value_ptr);
      if (s.ok()) {
        break;
      } else {
        feat_desc_->Deallocate(*value_ptr);
      }
    } while (!(kv_->Lookup(key, value_ptr)).ok());
  }

  Status GetOrCreate(K key, void** value_ptr) override {
    Status s = kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      return s;
    }

    *value_ptr = feat_desc_->Allocate();
    s = kv_->Insert(key, *value_ptr);
    if (s.ok()) {
      return s;
    }
    // Insert Failed, key already exist
    feat_desc_->Deallocate(*value_ptr);
    return kv_->Lookup(key, value_ptr);
  }
 
  Status Remove(K key) override {
    return kv_->Remove(key);
  }

  int64 Size() const override {
    return kv_->Size();
  }
  
  int64 Size(int level) const override {
    if (level > 0) {
      LOG(FATAL) << "Unsupport level>0 in SingleTierStorage.";
    }
    return kv_->Size();
  }

  int64 CacheSize() const override {
    LOG(FATAL) << "Unsupport cachesize in SingleTierStorage.";
    return 0;
  }

  int LookupTier(K key) const override {
    Status s = kv_->Contains(key);
    return (s.ok()) ? 0 : -1;
  }

  void CopyEmbeddingsFromCPUToGPU(
      int total, const K* keys,
      const std::list<int64>& copyback_cursor,
      V** memcpy_address, size_t value_len,
      void **gpu_value_ptrs,
      V* memcpy_buffer_gpu,
      se::Stream* compute_stream,
      EventMgr* event_mgr,
      const DeviceBase::CpuWorkerThreads* worker_threads) override {
    LOG(FATAL) << "Unsupport CopyEmbeddingsFromCPUToGPU in SingleTierStorage.";
  };

  BatchCache<K>* Cache() override {
    LOG(FATAL) << "Unsupport Cache in SingleTierStorage.";
    return nullptr;
  }

  void InitCache(embedding::CacheStrategy cache_strategy) override {
    LOG(FATAL) << "Unsupport InitCache in SingleTierStorage.";
  }

  virtual Status BatchCommit(const std::vector<K>& keys,
      const std::vector<void*>& value_ptrs) override {
    LOG(FATAL) << "Unsupport BatchCommit in Storage: "
               << typeid(this).name();
    return Status::OK();
  }

  virtual Status Commit(K keys, const void* value_ptr) {
     LOG(FATAL) << "Unsupport Commit in Storage: "
                << typeid(this).name();
    return Status::OK();
  }

  Status Eviction(K* evict_ids, int64 evict_size) override {
    LOG(FATAL) << "Unsupport Eviction in SingleTierStorage.";
    return Status::OK();
  }

  void CreateEmbeddingMemoryPool(
      Allocator* alloc,
      int64 value_len,
      int64 block_size) override {
    return;
  }

  virtual void Import(K key, V* value,
                      int64 freq, int64 version,
                      int emb_index) override {}

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<void*>* value_ptr_list) override {
    mutex_lock l(Storage<K, V>::mu_);
    return kv_->GetSnapshot(key_list, value_ptr_list);
  }

  Status GetShardedSnapshot(
      std::vector<std::vector<K>>& key_list,
      std::vector<std::vector<void*>>& value_ptr_list,
      int partition_id, int partition_nums) override {
    mutex_lock l(Storage<K, V>::mu_);
    return kv_->GetShardedSnapshot(key_list, value_ptr_list,
                                   partition_id, partition_nums);
  }

  Status Save(
      const std::string& tensor_name,
      const std::string& prefix,
      BundleWriter* writer,
      const EmbeddingConfig& emb_config,
      ShrinkArgs& shrink_args,
      int64 value_len,
      V* default_value) override {
    std::vector<void*> value_ptr_list;
    std::vector<K> key_list_tmp;
    TF_CHECK_OK(kv_->GetSnapshot(
        &key_list_tmp, &value_ptr_list));

    if (emb_config.is_primary()) {
      Shrink(key_list_tmp, value_ptr_list, shrink_args, value_len);
    }
    TF_CHECK_OK((Storage<K, V>::SaveToCheckpoint(
        tensor_name, writer,
        emb_config,
        value_len, default_value,
        key_list_tmp,
        value_ptr_list,
        SingleTierStorage<K, V>::feat_desc_)));
    return Status::OK();
  }

  bool IsMultiLevel() override {
    return false;
  }

  bool IsUseHbm() override {
    return false;
  }

  bool IsSingleHbm() override {
    return false;
  }

  bool IsUsePersistentStorage() override {
    return false;
  }

  void Schedule(std::function<void()> fn) override {
    LOG(FATAL) << "Unsupport Schedule in SingleTierStorage.";
  }

  void UpdateValuePtr(K key, void* new_value_ptr,
                      void* old_value_ptr) override {
    kv_->UpdateValuePtr(key, new_value_ptr, old_value_ptr);
  }

 protected:
  virtual void* CreateValuePtr() {
    return feat_desc_->Allocate();
  }

  virtual void DestroyValuePtr(void* value_ptr) {
    feat_desc_->Deallocate(value_ptr);
  }

  FeatureDescriptor<V>* feature_descriptor() {
    return feat_desc_;
  }

  virtual Status RestoreFeatures(int64 key_num, int bucket_num, int64 partition_id,
                                 int64 partition_num, int64 value_len, bool is_filter,
                                 bool is_incr, const EmbeddingConfig& emb_config, 
                                 const Eigen::GpuDevice* device,
                                 FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
                                 RestoreBuffer& restore_buff) override {
    Status s = filter->Restore(key_num, bucket_num, partition_id,
                               partition_num, value_len, is_filter,
                               false/*to_dram*/, is_incr, restore_buff);
    return s;
  }

  void CleanUp() override {
    std::vector<K> key_list;
    std::vector<void*> value_ptr_list;
    kv_->GetSnapshot(&key_list, &value_ptr_list);

    int list_size = key_list.size();
    for (int i = 0; i < list_size; i++) {
      kv_->Remove(key_list[i]);
      feat_desc_->Deallocate(value_ptr_list[i]);
    }
  }
 
 protected:
  virtual void Shrink(std::vector<K>& key_list,
                      std::vector<void*>& value_ptr_list,
                      ShrinkArgs& shrink_args,
                      int64 value_len) {
    mutex_lock l(Storage<K, V>::mu_);
    shrink_args.value_len = value_len;
    shrink_policy_->Shrink(
        key_list,
        value_ptr_list,
        shrink_args);
  }

 protected:
  KVInterface<K, V>* kv_;
  ShrinkPolicy<K, V>* shrink_policy_;
  Allocator* alloc_;
  FeatureDescriptor<V>* feat_desc_;
};

template<typename K, typename V>
class DramStorage : public SingleTierStorage<K, V> {
 public:
  DramStorage(const StorageConfig& sc,
      FeatureDescriptor<V>* feat_desc)
      : SingleTierStorage<K, V>(sc, new LocklessHashMap<K, V>(feat_desc), feat_desc) {}

  ~DramStorage() override {}

  Status BatchCommit(const std::vector<K>& keys,
      const std::vector<void*>& value_ptrs) {
    return SingleTierStorage<K, V>::kv_->BatchCommit(keys, value_ptrs);
  }

  Status TryInsert(K key, void* value_ptr) {
    return SingleTierStorage<K, V>::kv_->Insert(key, value_ptr);
  }

  Status Commit(K keys, const void* value_ptr) override{
    return SingleTierStorage<K, V>::kv_->Commit(keys, value_ptr);
  }

  void Import(K key, V* value,
              int64 freq, int64 version,
              int emb_index) override {
    void* value_ptr = SingleTierStorage<K, V>::feat_desc_->Allocate(freq);
    SingleTierStorage<K, V>::Insert(key, &value_ptr);
    SingleTierStorage<K, V>::feat_desc_->SetValue(value_ptr, emb_index, value);
    SingleTierStorage<K, V>::feat_desc_->SetFreq(value_ptr, freq);
    SingleTierStorage<K, V>::feat_desc_->UpdateVersion(value_ptr, version);
  }
 
  TF_DISALLOW_COPY_AND_ASSIGN(DramStorage);
 public:
  friend class DramSsdHashStorage<K, V>;
  friend class DramPmemStorage<K, V>;
  friend class DramLevelDBStore<K, V>;
#if GOOGLE_CUDA
  friend class HbmDramStorage<K, V>;
  friend class HbmDramSsdStorage<K, V>;
#endif
 protected:
  void Shrink(std::vector<K>& key_list,
              std::vector<void*>& value_ptr_list,
              ShrinkArgs& shrink_args,
              int64 value_len) override {
    SingleTierStorage<K, V>::Shrink(
        key_list,
        value_ptr_list,
        shrink_args,
        value_len);
  }
};

#if GOOGLE_CUDA
template<typename K, typename V>
class HbmStorage : public SingleTierStorage<K, V> {
 public:
  HbmStorage(const StorageConfig& sc, Allocator* gpu_allocator,
      FeatureDescriptor<V>* feat_desc) : SingleTierStorage<K, V>(
        sc, new GPUHashMapKV<K, V>(
            sc.embedding_config, gpu_allocator), feat_desc) {
  }
  ~HbmStorage() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(HbmStorage);

  bool IsSingleHbm() override {
    return true;
  }

  void SetValueLen(int64 value_len) override {
    SingleTierStorage<K, V>::kv_->SetValueLen(value_len);
  }

  void BatchLookupOrCreate(const K* key, V* val, V* default_v,
      int32 default_v_num,
      size_t n, const Eigen::GpuDevice& device) override {
    SingleTierStorage<K, V>::kv_->BatchLookupOrCreate(key, val,
                                                      default_v,
                                                      default_v_num,
                                                      n, device);
  }

  void BatchLookupOrCreateKeys(const K* key, int32* item_idxs, size_t n,
      const Eigen::GpuDevice& device) override {
    SingleTierStorage<K, V>::kv_->BatchLookupOrCreateKeys(key, n, item_idxs, device);
  }

  void BatchLookup(const Eigen::GpuDevice& device, const K* keys, V* val,
                   size_t n, const V* default_v) override {
    SingleTierStorage<K, V>::kv_->BatchLookup(device, keys, val, n, default_v);
  }

  Status Save(
      const string& tensor_name,
      const string& prefix,
      BundleWriter* writer,
      const EmbeddingConfig& emb_config,
      ShrinkArgs& shrink_args,
      int64 value_len,
      V* default_value) override {
    std::vector<V*> value_ptr_list;
    std::vector<K> key_list_tmp;
    GPUHashMapKV<K, V>* gpu_kv =
        dynamic_cast<GPUHashMapKV<K, V>*>(SingleTierStorage<K, V>::kv_);
    gpu_kv->GetSnapshot(&key_list_tmp, &value_ptr_list, emb_config);

    TF_CHECK_OK((Storage<K, V>::SaveToCheckpoint(
        tensor_name, writer,
        value_len,
        key_list_tmp,
        value_ptr_list)));

    if (value_ptr_list.size() > 0) {
      TypedAllocator::Deallocate(
          cpu_allocator(), value_ptr_list[0],
          value_ptr_list.size() * value_len);
    }
    return Status::OK();
  }

  GPUHashTable<K, V>* HashTable() override {
    return SingleTierStorage<K, V>::kv_->HashTable();
  }

  void CleanUp() override {
    LOG(FATAL) << "Function [CleanUp] of HbmStorage is not implemented.";
  }
 protected:
  Status RestoreFeatures(int64 key_num, int bucket_num, int64 partition_id,
                         int64 partition_num, int64 value_len, bool is_filter,
                         bool is_incr, const EmbeddingConfig& emb_config,
                         const Eigen::GpuDevice* device,
                         FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
                         RestoreBuffer& restore_buff) override {
    K* key_buff = (K*)restore_buff.key_buffer;
    V* value_buff = (V*)restore_buff.value_buffer;
    std::vector<K> key_import;
    std::vector<V> value_import;
    for (auto i = 0; i < key_num; ++i) {
      if (*(key_buff + i) % bucket_num % partition_num != partition_id) {
        LOG(INFO) << "skip EV key:" << *(key_buff + i);
        continue;
      }
      key_import.emplace_back(*(key_buff + i));
      auto row_offset = value_buff + i * value_len;
      for (int j = 0; j < value_len; j++) {
        value_import.emplace_back(*(row_offset + j));
      }
    }
    GPUHashMapKV<K, V>* gpu_kv =
        dynamic_cast<GPUHashMapKV<K, V>*>(SingleTierStorage<K, V>::kv_);
    gpu_kv->Import(key_import, value_import, device, emb_config);
    return Status::OK();
  }
};

template<typename K, typename V>
class HbmStorageWithCpuKv: public SingleTierStorage<K, V> {
 public:
  HbmStorageWithCpuKv(const StorageConfig& sc,
      FeatureDescriptor<V>* feat_desc) : SingleTierStorage<K, V>(
        sc, new LocklessHashMap<K, V>(feat_desc), feat_desc) {
  }

  ~HbmStorageWithCpuKv() override {}

  Status TryInsert(K key, void* value_ptr) {
    return SingleTierStorage<K, V>::kv_->Insert(key, value_ptr);
  }

  void CleanUp() override {
    LOG(FATAL) << "Function [CleanUp] of HbmStorageWithCPUKv is not implemented.";
  }

 public:
  friend class HbmDramStorage<K, V>;
  friend class HbmDramSsdStorage<K, V>;
 protected:
  void Shrink(std::vector<K>& key_list,
              std::vector<void*>& value_ptr_list,
              ShrinkArgs& shrink_args,
              int64 value_len) override {
    SingleTierStorage<K, V>::Shrink(
        key_list,
        value_ptr_list,
        shrink_args,
        value_len);
  }
};
#endif // GOOGLE_CUDA

template<typename K, typename V>
class PmemMemkindStorage : public SingleTierStorage<K, V> {
 public:
  PmemMemkindStorage(const StorageConfig& sc,
      FeatureDescriptor<V>* feat_desc) : SingleTierStorage<K, V>(
          sc, new LocklessHashMap<K, V>(feat_desc), feat_desc) {
  }
  ~PmemMemkindStorage() override {}

  void CleanUp() override {
    LOG(FATAL) << "Function [CleanUp] of PmemMemkindStorage is not implemented.";
  }

  TF_DISALLOW_COPY_AND_ASSIGN(PmemMemkindStorage);
};

template<typename K, typename V>
class PmemLibpmemStorage : public SingleTierStorage<K, V> {
 public:
  PmemLibpmemStorage(const StorageConfig& sc,
      FeatureDescriptor<V>* feat_desc) : SingleTierStorage<K, V>(
          sc, new LocklessHashMap<K, V>(feat_desc), feat_desc) {
  }
  ~PmemLibpmemStorage() override {}

  Status Commit(K keys, const void* value_ptr) {
    return SingleTierStorage<K, V>::kv_->Commit(keys, value_ptr);
  }

  void CleanUp() override {
    LOG(FATAL) << "Function [CleanUp] of PmemLibpmemStorage is not implemented.";
  }

  TF_DISALLOW_COPY_AND_ASSIGN(PmemLibpmemStorage);
 
 protected:
  friend class DramPmemStorage<K, V>;
  void Shrink(std::vector<K>& key_list,
              std::vector<void*>& value_ptr_list,
              ShrinkArgs& shrink_args,
              int64 value_len) override {
    SingleTierStorage<K, V>::Shrink(
        key_list,
        value_ptr_list,
        shrink_args,
        value_len);
  }
};

template<typename K, typename V>
class LevelDBStore : public SingleTierStorage<K, V> {
 public:
  LevelDBStore(const StorageConfig& sc,
      FeatureDescriptor<V>* feat_desc) : SingleTierStorage<K, V>(
          sc, new LevelDBKV<K, V>(sc.path, feat_desc), feat_desc) {
  }
  ~LevelDBStore() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(LevelDBStore);

  Status Commit(K keys, const void* value_ptr) {
    return SingleTierStorage<K, V>::kv_->Commit(keys, value_ptr);
  }

  embedding::ValueIterator<V>* GetValueIterator(
      const std::vector<K>& key_list,
      int64 emb_index, int64 value_len) {
    LevelDBKV<K, V>* leveldb_kv =
        reinterpret_cast<LevelDBKV<K, V>*>(SingleTierStorage<K, V>::kv_);
    return new DBValueIterator<K, V>(
        key_list, emb_index, value_len,
        leveldb_kv, SingleTierStorage<K, V>::feat_desc_);
  }

  void CleanUp() override {
    LOG(FATAL) << "Function [CleanUp] of LevelDBStorage is not implemented.";
  }

 public:
  friend class DramLevelDBStore<K, V>;
};

template<typename K, typename V>
class SsdHashStorage : public SingleTierStorage<K, V> {
 public:
  SsdHashStorage(const StorageConfig& sc,
      FeatureDescriptor<V>* feat_desc) : SingleTierStorage<K, V>(
          sc, new SSDHashKV<K, V>(sc.path, feat_desc), feat_desc) {
  }
  ~SsdHashStorage() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(SsdHashStorage);

  Status Commit(K keys, const void* value_ptr) {
    return SingleTierStorage<K, V>::kv_->Commit(keys, value_ptr);
  }

  Status Save(
      const string& tensor_name,
      const string& prefix,
      BundleWriter* writer,
      const EmbeddingConfig& emb_config,
      ShrinkArgs& shrink_args,
      int64 value_len,
      V* default_value) override {
    if (emb_config.is_primary()) {
      SSDHashKV<K, V>* ssd_kv =
          reinterpret_cast<SSDHashKV<K, V>*>(SingleTierStorage<K, V>::kv_);
      SsdRecordDescriptor<K> ssd_rec_desc;
      {
        mutex_lock l(Storage<K, V>::mu_);
        ssd_kv->SetSsdRecordDescriptor(&ssd_rec_desc);
      }
      ssd_rec_desc.GenerateCheckpoint(prefix, tensor_name);
    }
    return Status::OK();
  }

  void Import(K* key_list, int64* key_file_id_list,
      int64* key_offset_list, int64 num_of_keys,
      std::map<int64, int64>& file_id_map) {
    SSDHashKV<K, V>* ssd_kv =
        reinterpret_cast<SSDHashKV<K, V>*>(SingleTierStorage<K, V>::kv_);

    ssd_kv->Import(key_list, key_file_id_list,
                   key_offset_list, num_of_keys,
                   file_id_map);
  }

  void CopyEmbFilesFromCkpt(
      int64* file_list, int64* invalid_record_count_list,
      int64* record_count_list, int64 num_of_files,
      const std::string& ssd_emb_file_name) {
    SSDHashKV<K, V>* ssd_kv =
        reinterpret_cast<SSDHashKV<K, V>*>(SingleTierStorage<K, V>::kv_);

    ssd_kv->CopyEmbFilesFromCkpt(
        file_list, invalid_record_count_list,
        record_count_list, num_of_files,
        ssd_emb_file_name);
  }

  void SetSsdRecordDescriptor(SsdRecordDescriptor<K>* ssd_rec_desc) {
    SSDHashKV<K, V>* ssd_kv =
        reinterpret_cast<SSDHashKV<K, V>*>(SingleTierStorage<K, V>::kv_);
    ssd_kv->SetSsdRecordDescriptor(ssd_rec_desc);
  }

  void CleanUp() override {
    LOG(FATAL) << "Function [CleanUp] of SsdHashStorage is not implemented.";
  }
 public:
  friend class DramSsdHashStorage<K, V>;
#if GOOGLE_CUDA
  friend class HbmDramSsdStorage<K, V>;
#endif

 protected:
  void Init() override {
    dynamic_cast<SSDHashKV<K, V>*>(
        SingleTierStorage<K, V>::kv_)->Init();
  }
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_H_
