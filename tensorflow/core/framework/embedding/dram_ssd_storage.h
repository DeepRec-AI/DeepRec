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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_DRAM_SSD_STORAGE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_DRAM_SSD_STORAGE_H_

#include "tensorflow/core/framework/embedding/ssd_hash_kv.h"
#include "tensorflow/core/framework/embedding/cpu_hash_map_kv.h"
#include "tensorflow/core/framework/embedding/multi_tier_storage.h"
#include "tensorflow/core/framework/embedding/single_tier_storage.h"

namespace tensorflow {
template <class V>
class ValuePtr;

template <class K, class V>
class EmbeddingVar;

namespace embedding {
template<typename K, typename V>
class DramSsdHashStorage : public MultiTierStorage<K, V> {
 public:
  DramSsdHashStorage(const StorageConfig& sc, Allocator* alloc,
      LayoutCreator<V>* lc, const std::string& name)
      : MultiTierStorage<K, V>(sc, name) {
    dram_= new DramStorage<K, V>(sc, alloc, lc, new LocklessHashMap<K, V>());
    ssd_hash_ = new SsdHashStorage<K, V>(sc, alloc, lc);
  }

  ~DramSsdHashStorage() override {
    MultiTierStorage<K, V>::DeleteFromEvictionManager();
    delete dram_;
    delete ssd_hash_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(DramSsdHashStorage);

  Status Get(K key, ValuePtr<V>** value_ptr) override {
    Status s = dram_->Get(key, value_ptr);
    if (s.ok()) {
      return s;
    }
    s = ssd_hash_->Get(key, value_ptr);
    if(s.ok()) {
      s = dram_->TryInsert(key, *value_ptr);
      if (s.ok()) {
        return s;
      }
      //Insert Failed, the key is already in Dram;
      ssd_hash_->DestroyValuePtr(*value_ptr);
      return dram_->Get(key, value_ptr);
    }
    return s;
  }

  void Insert(K key, ValuePtr<V>* value_ptr) override {
    LOG(FATAL)<<"Unsupport Insert(K, ValuePtr<V>*) in DramSsdHashStorage.";
  }

  void Insert(K key, ValuePtr<V>** value_ptr,
              size_t alloc_len) override {
    dram_->Insert(key, value_ptr, alloc_len);
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size, CopyBackFlag &need_copyback) override {
    LOG(FATAL)<<"GetOrCreate(K key, ValuePtr<V>** value_ptr, "
              <<"size_t size, CopyBackFlag &need_copyback) "
              <<"in DramSsdStorage can not be called.";
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size) override {
    Status s = dram_->Get(key, value_ptr);
    if (s.ok()) {
      return s;
    }
    s = ssd_hash_->Get(key, value_ptr);
    if(s.ok()) {
      s = dram_->TryInsert(key, *value_ptr);
      if (s.ok()) {
        return s;
      }
      //Insert Failed, the key is already in Dram;
      ssd_hash_->DestroyValuePtr(*value_ptr);
      return dram_->Get(key, value_ptr);
    }
    dram_->Insert(key, value_ptr, size);
    return Status::OK();
  }

  Status Remove(K key) override {
    dram_->Remove(key);
    ssd_hash_->Remove(key);
    return Status::OK();
  }

  int64 Size() const override {
    int64 total_size = dram_->Size();
    total_size += ssd_hash_->Size();
    return total_size;
  }

  int64 Size(int level) const override {
    if (level == 0) {
      return dram_->Size();
    } else if (level == 1) {
      return ssd_hash_->Size();
    } else {
      return -1;
    }
  }

  int LookupTier(K key) const override {
    Status s = dram_->Contains(key);
    if (s.ok())
      return 0;
    s = ssd_hash_->Contains(key);
    if (s.ok())
      return 1;
    return -1;
  }

  bool IsUseHbm() override {
    return false;
  }

  bool IsSingleHbm() override {
    return false;
  }

  bool IsUsePersistentStorage() override {
    return true;
  }

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<ValuePtr<V>*>* value_ptr_list) override {
    {
     mutex_lock l(*(dram_->get_mutex()));
      TF_CHECK_OK(dram_->GetSnapshot(key_list, value_ptr_list));
    }
    {
      mutex_lock l(*(ssd_hash_->get_mutex()));
      TF_CHECK_OK(ssd_hash_->GetSnapshot(key_list, value_ptr_list));
    }
    return Status::OK();
  }

  Status Shrink(const ShrinkArgs& shrink_args) override {
    dram_->Shrink(shrink_args);
    ssd_hash_->Shrink(shrink_args);
    return Status::OK();
  }

  int64 GetSnapshot(std::vector<K>* key_list,
      std::vector<V* >* value_list,
      std::vector<int64>* version_list,
      std::vector<int64>* freq_list,
      const EmbeddingConfig& emb_config,
      FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
      embedding::Iterator** it) override {
    {
      mutex_lock l(*(dram_->get_mutex()));
      std::vector<ValuePtr<V>*> value_ptr_list;
      std::vector<K> key_list_tmp;
      TF_CHECK_OK(dram_->GetSnapshot(&key_list_tmp, &value_ptr_list));
      MultiTierStorage<K, V>::SetListsForCheckpoint(
          key_list_tmp, value_ptr_list, emb_config,
          key_list, value_list, version_list, freq_list);
    }
    {
      mutex_lock l(*(ssd_hash_->get_mutex()));
      *it = ssd_hash_->GetIterator();
    }
    return key_list->size();
  }

  int64 GetSnapshotWithoutFetchPersistentEmb(
      std::vector<K>* key_list,
      std::vector<V*>* value_list,
      std::vector<int64>* version_list,
      std::vector<int64>* freq_list,
      const EmbeddingConfig& emb_config,
      SsdRecordDescriptor<K>* ssd_rec_desc) override {
    {
      mutex_lock l(*(dram_->get_mutex()));
      std::vector<ValuePtr<V>*> value_ptr_list;
      std::vector<K> temp_key_list;
      TF_CHECK_OK(dram_->GetSnapshot(&temp_key_list, &value_ptr_list));
      MultiTierStorage<K, V>::SetListsForCheckpoint(
          temp_key_list, value_ptr_list, emb_config,
          key_list, value_list, version_list,
          freq_list);
    }
    {
      mutex_lock l(*(ssd_hash_->get_mutex()));
      ssd_hash_->SetSsdRecordDescriptor(ssd_rec_desc);
    }
    return key_list->size() + ssd_rec_desc->key_list.size();
  }

  void RestoreSsdHashmap(
      K* key_list, int64* key_file_id_list,
      int64* key_offset_list, int64 num_of_keys,
      int64* file_list, int64* invalid_record_count_list,
      int64* record_count_list, int64 num_of_files,
      const std::string& ssd_emb_file_name) override {
    std::map<int64, int64> file_id_map;
    for (int64 i = 0; i < num_of_files; i++) {
      file_id_map[file_list[i]] = i;
    }

    ssd_hash_->CopyEmbFilesFromCkpt(
        file_list, invalid_record_count_list,
        record_count_list, num_of_files,
        ssd_emb_file_name);

    ssd_hash_->Import(key_list, key_file_id_list,
                    key_offset_list, num_of_keys,
                    file_id_map);
  }

  Status Eviction(K* evict_ids, int64 evict_size) override {
    ValuePtr<V>* value_ptr = nullptr;
    for (int64 i = 0; i < evict_size; ++i) {
      if (dram_->Get(evict_ids[i], &value_ptr).ok()) {
        TF_CHECK_OK(ssd_hash_->Commit(evict_ids[i], value_ptr));
        TF_CHECK_OK(dram_->Remove(evict_ids[i]));
        dram_->DestroyValuePtr(value_ptr);
      }
    }
    return Status::OK();
  }

  Status EvictionWithDelayedDestroy(K* evict_ids, int64 evict_size) override {
    mutex_lock l(*(dram_->get_mutex()));
    mutex_lock l1(*(ssd_hash_->get_mutex()));
    MultiTierStorage<K, V>::ReleaseInvalidValuePtr(dram_->alloc_);
    ValuePtr<V>* value_ptr = nullptr;
    for (int64 i = 0; i < evict_size; ++i) {
      if (dram_->Get(evict_ids[i], &value_ptr).ok()) {
        TF_CHECK_OK(ssd_hash_->Commit(evict_ids[i], value_ptr));
        TF_CHECK_OK(dram_->Remove(evict_ids[i]));
        MultiTierStorage<K, V>::KeepInvalidValuePtr(value_ptr);
      }
    }
    return Status::OK();
  }

 protected:
  void SetTotalDims(int64 total_dims) override {
    ssd_hash_->SetTotalDims(total_dims);
  }

 private:
  DramStorage<K, V>* dram_ = nullptr;
  SsdHashStorage<K, V>* ssd_hash_ = nullptr;
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_DRAM_SSD_STORAGE_H_
