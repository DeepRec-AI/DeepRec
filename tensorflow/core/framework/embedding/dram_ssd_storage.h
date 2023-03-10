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
#include "tensorflow/core/framework/embedding/multi_tier_storage.h"

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
      : alloc_(alloc), layout_creator_(lc),
        MultiTierStorage<K, V>(sc, name) {
    dram_kv_ = new LocklessHashMap<K, V>();
    ssd_kv_ = new SSDHashKV<K, V>(sc.path, alloc_);
    if (sc.embedding_config.steps_to_live != 0) {
      dram_policy_ = new GlobalStepShrinkPolicy<K, V>(dram_kv_, alloc_,
          sc.embedding_config.slot_num + 1);
      ssd_policy_ = new GlobalStepShrinkPolicy<K, V>(ssd_kv_, alloc_,
         sc.embedding_config.slot_num + 1);
    } else if (sc.embedding_config.l2_weight_threshold != -1.0) {
      dram_policy_ =
          new L2WeightShrinkPolicy<K, V>(
              sc.embedding_config.l2_weight_threshold,
              sc.embedding_config.primary_emb_index,
              Storage<K, V>::GetOffset(sc.embedding_config.primary_emb_index),
              dram_kv_, alloc_,
              sc.embedding_config.slot_num + 1);
      ssd_policy_ =
          new L2WeightShrinkPolicy<K, V>(
              sc.embedding_config.l2_weight_threshold,
              sc.embedding_config.primary_emb_index,
              Storage<K, V>::GetOffset(sc.embedding_config.primary_emb_index),
              ssd_kv_, alloc_,
              sc.embedding_config.slot_num + 1);
    } else {
      dram_policy_ = nullptr;
      ssd_policy_ = nullptr;
    }

    MultiTierStorage<K, V>::kvs_.emplace_back(
        KVInterfaceDescriptor<K, V>(dram_kv_, alloc_, dram_mu_, dram_policy_));
    MultiTierStorage<K, V>::kvs_.emplace_back(
        KVInterfaceDescriptor<K, V>(ssd_kv_, alloc_, ssd_mu_, ssd_policy_));
  }

  ~DramSsdHashStorage() override {
    MultiTierStorage<K, V>::ReleaseValues(
        {std::make_pair(dram_kv_, alloc_)});
    if (dram_policy_ != nullptr) delete dram_policy_;
    if (ssd_policy_ != nullptr) delete ssd_policy_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(DramSsdHashStorage);

  Status Get(K key, ValuePtr<V>** value_ptr) override {
    Status s = dram_kv_->Lookup(key, value_ptr);
    if (!s.ok()) {
      s = ssd_kv_->Lookup(key, value_ptr);
    }
    return s;
  }

  void Insert(K key, ValuePtr<V>* value_ptr) override {
    LOG(FATAL)<<"Unsupport Insert(K, ValuePtr<V>*) in DramSsdHashStorage.";
  }

  void Insert(K key, ValuePtr<V>** value_ptr,
              int64 alloc_len) override {
    do {
      *value_ptr = layout_creator_->Create(alloc_, alloc_len);
      Status s = dram_kv_->Insert(key, *value_ptr);
      if (s.ok()) {
        break;
      } else {
        (*value_ptr)->Destroy(alloc_);
        delete *value_ptr;
      }
    } while (!(dram_kv_->Lookup(key, value_ptr)).ok());
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size, CopyBackFlag &need_copyback) override {
    need_copyback = NOT_COPYBACK;
    return GetOrCreate(key, value_ptr, size);
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size) override {
    Status s = dram_kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      return s;
    }
    s = ssd_kv_->Lookup(key, value_ptr);
    if (!s.ok()) {
      *value_ptr = layout_creator_->Create(alloc_, size);
    }
    s = dram_kv_->Insert(key, *value_ptr);
    if (s.ok()) {
      return s;
    }
    // Insert Failed, key already exist
    (*value_ptr)->Destroy(alloc_);
    delete *value_ptr;
    return dram_kv_->Lookup(key, value_ptr);
  }

  Status Remove(K key) override {
    dram_kv_->Remove(key);
    ssd_kv_->Remove(key);
    return Status::OK();
  }

  int64 Size() const override {
    int64 total_size = dram_kv_->Size();
    total_size += ssd_kv_->Size();
    return total_size;
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

  void iterator_mutex_lock() override {
    ssd_mu_.lock();
  }

  void iterator_mutex_unlock() override {
    ssd_mu_.unlock();
  }

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<ValuePtr<V>*>* value_ptr_list) override {
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

  int64 GetSnapshotWithoutFetchPersistentEmb(
      std::vector<K>* key_list,
      std::vector<V*>* value_list,
      std::vector<int64>* version_list,
      std::vector<int64>* freq_list,
      const EmbeddingConfig& emb_config,
      SsdRecordDescriptor<K>* ssd_rec_desc) override {
    {
      mutex_lock l(dram_mu_);
      std::vector<ValuePtr<V>*> value_ptr_list;
      std::vector<K> temp_key_list;
      TF_CHECK_OK(dram_kv_->GetSnapshot(&temp_key_list, &value_ptr_list));
      MultiTierStorage<K, V>::SetListsForCheckpoint(
          temp_key_list, value_ptr_list, emb_config,
          key_list, value_list, version_list,
          freq_list);
    }
    {
      mutex_lock l(ssd_mu_);
      ssd_kv_->SetSsdRecordDescriptor(ssd_rec_desc);
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

    ssd_kv_->CopyEmbFilesFromCkpt(
        file_list, invalid_record_count_list,
        record_count_list, num_of_files,
        ssd_emb_file_name);

    ssd_kv_->Import(key_list, key_file_id_list,
                    key_offset_list, num_of_keys,
                    file_id_map);
  }

 protected:
  void SetTotalDims(int64 total_dims) override {
    ssd_kv_->SetTotalDims(total_dims);
  }

 private:
  KVInterface<K, V>* dram_kv_;
  SSDHashKV<K, V>* ssd_kv_;
  Allocator* alloc_;
  ShrinkPolicy<K, V>* dram_policy_;
  ShrinkPolicy<K, V>* ssd_policy_;
  LayoutCreator<V>* layout_creator_;
  mutex dram_mu_; // must be locked before ssd_mu_ is locked
  mutex ssd_mu_;
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_DRAM_SSD_STORAGE_H_
