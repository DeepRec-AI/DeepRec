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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_DRAM_LEVELDB_STORAGE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_DRAM_LEVELDB_STORAGE_H_

#include "tensorflow/core/framework/embedding/leveldb_kv.h"
#include "tensorflow/core/framework/embedding/multi_tier_storage.h"

namespace tensorflow {
template <class V>
class ValuePtr;

template <class K, class V>
class EmbeddingVar;

namespace embedding {
template<typename K, typename V>
class DramLevelDBStore : public MultiTierStorage<K, V> {
 public:
  DramLevelDBStore(const StorageConfig& sc, Allocator* alloc,
      LayoutCreator<V>* lc, const std::string& name)
      : alloc_(alloc), layout_creator_(lc),
        MultiTierStorage<K, V>(sc, name) {
    dram_kv_ = new LocklessHashMap<K, V>();
    leveldb_ = new LevelDBKV<K, V>(sc.path);
    if (sc.embedding_config.steps_to_live != 0) {
      dram_policy_ = new GlobalStepShrinkPolicy<K, V>(dram_kv_, alloc_,
          sc.embedding_config.slot_num + 1);
      leveldb_policy_ = new GlobalStepShrinkPolicy<K, V>(leveldb_, alloc_,
          sc.embedding_config.slot_num + 1);
    } else if (sc.embedding_config.l2_weight_threshold != -1.0) {
      dram_policy_ =
          new L2WeightShrinkPolicy<K, V>(
              sc.embedding_config.l2_weight_threshold,
              sc.embedding_config.primary_emb_index,
              Storage<K, V>::GetOffset(sc.embedding_config.primary_emb_index),
              dram_kv_, alloc_,
              sc.embedding_config.slot_num + 1);
      leveldb_policy_ =
          new L2WeightShrinkPolicy<K, V>(
              sc.embedding_config.l2_weight_threshold,
              sc.embedding_config.primary_emb_index,
              Storage<K, V>::GetOffset(sc.embedding_config.primary_emb_index),
              leveldb_, alloc_,
              sc.embedding_config.slot_num + 1);
    } else {
      dram_policy_ = nullptr;
      leveldb_policy_ = nullptr;
    }

    MultiTierStorage<K, V>::kvs_.emplace_back(
        KVInterfaceDescriptor<K, V>(dram_kv_, alloc_, dram_mu_, dram_policy_));
    MultiTierStorage<K, V>::kvs_.emplace_back(
        KVInterfaceDescriptor<K, V>(leveldb_, alloc_,
                                    leveldb_mu_, leveldb_policy_));
  }

  ~DramLevelDBStore() override {
    MultiTierStorage<K, V>::ReleaseValues(
        {std::make_pair(dram_kv_, alloc_)});
    if (dram_policy_ != nullptr) delete dram_policy_;
    if (leveldb_policy_ != nullptr) delete leveldb_policy_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(DramLevelDBStore);

  Status Get(K key, ValuePtr<V>** value_ptr) override {
    Status s = dram_kv_->Lookup(key, value_ptr);
    if (!s.ok()) {
      s = leveldb_->Lookup(key, value_ptr);
    }
    return s;
  }

  void Insert(K key, ValuePtr<V>* value_ptr) override {
    LOG(FATAL)<<"Unsupport Insert(K, ValuePtr<V>*) in DramLevelDBStore.";
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
    s = leveldb_->Lookup(key, value_ptr);
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
    leveldb_->Remove(key);
    return Status::OK();
  }

  bool IsUseHbm() override {
    return false;
  }

  bool IsSingleHbm() override {
    return false;
  }

  bool IsUsePersistentStorage() override {
    /*The return value is set to false temporarily,
      because the corresponding interface is not implemented.*/
    return false;
  }

  void iterator_mutex_lock() override {
    leveldb_mu_.lock();
  }

  void iterator_mutex_unlock() override {
    leveldb_mu_.unlock();
  }

  int64 Size() const override {
    int64 total_size = dram_kv_->Size();
    total_size += leveldb_->Size();
    return total_size;
  }

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<ValuePtr<V>*>* value_ptr_list) override {
    {
      mutex_lock l(dram_mu_);
      TF_CHECK_OK(dram_kv_->GetSnapshot(key_list, value_ptr_list));
    }
    {
      mutex_lock l(leveldb_mu_);
      TF_CHECK_OK(leveldb_->GetSnapshot(key_list, value_ptr_list));
    }
    return Status::OK();
  }

 protected:
  void SetTotalDims(int64 total_dims) override {
    leveldb_->SetTotalDims(total_dims);
  }

 private:
  KVInterface<K, V>* dram_kv_;
  KVInterface<K, V>* leveldb_;
  Allocator* alloc_;
  ShrinkPolicy<K, V>* dram_policy_;
  ShrinkPolicy<K, V>* leveldb_policy_;
  LayoutCreator<V>* layout_creator_;
  mutex dram_mu_; //must be locked before leveldb_mu_ is locked
  mutex leveldb_mu_;
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_DRAM_LEVELDB_STORAGE_H_
