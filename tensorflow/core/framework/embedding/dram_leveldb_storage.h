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

    MultiTierStorage<K, V>::kvs_.emplace_back(
        std::make_pair(dram_kv_, alloc_));
    MultiTierStorage<K, V>::kvs_.emplace_back(
        std::make_pair(leveldb_, alloc_));
  }

  ~DramLevelDBStore() override {
    MultiTierStorage<K, V>::ShutdownEvictionThread();
    MultiTierStorage<K, V>::ReleaseValues(
        {std::make_pair(dram_kv_, alloc_)});
  }

  Status Get(K key, ValuePtr<V>** value_ptr) override {
    Status s = dram_kv_->Lookup(key, value_ptr);
    if (!s.ok()) {
      s = leveldb_->Lookup(key, value_ptr);
    }
    return s;
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size, bool &need_copyback) override {
    need_copyback = false;
    return GetOrCreate(key, value_ptr, size);
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size) override {
    Status s = dram_kv_->Lookup(key, value_ptr);
    if (s.ok()) {
      return s;
    }
    s = leveldb_->Lookup(key, value_ptr);
    if (s.ok()) {
      return s;
    }

    *value_ptr = layout_creator_->Create(alloc_, size);
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

  int64 Size() const override {
    int64 total_size = dram_kv_->Size();
    total_size += leveldb_->Size();
    return total_size;
  }

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<ValuePtr<V>*>* value_ptr_list) override {
    TF_CHECK_OK(dram_kv_->GetSnapshot(key_list, value_ptr_list));
    TF_CHECK_OK(leveldb_->GetSnapshot(key_list, value_ptr_list));
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
  LayoutCreator<V>* layout_creator_;
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_DRAM_LEVELDB_STORAGE_H_
