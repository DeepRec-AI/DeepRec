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
              size_t alloc_len, bool to_dram = false) override {
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

  Status Save(
      const string& tensor_name,
      const string& prefix,
      BundleWriter* writer,
      const EmbeddingConfig& emb_config,
      ShrinkArgs& shrink_args,
      int64 value_len,
      V* default_value) override {
    dram_->Save(tensor_name, prefix, writer, emb_config,
                shrink_args, value_len, default_value);

    ssd_hash_->Save(tensor_name, prefix, writer, emb_config,
                    shrink_args, value_len, default_value);

    return Status::OK();
  }

  Status RestoreSSD(int64 emb_index, int64 emb_slot_num, int64 value_len,
                    const std::string& ssd_emb_file_name, EmbeddingVar<K, V>* ev,
                    RestoreSSDBuffer<K>& restore_buff) override {
    int64 alloc_len = Storage<K, V>::ComputeAllocLen(value_len);
    std::map<int64, int64> file_id_map;
    for (int64 i = 0; i < restore_buff.num_of_files; i++) {
      file_id_map[restore_buff.file_list_buf[i]] = i;
    }

    ssd_hash_->CopyEmbFilesFromCkpt(restore_buff.file_list_buf,
                                    restore_buff.invalid_record_count_list_buf,
                                    restore_buff.record_count_list_buf,
                                    restore_buff.num_of_files,
                                    ssd_emb_file_name);

    ssd_hash_->Import(restore_buff.key_list_buf,
                      restore_buff.key_file_id_list_buf,
                      restore_buff.key_offset_list_buf,
                      restore_buff.num_of_keys,
                      file_id_map);
    return Status::OK();
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
