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
=======================================================================*/

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CPU_HASH_MAP_KV_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CPU_HASH_MAP_KV_H_

#include "sparsehash/dense_hash_map_lockless"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace embedding {

template <class K, class V>
class LocklessHashMap : public KVInterface<K, V> {
 public:
  LocklessHashMap(FeatureDescriptor<V>* feat_desc): feat_desc_(feat_desc) {
    hash_map_.max_load_factor(0.8);
    hash_map_.set_empty_key_and_value(
        LocklessHashMap<K, V>::EMPTY_KEY_, nullptr);
    hash_map_.set_counternum(16);
    hash_map_.set_deleted_key(LocklessHashMap<K, V>::DELETED_KEY_);
    pthread_key_create(&key_, NULL);
  }

  ~LocklessHashMap() override {
    pthread_key_delete(key_);
  }

  Status Lookup(K key, void** value_ptr) override {
    auto iter = hash_map_.find_wait_free(key);
    if (iter.first == LocklessHashMap<K, V>::EMPTY_KEY_) {
      return errors::NotFound(
          "Unable to find Key: ", key, " in LocklessHashMap.");
    } else {
      *value_ptr = iter.second;
      return Status::OK();
    }
  }

  Status Contains(K key) override {
    auto iter = hash_map_.find_wait_free(key);
    if (iter.first == LocklessHashMap<K, V>::EMPTY_KEY_) {
      return errors::NotFound(
          "Unable to find Key: ", key, " in LocklessHashMap.");
    } else {
      return Status::OK();
    }
  }

  Status Insert(K key, const void* value_ptr) override {
    auto iter = hash_map_.insert_lockless(
        std::move(std::pair<K, void*>(key,
            const_cast<void*>(value_ptr))));
    // insert fail, exist key
    if ((*(iter.first)).second != value_ptr){
      return errors::AlreadyExists(
          "already exists Key: ", key, " in LocklessHashMap.");
    } else {
      return Status::OK();
    }
  } 

  // Other Method
  int64 Size() const override {
    return hash_map_.size_lockless();
  }

  // Remove KV
  Status Remove(K key) override {
    if (hash_map_.erase_lockless(key)) {
      return Status::OK();
    } else {
      return errors::NotFound(
          "Unable to find Key: ", key, " in LocklessHashMap.");
    }
  }

  Status Commit(K key, const void* value_ptr) override {
    auto iter = hash_map_.insert_lockless(std::move(
        std::pair<K, void*>(key,
            const_cast<void*>(value_ptr))));
    if ((*(iter.first)).second != value_ptr) {
      AppendToValuePtrQueue((*(iter.first)).second);
      __sync_bool_compare_and_swap(
          &((*(iter.first)).second),
          (*(iter.first)).second,
          value_ptr);
    }
    return Status::OK();
  }

  Status BatchCommit(const std::vector<K>& keys,
      const std::vector<void*>& value_ptrs) override {
    for(int i = 0; i < keys.size(); ++i) {
      auto iter = hash_map_.insert_lockless(std::move(
          std::pair<K, void*>(keys[i],
              const_cast<void*>(value_ptrs[i]))));
      if ((*(iter.first)).second != value_ptrs[i]) {
        AppendToValuePtrQueue((*(iter.first)).second);
        __sync_bool_compare_and_swap(
            &((*(iter.first)).second),
            (*(iter.first)).second,
            value_ptrs[i]);
      }
    }
    return Status::OK();
  }

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<void*>* value_ptr_list) override {
    std::pair<const K, void*> *hash_map_dump;
    int64 bucket_count;
    auto it = hash_map_.GetSnapshot();
    hash_map_dump = it.first;
    bucket_count = it.second;
    for (int64 j = 0; j < bucket_count; j++) {
      if (hash_map_dump[j].first != LocklessHashMap<K, V>::EMPTY_KEY_ 
           && hash_map_dump[j].first != LocklessHashMap<K, V>::DELETED_KEY_) {
        key_list->emplace_back(hash_map_dump[j].first);
        value_ptr_list->emplace_back(hash_map_dump[j].second);
      }
    }
    free(hash_map_dump);
    return Status::OK();
  }

  Status GetShardedSnapshot(
      std::vector<std::vector<K>>& key_list,
      std::vector<std::vector<void*>>& value_ptr_list,
      int partition_id, int partition_nums) override {
    std::pair<const K, void*> *hash_map_dump;
    int64 bucket_count;
    auto it = hash_map_.GetSnapshot();
    hash_map_dump = it.first;
    bucket_count = it.second;
    for (int64 j = 0; j < bucket_count; j++) {
      if (hash_map_dump[j].first != LocklessHashMap<K, V>::EMPTY_KEY_ 
          && hash_map_dump[j].first != LocklessHashMap<K, V>::DELETED_KEY_) {
        int part_id = hash_map_dump[j].first % kSavedPartitionNum % partition_nums;
        if (part_id != partition_id) {
          key_list[part_id].emplace_back(hash_map_dump[j].first);
          value_ptr_list[part_id].emplace_back(hash_map_dump[j].second);
        }
      }
    }

    free(hash_map_dump);
    return Status::OK();
  }

  std::string DebugString() const override {
    LOG(INFO) << "map info size:" << Size()
              << "map info bucket_count:" << hash_map_.bucket_count()
              << "map info load_factor:" << hash_map_.load_factor()
              << "map info max_load_factor:" << hash_map_.max_load_factor()
              << "map info min_load_factor:" << hash_map_.min_load_factor();
    return "";
  }

  void UpdateValuePtr(
      K key, void* new_value_ptr, 
      void* old_value_ptr) override {
    auto iter = hash_map_.insert_lockless(
        std::move(std::pair<K, void*>(key, old_value_ptr)));
    bool flag = __sync_bool_compare_and_swap(
        &((*(iter.first)).second), old_value_ptr, new_value_ptr);
    if (flag) {
      AppendToValuePtrQueue(old_value_ptr);
    } else {
      feat_desc_->Deallocate(new_value_ptr);
    }
  }

 private:
  void AppendToValuePtrQueue(void* old_value_ptr) {
    //A parameter that can be adjusted in the future
    std::deque<void*>* value_ptr_queue = GetOutOfDateValuePtrQueue();
    if (value_ptr_queue->size() > CAP_INVALID_VALUEPTR) {
      void* value_ptr = value_ptr_queue->front();
      feat_desc_->Deallocate(value_ptr);
      value_ptr_queue->pop_front();
    }
    value_ptr_queue->emplace_back(old_value_ptr);
  }

  std::deque<void*>* GetOutOfDateValuePtrQueue() {
    std::deque<void*>* value_ptr_queue = 
        static_cast<std::deque<void*>*>(pthread_getspecific(key_));
    if (value_ptr_queue == nullptr) {
      value_ptr_queue = new std::deque<void*>();
      pthread_setspecific(key_, value_ptr_queue);
    }
    return value_ptr_queue;
  }

 private:
  typedef google::dense_hash_map_lockless<K, void*> LockLessHashMap;
  static const int EMPTY_KEY_;
  static const int DELETED_KEY_;
  LockLessHashMap hash_map_;
  const int CAP_INVALID_VALUEPTR = 20000;
  FeatureDescriptor<V>* feat_desc_;
  pthread_key_t key_;
};
template <class K, class V>
const int LocklessHashMap<K, V>::EMPTY_KEY_ = -1;
template <class K, class V>
const int LocklessHashMap<K, V>::DELETED_KEY_ = -2;

}  // namespace embedding
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CPU_HASH_MAP_KV_H_
