/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_LOCKLESS_HASH_MAP_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_LOCKLESS_HASH_MAP_H_

#include "sparsehash/dense_hash_map_lockless"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
template <class V>
class ValuePtr;

namespace embedding {

template <class K, class V>
class LocklessHashMap : public KVInterface<K, V> {
 public:
  LocklessHashMap() {
    hash_map_.max_load_factor(0.8);
    hash_map_.set_empty_key_and_value(
        LocklessHashMap<K, V>::EMPTY_KEY_, nullptr);
    hash_map_.set_counternum(16);
    hash_map_.set_deleted_key(LocklessHashMap<K, V>::DELETED_KEY_);
  }

  ~LocklessHashMap() {
  }

  Status Lookup(K key, ValuePtr<V>** value_ptr) {
    auto iter = hash_map_.find_wait_free(key);
    if (iter.first == LocklessHashMap<K, V>::EMPTY_KEY_) {
      return errors::NotFound(
          "Unable to find Key: ", key, " in LocklessHashMap.");
    } else {
      *value_ptr = iter.second;
      return Status::OK();
    }
  }

  Status Insert(K key, const ValuePtr<V>* value_ptr) {
    auto iter = hash_map_.insert_lockless(
        std::move(std::pair<K, ValuePtr<V>*>(key,
            const_cast<ValuePtr<V>*>(value_ptr))));
    // insert fail, exist key
    if ((*(iter.first)).second != value_ptr){
      return errors::AlreadyExists(
          "already exists Key: ", key, " in LocklessHashMap.");
    } else {
      return Status::OK();
    }
  } 

  // Other Method
  int64 Size() const {
    return hash_map_.size_lockless();
  }

  // Remove KV
  Status Remove(K key) {
    if (hash_map_.erase_lockless(key)) {
      return Status::OK();
    } else {
      return errors::NotFound(
          "Unable to find Key: ", key, " in LocklessHashMap.");
    }
  }

  Status BatchCommit(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) {
    return Status::OK();
  }

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<ValuePtr<V>* >* value_ptr_list) {
    std::pair<const K, ValuePtr<V>*> *hash_map_dump;
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

  std::string DebugString() const {
    LOG(INFO) << "map info size:" << Size()
              << "map info bucket_count:" << hash_map_.bucket_count()
              << "map info load_factor:" << hash_map_.load_factor()
              << "map info max_load_factor:" << hash_map_.max_load_factor()
              << "map info min_load_factor:" << hash_map_.min_load_factor();
    return "";
  }

 private:
  typedef google::dense_hash_map_lockless<K, ValuePtr<V>* > LockLessHashMap;
  static const int EMPTY_KEY_;
  static const int DELETED_KEY_;
  LockLessHashMap hash_map_;
};
template <class K, class V>
const int LocklessHashMap<K, V>::EMPTY_KEY_ = -1;
template <class K, class V>
const int LocklessHashMap<K, V>::DELETED_KEY_ = -2;

}  // namespace embedding
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_LOCKLESS_HASH_MAP_H_
