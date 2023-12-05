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

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_DENSE_HASH_MAP_KV_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_DENSE_HASH_MAP_KV_H_

#include "sparsehash/dense_hash_map"
#include "tensorflow/core/framework/typed_allocator.h"
#include "tensorflow/core/lib/core/spin_rw_lock.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"

namespace tensorflow {
namespace embedding {

template <class K, class V>
class DenseHashMap : public KVInterface<K, V> {
 public:
  DenseHashMap()
  : hash_map_(nullptr) {
    hash_map_ = new dense_hash_map[partition_num_];
    for (int i = 0; i< partition_num_; i++) {
      hash_map_[i].hash_map.max_load_factor(0.8);
      hash_map_[i].hash_map.set_empty_key(-1);
      hash_map_[i].hash_map.set_deleted_key(-2);
    }
  }

  ~DenseHashMap() override {
    delete []hash_map_;
  }

  Status Lookup(K key, void** value_ptr) override {
    int64 l_id = std::abs(key)%partition_num_;
    spin_rd_lock l(hash_map_[l_id].mu);
    auto iter = hash_map_[l_id].hash_map.find(key);
    if (iter == hash_map_[l_id].hash_map.end()) {
      return errors::NotFound(
          "Unable to find Key: ", key, " in DenseHashMap.");
    } else {
      *value_ptr = iter->second;
      return Status::OK();
    }
  }

  Status Contains(K key) override {
    int64 l_id = std::abs(key)%partition_num_;
    spin_rd_lock l(hash_map_[l_id].mu);
    auto iter = hash_map_[l_id].hash_map.find(key);
    if (iter == hash_map_[l_id].hash_map.end()) {
      return errors::NotFound(
          "Unable to find Key: ", key, " in DenseHashMap.");
    } else {
      return Status::OK();
    }
  }

  Status Insert(K key, const void* value_ptr) override {
    int64 l_id = std::abs(key)%partition_num_;
    spin_wr_lock l(hash_map_[l_id].mu);
    auto iter = hash_map_[l_id].hash_map.find(key);
    // insert fail, exist key
    if (iter != hash_map_[l_id].hash_map.end()) {
      return errors::AlreadyExists(
          "already exists Key: ", key, " in DenseHashMap.");
    } else {
      auto iter = hash_map_[l_id].hash_map.insert(
        std::move(std::pair<K, void*>(key,
            const_cast<void*>(value_ptr))));
      return Status::OK();
    }
  }

  // Other Method
  int64 Size() const override {
    int64 ret = 0;
    for (int i = 0; i< partition_num_; i++) {
      spin_rd_lock l(hash_map_[i].mu);
      ret += hash_map_[i].hash_map.size();
    }
    return ret;
  }

  // Remove KV
  Status Remove(K key) override {
    int64 l_id = std::abs(key)%partition_num_;
    spin_wr_lock l(hash_map_[l_id].mu);
    if (hash_map_[l_id].hash_map.erase(key)) {
      return Status::OK();
    } else {
      return errors::NotFound(
          "Unable to find Key: ", key, " in DenseHashMap.");
    }
  }

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<void*>* value_ptr_list) override {
    dense_hash_map hash_map_dump[partition_num_];
    for (int i = 0; i< partition_num_; i++) {
      spin_rd_lock l(hash_map_[i].mu);
      hash_map_dump[i].hash_map = hash_map_[i].hash_map;
    }
    for (int i = 0; i< partition_num_; i++) {
      for (const auto it : hash_map_dump[i].hash_map) {
        key_list->push_back(it.first);
        value_ptr_list->push_back(it.second);
      }
    }
    return Status::OK();
  }

  Status GetShardedSnapshot(
      std::vector<std::vector<K>>& key_list,
      std::vector<std::vector<void*>>& value_ptr_list,
      int partition_id, int partition_nums) override {
    dense_hash_map hash_map_dump[partition_num_];
    for (int i = 0; i< partition_num_; i++) {
      spin_rd_lock l(hash_map_[i].mu);
      hash_map_dump[i].hash_map = hash_map_[i].hash_map;
    }
    for (int i = 0; i< partition_num_; i++) {
      for (const auto it : hash_map_dump[i].hash_map) {
        int part_id = it.first % kSavedPartitionNum % partition_nums;
        if (part_id != partition_id) {
          key_list[part_id].emplace_back(it.first);
          value_ptr_list[part_id].emplace_back(it.second);
        }
      }
    }
    return Status::OK();
  }

  std::string DebugString() const override {
    return "";
  }

 private:
  const int partition_num_ = 1000;
  struct dense_hash_map {
    mutable easy_spinrwlock_t mu = EASY_SPINRWLOCK_INITIALIZER;
    google::dense_hash_map<K, void* > hash_map;
  };
  dense_hash_map* hash_map_;
};

}  // namespace embedding
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_DENSE_HASH_MAP_KV_H_
