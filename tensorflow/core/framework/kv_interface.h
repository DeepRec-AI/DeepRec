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

#ifndef TENSORFLOW_CORE_FRAMEWORK_KV_INTERFACE_H_
#define TENSORFLOW_CORE_FRAMEWORK_KV_INTERFACE_H_

#include <pthread.h>

#include "leveldb/db.h"
#include "leveldb/comparator.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "sparsehash/dense_hash_map"
#include "sparsehash/sparse_hash_map"
#include "tensorflow/core/framework/typed_allocator.h"
#include "tensorflow/core/lib/core/spin_rw_lock.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

template <class K, class V>
class KVInterface {
 public:
  virtual ~KVInterface() {}
  // Lookup KV
  // return no-Ok if not exist
  virtual Status Lookup(K key, V** val) = 0;
  virtual Status Lookup(K key, std::string* value) { return Status::OK(); }
  // Insert KV
  virtual Status Insert(K key, const V* val, V** exist_val) = 0;
  virtual Status Insert(K key, const V* val, size_t value_len) {
    return Status::OK();
  }

  // Other Method
  virtual int64 Size() const = 0;
  virtual Status Shrink(int64 step_to_live, int64 gs, int64 value_and_version_len, int64 value_len) { return Status::OK();}

  // hold all partition lock, return size after all lock hold
  virtual int64 GetSnapshot(std::vector<K>* key_list, std::vector<V* >* value_list) {return 0;}
  virtual Status ResetIterator() { return Status::OK();}
  virtual Status HasNext() = 0;
  virtual Status Next(K& key, V** value) = 0;

  virtual std::string DebugString() const = 0;

};

template <class K, class V>
class DenseHashMap : public KVInterface<K, V> {
 public:
  DenseHashMap(int partition_num = 1000)
  : partition_num_(partition_num),
    hash_map_(nullptr) {
    hash_map_ = new dense_hash_map[partition_num_];
    for (int i = 0; i< partition_num_; i++) {
      hash_map_[i].hash_map.max_load_factor(0.8);
      hash_map_[i].hash_map.set_empty_key(-1);
      hash_map_[i].hash_map.set_deleted_key(-2);
    }
  }
  ~DenseHashMap() {
    delete []hash_map_;
  }
  // Lookup KV
  // return no-Ok if not exist
  Status Lookup(K key, V** val) {
    int64 l_id = std::abs(key)%partition_num_;
    spin_rd_lock l(hash_map_[l_id].mu);
    //LOG(INFO) << "DenseHashMap::Lookup" << key;
    auto iter = hash_map_[l_id].hash_map.find(key);
    if (iter == hash_map_[l_id].hash_map.end()) {
      return errors::NotFound(
          "Unable to find Key: ", key, " in DenseHashMap.");
    } else {
      *val = iter->second;
      return Status::OK();
    }
  }
  // Insert KV
  Status Insert(K key, const V* val, V** exist_val) {
    int64 l_id = std::abs(key)%partition_num_;
    spin_wr_lock l(hash_map_[l_id].mu);
    auto iter = hash_map_[l_id].hash_map.find(key);
    if (iter == hash_map_[l_id].hash_map.end()) {
      hash_map_[l_id].hash_map.insert(std::move(std::pair<K, V*>(key, const_cast<V*>(val))));
    } else {
      *exist_val = iter->second;
      return errors::AlreadyExists(
          "already exists Key: ", key, " in DenseHashMap.");
    }
    //LOG(INFO) << "DenseHashMap::Insert" << key;
    return Status::OK();
  }
  // Other Method
  int64 Size() const {
    int64 ret = 0;
    for (int i = 0; i< partition_num_; i++) {
      spin_rd_lock l(hash_map_[i].mu);
      ret += hash_map_[i].hash_map.size();
    }
    return ret;
  }
  Status Shrink(int64 step_to_live, int64 gs, int64 value_and_version_len, int64 value_len) {
    for (int i = 0; i< partition_num_; i++) {
      spin_wr_lock l(hash_map_[i].mu);
      std::vector<K> to_deleted;
      for (const auto it : hash_map_[i].hash_map) {
        V* val = it.second;
        int64* version = reinterpret_cast<int64*>(val + value_len);
        VLOG(2) << i << " shrink version:" << *version;
        if (gs - *version > step_to_live) {
          VLOG(2) << i << " shrink remove:" << *version;
          TypedAllocator::Deallocate<V>(cpu_allocator(), val, value_and_version_len);
          to_deleted.push_back(it.first);
        }
      }
      for (const auto key : to_deleted) {
        if (!hash_map_[i].hash_map.erase(key)) {
          LOG(ERROR) << "dense hash map erase key failed: " << key;
        }
      }
    }
    return Status::OK();
  }

  int64 GetSnapshot(std::vector<K>* key_list, std::vector<V* >* value_list) {
    int64 tot_size = 0;
    dense_hash_map hash_map_dump[partition_num_];
    for (int i = 0; i< partition_num_; i++) {
      spin_rd_lock l(hash_map_[i].mu);
      tot_size += hash_map_[i].hash_map.size();
      hash_map_dump[i].hash_map = hash_map_[i].hash_map; 
    }
    for (int i = 0; i< partition_num_; i++) {
      for (const auto it : hash_map_dump[i].hash_map) {
        key_list->push_back(it.first);
        value_list->push_back(it.second);
      } 
    }
    return tot_size;
  }
  
  Status ResetIterator() {
    dense_iter_.curr_partition = 0;
    dense_iter_.curr_it = hash_map_[0].hash_map.begin();
    return Status::OK();
  }

  Status HasNext() {
   
    if (dense_iter_.curr_partition >= partition_num_)
      return errors::NotFound("");
   
    while (dense_iter_.curr_it == 
        hash_map_[dense_iter_.curr_partition].hash_map.end()) {
      dense_iter_.curr_partition++;
      if (dense_iter_.curr_partition < partition_num_) {
        dense_iter_.curr_it =
          hash_map_[dense_iter_.curr_partition].hash_map.begin();
      } else {
        return errors::NotFound("");
      }
    }
   
    if (dense_iter_.curr_it != 
        hash_map_[dense_iter_.curr_partition].hash_map.end()) {
      return Status::OK(); 
    } else {
        return errors::NotFound("");
    } 
  }
  
  Status Next(K& key, V** value) {
    key = dense_iter_.curr_it->first;
    *value = dense_iter_.curr_it->second;

    dense_iter_.curr_it++;
    return Status::OK();
  }
  
  std::string DebugString() const {
    LOG(INFO) << "map info size:" << Size();
    int64 bucket_count = 0;
    for (int i = 0; i< partition_num_; i++) {
      spin_rd_lock l(hash_map_[i].mu);
      bucket_count += hash_map_[i].hash_map.bucket_count();
    }
    LOG(INFO) << "map info bucket_count:" << bucket_count;
    LOG(INFO) << "map info bucket_bytes:" << bucket_count *sizeof(typename google::dense_hash_map<K, V*>::value_type)/1024<< "KB";
    LOG(INFO) << "map info load_factor:" << hash_map_[0].hash_map.load_factor();
    LOG(INFO) << "map info max_load_factor:" << hash_map_[0].hash_map.max_load_factor();
    LOG(INFO) << "map info min_load_factor:" << hash_map_[0].hash_map.min_load_factor();

    return "";
  }

 private:
  const int partition_num_;
  struct dense_hash_map {
    mutable easy_spinrwlock_t mu = EASY_SPINRWLOCK_INITIALIZER;
    google::dense_hash_map<K, V*> hash_map;
  };
  dense_hash_map* hash_map_;

  struct dense_iterator {
    int curr_partition;
    typename google::dense_hash_map<K, V*>::iterator curr_it;
  };
  dense_iterator dense_iter_;
};

template <class K, class V>
class SparseHashMap : public KVInterface<K, V> {
 public:
  SparseHashMap() {
    hash_map_.max_load_factor(0.8);
  }
  // Lookup KV
  // return no-Ok if not exist
  Status Lookup(K key, V** val) {
    tf_shared_lock l(mu_);
    //mutex_lock l(mu_);
    //LOG(INFO) << "SparseHashMap::Lookup" << key;
    auto iter = hash_map_.find(key);
    if (iter == hash_map_.end()) {
      return errors::NotFound(
          "Unable to find Key: ", key, " in SparseHashMap.");
    } else { *val = iter->second;
      return Status::OK();
    }
  }
  // Insert KV
  Status Insert(K key, const V* val, V** exist_val) {
    mutex_lock l(mu_);
    //LOG(INFO) << "SparseHashMap::Insert" << key;
    hash_map_.insert(std::move(std::pair<K, V*>(key, const_cast<V*>(val))));
    return Status::OK();
  }
  // Other Method
  int64 Size() const {
    mutex_lock l(mu_);
    return hash_map_.size();
  }
  Status Shrink() {
    return Status::OK();
  }

  virtual Status HasNext() { return Status::OK(); }
  virtual Status Next(K& key, V** value) { return Status::OK();}

  std::string DebugString() const {
    LOG(INFO) << "map info size:" << hash_map_.size();
    LOG(INFO) << "map info bucket_count:" << hash_map_.bucket_count();
    LOG(INFO) << "map info bucket_bytes:" << hash_map_.bucket_count() *sizeof(typename google::dense_hash_map<K, V*>::value_type)/1024<< "KB";
    LOG(INFO) << "map info load_factor:" << hash_map_.load_factor();
    LOG(INFO) << "map info max_load_factor:" << hash_map_.max_load_factor();
    LOG(INFO) << "map info min_load_factor:" << hash_map_.min_load_factor();

    return "";
  }

 private:
  mutable mutex mu_;
  google::sparse_hash_map<K, V*> hash_map_;
};

template <class K, class V>
class CuckooHashMap : public KVInterface<K, V> {
 public:
  CuckooHashMap() {
  }
  // Lookup KV
  // return no-Ok if not exist
  Status Lookup(K key, V** val) {
    //LOG(INFO) << "CuckooHashMap::Lookup" << key;
    auto flag = hash_map_.find(key, *val);
    if (!flag) {
      return errors::NotFound(
          "Unable to find Key: ", key, " in CuckooHashMap.");
    } else {
      return Status::OK();
    }
  }
  // Insert KV
  Status Insert(K key, const V* val, V** exist_val) {
    //LOG(INFO) << "CuckooHashMap::Insert" << key;
    //hash_map_.insert(std::move(std::pair<K, V*>(key, const_cast<V*>(val))));
    hash_map_.insert(key, const_cast<V*>(val));
    return Status::OK();
  }
  // Other Method
  int64 Size() const {
    return hash_map_.size();
  }
  std::string DebugString() const {
    LOG(INFO) << "map info size:" << hash_map_.size();
    LOG(INFO) << "map info bucket_count:" << hash_map_.bucket_count();
    LOG(INFO) << "map info bucket_bytes:" << hash_map_.bucket_count() *sizeof(typename google::dense_hash_map<K, V*>::value_type)/1024<< "KB";
    LOG(INFO) << "map info load_factor:" << hash_map_.load_factor();
    LOG(INFO) << "map info capacity:" << hash_map_.capacity();
    LOG(INFO) << "map info minimum_load_factor:" << hash_map_.minimum_load_factor();
    //LOG(INFO) << "map info maximum_hashpower:" << hash_map_.maximum_hashpower();
    LOG(INFO) << "map info max_num_worker_threads:" << hash_map_.max_num_worker_threads();

    return "";
  }

  virtual Status HasNext() { return Status::OK(); }
  virtual Status Next(K& key, V** value) { return Status::OK();}

 private:
  libcuckoo::cuckoohash_map<K, V*> hash_map_;
};

template <class K, class V>
class LevelDBHashMap: public KVInterface<K, V> {
 public:
  LevelDBHashMap(const std::string& db_name) {
    leveldb::Status st;
    leveldb::Options options;
    options.create_if_missing = true;
    //options.write_buffer_size = 1024 * 1024 * 1024;
    //options.error_if_exists = true;
    st = leveldb::DB::Open(options, db_name.c_str(), &db_hash_map_);
    if (!st.ok()) {
      LOG(FATAL) << "Fail to open hashmap_db: " << st.ToString();
    }
  }

  ~LevelDBHashMap() {
    delete db_hash_map_;
  }

  // Insert KV
  Status Insert(K key, const V* val, size_t value_len) {
    leveldb::WriteOptions options;
    options.sync = false;

    leveldb::Slice db_key((char*)&key, sizeof(key));
    leveldb::Slice db_value((char*)val, value_len);
  
    leveldb::Status st;
    st = db_hash_map_->Put(options, db_key, db_value);
    if (!st.ok()) {
      return errors::NotFound("");
    } else {
      return Status::OK();
    }
  }

  Status Lookup(K key, std::string* value) {
    leveldb::Slice db_key((char*)&key, sizeof(key));
    leveldb::ReadOptions options;
    //options.fill_cache = false;

    leveldb::Status st = db_hash_map_->Get(options,  db_key, value);
    if (!st.ok()){
      return errors::NotFound("");
    } else {
      return Status::OK();
    }
  }
 
  virtual Status Lookup(K key, V** val) { return Status::OK(); }
  Status Insert(K key, const V* val, V** exist_val) {
    return Status::OK();
  }
  
  Status Remove(K key) {
    return Status::OK();
  }

  virtual int64 Size() const {
    int64 count = 0;
    leveldb::Iterator* it = db_hash_map_->NewIterator(leveldb::ReadOptions());
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      count++;
    }
    delete it;
    return count;
  }

  std::string DebugString() const { return "";}
  virtual int64 LockAll() { return 0;}
  virtual Status HasNext() { return Status::OK(); }
  virtual Status Next(K& key, V** value) { return Status::OK();}
  virtual Status UnlockAll()  {return Status::OK();}

 private:
  leveldb::DB* db_hash_map_;
};

template <class K, class V>
class HashMapFactory {
 public:
  ~HashMapFactory() {}
  static KVInterface<K, V>* CreateHashMap(const std::string& ht_type,
                                          int partition_num) {
    if ("dense_hash_map" == ht_type || ht_type.empty()) {
      VLOG(2) << "Use dense_hash_map as EV data struct";
      return new DenseHashMap<K, V>(partition_num);
    } else if ("sparse_hash_map" == ht_type) {
      return nullptr;
      // TODO
      //return new SparseHashMap<K, V>();
    } else if ("cuckoo_hash_map" == ht_type) {
      return nullptr;
      // TODO
      //return new CuckooHashMap<K, V>();
    } else {
      LOG(WARNING) << "Not match any ht_type, use default 'dense_hash_map'";
      return new DenseHashMap<K, V>(partition_num);
    }
  }
 private:
  HashMapFactory() {}
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_KV_INTERFACE_H_
