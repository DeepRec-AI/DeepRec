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
#include <bitset>
#include <atomic>
#include <memory>

#include "leveldb/db.h"
#include "leveldb/comparator.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "sparsehash/dense_hash_map"
#include "sparsehash/sparse_hash_map"
#include "tensorflow/core/framework/typed_allocator.h"
#include "tensorflow/core/lib/core/spin_rw_lock.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

template <class V> 
class ValuePtr {
 public:  
  ValuePtr(size_t size) {
    /* ____________________________________________________________________
      |           |               |                  |                  |
      | number of | each bit a V* |        V*        |        V*        |
      | embedding |    1 valid    | actually pointer | actually pointer |...
      |  columns  |   0 no-valid  |    by alloctor   |    by alloctor   |
      |  (8 bits) |   (56 bits)   |     (8 bytes)    |     (8 bytes)    |
       --------------------------------------------------------------------
     */
    // we make sure that we store at least one embedding column
    ptr_ = (void*) malloc(sizeof(int64) * (1 + size));
    memset(ptr_, 0, sizeof(int64) * (1 + size));
  }

  ~ValuePtr() {
    free(ptr_);
  }
  
  V* GetOrAllocate(Allocator* allocator, int64 value_len, const V* default_v, int emb_index, bool allocate_version) {
    // fetch meta
    unsigned long metaorig = ((unsigned long*)ptr_)[0];
    unsigned int embnum = metaorig & 0xff;
    std::bitset<56> metadata(metaorig >> 8);
   
    if (!metadata.test(emb_index)) {

      while(flag_.test_and_set(std::memory_order_acquire)); 

      // need to realloc
      /*
      if (emb_index + 1 > embnum) {
        ptr_ = (void*)realloc(ptr_, sizeof(int64) * (1 + emb_index + 1));
      }*/
      embnum++ ;
      int64 alloc_value_len = value_len;
      if (allocate_version) {
        alloc_value_len = value_len + (sizeof(int64) + sizeof(V) - 1) / sizeof(V);
      }
      V* tensor_val = TypedAllocator::Allocate<V>(allocator, alloc_value_len, AllocationAttributes());
      memcpy(tensor_val, default_v, sizeof(V) * value_len);

      ((V**)((int64*)ptr_ + 1))[emb_index]  = tensor_val;

      metadata.set(emb_index);
      // NOTE:if we use ((unsigned long*)((char*)ptr_ + 1))[0] = metadata.to_ulong();
      // the ptr_ will be occaionally  modified from 0x7f18700912a0 to 0x700912a0
      // must use  ((V**)ptr_ + 1 + 1)[emb_index] = tensor_val;  to avoid
      ((unsigned long*)(ptr_))[0] = (metadata.to_ulong() << 8) | embnum;

      flag_.clear(std::memory_order_release);
      return tensor_val;
    } else {
      return ((V**)((int64*)ptr_ + 1))[emb_index];
    }
  }


  // simple getter for V* and version
  V* GetValue(int emb_index) {
    unsigned long metaorig = ((unsigned long*)ptr_)[0];
    std::bitset<56> metadata(metaorig >> 8);
    if (metadata.test(emb_index)) {
      return ((V**)((int64*)ptr_ + 1))[emb_index];
    } else {
      return nullptr;
    }
  }
  	
  void Destroy(int64 value_len, int64 value_version_len) {
    unsigned long metaorig = ((unsigned long*)ptr_)[0];
    unsigned int embnum = metaorig & 0xff;
    std::bitset<56> metadata(metaorig >> 8);
    
    for (int i = 0; i< embnum; i++) {
      if (metadata.test(i)) {
        V* val = ((V**)((int64*)ptr_ + 1))[i];
        if (i == 0) {
          TypedAllocator::Deallocate(cpu_allocator(), val, value_version_len);
        } else {
          TypedAllocator::Deallocate(cpu_allocator(), val, value_len);
        }
      }
    }
  }

 private:
  void* ptr_;
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

template <class K, class V>
class KVInterface {
 public:
  virtual ~KVInterface() {}
  // Lookup KV
  // return no-Ok if not exist
  virtual Status Lookup(K key, V** val) = 0;
  virtual Status Lookup(K key, std::string* value) { return Status::OK(); }
   virtual ValuePtr<V>* Lookup(K key, size_t size) { return NULL; }
  // Insert KV
  virtual Status Insert(K key, const V* val, V** exist_val) { return Status::OK();}
  virtual Status Insert(K key, const V* val, size_t value_len) {
    return Status::OK();
  }

  // Other Method
  virtual int64 Size() const = 0;
  virtual Status Shrink(int64 step_to_live, int64 gs, int64 value_and_version_len, int64 value_len) { return Status::OK();}


  virtual int64 GetSnapshot(std::vector<K>* key_list, std::vector<V* >* value_list, std::vector<int64>* version_list, 
                            int emb_index, int primary_emb_index, int64 value_len, int64 steps_to_live) {
    return 0;
  }
  virtual Status Destroy(int64 value_len, int64 value_version_len) { return Status::OK();}

  // hold all partition lock, return size after all lock hold
  virtual int64 GetSnapshot(std::vector<K>* key_list, std::vector<V* >* value_list) {return 0;}
  virtual Status ResetIterator() { return Status::OK();}

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

  Status Destroy(int64 value_len, int64 value_version_len) {
    ResetIterator();
    while (HasNext() == Status::OK()) {
      K unused;
      V* v = nullptr;
      Next(unused, &v);
      TypedAllocator::Deallocate(cpu_allocator(), v, value_version_len);
    }
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
class DynamicDenseHashMap : public KVInterface<K, V> {
 public:
  DynamicDenseHashMap(int partition_num = 1000) 
  : partition_num_(partition_num),
    value_ptr_map_(nullptr) {
    value_ptr_map_ = new value_ptr_map[partition_num_];
    for (int i = 0; i< partition_num_; i++) {
      value_ptr_map_[i].hash_map.max_load_factor(0.8);
      value_ptr_map_[i].hash_map.set_empty_key(-1);
      value_ptr_map_[i].hash_map.set_deleted_key(-2); 
    }
  }
  
  ~DynamicDenseHashMap() {
    delete []value_ptr_map_;
  }

  // Lookup ValuePtr 
  ValuePtr<V>* Lookup(K key, size_t size) {
    int64 l_id = std::abs(key)%partition_num_;
    mutex_lock l(value_ptr_map_[l_id].mu);
    auto iter = value_ptr_map_[l_id].hash_map.find(key);
    if (iter == value_ptr_map_[l_id].hash_map.end()) {
      // insert ValuePtr in-place
      ValuePtr<V>* newval = new ValuePtr<V>(size);
      value_ptr_map_[l_id].hash_map[key] = newval;
      return newval;
    } else {
      return iter->second;
    }
  }

  virtual Status Lookup(K key, V** val) {
    return Status::OK();
  }

  // Other Method
  int64 Size() const {
    int64 ret = 0;
    for (int i = 0; i< partition_num_; i++) {
      mutex_lock l(value_ptr_map_[i].mu);
      ret += value_ptr_map_[i].hash_map.size();
    }
    return ret;
  }
 
  // Remove KV
  Status Remove(K key, int64 value_len, int64 value_and_version_len) {
    int64 l_id = std::abs(key) % partition_num_;
    mutex_lock l(value_ptr_map_[l_id].mu);
    auto iter = value_ptr_map_[l_id].hash_map.find(key);
    if (iter != value_ptr_map_[l_id].hash_map.end()) {
      (iter->second)->Destroy(value_len, value_and_version_len);
      delete iter->second;
      value_ptr_map_[l_id].hash_map.erase(key);
    }

    return Status::OK();
  }

  int64 GetSnapshot(std::vector<K>* key_list, std::vector<V* >* value_list, std::vector<int64>* version_list, int emb_index, int primary_emb_index, int64 value_len, int64 steps_to_live) {
    int64 tot_size = 0;
    for (int i = 0; i< partition_num_; i++) {
      mutex_lock l(value_ptr_map_[i].mu);
      for (const auto it : value_ptr_map_[i].hash_map) {
        V* val = (it.second)->GetValue(emb_index);
        V* primary_val = (it.second)->GetValue(primary_emb_index);
        if (val != nullptr && primary_val != nullptr) {
          key_list->push_back(it.first);
          value_list->push_back((it.second)->GetValue(emb_index));
          // for version
          if (steps_to_live != 0) {
            int64 dump_version = *(reinterpret_cast<int64*>(primary_val + value_len));
            version_list->push_back(dump_version);
          } else {
            version_list->push_back(0);
          }
          tot_size++;
        }
      }
    }
    return tot_size;
  } 

  Status Shrink(int64 step_to_live, int64 gs, int64 value_and_version_len, int64 value_len) {
    for (int i = 0; i< partition_num_; i++) {
      mutex_lock l(value_ptr_map_[i].mu);
      std::vector<std::pair<K, ValuePtr<V>* > > to_deleted;
      for (const auto it : value_ptr_map_[i].hash_map) {
        ValuePtr<V>* valptr = it.second;
        V* primary_val = valptr->GetValue(0);        
        if (primary_val != nullptr) {
          int64 version = *(reinterpret_cast<int64*>(primary_val + value_len));
          VLOG(2) << "key:" << it.first << ", primary_val:" << primary_val << ", gs:" << gs << ", version:" << version 
              <<", step_to_live:" << step_to_live << ", value_len:" << value_len;
          if (gs - version > step_to_live) {
            to_deleted.push_back(std::pair<K, ValuePtr<V>* >(it.first, it.second));
            VLOG(2) << i << " shrink remove:" << version;
          }
        }
      }
      for (const auto it : to_deleted) {
        (it.second)->Destroy(value_len, value_and_version_len);
        delete it.second;
        if (!value_ptr_map_[i].hash_map.erase(it.first)) {
          LOG(ERROR) << "dense hash map erase key failed: " << it.first;
        }
      }
    }
    return Status::OK();
  }

  std::string DebugString() const {
    LOG(INFO) << "map info size:" << Size();
    int64 bucket_count = 0;
    for (int i = 0; i< partition_num_; i++) {
      mutex_lock l(value_ptr_map_[i].mu);
      bucket_count += value_ptr_map_[i].hash_map.bucket_count();
    }
    LOG(INFO) << "map info bucket_count:" << bucket_count;
    LOG(INFO) << "map info bucket_bytes:" << bucket_count *sizeof(typename google::dense_hash_map<K, V*>::value_type)/1024<< "KB";
    LOG(INFO) << "map info load_factor:" << value_ptr_map_[0].hash_map.load_factor();
    LOG(INFO) << "map info max_load_factor:" << value_ptr_map_[0].hash_map.max_load_factor();
    LOG(INFO) << "map info min_load_factor:" << value_ptr_map_[0].hash_map.min_load_factor();

    return "";
  }

  Status Destroy(int64 value_len, int64 value_version_len) {
    for (int i = 0; i< partition_num_; i++) {
      mutex_lock l(value_ptr_map_[i].mu);
      for (const auto it : value_ptr_map_[i].hash_map) {
        (it.second)->Destroy(value_len, value_version_len);
        delete it.second;
      }
    }
    return Status::OK();
  }

 private:
  const int partition_num_;
  // new ValuePtr
  struct value_ptr_map {
    mutable mutex mu;
    google::dense_hash_map<K, ValuePtr<V>* > hash_map;
  };
  value_ptr_map* value_ptr_map_;

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
    if ("dense_hash_map" == ht_type) {
      VLOG(2) << "Use dense_hash_map as EV data struct";
      return new DenseHashMap<K, V>(partition_num);
    } else if ("dynamic_dense_hash_map" == ht_type) {
      VLOG(2) << "Use dynamic dense_hash_map as EV data struct";
      return new DynamicDenseHashMap<K, V>(partition_num);
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
