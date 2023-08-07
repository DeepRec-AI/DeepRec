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

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_LEVELDB_KV_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_LEVELDB_KV_H_

#include "tensorflow/core/lib/io/path.h"

#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"
#include "tensorflow/core/lib/core/status.h"

#include "leveldb/db.h"
#include "leveldb/comparator.h"
#include "leveldb/write_batch.h"

#include <sstream>

using leveldb::DB;
using leveldb::Options;
using leveldb::ReadOptions;
using leveldb::WriteBatch;
using leveldb::WriteOptions;

namespace tensorflow {
template <class V>
class ValuePtr;

namespace embedding {

template <class K>
class SizeCounter {
 public:
  SizeCounter(int num_parts) {
    num_parts_ = num_parts;
    for (int i = 0; i < num_parts_; i++) {
      counter_.emplace_back(0);
    }
  }

  void add(K key, int64 count) {
    int part = key % num_parts_;
     __sync_fetch_and_add(&counter_[part], count);
  }

  void sub(K key, int64 count) {
    int part = key % num_parts_;
     __sync_fetch_and_sub(&counter_[part], count);
  }

  int64 size() {
    int64 total = 0;
    for (int i = 0; i < num_parts_; i++) {
      total += counter_[i];
    }
    return total;
  }

 private:
  std::vector<int64> counter_;
  int num_parts_;  
};

template <class K, class V>
class LevelDBKV : public KVInterface<K, V> {
 public:
  LevelDBKV(std::string path) {
    path_ = io::JoinPath(path,
        "level_db_" + std::to_string(Env::Default()->NowMicros()));;
    options_.create_if_missing = true;
    leveldb::Status s = leveldb::DB::Open(options_, path_, &db_);
    CHECK(s.ok());
    counter_ =  new SizeCounter<K>(8);
    new_value_ptr_fn_ = [] (size_t size) {
      return new NormalContiguousValuePtr<V>(ev_allocator(), size);
    };
    total_dims_ = 0;
  }

  void SetTotalDims(int total_dims) {
    total_dims_ = total_dims;
  }

  ~LevelDBKV() override {
    delete db_;
  }

  Status Lookup(K key, ValuePtr<V>** value_ptr) override {
    std::string val_str;
    leveldb::Slice db_key((char*)(&key), sizeof(void*));
    leveldb::ReadOptions options;
    leveldb::Status s = db_->Get(options, db_key, &val_str);
    if (s.IsNotFound()) {
      return errors::NotFound(
          "Unable to find Key: ", key, " in LevelDB.");
    } else {
      ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
      memcpy((int64 *)(val->GetPtr()), &val_str[0], val_str.length());
      *value_ptr = val;
      return Status::OK();
    }
  }

  Status Contains(K key) override {
    std::string val_str;
    leveldb::Slice db_key((char*)(&key), sizeof(void*));
    leveldb::ReadOptions options;
    leveldb::Status s = db_->Get(options, db_key, &val_str);
    if (s.IsNotFound()) {
      return errors::NotFound(
          "Unable to find Key: ", key, " in LevelDB.");
    } else {
      return Status::OK();
    }
  }

  Status Insert(K key, const ValuePtr<V>* value_ptr) override {
    counter_->add(key, 1);
    return Status::OK();
  }

  Status BatchInsert(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) override {
    return BatchCommit(keys, value_ptrs);
  } 

  Status BatchCommit(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) override {
    WriteBatch batch;
    for (int i = 0; i < keys.size(); i++) {
      std::string value_res((char*)value_ptrs[i]->GetPtr(),
          sizeof(FixedLengthHeader) + total_dims_ * sizeof(V));
      leveldb::Slice db_key((char*)(&keys[i]), sizeof(void*));
      batch.Put(db_key, value_res);
      delete value_ptrs[i];
    }
    db_->Write(WriteOptions(),&batch);
    return Status::OK();
  }

  Status Commit(K key, const ValuePtr<V>* value_ptr) override {
    std::string value_res((char*)value_ptr->GetPtr(),
        sizeof(FixedLengthHeader) + total_dims_ * sizeof(V));
    leveldb::Slice db_key((char*)(&key), sizeof(void*));
    leveldb::Status s = db_->Put(WriteOptions(), db_key, value_res);
    if (!s.ok()){
      return errors::AlreadyExists(
          "already exists Key: ", key, " in RocksDB.");
    } else {
      return Status::OK();
    }
  }

  Status Remove(K key) override {
    counter_->sub(key, 1);
    leveldb::Slice db_key((char*)(&key), sizeof(void*));
    leveldb::Status s = db_->Delete(WriteOptions(), db_key);
    if (s.ok()) {
      return Status::OK();
    } else {
      return errors::NotFound(
          "Unable to find Key: ", key, " in RocksDB.");
    }
  }

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<ValuePtr<V>*>* value_ptr_list) override {
    ReadOptions options;
    options.snapshot = db_->GetSnapshot();
    leveldb::Iterator* it = db_->NewIterator(options);
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      K key;
      memcpy((char*)&key, it->key().ToString().data(), sizeof(K));
      key_list->emplace_back(key);
      ValuePtr<V>* value_ptr =
          new NormalGPUValuePtr<V>(ev_allocator(), 1);
      memcpy((char *)value_ptr->GetPtr(),
             it->value().ToString().data(),
             sizeof(FixedLengthHeader));
      value_ptr_list->emplace_back(value_ptr);
    }
    delete it;
    return Status::OK();
  }

  int64 Size() const override {
    return counter_->size();
  }

  void FreeValuePtr(ValuePtr<V>* value_ptr) override {
    delete value_ptr;
  }

  std::string DebugString() const override{
    return "";
  }

 private:
  DB* db_;
  SizeCounter<K>* counter_;
  Options options_;
  std::string path_;
  std::function<ValuePtr<V>*(size_t)> new_value_ptr_fn_;
  int total_dims_;
};

template<class K, class  V>
class DBValueIterator: public ValueIterator<V> {
 public:
  DBValueIterator(
      const std::vector<K>& key_list,
      int64 emb_index,
      int64 value_len,
      LevelDBKV<K, V>* leveldb_kv)
      : value_len_(value_len),
        emb_index_(emb_index),
        leveldb_kv_(leveldb_kv) {
    int64 emb_offset = value_len_ * emb_index;
    std::vector<std::list<K>> keys_parts_vec(kSavedPartitionNum);
    for (int64 i = 0; i < key_list.size(); i++) {
      for (int part_id = 0; part_id < kSavedPartitionNum; part_id++) {
        if (key_list[i] % kSavedPartitionNum == part_id) {
          keys_parts_vec[part_id].emplace_back(key_list[i]);
          break;
        }
      }
    }

    for (int64 i = 0; i < kSavedPartitionNum; i++) {
      keys_.splice(keys_.end(), keys_parts_vec[i]);
    }

    keys_iter_= keys_.begin();
  }

  ~DBValueIterator() {
    delete value_ptr_;
  }

  V* Next() {
    if (value_ptr_ != nullptr) {
      value_ptr_->Destroy(ev_allocator());
      delete value_ptr_;
    }
    K key = *(keys_iter_++);

    Status s = leveldb_kv_->Lookup(key, &value_ptr_);
    if (!s.ok()) {
      LOG(FATAL)<<"Not found value in LevelDB when Save.";
    }
    return value_ptr_->GetValue(emb_index_, value_len_ * emb_index_);
  }

 private:
  int64 value_len_;
  int64 emb_index_;
  LevelDBKV<K, V>* leveldb_kv_;
  std::list<K> keys_;
  typename std::list<K>::const_iterator keys_iter_;
  ValuePtr<V>* value_ptr_ = nullptr;
  int64 key_cursor_ = 0;
};

} //namespace embedding
} //namespace tensorflow

#endif  //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_LEVELDB_KV_H_
