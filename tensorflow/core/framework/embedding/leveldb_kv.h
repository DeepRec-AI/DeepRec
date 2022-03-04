#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_LEVELDB_KV_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_LEVELDB_KV_H_

#include "tensorflow/core/lib/io/path.h"

#include "tensorflow/core/framework/embedding/kv_interface.h"
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
using leveldb::Iterator;

namespace tensorflow {
template <class V>
class ValuePtr;

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
    path_ = io::JoinPath(path, "level_db_" + std::to_string(Env::Default()->NowMicros()));;
    options_.create_if_missing = true;
    leveldb::Status s = leveldb::DB::Open(options_, path_, &db_);
    KVInterface<K, V>::total_dims_ = 0;
    counter_ =  new SizeCounter<K>(8);
    CHECK(s.ok());
  }

  void SetNewValuePtrFunc(std::function<ValuePtr<V>*(size_t)> new_value_ptr_fn) {
    new_value_ptr_fn_ = new_value_ptr_fn;
  }

  ~LevelDBKV() {
    delete db_;
  }

  Status Lookup(K key, ValuePtr<V>** value_ptr) {
    std::string val_str;
    leveldb::Slice db_key((char*)(&key), sizeof(void*));
    ValuePtr<V>* val = new_value_ptr_fn_(KVInterface<K, V>::total_dims_);
    leveldb::ReadOptions options;
    leveldb::Status s = db_->Get(options, db_key, &val_str);
    if (s.IsNotFound()) {
      delete val;
      return errors::NotFound(
          "Unable to find Key: ", key, " in LevelDB.");
    } else {
      memcpy((int64 *)(val->GetPtr()), &val_str[0], val_str.length());
      *value_ptr = val;
      return Status::OK();
    }
  }

  Status Insert(K key, const ValuePtr<V>* value_ptr) {
    counter_->add(key, 1);
    return Status::OK();
  }

  Status BatchInsert(std::vector<K> keys, std::vector<ValuePtr<V>*> value_ptrs) {
    return BatchCommit(keys, value_ptrs);
  } 

  Status BatchCommit(std::vector<K> keys, std::vector<ValuePtr<V>*> value_ptrs) {
    WriteBatch batch;
    for (int i = 0; i < keys.size(); i++) {
      std::string value_res((char*)value_ptrs[i]->GetPtr(), sizeof(FixedLengthHeader) +  KVInterface<K, V>::total_dims_ * sizeof(V));
      leveldb::Slice db_key((char*)(&keys[i]), sizeof(void*));
      batch.Put(db_key, value_res);
      delete value_ptrs[i];
    }
    db_->Write(WriteOptions(),&batch);
    return Status::OK();
  }

  Status Commit(K key, const ValuePtr<V>* value_ptr) {
    std::string value_res((char*)value_ptr->GetPtr(), sizeof(FixedLengthHeader) +  KVInterface<K, V>::total_dims_ * sizeof(V));
    leveldb::Slice db_key((char*)(&key), sizeof(void*));
    leveldb::Status s = db_->Put(WriteOptions(), db_key, value_res);
    delete value_ptr;
    if (!s.ok()){
      return errors::AlreadyExists(
          "already exists Key: ", key, " in RocksDB.");
    } else {
      return Status::OK();
    }
  }

  Status Remove(K key) {
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

  Status GetSnapshot(std::vector<K>* key_list, std::vector<ValuePtr<V>* >* value_ptr_list) {
    ReadOptions options;
    options.snapshot = db_->GetSnapshot();
    Iterator* it = db_->NewIterator(options);
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      std::string key_str, value_str;
      ValuePtr<V>* value_ptr = new_value_ptr_fn_(KVInterface<K, V>::total_dims_);
      key_str = it->key().ToString();
      value_str = it->value().ToString();
      key_list->emplace_back(*((long*)&key_str[0]));
      void* ptr = value_ptr->GetPtr();
      memcpy(ptr, &value_str[0], KVInterface<K, V>::total_dims_ * sizeof(V) + sizeof(FixedLengthHeader));
      value_ptr_list->emplace_back(value_ptr);  
    }
    assert(it->status().ok());
    delete it;
    db_->ReleaseSnapshot(options.snapshot);
    return Status::OK();
  }

  int64 Size() const {
    return counter_->size();
  }

  void FreeValuePtr(ValuePtr<V>* value_ptr) {
    delete value_ptr;
  }

  std::string DebugString() const {
    return "";
  }
 private:
  DB* db_;
  SizeCounter<K>* counter_;
  Options options_;
  std::string path_;
  std::function<ValuePtr<V>*(size_t)> new_value_ptr_fn_;
};

} //namespace tensorflow

#endif  TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_LEVELDB_KV_H_
