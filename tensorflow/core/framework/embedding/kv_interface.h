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

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_KV_INTERFACE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_KV_INTERFACE_H_

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

template <class V>
class ValuePtr;

namespace embedding {
class Iterator {
 public:
  Iterator() {};
  virtual ~Iterator() {};
  virtual bool Valid() {return true;};
  virtual void SeekToFirst() {};
  virtual void Next() {};
  virtual void Key(char* val, int64 dim) {};
  virtual void Freq(char* val, int64 dim) {};
  virtual void Version(char* val, int64 dim) {};
  virtual void Value(char* val, int64 dim, int64 value_offset) {};
};

template <class K, class V>
class KVInterface {
 public:
  virtual ~KVInterface() {}
  virtual Status Lookup(K key, ValuePtr<V>** value_ptr) = 0;
  virtual Status Contains(K key) = 0;
  virtual Status Insert(K key, const ValuePtr<V>* value_ptr) = 0;
  virtual Status Remove(K key) = 0;

  virtual Status BatchLookup(const K* keys, size_t size,
                             ValuePtr<V>** value_ptrs) {
    return Status(error::Code::UNIMPLEMENTED,
                  "Unimplemented for BatchLookup in KVInterface.");
  }
  // KV Batch Insert
  virtual Status BatchInsert(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) {
    return Status(error::Code::UNIMPLEMENTED,
                  "Unimplemented for BatchInsert in KVInterface.");
  }
  // KV Batch Remove
  virtual Status BatchRemove(const K* keys, size_t size) {
    return Status(error::Code::UNIMPLEMENTED,
                  "Unimplemented for BatchRemove in KVInterface.");
  }

  virtual Status BatchLookupOrCreate(const K* keys, size_t size,
      ValuePtr<V>** value_ptrs) {
    return Status(error::Code::UNIMPLEMENTED,
                  "Unimplemented for BatchLookupOrInsert in KVInterface.");
  }

  virtual Status BatchCommit(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) = 0;

  // KV Size
  virtual int64 Size() const = 0;

  virtual void SetTotalDims(int total_dims) {}

  virtual void FreeValuePtr(ValuePtr<V>* value_ptr) {}

  virtual Status Commit(K key, const ValuePtr<V>* value_ptr) {
    return Status::OK();
  }

  virtual Status GetSnapshot(std::vector<K>* key_list,
      std::vector<ValuePtr<V>*>* value_ptr_list) = 0;

  virtual std::string DebugString() const = 0;

  virtual Iterator* GetIterator() { return nullptr; }

  virtual Status BatchLookupOrCreate(const K* keys, V* val, V* default_v,
      int32 default_v_num, bool is_use_default_value_tensor,
      size_t n, const Eigen::GpuDevice& device) {
    return Status::OK();
  }
  virtual Status BatchLookupOrCreateKeys(const K* keys, size_t n,
      int32* item_idxs, const Eigen::GpuDevice& device) {
    return Status::OK();
  }

};

}  // namespace embedding
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_KV_INTERFACE_H_
