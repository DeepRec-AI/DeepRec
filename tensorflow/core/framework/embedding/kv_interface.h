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

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_KV_INTERFACE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_KV_INTERFACE_H_

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

template <class V>
class ValuePtr;

template <class K, class V>
class KVInterface {
 public:
  virtual ~KVInterface() {}
  // KV Lookup
  virtual Status Lookup(K key, ValuePtr<V>** value_ptr) = 0;
  // KV Insert
  virtual Status Insert(K key, const ValuePtr<V>* value_ptr) = 0;
  // KV Remove
  virtual Status Remove(K key) = 0;

  // KV Batch Lookup
  virtual Status BatchLookup(std::vector<K> keys, std::vector<ValuePtr<V>**> value_ptrs) {
    return Status(error::Code::UNIMPLEMENTED,
                      "Unimplemented for BatchLookup in KVInterface.");
  }
  // KV Batch Insert
  virtual Status BatchInsert(std::vector<K> keys, std::vector<const ValuePtr<V>*> value_ptrs) {
    return Status(error::Code::UNIMPLEMENTED,
                      "Unimplemented for BatchInsert in KVInterface.");
  }
  // KV Batch Remove
  virtual Status BatchRemove(std::vector<K> keys) {
    return Status(error::Code::UNIMPLEMENTED,
                      "Unimplemented for BatchRemove in KVInterface.");
  }

  virtual Status BatchCommit(std::vector<K> keys, std::vector<ValuePtr<V>*> value_ptrs) {return Status::OK();}

  // KV Size
  virtual int64 Size() const = 0;

  virtual void SetNewValuePtrFunc(std::function<ValuePtr<V>*(size_t)> new_value_ptr_fn) {}

  virtual void FreeValuePtr(ValuePtr<V>* value_ptr) {}

  virtual Status Commit(K key, const ValuePtr<V>* value_ptr) {return Status::OK();}

  virtual Status GetSnapshot(std::vector<K>* key_list,
                             std::vector<ValuePtr<V>* >* value_ptr_list) = 0;

  virtual std::string DebugString() const = 0;

  virtual void SetDim(int index, int dim, int slotnum) {
    int i;
    while (flag_.test_and_set(std::memory_order_acquire));
    if (slotnum != slot_dims_.size()) {
      for (i = slot_dims_.size(); i < slotnum; i++) {
        slot_dims_.emplace_back(0);
        slot_offset_.emplace_back(0);
      }
    }
    dim +=  (16 - (sizeof(V) * dim) % 16) / sizeof(V); 
    slot_dims_[index] = dim;
    total_dims_ += dim;
    for (i = 0; i < slotnum; i++) {
      if (slot_dims_[i] == 0)
        break;
    }
    if (i == slotnum) {
      for (int j = 1; j < slotnum; j++) {
        slot_offset_[j] += slot_dims_[j-1] + slot_offset_[j-1];
      }
    }
    flag_.clear(std::memory_order_release);
  }

  virtual int GetOffset(int index) {
    if (slot_offset_.size() == 0)
      return 0;
    else
      return slot_offset_[index];
  }

  virtual int GetTotalDims() {
    return total_dims_;
  }
  
  public:
    std::vector<int> slot_dims_;
    std::vector<int> slot_offset_;
    int total_dims_;
  private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT; 
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_KV_INTERFACE_H_
