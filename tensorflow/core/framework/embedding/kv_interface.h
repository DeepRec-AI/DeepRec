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

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/embedding/feature_descriptor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace {
const char* kInferenceMode = "INFERENCE_MODE";
}

template <class K, class V>
class GPUHashTable;

using GPUDevice = Eigen::GpuDevice;
namespace embedding {

template<class V>
class ValueIterator {
 public:
  virtual V* Next() = 0;
};

template <class K, class V>
class KVInterface {
 public:
  virtual ~KVInterface() {}
  virtual Status Lookup(K key, void** value_ptr) = 0;
  virtual Status Contains(K key) = 0;
  virtual Status Insert(K key, const void* value_ptr) = 0;
  virtual Status Remove(K key) = 0;

  virtual Status BatchLookup(const K* keys, size_t size,
                             void** value_ptrs) {
    return Status(error::Code::UNIMPLEMENTED,
                  "Unimplemented for BatchLookup in KVInterface.");
  }
  // KV Batch Insert
  virtual Status BatchInsert(const std::vector<K>& keys,
      const std::vector<void*>& value_ptrs) {
    return Status(error::Code::UNIMPLEMENTED,
                  "Unimplemented for BatchInsert in KVInterface.");
  }
  // KV Batch Remove
  virtual Status BatchRemove(const K* keys, size_t size) {
    return Status(error::Code::UNIMPLEMENTED,
                  "Unimplemented for BatchRemove in KVInterface.");
  }

  virtual Status BatchLookupOrCreate(const K* keys, size_t size,
      void** value_ptrs) {
    return Status(error::Code::UNIMPLEMENTED,
                  "Unimplemented for BatchLookupOrInsert in KVInterface.");
  }

  virtual void UpdateValuePtr(K key, void* new_value_ptr,
                              void* old_value_ptr) {
    LOG(FATAL)<<"Unimplemented for UpdateValuePtr in KVInterface.";
  }

  virtual Status BatchCommit(const std::vector<K>& keys,
      const std::vector<void*>& value_ptrs) = 0;

  // KV Size
  virtual int64 Size() const = 0;

  virtual void FreeValuePtr(void* value_ptr) {}

  virtual Status Commit(K key, const void* value_ptr) {
    return Status::OK();
  }

  virtual Status GetSnapshot(std::vector<K>* key_list,
      std::vector<void*>* value_ptr_list) = 0;

  virtual std::string DebugString() const = 0;

  virtual Status BatchLookupOrCreate(const K* keys, V* val, V* default_v,
      int32 default_v_num,
      size_t n, const GPUDevice& device) {
    return Status::OK();
  }
  virtual Status BatchLookupOrCreateKeys(const K* keys, size_t n,
      int32* item_idxs, const GPUDevice& device) {
    return Status::OK();
  }

  virtual Status BatchLookup(const GPUDevice& device,
      const K* keys, V* val, size_t n, const V* default_v) {
    return Status(error::Code::UNIMPLEMENTED,
                  "Unimplemented for BatchLookup in KVInterface.");
  }
  
  virtual GPUHashTable<K, V>* HashTable() {
    return nullptr;
  }

  virtual void SetValueLen(int64 value_len) {}
};

}  // namespace embedding
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_KV_INTERFACE_H_
