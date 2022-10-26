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
======================================================================*/
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_GPU_HASH_MAP_KV_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_GPU_HASH_MAP_KV_H_

#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"

namespace tensorflow {
namespace embedding {

template<typename K, typename V>
class GPUHashMapKV : public KVInterface<K, V> {
 public:
  ~GPUHashMapKV() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(GPUHashMapKV);

  // Add device here
  Status BatchLookupOrCreate(const K* keys, size_t size,
      ValuePtr<V>** value_ptrs) override {
    // const cudaStream_t& stream = device.stream();

    // BatchLookupOrCreateKeys
    // BatchLookupOrCreateValues
  }

  Status Lookup(K key, ValuePtr<V>** value_ptr) override {
    return Status::OK();
  }

  Status Contains(K key) override {
    return Status::OK();
  }

  Status Insert(K key, const ValuePtr<V>* value_ptr) override {
    return Status::OK();
  }

  Status Remove(K key) override {
    return Status::OK();
  }

  Status BatchLookup(const K* keys, size_t size,
      ValuePtr<V>** value_ptrs) override {
    return Status::OK();
  }

  Status BatchInsert(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) override {
    return Status::OK();
  }

  Status BatchRemove(const K* keys, size_t size) override {
    return Status::OK();
  }

  Status BatchCommit(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) override {
    return Status::OK();
  }

  int64 Size() const override {
    return 0;
  }

  void SetTotalDims(int total_dims) override {
  }

  void FreeValuePtr(ValuePtr<V>* value_ptr) override {
  }

  Status Commit(K key, const ValuePtr<V>* value_ptr) override {
    return Status::OK();
  }

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<ValuePtr<V>*>* value_ptr_list) override {
    return Status::OK();
  }

  std::string DebugString() const override {
    return std::string();
  }

  Iterator* GetIterator() override {
    return nullptr;
  }

 private:
  void Resize(size_t hint) {
  }

  void BatchLookupOrCreateKey(const K* keys, size_t size,
      ValuePtr<V>** value_ptrs, cudaStream_t stream) {
  }

 private:
  // GPUHashTable<K, V> hash_table_;
  mutex lock_;
};

} // namespace embedding
} // namespace tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_GPU_HASH_MAP_KV_H_
