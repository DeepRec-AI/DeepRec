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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_H_

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/storage_config.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
template <class V>
class ValuePtr;

template <class K, class V>
class EmbeddingVar;

template<typename K, typename V, typename EV>
class FilterPolicy;

namespace embedding {
template<typename K, typename V>
class Storage {
 public:
  explicit Storage(const StorageConfig& storage_config)
      : storage_config_(storage_config) {}
  virtual ~Storage() {}
  TF_DISALLOW_COPY_AND_ASSIGN(Storage);

  virtual Status Get(K key, ValuePtr<V>** value_ptr) = 0;
  virtual void SetAllocLen(int64 value_len, int slot_num) = 0;
  virtual Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size) = 0;
  virtual Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size, bool &need_copyback) = 0;
  virtual int LookupTier(K key) const = 0;
  virtual Status Remove(K key) = 0;
  virtual int64 Size() const = 0;
  virtual int64 Size(int level) const = 0;
  virtual Status GetSnapshot(std::vector<K>* key_list,
      std::vector<ValuePtr<V>*>* value_ptr_list) = 0;
  virtual int64 GetSnapshot(std::vector<K>* key_list,
      std::vector<V* >* value_list,
      std::vector<int64>* version_list,
      std::vector<int64>* freq_list,
      const EmbeddingConfig& emb_config,
      FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
      embedding::Iterator** it) = 0;
  virtual Status Shrink(const EmbeddingConfig& emb_config,
      int64 value_len) = 0;
  virtual Status Shrink(int64 gs, int64 steps_to_live) = 0;

  virtual Status BatchCommit(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) = 0;

  virtual Status Eviction(K* evict_ids, int64 evict_size) = 0;

  virtual void CopyBackToGPU(int total, K* keys, int64 size,
      bool* copyback_flags, V** memcpy_address, size_t value_len,
      int *copyback_cursor, ValuePtr<V> **gpu_value_ptrs,
      V* memcpy_buffer_gpu) = 0;

  virtual void InitCache(embedding::CacheStrategy cache_strategy) = 0;
  virtual int64 CacheSize() const = 0;
  virtual BatchCache<K>* Cache() = 0;
  virtual bool IsMultiLevel() = 0;
  virtual void Schedule(std::function<void()> fn) = 0;

  inline mutex* get_mutex() { return &mu_; }
  inline int64 GetAllocLen() { return alloc_len_; }
  inline int64 GetOffset(int64 index) { return alloc_len_ * index; }
  inline int64 GetTotalDims() { return total_dims_; }
  inline LayoutType GetLayoutType() { return storage_config_.layout_type; }
  inline embedding::StorageType GetStorageType() { return storage_config_.type; }
  inline std::string GetStoragePath() { return storage_config_.path; }

  inline std::string DebugString() const {
    return strings::StrCat("class type: ", typeid(this).name(),
                          " alloc len: ", alloc_len_,
                          " total dims: ", total_dims_,
                          " storage type: ", storage_config_.type,
                          " storage path: ", storage_config_.path,
                          " storage capacity: ", storage_config_.size);
  }

 protected:
  inline int64 ComputeAllocLen(int64 value_len) {
    return (value_len * sizeof(V) % 16 == 0) 
        ? value_len
        : value_len + (16 - (sizeof(V) * value_len) % 16) / sizeof(V);
  }

 protected:
  int64 alloc_len_ = 0;
  int64 total_dims_ = 0;
  StorageConfig storage_config_;

  mutex mu_;
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_H_
