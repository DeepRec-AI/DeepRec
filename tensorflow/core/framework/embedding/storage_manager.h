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

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_MANAGER_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_MANAGER_H_

#include "tensorflow/core/framework/embedding/storage.h"
#include "tensorflow/core/framework/embedding/storage_factory.h"

namespace tensorflow {
template <class V>
class ValuePtr;

template <class K, class V>
class EmbeddingVar;

namespace embedding {
template <class K, class V>
class StorageManager {
 public:
  StorageManager(const string& name,
                 StorageConfig sc,
                 Allocator* gpu_allocator = nullptr) {
    storage_ = StorageFactory::Create<K, V>(sc, gpu_allocator, name);
  }

  ~StorageManager() {
    delete storage_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(StorageManager);

  void SetAllocLen(int64 value_len, int slot_num){
    storage_->SetAllocLen(value_len, slot_num);
  }

  void InitCache(embedding::CacheStrategy cache_strategy) {
    storage_->InitCache(cache_strategy);
  }

  int64 GetAllocLen(){
    return storage_->GetAllocLen();
  }

  int64 GetOffset(int64 index) {
    return storage_->GetOffset(index);
  }

  int64 GetTotalDims() {
    return storage_->GetTotalDims();
  }

  LayoutType GetLayoutType() {
    return storage_->GetLayoutType();
  }

  embedding::StorageType GetStorageType() {
    return storage_->GetStorageType();
  }

  std::string GetStoragePath() {
    return storage_->GetStoragePath();
  }

  int64 Size(int level){
    return storage_->Size(level);
  }

  bool IsMultiLevel() {
    return storage_->IsMultiLevel();
  }

  bool IsUseHbm() {
    return storage_->IsUseHbm();
  }

  std::string DebugString() const{
    return storage_->DebugString();
  }

  void Schedule(std::function<void()> fn) {
    storage_->Schedule(fn);
  }

  int LookupTier(K key) const {
    return storage_->LookupTier(key);
  }

  Status Get(K key, ValuePtr<V>** value_ptr) {
    return storage_->Get(key, value_ptr);
  }

  void Insert(const std::vector<K>& keys,
                ValuePtr<V>** value_ptrs) {
    storage_->Insert(keys, value_ptrs);
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr, size_t size) {
    return storage_->GetOrCreate(key, value_ptr, size);
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size, CopyBackFlag &need_copyback) {
    return storage_->GetOrCreate(key, value_ptr, size, need_copyback);
  }

#if GOOGLE_CUDA
  void CopyBackToGPU(int total, const K* keys,
       const std::list<int64>& copyback_cursor,
       V** memcpy_address, size_t value_len,
       ValuePtr<V> **gpu_value_ptrs, V* memcpy_buffer_gpu){
    return storage_->CopyBackToGPU(total, keys, copyback_cursor,
        memcpy_address, value_len, gpu_value_ptrs,
        memcpy_buffer_gpu);
  }
#endif  // GOOGLE_CUDA

  Status Remove(K key) {
    return storage_->Remove(key);
  }

  int64 Size() const {
    return storage_->Size();
  }

  int64 CacheSize() const {
    return storage_->CacheSize();
  }

  Status GetSnapshot(std::vector<K>* key_list,
                     std::vector<ValuePtr<V>* >* value_ptr_list) {
    return storage_->GetSnapshot(key_list, value_ptr_list);
  }

  int64 GetSnapshot(std::vector<K>* key_list,
      std::vector<V* >* value_list,
      std::vector<int64>* version_list,
      std::vector<int64>* freq_list,
      const EmbeddingConfig& emb_config,
      FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
      embedding::Iterator** it) {
    return storage_->GetSnapshot(key_list, value_list, version_list,
        freq_list, emb_config, filter, it);
  }

  Status Shrink(const EmbeddingConfig& emb_config, int64 value_len) {
    return storage_->Shrink(emb_config, value_len);
  }

  Status Shrink(int64 gs, int64 steps_to_live) {
    return storage_->Shrink(gs, steps_to_live);
  }

  Status BatchCommit(const std::vector<K>& keys,
                     const std::vector<ValuePtr<V>*>& value_ptrs) {
    return storage_->BatchCommit(keys, value_ptrs);
  }

  BatchCache<K>* Cache() {
    return storage_->Cache();
  }

  Status Eviction(K* evict_ids, int64 evict_size) {
    return storage_->Eviction(evict_ids, evict_size);
  }

  void FreeValuePtr(ValuePtr<V>* value_ptr) {
    storage_->FreeValuePtr(value_ptr);
  }

  void iterator_mutex_lock() {
    storage_->iterator_mutex_lock();
  }

  void iterator_mutex_unlock() {
    storage_->iterator_mutex_unlock();
  }

 private:
  Storage<K, V>* storage_ = nullptr;
};

} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_MANAGER_H_
