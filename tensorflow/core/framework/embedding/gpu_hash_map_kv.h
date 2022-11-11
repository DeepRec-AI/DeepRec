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

#if GOOGLE_CUDA
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/gpu_hash_table.h"

namespace tensorflow {
namespace embedding {

template<typename K, typename V>
class GPUHashMapKV : public KVInterface<K, V> {
 public:
  GPUHashMapKV(const EmbeddingConfig& config, Allocator* alloc)
      : config_(config), alloc_(alloc) {
    hash_table_ = new GPUHashTable<K, V>(-1, alloc);
  }

  ~GPUHashMapKV() override {
    for (int i = 0; i < hash_table_->bank_ptrs.size(); ++i) {
      TypedAllocator::Deallocate(
          alloc_, hash_table_->bank_ptrs[i],
          value_len_ * hash_table_->initial_bank_size);
      TypedAllocator::Deallocate(
          alloc_, hash_table_->existence_flag_ptrs[i],
          hash_table_->initial_bank_size);
    }
    if (hash_table_->mem_bank_num != 0) {
      auto num_elements = hash_table_->mem_bank_num *
          (config_.block_num * (1 + config_.slot_num));
      TypedAllocator::Deallocate(
          alloc_, hash_table_->d_bank_ptrs, num_elements);
      TypedAllocator::Deallocate(
          alloc_, hash_table_->d_existence_flag_ptrs, num_elements);
    }
    delete hash_table_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(GPUHashMapKV);

  void SetValueLen(int64 value_len) {
    value_len_ = value_len;
  }

  Status BatchLookupOrCreateKeys(const K* keys, size_t n,
      int32* item_idxs, const Eigen::GpuDevice& device) {
    mutex_lock lock(lock_);
    int remaining_size = n + *(hash_table_->start_idx) -
        hash_table_->mem_bank_num * hash_table_->initial_bank_size;
    if (remaining_size > 0) {
      Resize(remaining_size);
    }
    functor::KvLookupInsertKey<Eigen::GpuDevice, K, V>()(
        keys, item_idxs, n, hash_table_, hash_table_->start_idx,
        device.stream());
  }

  Status BatchLookupOrCreate(const K* keys, V* val, V* default_v,
      int32 default_v_num, bool is_use_default_value_tensor,
      size_t n, const Eigen::GpuDevice& device) {
    int32* item_idxs = TypedAllocator::Allocate<int32>(alloc_, n,
        AllocationAttributes());
    BatchLookupOrCreateKeys(keys, n, item_idxs, device);
    functor::KvLookupCreateEmb<Eigen::GpuDevice, K, V>()(
        keys, val, default_v, value_len_, item_idxs, n,
        config_.emb_index, default_v_num, is_use_default_value_tensor,
        hash_table_->d_bank_ptrs, hash_table_->d_existence_flag_ptrs,
        (config_.block_num * (1 + config_.slot_num)),
        hash_table_->initial_bank_size, device.stream());
    TypedAllocator::Deallocate(alloc_, item_idxs, n);
  }

  void GetSnapshot(K* keys, V* values, const Eigen::GpuDevice& device) {
    auto stream = device.stream();
    auto size = hash_table_->Size();
    int32* item_idxs = TypedAllocator::Allocate<int32>(
        alloc_, size, AllocationAttributes());
    K* keys_gpu = TypedAllocator::Allocate<K>(
        alloc_, size, AllocationAttributes());
    V* values_gpu = TypedAllocator::Allocate<V>(
        alloc_, size * value_len_, AllocationAttributes());

    auto slot_num = config_.block_num * (1 + config_.slot_num);
    functor::KvKeyGetSnapshot<Eigen::GpuDevice, K, V>()(
        keys_gpu, item_idxs, config_.emb_index, config_.primary_emb_index,
        hash_table_->d_existence_flag_ptrs, hash_table_->mem_bank_num,
        slot_num, hash_table_->initial_bank_size, hash_table_, size,
        stream);
    functor::KvEmbGetSnapshot<Eigen::GpuDevice, K, V>()(
        keys_gpu, values_gpu, -1, value_len_, item_idxs,size,
        config_.emb_index, hash_table_->d_bank_ptrs, hash_table_->mem_bank_num,
        slot_num, hash_table_->initial_bank_size, stream);
    cudaMemcpy(keys, keys_gpu, size * sizeof(K), cudaMemcpyDeviceToHost);
    cudaMemcpy(values, values_gpu, size * value_len_ * sizeof(V),
        cudaMemcpyDeviceToHost);

    TypedAllocator::Deallocate(alloc_, item_idxs, size);
    TypedAllocator::Deallocate(alloc_, keys_gpu, size);
    TypedAllocator::Deallocate(alloc_, values_gpu, size * value_len_);
  }

  Status Import(RestoreBuffer& restore_buff, int64 key_num,
      int bucket_num, int64 partition_id, int64 partition_num,
      bool is_filter, const Eigen::GpuDevice& device) {
    K* key_buff = (K*)restore_buff.key_buffer;
    V* value_buff = (V*)restore_buff.value_buffer;
    std::vector<K> key_import;
    std::vector<V> value_import;
    for (auto i = 0; i < key_num; ++ i) {
      if (*(key_buff + i) % bucket_num % partition_num != partition_id) {
        LOG(INFO) << "skip EV key:" << *(key_buff + i);
        continue;
      }
      key_import.emplace_back(*(key_buff + i));
      register auto row_offset = value_buff + i * value_len_;
      for (int j = 0; j < value_len_; j++) {
        value_import.emplace_back(*(row_offset + j));
      }
    }
    int n = key_import.size();
    int32* item_idxs = TypedAllocator::Allocate<int32>(
        alloc_, n, AllocationAttributes());
    K* key_gpu = TypedAllocator::Allocate<K>(
        alloc_, n, AllocationAttributes());
    cudaMemcpy(key_gpu, key_import.data(),
        key_import.size() * sizeof(K), cudaMemcpyHostToDevice);
    BatchLookupOrCreateKeys(key_gpu, n, item_idxs, device);
    V* value_gpu = TypedAllocator::Allocate<V>(
        alloc_, value_import.size(), AllocationAttributes());
    cudaMemcpy(value_gpu, value_import.data(),
        value_import.size() * sizeof(V), cudaMemcpyHostToDevice);

    functor::KvUpdateEmb<Eigen::GpuDevice, K, V>()(
        key_import.data(), value_gpu, value_len_, item_idxs, n,
        config_.emb_index, key_import.size(),
        hash_table_->d_bank_ptrs, hash_table_->d_existence_flag_ptrs,
        (config_.block_num * (1 + config_.slot_num)),
        hash_table_->initial_bank_size, device.stream());
    TypedAllocator::Deallocate(alloc_, item_idxs, n);
    TypedAllocator::Deallocate(alloc_, value_gpu, value_import.size());
    TypedAllocator::Deallocate(alloc_, key_gpu, n);
    return Status::OK();
  }

  Status BatchLookupOrCreate(const K* keys, size_t n,
      ValuePtr<V>** value_ptrs) override {
    return Status::OK();
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

  GPUHashTable<K, V>* HashTable() {
    return hash_table_;
  }

 private:
  void Resize(size_t hint) {
    while (hint > 0) {
      for (int i = 0; i < (config_.block_num *
            (1 + config_.slot_num)); ++i) {
        V* ptr = TypedAllocator::Allocate<V>(
            alloc_, value_len_ * hash_table_->initial_bank_size,
            AllocationAttributes());
        hash_table_->bank_ptrs.push_back(ptr);
        bool* ptr2 = TypedAllocator::Allocate<bool>(
            alloc_, hash_table_->initial_bank_size, AllocationAttributes());
        hash_table_->existence_flag_ptrs.push_back(ptr2);
        cudaMemset(ptr2, 0, sizeof(bool) * hash_table_->initial_bank_size);
      }
      hint -= hash_table_->initial_bank_size;
      ++hash_table_->mem_bank_num;
    }

    auto num_elements = hash_table_->mem_bank_num * (
        config_.block_num * (1 + config_.slot_num));
    if (hash_table_->d_bank_ptrs) {
      TypedAllocator::Deallocate(alloc_, hash_table_->d_bank_ptrs,
          num_elements);
      TypedAllocator::Deallocate(alloc_, hash_table_->d_existence_flag_ptrs,
          num_elements);
    }
    hash_table_->d_bank_ptrs = TypedAllocator::Allocate<V*>(
        alloc_, num_elements, AllocationAttributes());
    cudaMemcpy(hash_table_->d_bank_ptrs, hash_table_->bank_ptrs.data(),
        num_elements * sizeof(V*), cudaMemcpyHostToDevice);
    hash_table_->d_existence_flag_ptrs = TypedAllocator::Allocate<bool*>(
        alloc_, num_elements, AllocationAttributes());
    cudaMemcpy(hash_table_->d_existence_flag_ptrs,
        hash_table_->existence_flag_ptrs.data(),
    num_elements * sizeof(bool*), cudaMemcpyHostToDevice);
  }

 private:
  EmbeddingConfig config_;
  GPUHashTable<K, V>* hash_table_;
  Allocator* alloc_;
  int64 value_len_;
  mutex lock_;
};

} // namespace embedding
} // namespace tensorflow

#endif // GOOGLE_CUDA
#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_GPU_HASH_MAP_KV_H_
