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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_GPU_HASH_TABLE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_GPU_HASH_TABLE_H_

#if GOOGLE_CUDA
#include <cuda/std/atomic>

#include "tensorflow/core/framework/typed_allocator.h"
#include "tensorflow/core/lib/core/status.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
template <typename T>
class gpu_hash_map_tf_allocator;

template <typename KeyType, typename ValueType, typename Allocator>
class DynamicHashTable;

template <typename KeyType, typename ValueType, typename Allocator>
class StaticHashTable;

template <typename K, typename V>
class GPUStaticHashTable {
 public:
  GPUStaticHashTable(size_t capacity, int dimension, K empty_key_sentinel,
                     int32 empty_value_sentinel, Allocator* alloc,
                     cudaStream_t stream);

  ~GPUStaticHashTable();

  std::size_t Size();

  StaticHashTable<K, V, gpu_hash_map_tf_allocator<uint8_t>>* hash_table;
  V* values_d{nullptr};
  int dimension_;
  V* default_values{nullptr};
  int capacity_;
};

template <typename K, typename V>
class GPUHashTable {
 public:
  GPUHashTable(K empty_key_sentinel, Allocator* alloc,
               size_t initial_capacity = 50000);

  ~GPUHashTable();

  int32 Size();

  DynamicHashTable<K, int32, gpu_hash_map_tf_allocator<uint8_t>>* hash_table;

  const int32 initial_bank_size;
  cuda::atomic<std::size_t, cuda::thread_scope_device>* start_idx;
  int32 mem_bank_num = 0;
  std::vector<V*> bank_ptrs;
  V** d_bank_ptrs = nullptr;
  std::vector<bool*> existence_flag_ptrs;
  bool** d_existence_flag_ptrs = nullptr;
};

namespace functor {

template <typename Device, typename Key, typename V>
struct KvLookupKey {
  void operator()(const Key* key_first, V* value_first, int32 num_items,
                  int32 dimension, GPUStaticHashTable<Key, V>* hash_table,
                  cudaStream_t stream);
};

template <typename Device, typename Key, typename V>
struct KvInitStaticMap {
  void operator()(const Key* key_first, GPUStaticHashTable<Key, V>* hash_table,
                  int32 num_items, int32 dimension, cudaStream_t stream);
};

template <typename Device, typename Key, typename V>
struct KvLookupInsertKey {
  void operator()(
      const Key* key_first, int32* value_first, int32 num_items,
      GPUHashTable<Key, V>* hash_table,
      cuda::atomic<std::size_t, cuda::thread_scope_device>* start_idx,
      cudaStream_t stream);
};

template <typename Device, typename Key, typename Value>
struct KvLookupCreateEmb {
  void operator()(const Key* key_first, Value* val, Value* default_v, int64 dim,
                  int32* item_idxs, int32 num_items, int32 slot_idx,
                  int32 default_v_num, bool is_use_default_value_tensor,
                  Value** d_banks, bool** d_flags, int32 slot_num,
                  int32 bank_size, cudaStream_t stream);
};

template <typename Device, typename Key, typename Value>
struct KvUpdateEmb {
  void operator()(const Key* key_first, Value* default_v, int64 dim,
                  int32* item_idxs, int32 num_items, int32 slot_idx,
                  int32 default_v_num, Value** d_banks, bool** d_flags,
                  int32 slot_num, int32 bank_size, cudaStream_t stream);
};

template <typename Device, typename Key, typename V>
struct KvKeyGetSnapshot {
  void operator()(Key* key_first, int32* value_first, int32 slot_idx,
                  int32 primary_slot_idx, bool** d_flags, int32 bank_num,
                  int32 slot_num, int32 bank_size,
                  GPUHashTable<Key, V>* hash_table, int32 ev_size,
                  cudaStream_t stream);
};

template <typename Device, typename Key, typename Value>
struct KvEmbGetSnapshot {
  void operator()(Key* key, Value* val, Key empty_key_sentinel, int64 dim,
                  int32* item_idxs, int32 num_items, int32 slot_idx,
                  Value** d_banks, int32 bank_num, int32 slot_num,
                  int32 bank_size, cudaStream_t stream);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_GPU_HASH_TABLE_H_