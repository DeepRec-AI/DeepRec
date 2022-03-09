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

#ifndef TENSORFLOW_CORE_KERNELS_KV_GPU_HASH_TABLE_H_
#define TENSORFLOW_CORE_KERNELS_KV_GPU_HASH_TABLE_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/typed_allocator.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/embedding_var.h"
#include "third_party/cuco_hash_table/cuco/allocator.hpp"
#include <cuda/std/atomic>

namespace tensorflow {

template <typename T>
class gpu_hash_map_tf_allocator;

template <typename KeyType, typename ValueType, typename Allocator>
class DynamicHashTable;

template <typename K, typename V>
class GPUHashTable {
public:
  GPUHashTable(K empty_key_sentinel, OpKernelContext *context, size_t initial_capacity=50000);

  ~GPUHashTable();

  DynamicHashTable<K, int32, gpu_hash_map_tf_allocator<uint8_t>>* hash_table;

  const int32 initial_bank_size = 50000;
  cuda::atomic<std::size_t, cuda::thread_scope_device>* start_idx;
  int32 mem_bank_num = 0;
  std::vector<int32> remaining_sizes;
  std::vector<V*> bank_ptrs;
  V** d_bank_ptrs;
  std::vector<bool*> existence_flag_ptrs;
  bool** d_existence_flag_ptrs;
};

namespace functor {
template <typename Device, typename Key, typename V>
struct KvLookupInsertKey {
  void operator()(const Key* key_first,
                  int32* value_first,
                  int32 num_items,
                  GPUHashTable<Key, V>* hash_table,
                  cuda::atomic<std::size_t, cuda::thread_scope_device>* start_idx,
                  int32 idx_offset,
                  cudaStream_t stream);
};

template <typename Device, typename Value>
struct KvLookupCreateEmb {
  void operator()(Value* val,
                  Value* default_v,
                  int64 dim,
                  int32* item_idxs,
                  int32 num_items,
                  int32 slot_idx,
                  int32 default_v_num,
                  Value** d_banks,
                  bool** d_flags,
                  int32 bank_num,
                  int32 slot_num,
                  int32 bank_size,
                  cudaStream_t stream);
};
} // namespace functor

template <class K, class V>
class EmbeddingVarGPU : public ResourceBase {
 public:
  EmbeddingVarGPU(const string& name,
                  GPUHashTable<K, V>* kv,
                  Allocator* alloc,
                  EmbeddingConfig emb_cfg = EmbeddingConfig()):
      name_(name),
      kv_(kv),
      default_value_(nullptr),
      value_len_(0),
      alloc_(alloc),
      emb_config_(emb_cfg) {}

  Status Init() {
    if (kv_ == nullptr) {
       return errors::InvalidArgument("Error to construct EmbeddingVarGPU");
    } else {
      return Status::OK();
    }
  }

  Status Init(const Tensor& default_tensor, int64 default_value_dim=1) {
    if (DataTypeToEnum<V>::v() != default_tensor.dtype()) {
       return errors::InvalidArgument("EV's default_tensor DTYPE must be same as EmbeddingVar Value Type");
    } else if (kv_ == nullptr) {
       return errors::InvalidArgument("Error to construct EmbeddingVarGPU");
    } else {
      emb_config_.default_value_dim = default_value_dim;
      value_len_ = default_tensor.NumElements()/emb_config_.default_value_dim;
      default_value_ = TypedAllocator::Allocate<V>(alloc_, default_tensor.NumElements(), AllocationAttributes());
      auto default_tensor_flat = default_tensor.flat<V>();
      cudaMemcpy(default_value_, &default_tensor_flat(0), default_tensor.TotalBytes(), cudaMemcpyDeviceToDevice);
      return Status::OK();
    }
  }

  void SetInitialized() {
    is_initialized_ = true;
  }
  bool IsInitialized() const {
    return is_initialized_;
  }

  void LookupOrCreateKey(const K* key, int32* item_idxs, size_t n, cudaStream_t stream, int64 update_version = -1) {
    mutex_lock lock(lock_);
    int32 nth_bank = 0;
    while (nth_bank < kv_->mem_bank_num) {
      if (kv_->remaining_sizes[nth_bank] >= n) break;
      ++nth_bank;
    }
    if (kv_->mem_bank_num == nth_bank) {
      for (int i = 0; i < emb_config_.total_num(); ++i) {
        V* ptr = TypedAllocator::Allocate<V>(alloc_, value_len_ * kv_->initial_bank_size, AllocationAttributes());
        kv_->bank_ptrs.push_back(ptr);
        bool* ptr2 = TypedAllocator::Allocate<bool>(alloc_, kv_->initial_bank_size, AllocationAttributes());
        kv_->existence_flag_ptrs.push_back(ptr2);
        cudaMemset(ptr2, 0, sizeof(bool) * kv_->initial_bank_size);
      }
      if (kv_->mem_bank_num != 0) {
        TypedAllocator::Deallocate(alloc_, kv_->d_bank_ptrs, kv_->mem_bank_num * emb_config_.total_num());
        TypedAllocator::Deallocate(alloc_, kv_->d_existence_flag_ptrs, kv_->mem_bank_num * emb_config_.total_num());
      }
      ++kv_->mem_bank_num;
      kv_->d_bank_ptrs = TypedAllocator::Allocate<V*>(alloc_, kv_->mem_bank_num * emb_config_.total_num(), AllocationAttributes());
      cudaMemcpy(kv_->d_bank_ptrs, kv_->bank_ptrs.data(), kv_->mem_bank_num * emb_config_.total_num() * sizeof(V*), cudaMemcpyHostToDevice);
      kv_->d_existence_flag_ptrs = TypedAllocator::Allocate<bool*>(alloc_, kv_->mem_bank_num * emb_config_.total_num(), AllocationAttributes());
      cudaMemcpy(kv_->d_existence_flag_ptrs, kv_->existence_flag_ptrs.data(), kv_->mem_bank_num * emb_config_.total_num() * sizeof(bool*), cudaMemcpyHostToDevice);
      kv_->remaining_sizes.push_back(kv_->initial_bank_size);
    }
    *(kv_->start_idx) = kv_->initial_bank_size - kv_->remaining_sizes[nth_bank];
    functor::KvLookupInsertKey<Eigen::GpuDevice, K, V>()(key, item_idxs, n, kv_, kv_->start_idx, nth_bank * kv_->initial_bank_size, stream);
  }

  void LookupOrCreate(const K* key, V* val, V* default_v, int32 default_v_num, size_t n, cudaStream_t stream) {
    int32* item_idxs = TypedAllocator::Allocate<int32>(alloc_, n, AllocationAttributes());
    LookupOrCreateKey(key, item_idxs, n, stream);
    functor::KvLookupCreateEmb<Eigen::GpuDevice, V>()(val, default_v, value_len_, item_idxs, n, emb_config_.emb_index, default_v_num,
                                                      kv_->d_bank_ptrs, kv_->d_existence_flag_ptrs, kv_->mem_bank_num,
                                                      emb_config_.total_num(), kv_->initial_bank_size, stream);
    TypedAllocator::Deallocate(alloc_, item_idxs, n);
  }

  int64 ValueLen() const {
    return value_len_;
  }

  std::string DebugString() const {
    return emb_config_.DebugString();
  }

  GPUHashTable<K, V>* kv() {
    return kv_;
  }

  int64 MinFreq() {
    return emb_config_.filter_freq;
  }

  float GetL2WeightThreshold() {
    return emb_config_.l2_weight_threshold;
  }

  int32 SlotNum() {
    return emb_config_.total_num();
  }

  int32 EmbIdx() {
    return emb_config_.emb_index;
  }

  V* DefaultValuePtr() {
    return default_value_;
  }

  void SetSlotNum(int64 slot_num) {
    emb_config_.slot_num = slot_num;
  }

  V* GetDefaultValuePtr() {
    return default_value_;
  }

  int64 GetDefaultValueDim() {
    return emb_config_.default_value_dim;
  }

 private:
  std::string name_;
  GPUHashTable<K, V>* kv_;
  bool is_initialized_ = false;

  V* default_value_;
  int64 value_len_;
  Allocator* alloc_;
  EmbeddingConfig emb_config_;

  mutex lock_;

  ~EmbeddingVarGPU() override {
    if (emb_config_.is_primary()) {
      delete kv_;
      for (int i = 0; i < kv_->bank_ptrs.size(); ++i) {
        TypedAllocator::Deallocate(alloc_, kv_->bank_ptrs[i], kv_->initial_bank_size);
        TypedAllocator::Deallocate(alloc_, kv_->existence_flag_ptrs[i], kv_->initial_bank_size);
      }
      if (kv_->mem_bank_num != 0) {
        TypedAllocator::Deallocate(alloc_, kv_->d_bank_ptrs, kv_->mem_bank_num * emb_config_.total_num());
        TypedAllocator::Deallocate(alloc_, kv_->d_existence_flag_ptrs, kv_->mem_bank_num * emb_config_.total_num());
      }
    }
    TypedAllocator::Deallocate(alloc_, default_value_, value_len_);
  }
  TF_DISALLOW_COPY_AND_ASSIGN(EmbeddingVarGPU);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_KV_GPU_HASH_TABLE_H_
