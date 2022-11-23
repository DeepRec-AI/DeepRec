/* Copyright 2019 The DeepRec Authors. All Rights Reserved.

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


#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"
#include "tensorflow/core/framework/embedding/filter_factory.h"
#include "tensorflow/core/framework/embedding/gpu_hash_map_kv.h"
#include "tensorflow/core/framework/embedding/embedding_config.h"
#include "tensorflow/core/framework/embedding/storage_manager.h"
#include "tensorflow/core/framework/typed_allocator.h"

namespace tensorflow {

template <class K, class V>
class EmbeddingVar : public ResourceBase {
 public:
  EmbeddingVar(const string& name,
               embedding::StorageManager<K, V>* storage_manager,
               EmbeddingConfig emb_cfg = EmbeddingConfig(),
               Allocator* alloc = nullptr):
      name_(name),
      storage_manager_(storage_manager),
      default_value_(nullptr),
      default_value_no_permission_(nullptr),
      value_len_(0),
      alloc_(alloc),
      emb_config_(emb_cfg) {
    if (IsMultiLevel() || emb_config_.record_freq) {
      add_freq_fn_ = [](ValuePtr<V>* value_ptr, int freq, int64 filter_freq) {
        value_ptr->AddFreq(freq);
      };
    } else if (emb_config_.is_counter_filter()) {
      add_freq_fn_ = [](ValuePtr<V>* value_ptr, int freq, int64 filter_freq) {
        if (value_ptr->GetFreq() < filter_freq)
          value_ptr->AddFreq(freq);
      };
    } else {
      add_freq_fn_ = [](ValuePtr<V>* value_ptr, int freq, int64 filter_freq) {};
    }
    if (emb_config_.steps_to_live != 0 || emb_config_.record_version) {
      update_version_fn_ = [](ValuePtr<V>* value_ptr, int64 gs) {
        value_ptr->SetStep(gs);
      };
    } else {
      update_version_fn_ = [](ValuePtr<V>* value_ptr, int64 gs) {};
    }
  }

  Status Init(const Tensor& default_tensor, int64 default_value_dim) {
    if (storage_manager_ == nullptr) {
      return errors::InvalidArgument(
          "Invalid ht_type to construct EmbeddingVar");
    }

    storage_type_ = storage_manager_->GetStorageType();
    filter_ = FilterFactory::CreateFilter<K, V, EmbeddingVar<K, V>>(
        emb_config_, this, storage_manager_);
    emb_config_.default_value_dim = default_value_dim;
    value_len_ =
        default_tensor.NumElements() / emb_config_.default_value_dim;

    if (LayoutType::NORMAL_CONTIGUOUS == storage_manager_->GetLayoutType() ||
        LayoutType::NORMAL_CONTIGUOUS_GPU == storage_manager_->GetLayoutType()) {
      storage_manager_->SetAllocLen(value_len_, emb_config_.slot_num + 1);
    }

    if (storage_manager_->IsUseHbm()) {
#if GOOGLE_CUDA
      default_value_ = TypedAllocator::Allocate<V>(alloc_,
          default_tensor.NumElements(), AllocationAttributes());
      auto default_tensor_flat = default_tensor.flat<V>();
      buffer1_ = nullptr;
      buffer2_ = nullptr;
      buffer3_ = nullptr;
      buffer1_size_ = 0;
      buffer2_size_ = 0;
      buffer3_size_ = 0;
      cudaMemcpy(default_value_, &default_tensor_flat(0),
          default_tensor.TotalBytes(), cudaMemcpyDeviceToDevice);
      storage_manager_->
          CreateMemoryPool(alloc_,
                           emb_config_.total_num(
                               storage_manager_->GetAllocLen()),
                           1024 * 1024);
#endif  // GOOGLE_CUDA
    } else {
      alloc_ = ev_allocator();
      default_value_ = TypedAllocator::Allocate<V>(alloc_,
          default_tensor.NumElements(), AllocationAttributes());

      auto default_tensor_flat = default_tensor.flat<V>();
      memcpy(default_value_, &default_tensor_flat(0),
          default_tensor.TotalBytes());

      default_value_no_permission_ = TypedAllocator::Allocate<V>(alloc_,
          value_len_, AllocationAttributes());
      for (int i = 0; i < value_len_; ++i) {
        default_value_no_permission_[i] = static_cast<V>(
            emb_config_.default_value_no_permission);
      }
    }

    return Status::OK();
  }

  void SetInitialized() {
    is_initialized_ = true;
  }

  bool IsInitialized() const {
    return is_initialized_;
  }

  Status LookupKey(K key, ValuePtr<V>** value_ptr) {
    return storage_manager_->Get(key, value_ptr);
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** value_ptr, bool* is_filter) {
    return filter_->LookupOrCreateKey(key, value_ptr, is_filter);
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** value_ptr) {
    Status s = storage_manager_->GetOrCreate(key, value_ptr,
        emb_config_.total_num(storage_manager_->GetAllocLen()));
    TF_CHECK_OK(s);
    return s;
  }

  void UpdateVersion(ValuePtr<V>* value_ptr, int64 gs) {
    update_version_fn_(value_ptr, gs);
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** value_ptr,
      int64 update_version, embedding::CopyBackFlag &need_copyback) {
    Status s = storage_manager_->GetOrCreate(key, value_ptr,
        emb_config_.total_num(storage_manager_->GetAllocLen()), need_copyback);
    TF_CHECK_OK(s);
    if (emb_config_.is_primary() &&
        emb_config_.steps_to_live != 0 &&
        update_version != -1) {
      (*value_ptr)->SetStep(update_version);
    }
    return s;
  }

  void BatchCommit(const std::vector<K>& keys,
                   const std::vector<ValuePtr<V>*>& value_ptrs) {
    TF_CHECK_OK(storage_manager_->BatchCommit(keys, value_ptrs));
  }

  void Eviction(K* evict_ids, int64 evict_size) {
    TF_CHECK_OK(storage_manager_->Eviction(evict_ids, evict_size));
  }

  int64 GetVersion(K key) {
    ValuePtr<V>* value_ptr = nullptr;
    TF_CHECK_OK(LookupOrCreateKey(key, &value_ptr));
    return value_ptr->GetStep();
  }

  int64 GetFreq(K key) {
    return filter_->GetFreq(key);
  }

  Status Lookup(K key, V* val, V* default_v)  {
    const V* default_value_ptr =
      (default_v == nullptr) ? default_value_ : default_v;
    return filter_->Lookup(this, key, val, default_value_ptr,
                           default_value_no_permission_);
  }

  void LookupOrCreate(K key, V* val, V* default_v, int count = 1)  {
    const V* default_value_ptr =
      (default_v == nullptr) ? default_value_ : default_v;
    ValuePtr<V>* value_ptr = nullptr;
    filter_->LookupOrCreate(key, val, default_value_ptr, &value_ptr, count,
                            default_value_no_permission_);
    add_freq_fn_(value_ptr, count, emb_config_.filter_freq);
  }

  void LookupWithFreqBatch(const K* keys,
      V** memcpy_address, int64 start, int64 limit,
      std::list<int64>& init_cursor,
      std::list<int64>& copyback_cursor) {
    ValuePtr<V>* value_ptr = nullptr;
    for (int64 i = start; i < limit; i++) {
      embedding::CopyBackFlag copyback_flag =
          embedding::CopyBackFlag::NOT_COPYBACK;
      bool init_flag = false;
      TF_CHECK_OK(LookupOrCreateKey(keys[i], &value_ptr, -1, copyback_flag));
      value_ptr->AddFreq();
      if (!copyback_flag) {
        memcpy_address[i] = LookupOrCreateEmb(value_ptr, init_flag); 
      } else {
        //memcpy_address[i] = LookupOrCreateEmb(value_ptr, init_flags[i]);
        memcpy_address[i] = value_ptr->GetValue(0,0);
        if (copyback_flag ==
          embedding::CopyBackFlag::COPYBACK_AND_DESTROY) {
          delete value_ptr;
          // If the 64th bit of cursor is set to 1,
          // the corresponding valueptr need to be deleted later.
          int64 tmp = 1;
          tmp = tmp << 63;
          copyback_cursor.emplace_back(i | tmp);
        } else {
          copyback_cursor.emplace_back(i);
        }   
      }
      if (init_flag) {
        init_cursor.emplace_back(i);
      }
    }
  }

  void BatchInitEmb(int64 size, V** memcpy_address, V* default_value,
      bool* init_flags, int64 value_len) {
    filter_->BatchInitEmb(size, memcpy_address, default_value,
        init_flags, value_len);
  }

#if GOOGLE_CUDA
  void CreateGPUBatch(V* val_base, int64 size,
      int64 slice_elems, V** memcpy_address) {
    filter_->CreateGPUBatch(val_base, size,
        slice_elems, value_len_, memcpy_address);
  }

  void InitializeEmbeddingOnGPU(const K* keys, int64 size,
       const std::list<int64>& init_cursor,
       V** memcpy_address, V* default_values,
       std::function<V*(V*, K, int64, int64, int64)> get_default_v_fn) {
    V** dev_default_value_address, **default_value_address;
    V** dev_value_address, **value_address;
    if (init_cursor.size() > 0) {
      int64 total = init_cursor.size();
      value_address = (V**)malloc(sizeof(V*) * total);
      default_value_address = (V**)malloc(sizeof(V*) * total);
      dev_value_address = TypedAllocator::Allocate<V*>(alloc_,
              total, AllocationAttributes());
      dev_default_value_address = TypedAllocator::Allocate<V*>(alloc_,
              total, AllocationAttributes());
      int64 i = 0;
      auto it = init_cursor.cbegin();
      for ( ; it != init_cursor.cend(); ++it, ++i) {
        ValuePtr<V>* value_ptr =
            reinterpret_cast<ValuePtr<V>*>(memcpy_address[*it]);
        value_address[i] = *((V**)((char*)(value_ptr->GetPtr())
                            + sizeof(FixedLengthHeader)))
                            + storage_manager_->GetOffset(emb_config_.emb_index);
        default_value_address[i] = get_default_v_fn(
                                       default_values,
                                       keys[*it],
                                       *it,
                                       GetDefaultValueDim(),
                                       ValueLen());
      }
      cudaMemcpy(dev_value_address, value_address, sizeof(V*) * total,
          cudaMemcpyHostToDevice);
      cudaMemcpy(dev_default_value_address, default_value_address,
                 sizeof(V*) * total, cudaMemcpyHostToDevice);
      int block_dim = 128;
      void* args[] = {(void*)&dev_default_value_address,
                       (void*)&dev_value_address,
                       (void*)&value_len_,
                       (void*)&total};
      cudaLaunchKernel((void *)CopyEmbedding<V>,
                       (total * value_len_ + block_dim - 1) / block_dim,
                       block_dim, args, 0, NULL);
      cudaDeviceSynchronize();
      // Set init meta of ValuePtrs
      for (auto it = init_cursor.cbegin();
          it != init_cursor.cend(); ++it) {
        ValuePtr<V>* value_ptr =
            reinterpret_cast<ValuePtr<V>*>(memcpy_address[*it]);
        value_ptr->SetInitialized(emb_config_.emb_index);
        memcpy_address[*it] = value_ptr->GetValue(
            emb_config_.emb_index,
            storage_manager_->GetOffset(emb_config_.emb_index));
      }

      TypedAllocator::Deallocate(alloc_, dev_value_address, total);
      TypedAllocator::Deallocate(alloc_, dev_default_value_address, total);
      free(value_address);
      free(default_value_address);
    }
  }

  void CopyBackToGPU(const K* keys,
      const std::list<int64>& copyback_cursor,
      V** memcpy_address) {
    size_t value_len = emb_config_.total_num(storage_manager_->GetAllocLen());
    V* memcpy_buffer_gpu;
    V** dev_value_address, **value_address;
    if (copyback_cursor.size() > 0) {
      int64 total = copyback_cursor.size();
      ValuePtr<V>** gpu_value_ptrs = new ValuePtr<V>* [total];
      memcpy_buffer_gpu = (V*)alloc_->AllocateRaw(Allocator::kAllocatorAlignment,
          total * value_len * sizeof(V));
      storage_manager_->CopyBackToGPU(total, keys, copyback_cursor,
          memcpy_address, value_len, gpu_value_ptrs,
          memcpy_buffer_gpu);

      value_address = (V**)malloc(sizeof(V*) * total);
      dev_value_address = (V**)alloc_->AllocateRaw(Allocator::kAllocatorAlignment,
          sizeof(V*) * total);
      std::vector<K> copyback_keys;
      int64 i = 0;
      auto it = copyback_cursor.cbegin();
      for (; it != copyback_cursor.cend(); ++it, ++i) {
        bool init;
        // Get the curosr
        int64 cursor = *it & 0x0fffffffffffffff;
        gpu_value_ptrs[i]->SetInitialized(emb_config_.emb_index);
        memcpy_address[cursor] = LookupOrCreateEmb(
            gpu_value_ptrs[i], init);
        value_address[i] = memcpy_address[cursor];
        copyback_keys.emplace_back(keys[cursor]);
      }

      cudaMemcpy(dev_value_address, value_address, sizeof(V*) * total,
          cudaMemcpyHostToDevice);
      int block_dim = 128;
      void* args[] = { (void*)&dev_value_address,
        (void*)&memcpy_buffer_gpu, (void*)&value_len, (void*)&total};

      cudaLaunchKernel((void *)BatchUnpack<V>,
          (total + block_dim - 1) / block_dim * value_len, block_dim,
          args, 0, NULL);
      cudaDeviceSynchronize();

      storage_manager_->Insert(copyback_keys, gpu_value_ptrs);
      alloc_->DeallocateRaw(dev_value_address);
      alloc_->DeallocateRaw(memcpy_buffer_gpu);
      delete []gpu_value_ptrs;
    }
  }

  void AllocateMemory(V** memcpy_address,
                      const std::list<int64>& init_cursor) {
    std::vector<ValuePtr<V>*> value_ptr_list;
    for (auto it = init_cursor.cbegin();
      it != init_cursor.cend(); ++it) {
      ValuePtr<V>* value_ptr =
          reinterpret_cast<ValuePtr<V>*>(memcpy_address[*it]);
      value_ptr_list.emplace_back(value_ptr);
    }
    storage_manager_->AllocateMemory(value_ptr_list);
  }
#endif  // GOOGLE_CUDA

  V* LookupOrCreateEmb(ValuePtr<V>* value_ptr, const V* default_v) {
    return value_ptr->GetOrAllocate(alloc_, value_len_, default_v,
        emb_config_.emb_index, storage_manager_->GetOffset(
          emb_config_.emb_index));
  }

  V* LookupOrCreateEmb(ValuePtr<V>* value_ptr, bool &need_initialize) {
    return value_ptr->GetOrAllocate(alloc_, value_len_, nullptr,
        emb_config_.emb_index,
        storage_manager_->GetOffset(emb_config_.emb_index),
        need_initialize);
  }

  V* LookupPrimaryEmb(ValuePtr<V>* value_ptr) {
    V* primary_val = value_ptr->GetValue(emb_config_.primary_emb_index,
        storage_manager_->GetOffset(emb_config_.primary_emb_index));
    return primary_val;
  }

  typename TTypes<V>::Flat flat(ValuePtr<V>* value_ptr) {
    V* val = LookupOrCreateEmb(value_ptr, default_value_);
    Eigen::array<Eigen::DenseIndex, 1> dims({value_len_});
    return typename TTypes<V>::Flat(val, dims);
  }

  int64 ValueLen() const {
    return value_len_;
  }

  int64 Size() const {
    return storage_manager_->Size();
  }

  int64 CacheSize() const {
    return storage_manager_->CacheSize();
  }

  int64 MinFreq() {
    return emb_config_.filter_freq;
  }

  int64 StepsToLive() const {
    return emb_config_.steps_to_live;
  }

  float GetL2WeightThreshold() {
    return emb_config_.l2_weight_threshold;
  }

  bool IsMultiLevel() {
    return storage_manager_->IsMultiLevel();
  }

  bool IsUseHbm() {
    return storage_manager_->IsUseHbm();
  }

  void InitCache(embedding::CacheStrategy cache_strategy) {
    storage_manager_->InitCache(cache_strategy);
  }

  std::string DebugString() const {
    return emb_config_.DebugString();
  }

  Status Import(RestoreBuffer& restore_buff,
                int64 key_num,
                int bucket_num,
                int64 partition_id,
                int64 partition_num,
                bool is_filter) {
    return filter_->Import(restore_buff, key_num, bucket_num,
        partition_id, partition_num, is_filter);
  }

  int64 GetSnapshot(std::vector<K>* key_list,
                    std::vector<V* >* value_list,
                    std::vector<int64>* version_list,
                    std::vector<int64>* freq_list,
                    embedding::Iterator** it) {
    return storage_manager_->GetSnapshot(key_list, value_list, version_list,
                                         freq_list, emb_config_, filter_, it);
  }

  mutex* mu() {
    return &mu_;
  }

  embedding::StorageManager<K, V>* storage_manager() {
    return storage_manager_;
  }

  Status Shrink() {
    return storage_manager_->Shrink(emb_config_, value_len_);
  }

  Status Shrink(int64 gs) {
    if (emb_config_.steps_to_live > 0) {
      return storage_manager_->Shrink(gs, emb_config_.steps_to_live);
    }
  }

  V* GetDefaultValuePtr() {
    return default_value_;
  }

  int64 GetDefaultValueDim() {
    return emb_config_.default_value_dim;
  }

  V* GetDefaultValue(int64 key) {
    return default_value_ + (key % emb_config_.default_value_dim) * value_len_;
  }

  void SetSlotNum(int64 slot_num) {
    emb_config_.slot_num = slot_num;
  }

  int64 GetSlotNum() {
    return emb_config_.slot_num;
  }

  embedding::BatchCache<K>* Cache() {
    return storage_manager_->Cache();
  }

  int64 GetEmbeddingIndex() {
    return emb_config_.emb_index;
  }

  Allocator* GetAllocator() {
    return alloc_;
  }

  int64 GetAllocLen() {
    return emb_config_.total_num(storage_manager_->GetAllocLen());
  }

  V** GetBuffer1(int64 size) {
    if (buffer1_size_ >= size) {
      return buffer1_;
    } else{
      if (buffer1_size_ != 0) {
        alloc_->DeallocateRaw(buffer1_);
      }
      buffer1_ = (V**)alloc_->AllocateRaw(Allocator::kAllocatorAlignment,
        size * sizeof(V*));
      buffer1_size_ = size;
      return buffer1_;
    }
  }

  V** GetBuffer2(int64 size) {
    if (buffer2_size_ >= size) {
      return buffer2_;
    } else {
      if (buffer2_size_ != 0) {
        alloc_->DeallocateRaw(buffer2_);
      }
      buffer2_ =(V**)alloc_->AllocateRaw(Allocator::kAllocatorAlignment,
        size * sizeof(V*));
      buffer2_size_ = size;
      return buffer2_;
    }
  }

  V** GetBuffer3(int64 size) {
    if (buffer3_size_ >= size) {
      return buffer3_;
    } else {
      if (buffer3_size_ != 0) {
        alloc_->DeallocateRaw(buffer3_);
      }
      buffer3_ = (V**)alloc_->AllocateRaw(Allocator::kAllocatorAlignment,
        size * sizeof(V*));
      buffer3_size_ = size;
      return buffer3_;
    }
  }

  void UpdateCache(const K* key_buff, int64 key_num,
      const int64* version_buff, const int64* freq_buff) {
    auto cache = Cache();
    if (cache) {
      cache->add_to_rank(key_buff, key_num, version_buff, freq_buff);
      auto cache_size = CacheSize();
      if (cache->size() > cache_size) {
        int64 evict_size = cache->size() - cache_size;
        K* evict_ids = new K[evict_size];
        size_t true_size = cache->get_evic_ids(evict_ids, evict_size);
        Eviction(evict_ids, true_size);
        delete []evict_ids;
      }
    }
  }

 protected:
  FilterPolicy<K, V, EmbeddingVar<K, V>>* GetFilter() const {
    return filter_;
  }

  ~EmbeddingVar() override {
    // When dynamic dimension embedding is used,
    // there will be more than one primary slot
    if (emb_config_.is_primary() && emb_config_.primary_emb_index == 0) {
      delete storage_manager_;
    }
    if (embedding::StorageType::HBM_DRAM == storage_type_) {
      buffer1_size_ = 0;
      buffer2_size_ = 0;
      buffer3_size_ = 0;
      if (buffer1_ != nullptr) {
        alloc_->DeallocateRaw(buffer1_);
        buffer1_ = nullptr;
      }
      if (buffer2_ != nullptr) {
        alloc_->DeallocateRaw(buffer2_);
        buffer2_ = nullptr;
      }
      if (buffer3_ != nullptr) {
        alloc_->DeallocateRaw(buffer3_);
        buffer3_ = nullptr;
      }
    }
    TypedAllocator::Deallocate(alloc_, default_value_,
        value_len_ * emb_config_.default_value_dim);
    if (default_value_no_permission_) {
      TypedAllocator::Deallocate(alloc_, default_value_no_permission_,
          value_len_);
    }
  }

 private:
  std::string name_;
  bool is_initialized_ = false;

  mutex mu_;

  V* default_value_;
  V* default_value_no_permission_;
  V** buffer1_;
  V** buffer2_;
  V** buffer3_;
  int64 buffer1_size_;
  int64 buffer2_size_;
  int64 buffer3_size_;
  int64 value_len_;
  Allocator* alloc_;
  embedding::StorageManager<K, V>* storage_manager_;
  embedding::StorageType storage_type_;
  EmbeddingConfig emb_config_;
  FilterPolicy<K, V, EmbeddingVar<K, V>>* filter_;
  std::function<void(ValuePtr<V>*, int, int64)> add_freq_fn_;
  std::function<void(ValuePtr<V>*, int64)> update_version_fn_;

  TF_DISALLOW_COPY_AND_ASSIGN(EmbeddingVar);
};

}  // namespace tensorflow

#if GOOGLE_CUDA
namespace tensorflow {

template <class K, class V>
class EmbeddingVarGPU : public ResourceBase {
 public:
  EmbeddingVarGPU(const string& name,
                  embedding::GPUHashMapKV<K, V>* kv,
                  Allocator* alloc,
                  const EmbeddingConfig& emb_cfg = EmbeddingConfig()):
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

  Status Init(const Tensor& default_tensor,
      int64 default_value_dim=1) {
    if (DataTypeToEnum<V>::v() != default_tensor.dtype()) {
       return errors::InvalidArgument(
           "EV's default_tensor DTYPE must be same as EmbeddingVar Value Type");
    } else if (kv_ == nullptr) {
       return errors::InvalidArgument("Error to construct EmbeddingVarGPU");
    } else {
      emb_config_.default_value_dim = default_value_dim;
      value_len_ =
        default_tensor.NumElements() / emb_config_.default_value_dim;
      kv_->SetValueLen(value_len_);
      default_value_ = TypedAllocator::Allocate<V>(
          alloc_, default_tensor.NumElements(), AllocationAttributes());
      auto default_tensor_flat = default_tensor.flat<V>();
      cudaMemcpy(default_value_, &default_tensor_flat(0),
          default_tensor.TotalBytes(), cudaMemcpyDeviceToDevice);
      return Status::OK();
    }
  }

  void SetInitialized() {
    is_initialized_ = true;
  }

  bool IsInitialized() const {
    return is_initialized_;
  }

  void LookupOrCreateKey(const K* key, int32* item_idxs, size_t n,
      const Eigen::GpuDevice& device, int64 update_version = -1) {
    kv_->BatchLookupOrCreateKeys(key, n, item_idxs, device);
  }

  void LookupOrCreate(const K* key, V* val, V* default_v,
      int32 default_v_num, bool is_use_default_value_tensor,
      size_t n, const Eigen::GpuDevice& device) {
    kv_->BatchLookupOrCreate(key, val, default_v, default_v_num,
        is_use_default_value_tensor, n, device);
  }

  void GetSnapshot(K* keys, V* values, const Eigen::GpuDevice& device) {
    kv_->GetSnapshot(keys, values, device);
  }

  int64 Size() const {
    return kv_->Size();
  }

  int64 ValueLen() const {
    return value_len_;
  }

  std::string DebugString() const {
    return emb_config_.DebugString();
  }

  embedding::GPUHashMapKV<K, V>* kv() {
    return kv_;
  }

  int64 MinFreq() {
    return emb_config_.filter_freq;
  }

  float GetL2WeightThreshold() {
    return emb_config_.l2_weight_threshold;
  }

  int32 SlotNum() {
    return (emb_config_.block_num * (1 + emb_config_.slot_num));
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

  int64 GetSlotNum() {
    return emb_config_.slot_num;
  }

  V* GetDefaultValuePtr() {
    return default_value_;
  }

  int64 GetDefaultValueDim() {
    return emb_config_.default_value_dim;
  }

  Status Import(RestoreBuffer& restore_buff, int64 key_num,
      int bucket_num, int64 partition_id, int64 partition_num,
      bool is_filter, const Eigen::GpuDevice& device) {
    return kv_->Import(restore_buff, key_num, bucket_num,
        partition_id, partition_num, is_filter, device);
  }

 private:
  ~EmbeddingVarGPU() override {
    delete kv_;
    TypedAllocator::Deallocate(alloc_, default_value_, value_len_);
  }
  TF_DISALLOW_COPY_AND_ASSIGN(EmbeddingVarGPU);

 private:
  bool is_initialized_ = false;
  std::string name_;
  embedding::GPUHashMapKV<K, V>* kv_ = nullptr;
  Allocator* alloc_ = nullptr;
  EmbeddingConfig emb_config_;
  V* default_value_ = nullptr;
  int64 value_len_;
};

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_H_
