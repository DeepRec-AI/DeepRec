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
#include "tensorflow/core/framework/embedding/embedding_var_context.h"
#include "tensorflow/core/framework/embedding/embedding_var_restore.h"
#include "tensorflow/core/framework/embedding/filter_factory.h"
#include "tensorflow/core/framework/embedding/gpu_hash_map_kv.h"
#include "tensorflow/core/framework/embedding/embedding_config.h"
#include "tensorflow/core/framework/embedding/storage.h"
#include "tensorflow/core/framework/typed_allocator.h"

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#if GOOGLE_CUDA
  void SyncWithEventMgr(se::Stream* stream,
      EventMgr* event_mgr);
#endif //GOOGLE_CUDA


template <class K, class V>
class GPUHashTable;

template <class K, class V>
class EmbeddingVar : public ResourceBase {
 public:
  EmbeddingVar(const string& name,
               embedding::Storage<K, V>* storage,
               EmbeddingConfig emb_cfg,
               Allocator* alloc,
               embedding::FeatureDescriptor<V>* feat_desc):
      name_(name),
      storage_(storage),
      default_value_(nullptr),
      default_value_no_permission_(nullptr),
      value_len_(0),
      alloc_(alloc),
      default_value_alloc_(alloc),
      emb_config_(emb_cfg),
      feat_desc_(feat_desc) {}

  Status Init(const Tensor& default_tensor, int64 default_value_dim) {
    if (storage_ == nullptr) {
      return errors::InvalidArgument(
          "Invalid ht_type to construct EmbeddingVar");
    }

    storage_type_ = storage_->GetStorageType();
    filter_ = FilterFactory::CreateFilter<K, V, EmbeddingVar<K, V>>(
        emb_config_, this, storage_, feat_desc_);
    emb_config_.default_value_dim = default_value_dim;
    value_len_ =
        default_tensor.NumElements() / emb_config_.default_value_dim;

    if (storage_->IsUseHbm()) {
#if GOOGLE_CUDA
      default_value_ = TypedAllocator::Allocate<V>(alloc_,
          default_tensor.NumElements(), AllocationAttributes());
      auto default_tensor_flat = default_tensor.flat<V>();
      dev_addr_buffer_ = nullptr;
      dev_addr_buffer_size_ = 0;
      cudaMemcpy(default_value_, &default_tensor_flat(0),
          default_tensor.TotalBytes(), cudaMemcpyDeviceToDevice);
#endif  // GOOGLE_CUDA
    } else if (storage_->IsSingleHbm()) {
#if GOOGLE_CUDA
      storage_->SetValueLen(value_len_);
      default_value_ = TypedAllocator::Allocate<V>(
          alloc_, default_tensor.NumElements(), AllocationAttributes());
      auto default_tensor_flat = default_tensor.flat<V>();
      cudaMemcpy(default_value_, &default_tensor_flat(0),
          default_tensor.TotalBytes(), cudaMemcpyDeviceToDevice);
#endif  // GOOGLE_CUDA
    } else {
      alloc_ = ev_allocator();
      default_value_ = TypedAllocator::Allocate<V>(default_value_alloc_,
          default_tensor.NumElements(), AllocationAttributes());

      auto default_tensor_flat = default_tensor.flat<V>();
      memcpy(default_value_, &default_tensor_flat(0),
          default_tensor.TotalBytes());

      default_value_no_permission_ = TypedAllocator::Allocate<V>(
          default_value_alloc_, value_len_, AllocationAttributes());
      for (int i = 0; i < value_len_; ++i) {
        default_value_no_permission_[i] = static_cast<V>(
            emb_config_.default_value_no_permission);
      }
    }
    bool is_all_slots_initialized = 
        feat_desc_->InitSlotInfo(
            emb_config_.emb_index, value_len_,
            std::pair<V*, int64>(
                default_value_, emb_config_.default_value_dim));
    if (is_all_slots_initialized) {
      storage_->Init();
    }

    return Status::OK();
  }

  void SetInitialized() {
    is_initialized_ = true;
  }

  bool IsInitialized() const {
    return is_initialized_;
  }

  Status LookupKey(K key, void** value_ptr) {
    return storage_->Get(key, value_ptr);
  }

  Status LookupOrCreateKey(K key, void** value_ptr,
                           bool* is_filter, bool indices_as_pointer,
                           int64 count = 1) {
    if (indices_as_pointer) {
      *value_ptr = (void*)key;
      *is_filter = filter_->is_admit(key, *value_ptr);
      return Status::OK();
    } else {
      Status s = filter_->LookupOrCreateKey(key, value_ptr, is_filter, count);
      return s;
    }
  }

  Status Insert(K key, V* value) {
    void* value_ptr = nullptr;
    CreateKey(key, &value_ptr, true);
    feat_desc_->SetValue(value_ptr, emb_config_.emb_index, value);
    return Status::OK();
  }

  Status LookupOrCreateKey(K key, void** value_ptr) {
    Status s = storage_->GetOrCreate(key, value_ptr);
    TF_CHECK_OK(s);
    return s;
  }

  void CreateKey(K key, void** value_ptr, bool to_dram) {
    storage_->CreateAndInsert(key, value_ptr, to_dram);
  }

  void UpdateVersion(void* value_ptr, int64 gs) {
    feat_desc_->UpdateVersion(value_ptr, gs);
  }

  void BatchCommit(const std::vector<K>& keys,
                   const std::vector<void*>& value_ptrs) {
    TF_CHECK_OK(storage_->BatchCommit(keys, value_ptrs));
  }

  void Eviction(K* evict_ids, int64 evict_size) {
    TF_CHECK_OK(storage_->Eviction(evict_ids, evict_size));
  }

  int64 GetVersion(K key) {
    void* value_ptr = nullptr;
    TF_CHECK_OK(LookupOrCreateKey(key, &value_ptr));
    return feat_desc_->GetVersion(value_ptr);
  }

  int64 GetFreq(K key) {
    return filter_->GetFreq(key);
  }

  Status Lookup(K key, V* val, V* default_v)  {
    const V* default_value_ptr =
      (default_v == nullptr) ? default_value_ : default_v;
    return filter_->Lookup(key, val, default_value_ptr,
                           default_value_no_permission_);
  }

  void GetEmbeddings(const EmbeddingVarContext<CPUDevice>& context,
                     const K* keys, V* output,
                     int64 num_of_keys) {
    auto do_work = [this, keys, output] (int64 start, int64 limit) {
      for (int64 i = start; i < limit; ++i) {
        V* default_v =
            default_value_ +
                (keys[i] % emb_config_.default_value_dim) * value_len_;
        filter_->Lookup(keys[i],
            output + i * value_len_, default_v,
            default_value_no_permission_);
      }
    };
    auto worker_threads = context.worker_threads;
    Shard(worker_threads->num_threads,
          worker_threads->workers, num_of_keys,
          value_len_ * sizeof(V), do_work);
  }

//Used for CPU Adaptive Embedding
  void GetEmbeddings(const EmbeddingVarContext<CPUDevice>& context,
                     const K* keys, V* output,
                     int64 num_of_keys, V* default_value) {
    auto do_work = [this, keys, output, default_value]
        (int64 start, int64 limit) {
      for (int64 i = start; i < limit; ++i) {
        V* default_v = default_value + i * value_len_;
        void* value_ptr = nullptr;
        filter_->LookupOrCreate(
            keys[i], output + i * value_len_, default_v, &value_ptr, 1,
            default_value_no_permission_);
        feat_desc_->AddFreq(value_ptr, 1);
      }
    };
    auto worker_threads = context.worker_threads;
    Shard(worker_threads->num_threads,
          worker_threads->workers, num_of_keys,
          value_len_ * sizeof(V), do_work);
  }

  void GetOrCreateKey(const EmbeddingVarContext<CPUDevice>& context,
                      const Tensor& keys_tensor,
                      void** value_ptrs,
                      int64 num_of_keys) {
    const K* keys = (K*)keys_tensor.data();
    auto do_work = [this, keys, value_ptrs] (int64 start, int64 limit) {
      for (int64 i = start; i < limit; ++i) {
        bool is_filter = false;
        filter_->LookupOrCreateKey(keys[i], &value_ptrs[i], &is_filter, 1);
      }
    };
    auto worker_threads = context.worker_threads;
    Shard(worker_threads->num_threads,
          worker_threads->workers,
          num_of_keys, value_len_ * sizeof(V), do_work);

    storage_->AddToCachePrefetchList(keys_tensor);
  }

  void GatherEmbeddings(const EmbeddingVarContext<CPUDevice>& context,
                        const Tensor& keys_tensor,
                        void** value_ptrs,
                        V* output,
                        int64 num_of_keys) {
    const K* keys = (K*)keys_tensor.data();
    auto do_work = [this, keys, value_ptrs, output]
        (int64 start, int64 limit) {
      for (int64 i = start; i < limit; ++i) {
        bool is_admit = filter_->is_admit(keys[i], value_ptrs[i]);
        V* value = nullptr;
        if (is_admit) {
          value = feat_desc_->GetEmbedding(
              value_ptrs[i], emb_config_.emb_index);
        } else {
          value = default_value_no_permission_;
        }
        memcpy(output + i * value_len_, value, sizeof(V) * value_len_);
      }
    };
    auto worker_threads = context.worker_threads;
    Shard(worker_threads->num_threads,
          worker_threads->workers, num_of_keys,
          value_len_ * sizeof(V), do_work);

    storage_->AddToCache(keys_tensor);
  }

#if GOOGLE_CUDA
  void GetEmbeddings(const EmbeddingVarContext<GPUDevice>& context,
                     const K* keys,
                     V* output,
                     int64 num_of_keys) {
    if (IsSingleHbm()) {
      storage_->BatchLookup(context.gpu_device, keys, 
		            output, num_of_keys, default_value_);
    } else {
      filter_->BatchLookup(context, keys, output,
                           num_of_keys, default_value_,
                           default_value_no_permission_);
    }
  }

  void GetOrCreateKey(const EmbeddingVarContext<GPUDevice>& context,
                      const Tensor& keys_tensor,
                      void** value_ptrs,
                      int64 num_of_keys,
                      bool indices_as_pointer = false) {
    const K* keys = (K*)keys_tensor.data();
    filter_->BatchLookupOrCreateKey(context, keys, value_ptrs, num_of_keys);
    storage_->AddToCachePrefetchList(keys_tensor);
  }

  void BatchLookupOrCreateKey(
      const EmbeddingVarContext<GPUDevice>& context,
      const K* keys,
      void** value_ptrs,
      int64 num_of_keys,
      std::vector<std::list<int64>>& not_found_cursor_list) {
    storage_->BatchGetOrCreate(context, keys, value_ptrs, num_of_keys,
                               value_len_,
                               not_found_cursor_list);
  }

  void GatherEmbeddings(const EmbeddingVarContext<GPUDevice>& context,
                        const Tensor& keys_tensor,
                        void** value_ptrs,
                        V* output,
                        int64 num_of_keys) {
    std::vector<V*> embedding_ptr(num_of_keys);
    const K* keys = (K*)keys_tensor.data();
    auto do_work = [this, keys, value_ptrs, output, &embedding_ptr]
        (int64 start, int64 limit) {
      for (int64 i = start; i < limit; ++i) {
        bool is_admit = filter_->is_admit(keys[i], value_ptrs[i]);
        feat_desc_->AddFreq(value_ptrs[i], 1);
        if (is_admit) {
          embedding_ptr[i] = feat_desc_->GetEmbedding(
              value_ptrs[i], emb_config_.emb_index);
        } else {
          embedding_ptr[i] = default_value_no_permission_;
        }
      }
    };
    auto worker_threads = context.worker_threads;
    Shard(worker_threads->num_threads,
          worker_threads->workers, num_of_keys,
          value_len_ * sizeof(V), do_work);

    auto stream = context.compute_stream;
    auto event_mgr = context.event_mgr;
    CopyEmbeddingsToBuffer(
        output, num_of_keys, embedding_ptr.data(),
        stream, event_mgr, context.gpu_device);

    storage_->AddToCache(keys_tensor);
  }

  void BatchLookupKey(const EmbeddingVarContext<GPUDevice>& ctx,
                      const K* keys,
                      void** value_ptr_list,
                      int64 num_of_keys) {
    storage_->BatchGet(ctx, keys, value_ptr_list, num_of_keys);
  }

  Status LookupOrCreateKey(const EmbeddingVarContext<GPUDevice>& context,
                           const K* keys,
                           void** value_ptrs,
                           int64 num_of_keys,
                           int64* indices_counts,
                           bool indices_as_pointer = false) {
    if (indices_as_pointer) {
      auto lookup_key_and_set_version_fn = [keys, value_ptrs]
          (int64 start, int64 limit) {
        for (int i = start; i < limit; i++) {
          value_ptrs[i] = (void*)keys[i];
        }
      };
      const int64 unit_cost = 1000; //very unreliable estimate for cost per step.
      auto worker_threads = context.worker_threads;
      Shard(worker_threads->num_threads,
            worker_threads->workers, num_of_keys, unit_cost,
            lookup_key_and_set_version_fn);
    } else {
      filter_->BatchLookupOrCreateKey(context, keys, value_ptrs, num_of_keys);
    }

    if (indices_counts != nullptr) {
      auto add_freq_fn = [this, value_ptrs, indices_counts]
          (int64 start, int64 limit) {
        for (int i = start; i < limit; i++) {
          feat_desc_->AddFreq(value_ptrs[i], indices_counts[i]);
        }
      };
      const int64 unit_cost = 1000; //very unreliable estimate for cost per step.
      auto worker_threads = context.worker_threads;
      Shard(worker_threads->num_threads,
            worker_threads->workers, num_of_keys, unit_cost,
            add_freq_fn);
    }
    return Status::OK();
  }
#endif

#if GOOGLE_CUDA
  void CopyEmbeddingsToBuffer(
      V* val_base, int64 size,
      V** memcpy_address,
      se::Stream* compute_stream,
      EventMgr* event_mgr,
      const Eigen::GpuDevice& gpu_device);
#endif  // GOOGLE_CUDA

  typename TTypes<V>::Flat flat(void* value_ptr) {
    V* val = feat_desc_->GetEmbedding(value_ptr, emb_config_.emb_index);
    Eigen::array<Eigen::DenseIndex, 1> dims({value_len_});
    return typename TTypes<V>::Flat(val, dims);
  }

  V* GetValuePtr(void* ptr) {
    return feat_desc_->GetEmbedding(ptr, emb_config_.emb_index);
  }

  int64 ValueLen() const {
    return value_len_;
  }

  int64 Size() const {
    return storage_->Size();
  }

  int64 CacheSize() const {
    return storage_->CacheSize();
  }

  int64 MemoryUsage() const {
    return storage_->Size() * (sizeof(K) + feat_desc_->data_bytes());
  }

  int64 MinFreq() {
    return emb_config_.filter_freq;
  }

  int64 StepsToLive() const {
    return emb_config_.steps_to_live;
  }

  bool IsMultiLevel() {
    return storage_->IsMultiLevel();
  }

  bool IsUseHbm() {
    return storage_->IsUseHbm();
  }

  bool IsSingleHbm() {
    return storage_->IsSingleHbm();
  }

  bool IsUsePersistentStorage() {
    return storage_->IsUsePersistentStorage();
  }

  void InitCache(embedding::CacheStrategy cache_strategy) {
    storage_->InitCache(cache_strategy);
  }

  std::string DebugString() const {
    return emb_config_.DebugString();
  }

  void Restore(const std::string& name_string,
               const std::string& file_name_string, int64 partition_id,
               int64 partition_num, bool is_incr, BundleReader* reader,
               bool reset_version = false,
               const Eigen::GpuDevice* device = nullptr) {
    return storage_->Restore(name_string, file_name_string, partition_id,
                             partition_num, value_len_, is_incr, reset_version,
                             emb_config_, device, reader, this, filter_);
  }

  Status Save(const string& tensor_name,
              const string& prefix,
              BundleWriter* writer,
              embedding::ShrinkArgs& shrink_args) {
    return storage_->Save(tensor_name, prefix,
                          writer, emb_config_,
                          shrink_args, value_len_,
                          default_value_);
  }

  void GetSnapshot(std::vector<K>* key_list,
                   std::vector<V*>* value_list,
                   std::vector<int64>* version_list,
                   std::vector<int64>* freq_list) {
    std::vector<void*> value_ptr_list;
    storage_->GetSnapshot(key_list, &value_ptr_list);
    bool is_save_freq = emb_config_.is_save_freq();
    bool is_save_version = emb_config_.is_save_version();
    for (int64 i = 0; i < key_list->size(); i++) {
      if (feat_desc_->IsAdmit(value_ptr_list[i])) {
        V* val = feat_desc_->GetEmbedding(
            value_ptr_list[i], emb_config_.emb_index);
        value_list->emplace_back(val);
      } else {
        value_list->emplace_back(default_value_);
      }

      if(is_save_version) {
        int64 dump_version = feat_desc_->GetVersion(value_ptr_list[i]);
        version_list->emplace_back(dump_version);
      }

      if(is_save_freq) {
        int64 dump_freq = feat_desc_->GetFreq(value_ptr_list[i]);
        freq_list->emplace_back(dump_freq);
      }
    }
  }

  Status GetShardedSnapshot(std::vector<std::vector<K>>& key_list,
                            std::vector<std::vector<void*>>& value_ptr_list,
                            int partition_id, int partition_num) {
    return storage_->GetShardedSnapshot(key_list, value_ptr_list,
                                        partition_id, partition_num);
  }

  void ExportAndRemove(K* key_list, V* value_list,
                     int64* version_list, int64* freq_list,
                     std::vector<K>& tot_keys_list,
                     std::vector<void*>& tot_value_ptr_list) {
    bool save_unfiltered_features = true;
    TF_CHECK_OK(ReadBoolFromEnvVar(
        "TF_EV_SAVE_FILTERED_FEATURES", true, &save_unfiltered_features));

    bool is_save_freq = emb_config_.is_save_freq();
    bool is_save_version = emb_config_.is_save_version();

    for (int64 i = 0; i < tot_keys_list.size(); ++i) {
      auto& value_ptr = tot_value_ptr_list[i];
      if((int64)value_ptr == embedding::ValuePtrStatus::IS_DELETED)
        continue;

      bool is_admit = feat_desc_->IsAdmit(value_ptr);
      bool is_in_dram = ((int64)value_ptr >> kDramFlagOffset == 0);

      if (is_admit) {
        key_list[i] = tot_keys_list[i];
        
        if (!is_in_dram) {
          auto tmp_value = value_list + i * value_len_;
          tmp_value = (V*)embedding::ValuePtrStatus::NOT_IN_DRAM;
          value_ptr = (void*)((int64)value_ptr & ((1L << kDramFlagOffset) - 1));
        } else if (feat_desc_->GetEmbedding(value_ptr, 0) == nullptr) {
          memcpy(value_list + i * value_len_, default_value_, sizeof(V) * value_len_);
        } else {
          V* val = feat_desc_->GetEmbedding(value_ptr, emb_config_.emb_index);
          memcpy(value_list + i * value_len_, val, sizeof(V) * value_len_);
        }

        if(is_save_version) {
          int64 dump_version = feat_desc_->GetVersion(value_ptr);
          version_list[i] = dump_version;
        }

        if(is_save_freq) {
          int64 dump_freq = feat_desc_->GetFreq(value_ptr);
          freq_list[i] = dump_freq;
        }
      } else {
        if (!save_unfiltered_features)
          continue;
        //TODO(JUNQI) : currently not export filtered keys
      }

      if (emb_config_.is_primary()) {
        Status s;
        s = storage_->Remove(tot_keys_list[i]);
        if (!s.ok()) {
          LOG(ERROR) << "Remove keys error: " << s.error_message();
        }
        feat_desc_->Deallocate(value_ptr);
      }
    }
    return;
  }

  Status RestoreFromKeysAndValues(int64 key_num, int partition_id,
                                  int partition_num, const K* key_list,
                                  const V* value_list, const int64* version_list,
                                  const int64* freq_list,
                                  const Eigen::GpuDevice* device = nullptr) {
    RestoreBuffer restore_buff((char*)key_list, (char*)value_list,
                                (char*)version_list, (char*)freq_list);
    return storage_->RestoreFeatures(key_num, kSavedPartitionNum, 
                                     partition_id, partition_num,
                                     value_len_, false/* is_filter*/, false/* is_incr*/,
                                     emb_config_, device, filter_, restore_buff);
  }

  mutex* mu() {
    return &mu_;
  }

  embedding::Storage<K, V>* storage() {
    return storage_;
  }

  embedding::FeatureDescriptor<V>* feature_descriptor() {
    return feat_desc_;
  }

  Status Shrink(embedding::ShrinkArgs& shrink_args) {
    if (emb_config_.is_primary()) {
      shrink_args.value_len = value_len_;
      return storage_->Shrink(shrink_args);
    } else {
      return Status::OK();
    }
  }

  string Name() {return name_; }

  V* GetDefaultValuePtr() {
    return default_value_;
  }

  int64 GetDefaultValueDim() {
    return emb_config_.default_value_dim;
  }

  V* GetDefaultValue(int64 key) {
    return default_value_ + (key % emb_config_.default_value_dim) * value_len_;
  }

  embedding::BatchCache<K>* Cache() {
    return storage_->Cache();
  }

  int64 GetEmbeddingIndex() {
    return emb_config_.emb_index;
  }

  int64 GetEmbeddingSlotNum() {
    return emb_config_.slot_num;
  }
  
  Allocator* GetAllocator() {
    return alloc_;
  }

  V** GetBuffer(int64 size) {
    if (dev_addr_buffer_size_ >= size) {
      return dev_addr_buffer_;
    } else {
      if (dev_addr_buffer_size_ != 0) {
        alloc_->DeallocateRaw(dev_addr_buffer_);
      }
      dev_addr_buffer_ =
          (V**)alloc_->AllocateRaw(
              Allocator::kAllocatorAlignment,
              size * sizeof(V*));
      dev_addr_buffer_size_ = size;
      return dev_addr_buffer_;
    }
  }

  void UpdateCache(const Tensor& indices,
                   const Tensor& indices_counts,
                   bool is_called_by_gather = false) {
    if (!is_called_by_gather ||
        (is_called_by_gather && emb_config_.is_inference)) {
      storage_->UpdateCache(indices, indices_counts);
    }
  }

  void UpdateCache(const Tensor& indices,
                   bool is_called_by_gather = false) {
    if (!is_called_by_gather ||
        (is_called_by_gather && emb_config_.is_inference)) {
      storage_->UpdateCache(indices);
    }
  }

  void UpdateCache(const K* key_buff, int64 key_num,
      const int64* version_buff, const int64* freq_buff) {
    auto cache = Cache();
    if (cache) {
      cache->update(key_buff, key_num, version_buff, freq_buff);
      auto cache_size = CacheSize();
      if (cache->size() > cache_size) {
        int64 evict_size = cache->size() - cache_size;
        K* evict_ids = new K[evict_size];
        size_t true_size = cache->get_evic_ids(evict_ids, evict_size);
        if (!IsUseHbm()) {
          Eviction(evict_ids, true_size);
        }
        delete []evict_ids;
      }
    }
  }

  void LookupOrCreate(const K* key, V* val, V* default_v,
      int32 default_v_num, size_t n, const Eigen::GpuDevice& device) {
    storage_->BatchLookupOrCreate(key, val, default_v, default_v_num,
        n, device);
  }

  void LookupOrCreateKey(const K* key, int32* item_idxs, size_t n,
      const Eigen::GpuDevice& device, int64 update_version = -1) {
    storage_->BatchLookupOrCreateKeys(key, item_idxs, n, device);
  }

  void Lookup(const K* key, V* val, V* default_v,
      int32 default_v_num,
      size_t n, const Eigen::GpuDevice& device) {
    storage_->BatchLookup(key, val, default_v, default_v_num,
        n, device);
  }
  
  int32 SlotNum() {
    return (emb_config_.block_num * (1 + emb_config_.slot_num));
  }

  int32 EmbIdx() {
    return emb_config_.emb_index;
  }

  GPUHashTable<K, V>* HashTable() {
    return storage_->HashTable();
  }
  FilterPolicy<K, V, EmbeddingVar<K, V>>* GetFilter() const {
    return filter_;
  }

  void CleanUp() {
    if (emb_config_.is_primary() && emb_config_.primary_emb_index == 0) {
      storage_->CleanUp();
    }
  }

 protected:
  ~EmbeddingVar() override {
    // When dynamic dimension embedding is used,
    // there will be more than one primary slot
    if (emb_config_.is_primary() && emb_config_.primary_emb_index == 0) {
      delete storage_;
      delete feat_desc_;
    }
    if (embedding::StorageType::HBM_DRAM == storage_type_) {
      alloc_->DeallocateRaw(dev_addr_buffer_);
    }
    TypedAllocator::Deallocate(default_value_alloc_, default_value_,
        value_len_ * emb_config_.default_value_dim);
    if (default_value_no_permission_) {
      TypedAllocator::Deallocate(default_value_alloc_,
          default_value_no_permission_,
          value_len_);
    }
    if (filter_) {
      delete filter_;
    }
  }

 private:
  void LookupThroughFilter(
      const EmbeddingVarContext<CPUDevice>& context,
      const Tensor& indices, V* output,
      int64 num_of_keys) {
    const K* keys = (K*)indices.data();
    auto do_work = [this, keys, output] (int64 start, int64 limit) {
      for (int64 i = start; i < limit; ++i) {
        V* default_v =
            default_value_ +
                (keys[i] % emb_config_.default_value_dim) * value_len_;
        filter_->Lookup(keys[i],
            output + i * value_len_, default_v,
            default_value_no_permission_);
      }
    };
    auto worker_threads = context.worker_threads;
    Shard(worker_threads->num_threads,
          worker_threads->workers, num_of_keys,
          value_len_ * sizeof(V), do_work);
  }

  std::string name_;
  bool is_initialized_ = false;

  mutex mu_;

  V* default_value_;
  V* default_value_no_permission_;
  V** dev_addr_buffer_;
  int64 dev_addr_buffer_size_;
  int64 value_len_;
  Allocator* alloc_;
  Allocator* default_value_alloc_;
  embedding::Storage<K, V>* storage_;
  embedding::StorageType storage_type_;
  EmbeddingConfig emb_config_;
  FilterPolicy<K, V, EmbeddingVar<K, V>>* filter_;
  embedding::FeatureDescriptor<V>* feat_desc_;

  TF_DISALLOW_COPY_AND_ASSIGN(EmbeddingVar);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMBEDDING_VAR_H_
