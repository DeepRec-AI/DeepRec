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
#include "tensorflow/core/framework/embedding/embedding_memory_pool.h"
#include "tensorflow/core/framework/embedding/embedding_var_ckpt_data.h"
#include "tensorflow/core/framework/embedding/embedding_var_restore.h"
#include "tensorflow/core/framework/embedding/filter_policy.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/shrink_policy.h"
#include "tensorflow/core/framework/embedding/storage_config.h"
#include "tensorflow/core/lib/core/status.h"

#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/framework/device_base.h"
#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#endif

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <class K, class V>
class CheckpointLoader;

template <class K, class V>
class EmbeddingVar;

template <class K>
struct SsdRecordDescriptor;

template <class K, class V>
class GPUHashTable;

class BundleWriter;
class BundleReader;

template<typename Device>
struct EmbeddingVarContext;
namespace embedding {

template<typename K, typename V>
class Storage {
 friend class CheckpointLoader<K, V>;
 public:
  explicit Storage(const StorageConfig& storage_config)
      : storage_config_(storage_config) {
    initialize_value_.resize(storage_config.embedding_config.slot_num + 1);    
  }
  virtual ~Storage() {}
  TF_DISALLOW_COPY_AND_ASSIGN(Storage);

  virtual Status Get(K key, void** value_ptr) = 0;
#if GOOGLE_CUDA
  virtual void BatchGet(const EmbeddingVarContext<GPUDevice>& ctx,
                        const K* key,
                        void** value_ptr_list,
                        int64 num_of_keys) {}

  virtual void BatchGetOrCreate(
      const EmbeddingVarContext<GPUDevice>& ctx,
      const K* key,
      void** value_ptr_list,
      int64 num_of_keys,
      int64 value_len,
      std::vector<std::list<int64>>& not_found_cursor_list) {}
#endif //GOOGLE_CUDA
  virtual Status Contains(K key) = 0;
  virtual void CreateAndInsert(K key, void** value_ptr,
                               bool to_dram=false) = 0;
  virtual void Insert(K key, void** value_ptr) = 0;
  virtual void Init() {}
  virtual void SetValueLen(int64 value_len) {}
  virtual Status GetOrCreate(K key, void** value_ptr) = 0;
  virtual int LookupTier(K key) const = 0;
  virtual Status Remove(K key) = 0;
  virtual int64 Size() const = 0;
  virtual int64 Size(int level) const = 0;
  virtual Status GetSnapshot(std::vector<K>* key_list,
      std::vector<void*>* value_ptr_list) = 0;
  virtual Status Save(
      const string& tensor_name,
      const string& prefix,
      BundleWriter* writer,
      const EmbeddingConfig& emb_config,
      ShrinkArgs& shrink_args,
      int64 value_len,
      V* default_value) = 0;

  virtual Status BatchCommit(const std::vector<K>& keys,
      const std::vector<void*>& value_ptrs) = 0;

  virtual Status Eviction(K* evict_ids, int64 evict_size) = 0;

  virtual void CopyEmbeddingsFromCPUToGPU(
      int total, const K* keys,
      const std::list<int64>& copyback_cursor,
      V** memcpy_address, size_t value_len,
      void **gpu_value_ptrs,
      V* memcpy_buffer_gpu,
      se::Stream* compute_stream,
      EventMgr* event_mgr,
      const DeviceBase::CpuWorkerThreads* worker_threads) = 0;

  virtual void BatchLookupOrCreate(const K* key, V* val, V* default_v,
      int32 default_v_num, size_t n, const Eigen::GpuDevice& device) {}
  virtual void BatchLookupOrCreateKeys(const K* key, int32* item_idxs, size_t n,
      const Eigen::GpuDevice& device) {}
  virtual void BatchLookup(const Eigen::GpuDevice& device, const K* keys, V* val,
                           size_t n, const V* default_v) {}
  virtual GPUHashTable<K, V>* HashTable() {
    return nullptr;
  }

  virtual void InitCache(embedding::CacheStrategy cache_strategy) = 0;
  virtual int64 CacheSize() const = 0;
  virtual BatchCache<K>* Cache() = 0;
  virtual bool IsMultiLevel() = 0;
  virtual bool IsUseHbm() = 0;
  virtual bool IsSingleHbm() = 0;
  virtual bool IsUsePersistentStorage() { return false; };
  virtual void Schedule(std::function<void()> fn) = 0;
  virtual void CreateEmbeddingMemoryPool(
      Allocator* alloc,
      int64 value_len,
      int64 block_size) = 0;
 
  inline mutex* get_mutex() { return &mu_; }
  inline int64 GetAllocLen() { return alloc_len_; }
  inline int64 GetOffset(int64 index) { return alloc_len_ * index; }
  inline int64 GetTotalDims() { return total_dims_; }
  inline embedding::StorageType GetStorageType() { return storage_config_.type; }
  inline std::string GetStoragePath() { return storage_config_.path; }
  inline embedding::CacheStrategy
      CacheStrategy() { return storage_config_.cache_strategy; }

  inline std::string DebugString() const {
    return strings::StrCat("class type: ", typeid(this).name(),
                          " alloc len: ", alloc_len_,
                          " total dims: ", total_dims_,
                          " storage type: ", storage_config_.type,
                          " storage path: ", storage_config_.path,
                          " storage capacity: ", storage_config_.size);
  }

  inline void Insert(const std::vector<K>& keys,
                     void** value_ptrs) {
    for (size_t i = 0; i < keys.size(); i++) {
      Insert(keys[i], value_ptrs[i]);
    }
  }

  virtual void UpdateCache(const Tensor& indices,
                           const Tensor& indices_counts) {}

  virtual void UpdateCache(const Tensor& indices) {}

  virtual void AddToCachePrefetchList(const Tensor& indices) {}

  virtual void AddToCache(const Tensor& indices) {}
  
  virtual void Restore(const std::string& name_string,
                       const std::string& file_name_string, int64 partition_id,
                       int64 partition_num, int64 value_len, bool is_incr,
                       bool reset_version, const EmbeddingConfig& emb_config,
                       const Eigen::GpuDevice* device, BundleReader* reader,
                       EmbeddingVar<K, V>* ev,
                       FilterPolicy<K, V, EmbeddingVar<K, V>>* filter) {
    CheckpointLoader<K, V> restorer(reinterpret_cast<Storage<K, V>*>(this), ev,
                                    filter, name_string, file_name_string,
                                    partition_id, partition_num, is_incr,
                                    reset_version, reader);
    restorer.RestoreCkpt(emb_config, device);
  };
  
  virtual void UpdateValuePtr(K key, void* new_value_ptr,
                              void* old_value_ptr) = 0;
  
  virtual void Import(K key, V* value,
                      int64 freq, int64 version,
                      int emb_index) = 0;

 protected:
  virtual Status RestoreFeatures(int64 key_num, int bucket_num, int64 partition_id,
                                 int64 partition_num, int64 value_len, bool is_filter,
                                 bool is_incr, const EmbeddingConfig& emb_config,
                                 const Eigen::GpuDevice* device,
                                 FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
                                 RestoreBuffer& restore_buff) {
    return Status::OK();
  }
  
  virtual Status RestoreSSD(int64 emb_index, int64 emb_slot_num,
                            int64 value_len,
                            const std::string& ssd_emb_file_name,
                            EmbeddingVar<K, V>* ev,
                            RestoreSSDBuffer<K>& restore_buff) {
    for (int64 i = 0; i < restore_buff.num_of_keys; i++) {
      int64 file_id = restore_buff.key_file_id_list_buf[i];
      int64 key_offset = restore_buff.key_offset_list_buf[i];
      // Read data from embedding files on SSD. Data are stored in
      // NormalContiguousValuePtr temporarily.
      std::stringstream ss;
      ss << ssd_emb_file_name << "/" << file_id << ".emb";
      int fd = open(ss.str().data(), O_RDONLY);
      EmbeddingConfig& emb_config = storage_config_.embedding_config;
      FeatureDescriptor<V> normal_feat_desc(
          emb_config.block_num, emb_config.slot_num + 1,
          ev_allocator(), StorageType::DRAM, true,
          true, {false, 0});
      void* value_ptr = normal_feat_desc.Allocate();
      char* file_addr = (char*)mmap(nullptr,
                                    normal_feat_desc.data_bytes() +
                                    key_offset,
                                    PROT_READ, MAP_PRIVATE, fd, 0);
      memcpy(value_ptr, file_addr + key_offset,
             normal_feat_desc.data_bytes());
      munmap(file_addr,
             normal_feat_desc.data_bytes() +
             key_offset);
      close(fd);
      // Copy Data to ValuePtr, data of slots are set by primary here.
      int64 import_freq = normal_feat_desc.GetFreq(value_ptr);
      int64 import_version = normal_feat_desc.GetVersion(value_ptr);
      V* value = normal_feat_desc.GetEmbedding(value_ptr, emb_index);
      Import(restore_buff.key_list_buf[i], value,
             import_freq, import_version, emb_index);
      normal_feat_desc.Deallocate(value_ptr);
    }
    return Status::OK();
  }

 private:
  void GeneratePartitionedCkptData(
      const std::vector<K>& key_list,
      const std::vector<void*>& value_ptr_list,
      EmbeddingVarCkptData<K, V>* partitioned_ckpt_data,
      const EmbeddingConfig& emb_config,
      V* default_value,
      FeatureDescriptor<V>* feat_desc) {
    std::vector<EmbeddingVarCkptData<K, V>>
        ev_ckpt_data_parts(kSavedPartitionNum);

    bool save_unfiltered_features = true;
    TF_CHECK_OK(ReadBoolFromEnvVar(
        "TF_EV_SAVE_FILTERED_FEATURES", true, &save_unfiltered_features));

    bool is_save_freq = emb_config.is_save_freq();
    bool is_save_version = emb_config.is_save_version();

    for (int64 i = 0; i < key_list.size(); i++) {
      for (int part_id = 0; part_id < kSavedPartitionNum; part_id++) {
        if (key_list[i] % kSavedPartitionNum == part_id) {
          ev_ckpt_data_parts[part_id].Emplace(
              key_list[i], value_ptr_list[i],
              emb_config, default_value,
              feat_desc,
              is_save_freq,
              is_save_version,
              save_unfiltered_features);
          break;
        }
      }
    }

    partitioned_ckpt_data->SetWithPartition(ev_ckpt_data_parts);
  }

  void GeneratePartitionedCkptData(
      const std::vector<K>& key_list,
      const std::vector<void*>& value_ptr_list,
      EmbeddingVarCkptData<K, V>* partitioned_ckpt_data,
      const EmbeddingConfig& emb_config,
      V* default_value,
      const std::vector<FeatureDescriptor<V>*>& feat_desc) {
    std::vector<EmbeddingVarCkptData<K, V>>
        ev_ckpt_data_parts(kSavedPartitionNum);

    bool save_unfiltered_features = true;
    TF_CHECK_OK(ReadBoolFromEnvVar(
        "TF_EV_SAVE_FILTERED_FEATURES", true, &save_unfiltered_features));

    bool is_save_freq = emb_config.is_save_freq();
    bool is_save_version = emb_config.is_save_version();

    for (int64 i = 0; i < key_list.size(); i++) {
      for (int part_id = 0; part_id < kSavedPartitionNum; part_id++) {
        if (key_list[i] % kSavedPartitionNum == part_id) {
          int feat_desc_type = (int64)value_ptr_list[i] >> kDramFlagOffset;
          ev_ckpt_data_parts[part_id].Emplace(
              key_list[i], value_ptr_list[i],
              emb_config, default_value,
              feat_desc[feat_desc_type],
              is_save_freq,
              is_save_version,
              save_unfiltered_features);
          break;
        }
      }
    }

    partitioned_ckpt_data->SetWithPartition(ev_ckpt_data_parts);
  }

  void GeneratePartitionedCkptData(
      const std::vector<K>& key_list,
      const std::vector<V*>& value_ptr_list,
      EmbeddingVarCkptData<K, V>* partitioned_ckpt_data) {
    std::vector<EmbeddingVarCkptData<K, V>>
        ev_ckpt_data_parts(kSavedPartitionNum);

    for (int64 i = 0; i < key_list.size(); i++) {
      for (int part_id = 0; part_id < kSavedPartitionNum; part_id++) {
        if (key_list[i] % kSavedPartitionNum == part_id) {
          ev_ckpt_data_parts[part_id].Emplace(
              key_list[i], value_ptr_list[i]);
          break;
        }
      }
    }

    partitioned_ckpt_data->SetWithPartition(ev_ckpt_data_parts);
  }

 protected:
  Status SaveToCheckpoint(
      const string& tensor_name,
      BundleWriter* writer,
      const EmbeddingConfig& emb_config,
      int64 value_len,
      V* default_value,
      const std::vector<K>& key_list,
      const std::vector<void*>& value_ptr_list,
      FeatureDescriptor<V>* feat_desc,
      ValueIterator<V>* value_iter = nullptr) {
    EmbeddingVarCkptData<K, V> partitioned_ckpt_data;
    GeneratePartitionedCkptData(key_list, value_ptr_list,
                                &partitioned_ckpt_data, emb_config,
                                default_value, feat_desc);
    Status s =
        partitioned_ckpt_data.ExportToCkpt(
            tensor_name, writer, value_len, value_iter);
    return Status::OK();
  }

  Status SaveToCheckpoint(
      const string& tensor_name,
      BundleWriter* writer,
      const EmbeddingConfig& emb_config,
      int64 value_len,
      V* default_value,
      const std::vector<K>& key_list,
      const std::vector<void*>& value_ptr_list,
      const std::vector<FeatureDescriptor<V>*>& feat_desc,
      ValueIterator<V>* value_iter = nullptr) {
    EmbeddingVarCkptData<K, V> partitioned_ckpt_data;
    GeneratePartitionedCkptData(key_list, value_ptr_list,
                                &partitioned_ckpt_data, emb_config,
                                default_value, feat_desc);
    Status s =
        partitioned_ckpt_data.ExportToCkpt(
            tensor_name, writer, value_len, value_iter);
    return Status::OK();
  }

  Status SaveToCheckpoint(
      const string& tensor_name,
      BundleWriter* writer,
      int64 value_len,
      const std::vector<K>& key_list,
      const std::vector<V*>& value_ptr_list) {
    EmbeddingVarCkptData<K, V> partitioned_ckpt_data;
    GeneratePartitionedCkptData(key_list, value_ptr_list,
                                &partitioned_ckpt_data);
    Status s =
        partitioned_ckpt_data.ExportToCkpt(tensor_name, writer, value_len);
    return Status::OK();
  }

 protected:
  int64 alloc_len_ = 0;
  int64 total_dims_ = 0;
  StorageConfig storage_config_;

  mutex mu_;
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
  std::vector<V*> initialize_value_;
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_H_
