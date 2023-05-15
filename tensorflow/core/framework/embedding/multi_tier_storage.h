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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_MULTI_TIER_STORAGE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_MULTI_TIER_STORAGE_H_

#include "tensorflow/core/framework/embedding/cache_factory.h"
#include "tensorflow/core/framework/embedding/cache_thread_pool_creator.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/cpu_hash_map_kv.h"
#include "tensorflow/core/framework/embedding/eviction_manager.h"
#include "tensorflow/core/framework/embedding/globalstep_shrink_policy.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/l2weight_shrink_policy.h"
#include "tensorflow/core/framework/embedding/storage_config.h"
#include "tensorflow/core/framework/embedding/storage.h"

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
template<typename V>
class ValuePtr;

template<typename K, typename V>
class EmbeddingVar;

template <class K>
struct SsdRecordDescriptor;

namespace embedding {
template<typename K, typename V>
class MultiTierStorage : public Storage<K, V> {
 public:
  MultiTierStorage(const StorageConfig& sc, const std::string& name)
      : Storage<K, V>(sc), name_(name) {}

  virtual ~MultiTierStorage() {
    delete cache_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(MultiTierStorage);

  void SetAllocLen(int64 value_len, int slot_num) override {
    while (Storage<K, V>::flag_.test_and_set(std::memory_order_acquire));
    // The start address of every slot should be aligned to 16 bytes,
    // otherwise a coredump will happen in the ApplyOp.
    Storage<K, V>::alloc_len_ = Storage<K, V>::ComputeAllocLen(value_len);

    int64 temp = Storage<K, V>::alloc_len_ * slot_num;
    if (temp > Storage<K, V>::total_dims_) {
      Storage<K, V>::total_dims_ = temp;
      SetTotalDims(Storage<K, V>::total_dims_);

      cache_capacity_ = Storage<K, V>::storage_config_.size[0]
                        / (Storage<K, V>::total_dims_ * sizeof(V));
      ready_eviction_ = true;
    }
    Storage<K, V>::flag_.clear(std::memory_order_release);
  }

  int64 CacheSize() const override {
    return cache_capacity_;
  }

  BatchCache<K>* Cache() override {
    return cache_;
  }

  void InsertToDram(K key, ValuePtr<V>** value_ptr,
              int64 alloc_len) override {
    LOG(FATAL)<<"InsertToDram in MultiTierStorage shouldn't be called";
  }

  void InitCache(embedding::CacheStrategy cache_strategy) override {
    cache_ = CacheFactory::Create<K>(cache_strategy, name_);
    eviction_manager_ = EvictionManagerCreator::Create<K, V>();
    eviction_manager_->AddStorage(this);
    cache_thread_pool_ = CacheThreadPoolCreator::Create();
  }

  Status BatchCommit(const std::vector<K>& keys,
      const std::vector<ValuePtr<V>*>& value_ptrs) override {
    LOG(FATAL)<<"BatchCommit isn't supported by MultiTierStorage.";
    return Status::OK();
  }

  embedding::Iterator* GetIterator() {
    LOG(FATAL)<<"GetIterator isn't support by MultiTierStorage.";
    return nullptr;
  }

  void CopyEmbeddingsFromCPUToGPU(
      int total, const K* keys,
      const std::list<int64>& copyback_cursor,
      V** memcpy_address, size_t value_len,
      ValuePtr<V> **gpu_value_ptrs,
      V* memcpy_buffer_gpu,
      se::Stream* compute_stream,
      EventMgr* event_mgr,
      const DeviceBase::CpuWorkerThreads* worker_threads) override {
    LOG(FATAL) << "Unsupport CopyEmbeddingsFromCPUToGPU in MultiTierStorage.";
  };

  void SetListsForCheckpoint(
      const std::vector<K>& input_key_list,
      const std::vector<ValuePtr<V>*>& input_value_ptr_list,
      const EmbeddingConfig& emb_config,
      std::vector<K>* output_key_list,
      std::vector<V*>* output_value_list,
      std::vector<int64>* output_version_list,
      std::vector<int64>* output_freq_list) {
    for (int64 i = 0; i < input_key_list.size(); ++i) {
      output_key_list->emplace_back(input_key_list[i]);

      //NormalContiguousValuePtr is used, GetFreq() is valid.
      int64 dump_freq = input_value_ptr_list[i]->GetFreq();
      output_freq_list->emplace_back(dump_freq);

      if (emb_config.steps_to_live != 0 || emb_config.record_version) {
        int64 dump_version = input_value_ptr_list[i]->GetStep();
        output_version_list->emplace_back(dump_version);
      }

      V* val = input_value_ptr_list[i]->GetValue(emb_config.emb_index,
          Storage<K, V>::GetOffset(emb_config.emb_index));
      V* primary_val = input_value_ptr_list[i]->GetValue(
          emb_config.primary_emb_index,
          Storage<K, V>::GetOffset(emb_config.primary_emb_index));
      /* Classify features into 3 categories:
        1. filtered
        2. not involved in backward
        3. normal
      */
      if (primary_val == nullptr) {
        output_value_list->emplace_back(nullptr);
      } else {
        if (val == nullptr) {
          output_value_list->emplace_back(reinterpret_cast<V*>(-1));
        } else {
          output_value_list->emplace_back(val);
        }
      }
    }
  }

  virtual int64 GetSnapshotWithoutFetchPersistentEmb(
      std::vector<K>* key_list,
      std::vector<V* >* value_list,
      std::vector<int64>* version_list,
      std::vector<int64>* freq_list,
      const EmbeddingConfig& emb_config,
      SsdRecordDescriptor<K>* ssd_rec_desc) override {
    LOG(FATAL)<<"The Storage dosen't use presisten memory"
              <<" or this storage hasn't suppported"
              <<" GetSnapshotWithoutFetchPersistentEmb yet";
    return -1;
  }

  Status Contains(K key) override {
    LOG(FATAL)<<"Contains is not support in MultiTierStorage.";
    return Status::OK();
  }

  void iterator_mutex_lock() override {
    return;
  }

  void iterator_mutex_unlock() override {
    return;
  }

  void RestoreSsdHashmap(
      K* key_list, int64* key_file_id_list,
      int64* key_offset_list, int64 num_of_keys,
      int64* file_list, int64* invalid_record_count_list,
      int64* record_count_list, int64 num_of_files,
      const std::string& ssd_emb_file_name) override {
    LOG(FATAL)<<"The Storage dosen't have ssd storage"
              <<" or this storage hasn't suppported"
              <<" RestoreSsdHashmap yet";
  }

  void ImportToHbm(
      K* ids, int64 size, int64 value_len, int64 emb_index) override {
    LOG(FATAL)<<"This Storage dosen't have a HBM storage.";
  }

  bool IsMultiLevel() override {
    return true;
  }

  void CreateEmbeddingMemoryPool(
      Allocator* alloc,
      int64 value_len,
      int64 block_size) override {
    return;
  }

  void AllocateMemoryForNewFeatures(
      const std::vector<ValuePtr<V>*>& value_ptr_list) override {
    return;
  }

  void AllocateMemoryForNewFeatures(
      ValuePtr<V>** value_ptr_list,
      int64 num_of_value_ptrs) override {
    return;
  }

  void Schedule(std::function<void()> fn) override {
    cache_thread_pool_->Schedule(std::move(fn));
  }

  virtual Status Eviction(K* evict_ids, int64 evict_size) override {
    LOG(FATAL)<<"Eviction isn't support by "<<typeid(this).name();
    return Status::OK();
  }

  virtual void BatchEviction() {
    constexpr int EvictionSize = 10000;
    K evic_ids[EvictionSize];
    if (!ready_eviction_)
      return;
    int cache_count = cache_->size();
    if (cache_count > cache_capacity_) {
      // eviction
      int k_size = cache_count - cache_capacity_;
      k_size = std::min(k_size, EvictionSize);
      size_t true_size = cache_->get_evic_ids(evic_ids, k_size);
      EvictionWithDelayedDestroy(evic_ids, true_size);
    }
  }

 protected:
  virtual void SetTotalDims(int64 total_dims) = 0;

  void DeleteFromEvictionManager() {
    eviction_manager_->DeleteStorage(this);
  }

  void ReleaseValuePtrs(std::deque<ValuePtr<V>*>& value_ptrs,
                        Allocator* allocator) {
    constexpr int CAP_INVALID_VALUEPTR = 64 * 1024;
    if (value_ptrs.size() > CAP_INVALID_VALUEPTR) {
      int64 num_of_deleted_value_ptrs =
          value_ptrs.size() - CAP_INVALID_VALUEPTR;
      for (int i = 0; i < num_of_deleted_value_ptrs; i++) {
        ValuePtr<V>* value_ptr = value_ptrs.front();
        value_ptr->Destroy(allocator);
        delete value_ptr;
        value_ptrs.pop_front();
      }
    }
  }

  void ReleaseInvalidValuePtr(Allocator* allocator) {
    ReleaseValuePtrs(value_ptr_out_of_date_, allocator);
  }

  void KeepInvalidValuePtr(ValuePtr<V>* value_ptr) {
    value_ptr_out_of_date_.emplace_back(value_ptr);
  }
 private:
  virtual Status EvictionWithDelayedDestroy(K* evict_ids, int64 evict_size) {}

 protected:
  std::deque<ValuePtr<V>*> value_ptr_out_of_date_;
  BatchCache<K>* cache_ = nullptr;

  EvictionManager<K, V>* eviction_manager_;
  thread::ThreadPool* cache_thread_pool_;

  condition_variable shutdown_cv_;
  volatile bool shutdown_ = false;

  int64 cache_capacity_ = -1;
  volatile bool ready_eviction_ = false;

  std::string name_;
  std::vector<mutex> mu_list_;
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_MULTI_TIER_STORAGE_H_
