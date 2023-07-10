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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_STORAGE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_STORAGE_H_

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/framework/embedding/lockless_hash_map_cpu.h"
#include "tensorflow/core/framework/embedding/multi_tier_storage.h"
#include "tensorflow/core/framework/embedding/single_tier_storage.h"
#include "tensorflow/core/framework/embedding/hbm_storage_iterator.h"
#include "tensorflow/core/framework/embedding/intra_thread_copy_id_allocator.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
using se::DeviceMemoryBase;
using se::Stream;

template <class V>
class ValuePtr;

template <typename K, typename V>
class CheckpointLoader;

void SyncWithEventMgr(se::Stream* stream, EventMgr* event_mgr);

namespace embedding {
template<typename K, typename V>
class HbmDramStorage : public MultiTierStorage<K, V> {
 public:
  HbmDramStorage(const StorageConfig& sc, Allocator* gpu_alloc,
                 Allocator* cpu_alloc, LayoutCreator<V>* lc,
                 const std::string& name)
      : gpu_alloc_(gpu_alloc), MultiTierStorage<K, V>(sc, name) {
    hbm_ = new HbmStorageWithCpuKv<K, V>(sc, gpu_alloc, lc);
    StorageConfig storage_config = StorageConfig();
    storage_config.layout_type = LayoutType::NORMAL_CONTIGUOUS;
    dram_ = new DramStorage<K, V>(sc, cpu_alloc,
                                  LayoutCreatorFactory::Create<V>(storage_config),
                                  new LocklessHashMapCPU<K, V>(gpu_alloc));
  }

  ~HbmDramStorage() override {
    MultiTierStorage<K, V>::DeleteFromEvictionManager();
    delete hbm_;
    delete dram_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(HbmDramStorage);

  Status Get(K key, ValuePtr<V>** value_ptr) override {
    Status s = hbm_->Get(key, value_ptr);
    if (s.ok()) {
      return s;
    }
    s = dram_->Get(key, value_ptr);
    if (s.ok()) {
      AddCopyBackFlagToValuePtr(value_ptr, COPYBACK);
      return s;
    }
    return s;
  }

  void BatchGet(const EmbeddingVarContext<GPUDevice>& ctx,
                const K* keys,
                ValuePtr<V>** value_ptr_list,
                int64 num_of_keys,
                int64 value_len) override {
    int num_worker_threads = ctx.worker_threads->num_threads;
    std::vector<std::list<int64>>
        copyback_cursor_list(num_worker_threads + 1);

    BatchGetValuePtrs(ctx, keys, value_ptr_list, num_of_keys,
                      copyback_cursor_list);

    CopyEmbeddingsFromDramToHbm(
        ctx, keys, value_ptr_list, copyback_cursor_list[0],
        value_len);
  }

  void Insert(K key, ValuePtr<V>* value_ptr) override {
    hbm_->Insert(key, value_ptr);
  }

  void BatchGetOrCreate(
      const EmbeddingVarContext<GPUDevice>& ctx,
      const K* keys,
      ValuePtr<V>** value_ptr_list,
      int64 num_of_keys,
      int64 value_len,
      std::vector<std::list<int64>>& not_fountd_cursor_list) override {
    int num_worker_threads = ctx.worker_threads->num_threads;
    std::vector<std::list<int64>>
        copyback_cursor_list(num_worker_threads + 1);

    BatchGetValuePtrs(ctx, keys, value_ptr_list, num_of_keys,
                      copyback_cursor_list, &not_fountd_cursor_list);

    CopyEmbeddingsFromDramToHbm(
        ctx, keys, value_ptr_list, copyback_cursor_list[0],
        value_len);

    CreateValuePtrs(ctx, keys, value_ptr_list,
                    not_fountd_cursor_list[0], value_len);
  }

  void Insert(K key, ValuePtr<V>** value_ptr,
              size_t alloc_len, bool to_dram = false) override {
    if (to_dram) {
      dram_->Insert(key, value_ptr, alloc_len);
    } else {
      hbm_->Insert(key, value_ptr, alloc_len);
    }
  }
  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size) override {
    Status s = hbm_->Get(key, value_ptr);
    if (s.ok()) {
      return s;
    }
    ValuePtr<V>* gpu_value_ptr = hbm_->CreateValuePtr(size);
    {
      mutex_lock l(memory_pool_mu_);
      gpu_value_ptr->SetPtr(embedding_mem_pool_->Allocate());
      *value_ptr = gpu_value_ptr;
    }

    s = hbm_->TryInsert(key, *value_ptr);
    if (s.ok()) {
      return s;
    }
    // Insert Failed, key already exist
    {
      mutex_lock l(memory_pool_mu_);
      embedding_mem_pool_->Deallocate((*value_ptr)->GetValue(0, 0));
    }
    delete *value_ptr;
    return hbm_->Get(key, value_ptr);
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr,
      size_t size, CopyBackFlag &need_copyback) override {
    need_copyback = NOT_COPYBACK;
    Status s = hbm_->Get(key, value_ptr);
    if (s.ok()) {
      return s;
    }
    s = dram_->Get(key, value_ptr);
    if (s.ok()) {
      need_copyback = COPYBACK;
      return s;
    }

    hbm_->Insert(key, value_ptr, size);
    return Status::OK();
  }

  void CopyEmbeddingsFromCPUToGPU(
      int total, const K* keys,
      const std::list<int64>& copyback_cursor,
      V** memcpy_address, size_t value_len,
      ValuePtr<V> **gpu_value_ptrs, V* memcpy_buffer_gpu,
      se::Stream* compute_stream,
      EventMgr* event_mgr,
      const DeviceBase::CpuWorkerThreads* worker_threads) override {
    auto memcpy_buffer_cpu = TypedAllocator::Allocate<V>(cpu_allocator(),
        total * value_len, AllocationAttributes());
    int64* memory_index = new int64[total];
    int64 i = 0;
    auto it = copyback_cursor.cbegin();
    {
      //Mutex with eviction thread
      mutex_lock l(memory_pool_mu_);
      for ( ; it != copyback_cursor.cend(); ++it, ++i) {
        int64 j = *it;
        memory_index[i] = j;
        ValuePtr<V>* gpu_value_ptr = hbm_->CreateValuePtr(value_len);
        V* val_ptr = embedding_mem_pool_->Allocate();
        bool flag = gpu_value_ptr->SetPtr(val_ptr);
        if (!flag) {
          embedding_mem_pool_->Deallocate(val_ptr);
        }
        memcpy((char *)gpu_value_ptr->GetPtr(),
               (char *)memcpy_address[j] - sizeof(FixedLengthHeader),
               sizeof(FixedLengthHeader));
        gpu_value_ptrs[i] = gpu_value_ptr;
      }
    }
    //Split from above for loop for minize the cost of mutex lock
    auto do_work = [memory_index, memcpy_address,
                    memcpy_buffer_cpu, gpu_value_ptrs,
                    value_len, this] (int64 start, int64 limit) {
      for (int i = start; i < limit; i++) {
        int j = memory_index[i];
        memcpy(memcpy_buffer_cpu + i * value_len,
               memcpy_address[j], value_len * sizeof(V));
      }
    };
    Shard(worker_threads->num_threads, worker_threads->workers, total,
          1000, do_work);
    DeviceMemoryBase gpu_dst_ptr(
        memcpy_buffer_gpu, total * value_len * sizeof(V));
    compute_stream->ThenMemcpy(
        &gpu_dst_ptr, memcpy_buffer_cpu, total * value_len * sizeof(V));
    SyncWithEventMgr(compute_stream, event_mgr);
    TypedAllocator::Deallocate(
        cpu_allocator(), memcpy_buffer_cpu, total * value_len);
    delete[] memory_index;
  }

  Status Remove(K key) override {
    hbm_->Remove(key);
    dram_->Remove(key);
    return Status::OK();
  }

  int64 Size() const override {
    int64 total_size = hbm_->Size();
    total_size += dram_->Size();
    return total_size;
  }

  int64 Size(int level) const override {
    if (level == 0) {
      return hbm_->Size();
    } else if (level == 1) {
      return dram_->Size();
    } else {
      return -1;
    }
  }

  int LookupTier(K key) const override {
    Status s = hbm_->Contains(key);
    if (s.ok())
      return 0;
    s = dram_->Contains(key);
    if (s.ok())
      return 1;
    return -1;
  }

  bool IsUseHbm() override {
    return true;
  }

  bool IsSingleHbm() override {
    return false;
  }

  void iterator_mutex_lock() override {
    return;
  }

  void iterator_mutex_unlock() override {
    return;
  }

  Status GetSnapshot(std::vector<K>* key_list,
      std::vector<ValuePtr<V>* >* value_ptr_list) override {
    {
      mutex_lock l(*(hbm_->get_mutex()));
      TF_CHECK_OK(hbm_->GetSnapshot(key_list, value_ptr_list));
    }
    {
      mutex_lock l(*(dram_->get_mutex()));
      TF_CHECK_OK(dram_->GetSnapshot(key_list, value_ptr_list));
    }
    return Status::OK();
  }

  int64 GetSnapshot(std::vector<K>* key_list,
      std::vector<V* >* value_list,
      std::vector<int64>* version_list,
      std::vector<int64>* freq_list,
      const EmbeddingConfig& emb_config,
      FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
      embedding::Iterator** it) override {
    std::vector<ValuePtr<V>*> hbm_value_ptr_list, dram_value_ptr_list;
    std::vector<K> temp_hbm_key_list, temp_dram_key_list;
    // Get Snapshot of HBM storage
    {
      mutex_lock l(*(hbm_->get_mutex()));
      TF_CHECK_OK(hbm_->GetSnapshot(&temp_hbm_key_list,
                                    &hbm_value_ptr_list));
    }
    // Get Snapshot of DRAM storage.
    {
      mutex_lock l(*(dram_->get_mutex()));
      TF_CHECK_OK(dram_->GetSnapshot(&temp_dram_key_list,
                                     &dram_value_ptr_list));
    }
    *it = new HbmDramIterator<K, V>(temp_hbm_key_list,
                                    temp_dram_key_list,
                                    hbm_value_ptr_list,
                                    dram_value_ptr_list,
                                    Storage<K, V>::alloc_len_,
                                    gpu_alloc_,
                                    emb_config.emb_index);
    // This return value is not the exact number of IDs
    // because the two tables intersect.
    return temp_hbm_key_list.size() + temp_dram_key_list.size();
  }

  Status Shrink(const ShrinkArgs& shrink_args) override {
    hbm_->Shrink(shrink_args);
    dram_->Shrink(shrink_args);
    return Status::OK();
  }

  void CreateEmbeddingMemoryPool(
      Allocator* alloc,
      int64 value_len,
      int64 block_size) override {
    embedding_mem_pool_ =
       new EmbeddingMemoryPool<V>(alloc, value_len, block_size);
  }

  void AllocateMemoryForNewFeatures(
      const std::vector<ValuePtr<V>*>& value_ptr_list) override {
    //Mutex with eviction thread
    mutex_lock l(memory_pool_mu_);
    for (auto it : value_ptr_list) {
      V* val_ptr = embedding_mem_pool_->Allocate();
      bool flag = it->SetPtr(val_ptr);
      if (!flag) {
        embedding_mem_pool_->Deallocate(val_ptr);
      }
    }
  }

  void AllocateMemoryForNewFeatures(
     ValuePtr<V>** value_ptr_list,
     int64 num_of_value_ptrs) override {
    //Mutex with other ImportOps
    mutex_lock l(memory_pool_mu_);
    for (int64 i = 0; i < num_of_value_ptrs; i++) {
      V* val_ptr = embedding_mem_pool_->Allocate();
      bool flag = value_ptr_list[i]->SetPtr(val_ptr);
      if (!flag) {
        embedding_mem_pool_->Deallocate(val_ptr);
      }
    }
  }

  void BatchEviction() override {
    constexpr int EvictionSize = 10000;
    K evic_ids[EvictionSize];
    if (!MultiTierStorage<K, V>::ready_eviction_) {
      return;
    }
    mutex_lock l(*(hbm_->get_mutex()));
    mutex_lock l1(*(dram_->get_mutex()));

    int64 cache_count = MultiTierStorage<K, V>::cache_->size();
    if (cache_count > MultiTierStorage<K, V>::cache_capacity_) {
      // eviction
      int k_size = cache_count - MultiTierStorage<K, V>::cache_capacity_;
      k_size = std::min(k_size, EvictionSize);
      size_t true_size =
          MultiTierStorage<K, V>::cache_->get_evic_ids(evic_ids, k_size);
      ValuePtr<V>* value_ptr;
      std::vector<K> keys;
      std::vector<ValuePtr<V>*> value_ptrs;

      for (int64 i = 0; i < true_size; ++i) {
        if (hbm_->Get(evic_ids[i], &value_ptr).ok()) {
          keys.emplace_back(evic_ids[i]);
          value_ptrs.emplace_back(value_ptr);
        }
      }
      dram_->BatchCommit(keys, value_ptrs);
      {
        //Mutex with main thread
        mutex_lock l_mem(memory_pool_mu_);
        embedding_mem_pool_->Deallocate(value_ptrs);
      }
      for (auto it : keys) {
        TF_CHECK_OK(hbm_->Remove(it));
      }
    }
  }

  void Restore(const std::string& name_string,
               const std::string& file_name_string,
               int64 partition_id, int64 partition_num,
               int64 value_len, bool is_incr, bool reset_version,
               const EmbeddingConfig& emb_config,
               const Eigen::GpuDevice* device,
               BundleReader* reader, EmbeddingVar<K, V>* ev,
               FilterPolicy<K, V, EmbeddingVar<K, V>>* filter) override {
                         
    CheckpointLoader<K, V> restorer(reinterpret_cast<Storage<K, V>*>(this),
                                    ev, filter, name_string, file_name_string,
                                    partition_id, partition_num,
                                    is_incr, reset_version, reader);

    restorer.RestoreCkpt(emb_config, device);

    int64 num_of_hbm_ids =
        std::min(MultiTierStorage<K, V>::cache_capacity_,
        (int64)MultiTierStorage<K, V>::cache_->size());
    if (num_of_hbm_ids > 0) {
      K* hbm_ids = new K[num_of_hbm_ids];
      int64* hbm_freqs = new int64[num_of_hbm_ids];
      int64* hbm_versions = nullptr;
      MultiTierStorage<K, V>::cache_->get_cached_ids(hbm_ids, num_of_hbm_ids,
                                                     hbm_versions, hbm_freqs);
      ImportToHbm(hbm_ids, num_of_hbm_ids, value_len, emb_config.emb_index);
      MultiTierStorage<K, V>::cache_thread_pool_->Schedule(
          [this, hbm_ids, num_of_hbm_ids, hbm_versions, hbm_freqs]() {
            MultiTierStorage<K, V>::cache_->update(hbm_ids, num_of_hbm_ids,
                                                   hbm_versions, hbm_freqs);
            delete[] hbm_ids;
            delete[] hbm_freqs;
          });
    }
  }

 protected:
  Status RestoreFeatures(int64 key_num, int bucket_num, int64 partition_id,
                         int64 partition_num, int64 value_len, bool is_filter,
                         bool is_incr, const EmbeddingConfig& emb_config,
                         const Eigen::GpuDevice* device,
                         FilterPolicy<K, V, EmbeddingVar<K, V>>* filter,
                         RestoreBuffer& restore_buff) override {
    Status s = filter->Restore(key_num, bucket_num, partition_id,
                               partition_num, value_len, is_filter,
                               true/*to_dram*/, is_incr, restore_buff);

    MultiTierStorage<K, V>::cache_->update((K*)restore_buff.key_buffer, key_num,
                                           (int64*)restore_buff.version_buffer,
                                           (int64*)restore_buff.freq_buffer);
    return s;
  }

  void SetTotalDims(int64 total_dims) override {
    dram_->SetTotalDims(total_dims);
  }
 private:
  void BatchGetValuePtrs(
      const EmbeddingVarContext<GPUDevice>& ctx,
      const K* keys,
      ValuePtr<V>** value_ptr_list,
      int64 num_of_keys,
      std::vector<std::list<int64>>& copyback_cursor_list,
      std::vector<std::list<int64>>* not_found_cursor_list = nullptr) {
    int num_worker_threads = ctx.worker_threads->num_threads;
    IntraThreadCopyIdAllocator thread_copy_id_alloc(num_worker_threads);
    uint64 main_thread_id = Env::Default()->GetCurrentThreadId();

    std::function<void(std::vector<std::list<int64>>*,
                       int64, int)> set_not_found_list = 0;
    if (not_found_cursor_list != nullptr) {
      set_not_found_list =
          [](std::vector<std::list<int64>>* not_found_cursor_list,
             int64 i, int copy_id) {
        (*not_found_cursor_list)[copy_id].emplace_back(i);
      };
    } else {
      set_not_found_list =
          [](std::vector<std::list<int64>>* not_found_cursor_list,
             int64 i, int copy_id) {};
    }

    auto do_work = [this, keys, value_ptr_list, &thread_copy_id_alloc,
                    main_thread_id, &copyback_cursor_list,
                    set_not_found_list, &not_found_cursor_list]
        (int64 start, int64 limit) {
      int copy_id =
          thread_copy_id_alloc.GetCopyIdOfThread(main_thread_id);
      for (int64 i = start; i < limit; i++) {
        Status s = Get(keys[i], &value_ptr_list[i]);
        if (s.ok()) {
          int64 copyback_flag =
              (int64)value_ptr_list[i] >> copyback_flag_offset_bits_;
          RemoveCopyBackFlagInValuePtr(&value_ptr_list[i]);
          if (copyback_flag == CopyBackFlag::COPYBACK) {
            copyback_cursor_list[copy_id].emplace_back(i);
          }
        } else {
          value_ptr_list[i] = nullptr;
          set_not_found_list(not_found_cursor_list, i, copy_id);
        }
      }
    };
    auto worker_threads = ctx.worker_threads;
    Shard(worker_threads->num_threads,
          worker_threads->workers, num_of_keys,
          1000, do_work);

    for (int i = 1; i < worker_threads->num_threads + 1; i++) {
      if (copyback_cursor_list[i].size()>0) {
        copyback_cursor_list[0].splice(copyback_cursor_list[0].end(),
                                       copyback_cursor_list[i]);
      }
    }

    if (not_found_cursor_list != nullptr) {
      for (int i = 1; i < worker_threads->num_threads + 1; i++) {
        if ((*not_found_cursor_list)[i].size()>0) {
          (*not_found_cursor_list)[0].splice(
              (*not_found_cursor_list)[0].end(),
              (*not_found_cursor_list)[i]);
        }
      }
    }
  }

  void CopyEmbeddingsFromDramToHbm(const EmbeddingVarContext<GPUDevice>& ctx,
                                   const K* keys,
                                   ValuePtr<V>** value_ptr_list,
                                   std::list<int64>& copyback_cursors,
                                   int64 value_len) {
    int64 total = copyback_cursors.size();
    std::vector<ValuePtr<V>*> gpu_value_ptrs(total);
    std::vector<K> copyback_keys(total);
    std::vector<int64> memory_index(total);
    //Create Hbm ValuePtrs.
    {
      int64 i = 0;
      auto it = copyback_cursors.cbegin();
      //Mutex with eviction thread
      mutex_lock l(memory_pool_mu_);
      for ( ; it != copyback_cursors.cend(); ++it, ++i) {
        int64 j = *it;
        memory_index[i] = j;
        ValuePtr<V>* gpu_value_ptr = hbm_->CreateValuePtr(value_len);
        V* val_ptr = embedding_mem_pool_->Allocate();
        bool flag = gpu_value_ptr->SetPtr(val_ptr);
        if (!flag) {
          embedding_mem_pool_->Deallocate(val_ptr);
        }
        memcpy((char *)gpu_value_ptr->GetPtr(),
               (char *)value_ptr_list[j]->GetPtr(),
               sizeof(FixedLengthHeader));
        gpu_value_ptrs[i] = gpu_value_ptr;
        copyback_keys[i] = keys[*it];
      }
    }
    MultiTierStorage<K, V>::CopyEmbeddingsFromDramToHbm(
        ctx, keys, value_ptr_list, copyback_cursors,
        memory_index, gpu_value_ptrs, value_len);

    //Insert copyback ids to hbm hash table.
    auto do_insert = [this, copyback_keys, gpu_value_ptrs,
                      memory_index, value_ptr_list]
        (int64 start, int64 limit) {
      for (int64 i = start; i < limit; i++) {
        Status s = hbm_->TryInsert(
            copyback_keys[i], gpu_value_ptrs[i]);
        if (!s.ok()) {
          {
            mutex_lock l(memory_pool_mu_);
            embedding_mem_pool_->Deallocate(
                gpu_value_ptrs[i]->GetValue(0, 0));
          }
          delete gpu_value_ptrs[i];
          hbm_->Get(copyback_keys[i], &value_ptr_list[memory_index[i]]);
        }
      }
    };
    auto worker_threads = ctx.worker_threads;
    Shard(worker_threads->num_threads, worker_threads->workers,
          total, 100000, do_insert);
  }

  void CreateValuePtrs(const EmbeddingVarContext<GPUDevice>& ctx,
                       const K* keys,
                       ValuePtr<V>** value_ptr_list,
                       std::list<int64>& not_found_cursors,
                       int64 value_len) {
    int64 total = not_found_cursors.size();
    if (total > 0) {
      std::vector<std::pair<int64, ValuePtr<V>*>> insert_pairs(total);
      std::vector<int64> cursor_index(total);
      //Create Hbm ValuePtrs.
      {
        int64 i = 0;
        auto it = not_found_cursors.cbegin();
        //Mutex with eviction thread
        mutex_lock l(memory_pool_mu_);
        for ( ; it != not_found_cursors.cend(); ++it, ++i) {
          int64 j = *it;
          cursor_index[i] = j;
          ValuePtr<V>* gpu_value_ptr = hbm_->CreateValuePtr(value_len);
          V* val_ptr = embedding_mem_pool_->Allocate();
          bool flag = gpu_value_ptr->SetPtr(val_ptr);
          if (!flag) {
            embedding_mem_pool_->Deallocate(val_ptr);
          }
          value_ptr_list[j] = gpu_value_ptr;
          insert_pairs[i].first = keys[j];
          insert_pairs[i].second = value_ptr_list[j];
        }
      }

      //Insert copyback ids to hbm hash table.
      auto do_insert = [this, insert_pairs, value_ptr_list, cursor_index]
          (int64 start, int64 limit) {
        for (int64 i = start; i < limit; i++) {
          Status s = hbm_->TryInsert(
              insert_pairs[i].first, insert_pairs[i].second);
          if (!s.ok()) {
            {
              mutex_lock l(memory_pool_mu_);
              embedding_mem_pool_->Deallocate(
                  insert_pairs[i].second->GetValue(0, 0));
            }
            delete insert_pairs[i].second;
            hbm_->Get(insert_pairs[i].first, &value_ptr_list[cursor_index[i]]);
          }
        }
      };
      auto worker_threads = ctx.worker_threads;
      Shard(worker_threads->num_threads, worker_threads->workers,
            total, 100000, do_insert);
    }
  }

  void AddCopyBackFlagToValuePtr(
      ValuePtr<V>** value_ptr, CopyBackFlag copyback_flag) {
    int64 tmp = ((int64)copyback_flag) << copyback_flag_offset_bits_;
    tmp = ((int64)*value_ptr) | tmp;
    *value_ptr = reinterpret_cast<ValuePtr<V>*>(tmp);
  }

  void RemoveCopyBackFlagInValuePtr(ValuePtr<V>** value_ptr) {
    int64 tmp = (1L << (copyback_flag_offset_bits_)) - 1;
    tmp = ((int64)*value_ptr) & tmp;
    *value_ptr = reinterpret_cast<ValuePtr<V>*>(tmp);
  }

  void ImportToHbm(K* ids, int64 size, int64 value_len, int64 emb_index) {
    V* memcpy_buffer_cpu = new V[size * value_len];
    V** value_address = new V*[size];
    V* memcpy_buffer_gpu =
        (V*)gpu_alloc_->AllocateRaw(
            Allocator::kAllocatorAlignment,
            size * value_len * sizeof(V));
    V* dev_value_address =
        (V*)gpu_alloc_->AllocateRaw(
            Allocator::kAllocatorAlignment,
            size * sizeof(V*));
    ValuePtr<V>** gpu_value_ptrs = new ValuePtr<V>*[size];
    ValuePtr<V>** cpu_value_ptrs = new ValuePtr<V>*[size];
    {
      //Mutex with other Import Ops
      mutex_lock l(memory_pool_mu_);
      for (int64 i = 0; i < size; i++) {
        dram_->Get(ids[i], &cpu_value_ptrs[i]);
        gpu_value_ptrs[i] = hbm_->CreateValuePtr(value_len);
        V* val_ptr = embedding_mem_pool_->Allocate();
        gpu_value_ptrs[i]->SetPtr(val_ptr);
        memcpy((char *)gpu_value_ptrs[i]->GetPtr(),
               (char *)cpu_value_ptrs[i]->GetPtr(),
               sizeof(FixedLengthHeader));
      }
    }
    //Split from above for loop for minize the cost of mutex lock
    //TODO: Speed up with intra parallelism
    std::vector<ValuePtr<V>*> invalid_value_ptrs;
    for (int64 i = 0; i < size; i++) {
      memcpy(memcpy_buffer_cpu + i * value_len,
             cpu_value_ptrs[i]->GetValue(emb_index,
             Storage<K, V>::GetOffset(emb_index)), value_len * sizeof(V));
      Status s = hbm_->TryInsert(ids[i], gpu_value_ptrs[i]);
      if (!s.ok()) {
        invalid_value_ptrs.emplace_back(gpu_value_ptrs[i]);
        hbm_->Get(ids[i], &gpu_value_ptrs[i]);
      }
      gpu_value_ptrs[i]->SetInitialized(emb_index);
      value_address[i] = gpu_value_ptrs[i]->GetValue(
          emb_index, Storage<K, V>::GetOffset(emb_index));
    }
    cudaMemcpy(memcpy_buffer_gpu, memcpy_buffer_cpu,
               size * value_len * sizeof(V), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_value_address, value_address,
               size * sizeof(V*), cudaMemcpyHostToDevice);
    {
      mutex_lock l(memory_pool_mu_);
      embedding_mem_pool_->Deallocate(invalid_value_ptrs);
    }
    int block_dim = 128;
    void* args[] = {(void*)&dev_value_address, (void*)&memcpy_buffer_gpu,
                    (void*)&value_len, (void*)&size};

    cudaLaunchKernel((void *)BatchUnpack<V>,
                     (size + block_dim - 1) / block_dim * value_len,
                     block_dim, args, 0, NULL);
    cudaDeviceSynchronize();

    delete[] memcpy_buffer_cpu;
    delete[] cpu_value_ptrs;
    delete[] gpu_value_ptrs;
    delete[] value_address;
    gpu_alloc_->DeallocateRaw(dev_value_address);
    gpu_alloc_->DeallocateRaw(memcpy_buffer_gpu);
  }

 private:
  HbmStorageWithCpuKv<K, V>* hbm_ = nullptr;
  DramStorage<K, V>* dram_ = nullptr;
  EmbeddingMemoryPool<V>* embedding_mem_pool_ = nullptr;
  Allocator* gpu_alloc_;
  mutex memory_pool_mu_; //ensure thread safety of embedding_mem_pool_
  const int copyback_flag_offset_bits_ = 60;
};
} // embedding
} // tensorflow

#endif  // GOOGLE_CUDA
#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_STORAGE_H_
