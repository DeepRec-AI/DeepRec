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
#include "tensorflow/core/framework/embedding/multi_tier_storage.h"
#include "tensorflow/core/framework/embedding/single_tier_storage.h"
#include "tensorflow/core/framework/embedding/hbm_storage_iterator.h"
#include "tensorflow/core/framework/embedding/intra_thread_copy_id_allocator.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
using se::DeviceMemoryBase;
using se::Stream;

template <typename K, typename V>
class CheckpointLoader;

void SyncWithEventMgr(se::Stream* stream, EventMgr* event_mgr);

namespace embedding {
template<typename K, typename V>
class HbmDramStorage : public MultiTierStorage<K, V> {
 public:
  HbmDramStorage(const StorageConfig& sc,
      Allocator* gpu_alloc,
      FeatureDescriptor<V>* feat_desc, const std::string& name)
      : gpu_alloc_(gpu_alloc),
        MultiTierStorage<K, V>(sc, name) {
    hbm_ = new HbmStorageWithCpuKv<K, V>(sc, feat_desc);
    hbm_feat_desc_ = feat_desc;
    dram_feat_desc_ = new FeatureDescriptor<V>(feat_desc);
    dram_ = new DramStorage<K, V>(sc, dram_feat_desc_);
  }

  ~HbmDramStorage() override {
    MultiTierStorage<K, V>::DeleteFromEvictionManager();
    delete hbm_;
    delete dram_;
    delete dram_feat_desc_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(HbmDramStorage);

  Status Get(K key, void** value_ptr) override {
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
                void** value_ptr_list,
                int64 num_of_keys) override {
    int num_worker_threads = ctx.worker_threads->num_threads;
    std::vector<std::list<int64>>
        copyback_cursor_list(num_worker_threads + 1);

    BatchGetValuePtrs(ctx, keys, value_ptr_list, num_of_keys,
                      copyback_cursor_list);

    CopyEmbeddingsFromDramToHbm(
        ctx, keys, value_ptr_list, copyback_cursor_list[0]);
  }

  void Insert(K key, void** value_ptr) override {
    hbm_->Insert(key, value_ptr);
  }

  void BatchGetOrCreate(
      const EmbeddingVarContext<GPUDevice>& ctx,
      const K* keys,
      void** value_ptr_list,
      int64 num_of_keys,
      int64 value_len,
      std::vector<std::list<int64>>& not_fountd_cursor_list) override {
    int num_worker_threads = ctx.worker_threads->num_threads;
    std::vector<std::list<int64>>
        copyback_cursor_list(num_worker_threads + 1);

    BatchGetValuePtrs(ctx, keys, value_ptr_list, num_of_keys,
                      copyback_cursor_list, &not_fountd_cursor_list);

    CopyEmbeddingsFromDramToHbm(
        ctx, keys, value_ptr_list, copyback_cursor_list[0]);
    CreateValuePtrs(ctx, keys, value_ptr_list,
                    not_fountd_cursor_list[0], value_len);
  }

  void CreateAndInsert(K key, void** value_ptr,
      bool to_dram=false) override {
    if (to_dram) {
      dram_->CreateAndInsert(key, value_ptr);
    } else {
      hbm_->CreateAndInsert(key, value_ptr);
    }
  }

  Status GetOrCreate(K key, void** value_ptr) override {
    LOG(FATAL)<<"Stroage with HBM only suppotrs batch APIs.";
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

  Status Save(
      const string& tensor_name,
      const string& prefix,
      BundleWriter* writer,
      const EmbeddingConfig& emb_config,
      ShrinkArgs& shrink_args,
      int64 value_len,
      V* default_value) override {
    std::vector<K> key_list, tmp_dram_key_list;
    std::vector<void*> value_ptr_list, tmp_dram_value_list;
    TF_CHECK_OK(hbm_->GetSnapshot(&key_list, &value_ptr_list));
    hbm_->Shrink(key_list, value_ptr_list, shrink_args, value_len);

    HbmValueIterator<K, V> hbm_value_iter(
        key_list, value_ptr_list,
        emb_config.emb_index, value_len,
        gpu_alloc_, hbm_feat_desc_);
    
    for (int64 i = 0; i < value_ptr_list.size(); i++) {
      void* value_ptr = cpu_allocator()->AllocateRaw(
          Allocator::kAllocatorAlignment, hbm_feat_desc_->data_bytes());
      hbm_feat_desc_->SetFreq(
          value_ptr, hbm_feat_desc_->GetFreq(value_ptr_list[i]));
      hbm_feat_desc_->UpdateVersion(
          value_ptr, hbm_feat_desc_->GetVersion(value_ptr_list[i]));
      value_ptr_list[i] = (void*)((int64)value_ptr | (1L << kDramFlagOffset));
    }

    TF_CHECK_OK(dram_->GetSnapshot(&tmp_dram_key_list,
                                   &tmp_dram_value_list));
    dram_->Shrink(tmp_dram_key_list, tmp_dram_value_list,
                  shrink_args, value_len);

    for (int64 i = 0; i < tmp_dram_key_list.size(); i++) {
      Status s = hbm_->Contains(tmp_dram_key_list[i]);
      if (!s.ok()) {
        key_list.emplace_back(tmp_dram_key_list[i]);
        value_ptr_list.emplace_back(tmp_dram_value_list[i]);
      }
    }

    {
      mutex_lock l(*(hbm_->get_mutex()));
      std::vector<FeatureDescriptor<V>*> feat_desc_list(2);
      feat_desc_list[0] = dram_feat_desc_;
      feat_desc_list[1] = hbm_feat_desc_;
      TF_CHECK_OK((Storage<K, V>::SaveToCheckpoint(
          tensor_name, writer,
          emb_config,
          value_len, default_value,
          key_list,
          value_ptr_list,
          feat_desc_list,
          &hbm_value_iter)));
    }

    for (auto value_ptr: value_ptr_list) {
      if ((int64)value_ptr >> kDramFlagOffset == 1)
        cpu_allocator()->DeallocateRaw(value_ptr);
    }
    return Status::OK();
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
      void* value_ptr;
      std::vector<K> keys;
      std::vector<void*> hbm_value_ptrs;
      std::vector<void*> dram_value_ptrs;

      for (int64 i = 0; i < true_size; ++i) {
        if (hbm_->Get(evic_ids[i], &value_ptr).ok()) {
          keys.emplace_back(evic_ids[i]);
          hbm_value_ptrs.emplace_back(value_ptr);
          void* dram_value_ptr = dram_->CreateValuePtr();
          dram_feat_desc_->SetFreq(dram_value_ptr,
              hbm_feat_desc_->GetFreq(value_ptr));
          dram_feat_desc_->UpdateVersion(dram_value_ptr,
              hbm_feat_desc_->GetVersion(value_ptr));
          dram_value_ptrs.emplace_back(dram_value_ptr);
        }
      }
      
      CopyEmbeddingFromHbmToDram(
          hbm_value_ptrs,
          dram_value_ptrs, gpu_alloc_,
          hbm_feat_desc_, dram_feat_desc_);

      dram_->BatchCommit(keys, dram_value_ptrs);
      hbm_feat_desc_->Deallocate(hbm_value_ptrs);
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

  void UpdateValuePtr(K key, void* new_value_ptr,
                      void* old_value_ptr) override {
    hbm_->UpdateValuePtr(key, new_value_ptr, old_value_ptr);
  }

  void Init() override {
    dram_feat_desc_->InitSlotInfo(hbm_feat_desc_);
    MultiTierStorage<K, V>::Init();
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

  int total_dim() override {
    return hbm_feat_desc_->total_dim();
  }
 private:
  void BatchGetValuePtrs(
      const EmbeddingVarContext<GPUDevice>& ctx,
      const K* keys,
      void** value_ptr_list,
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
                                   void** value_ptr_list,
                                   std::list<int64>& copyback_cursors) {
    int64 total = copyback_cursors.size();
    std::vector<void*> gpu_value_ptrs(total);
    std::vector<K> copyback_keys(total);
    std::vector<int64> memory_index(total);
    //Create Hbm ValuePtrs.
    int64 i = 0;
    auto it = copyback_cursors.cbegin();
    //Mutex with eviction thread
    for ( ; it != copyback_cursors.cend(); ++it, ++i) {
      int64 j = *it;
      memory_index[i] = j;
      void* gpu_value_ptr = hbm_->CreateValuePtr();
      hbm_feat_desc_->SetFreq(gpu_value_ptr,
          dram_feat_desc_->GetFreq(value_ptr_list[i]));
      hbm_feat_desc_->UpdateVersion(gpu_value_ptr,
          dram_feat_desc_->GetVersion(value_ptr_list[i]));
      gpu_value_ptrs[i] = gpu_value_ptr;
      copyback_keys[i] = keys[*it];
    }
    MultiTierStorage<K, V>::CopyEmbeddingsFromDramToHbm(
        ctx, keys, value_ptr_list, copyback_cursors,
        memory_index, gpu_value_ptrs, hbm_feat_desc_->total_dim(),
        hbm_feat_desc_, dram_feat_desc_);

    //Insert copyback ids to hbm hash table.
    auto do_insert = [this, copyback_keys, gpu_value_ptrs,
                      memory_index, value_ptr_list]
        (int64 start, int64 limit) {
      for (int64 i = start; i < limit; i++) {
        Status s = hbm_->TryInsert(
            copyback_keys[i], gpu_value_ptrs[i]);
        if (!s.ok()) {
          hbm_->DestroyValuePtr(gpu_value_ptrs[i]);
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
                       void** value_ptr_list,
                       std::list<int64>& not_found_cursors,
                       int64 value_len) {
    int64 total = not_found_cursors.size();
    if (total > 0) {
      std::vector<std::pair<int64, void*>> insert_pairs(total);
      std::vector<int64> cursor_index(total);
      //Create Hbm ValuePtrs.      
      int64 i = 0;
      auto it = not_found_cursors.cbegin();
      for ( ; it != not_found_cursors.cend(); ++it, ++i) {
        int64 j = *it;
        cursor_index[i] = j;
        void* gpu_value_ptr = hbm_->CreateValuePtr();
        value_ptr_list[j] = gpu_value_ptr;
        insert_pairs[i].first = keys[j];
        insert_pairs[i].second = value_ptr_list[j];
      }

      hbm_feat_desc_->SetDefaultValues(
          keys, not_found_cursors, value_ptr_list,
          ctx.compute_stream, ctx.event_mgr, ctx.gpu_device);

      //Insert copyback ids to hbm hash table.
      auto do_insert = [this, insert_pairs, value_ptr_list, cursor_index]
          (int64 start, int64 limit) {
        for (int64 i = start; i < limit; i++) {
          Status s = hbm_->TryInsert(
              insert_pairs[i].first, insert_pairs[i].second);
          if (!s.ok()) {
            hbm_->DestroyValuePtr(insert_pairs[i].second);
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
      void** value_ptr, CopyBackFlag copyback_flag) {
    int64 tmp = ((int64)copyback_flag) << copyback_flag_offset_bits_;
    tmp = ((int64)*value_ptr) | tmp;
    *value_ptr = reinterpret_cast<void*>(tmp);
  }

  void RemoveCopyBackFlagInValuePtr(void** value_ptr) {
    int64 tmp = (1L << (copyback_flag_offset_bits_)) - 1;
    tmp = ((int64)*value_ptr) & tmp;
    *value_ptr = reinterpret_cast<void*>(tmp);
  }

  void Import(K key, V* value,
              int64 freq, int64 version,
              int emb_index) override {
    dram_->Import(key, value, freq, version, emb_index);
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
    void** gpu_value_ptrs = new void*[size];
    void** cpu_value_ptrs = new void*[size];
    for (int64 i = 0; i < size; i++) {
      dram_->Get(ids[i], &cpu_value_ptrs[i]);
      gpu_value_ptrs[i] = hbm_->CreateValuePtr();
      Status s = hbm_->TryInsert(ids[i], gpu_value_ptrs[i]);
      if (!s.ok()) {
        hbm_feat_desc_->Deallocate(gpu_value_ptrs[i]);
        hbm_->Get(ids[i], &gpu_value_ptrs[i]);
      }
    }
    //Split from above for loop for minize the cost of mutex lock
    //TODO: Speed up with intra parallelism
    
    for (int64 i = 0; i < size; i++) {
      memcpy(memcpy_buffer_cpu + i * value_len,
             dram_feat_desc_->GetEmbedding(cpu_value_ptrs[i], emb_index),
             value_len * sizeof(V));
      value_address[i] = hbm_feat_desc_->GetEmbedding(gpu_value_ptrs[i], emb_index);
    }
    cudaMemcpy(memcpy_buffer_gpu, memcpy_buffer_cpu,
               size * value_len * sizeof(V), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_value_address, value_address,
               size * sizeof(V*), cudaMemcpyHostToDevice);
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
  FeatureDescriptor<V>* hbm_feat_desc_ = nullptr;
  FeatureDescriptor<V>* dram_feat_desc_ = nullptr;
  Allocator* gpu_alloc_;
  const int copyback_flag_offset_bits_ = 60;
};
} // embedding
} // tensorflow

#endif  // GOOGLE_CUDA
#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_HBM_DRAM_STORAGE_H_
