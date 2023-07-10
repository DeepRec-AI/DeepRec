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

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_NULLABLE_FILTER_POLICY_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_NULLABLE_FILTER_POLICY_H_

#include "tensorflow/core/framework/embedding/batch.h"
#include "tensorflow/core/framework/embedding/embedding_config.h"
#include "tensorflow/core/framework/embedding/filter_policy.h"

namespace tensorflow {
namespace embedding{
template <class K, class V>
class Storage;
}

template<typename K, typename V, typename EV>
class NullableFilterPolicy : public FilterPolicy<K, V, EV> {
 using FilterPolicy<K, V, EV>::ev_;
 using FilterPolicy<K, V, EV>::config_;
 using FilterPolicy<K, V, EV>::LookupOrCreateEmbInternal;

 public:
  NullableFilterPolicy(const EmbeddingConfig& config,
                       EV* ev, embedding::Storage<K, V>* storage) : 
      FilterPolicy<K, V, EV>(config, ev), storage_(storage) {}

  Status Lookup(K key, V* val, const V* default_value_ptr,
      const V* default_value_no_permission) override {
    ValuePtr<V>* value_ptr = nullptr;
    Status s = ev_->LookupKey(key, &value_ptr);
    if (s.ok()) {
      V* mem_val = ev_->LookupOrCreateEmb(value_ptr, default_value_ptr);
      memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
    } else {
      memcpy(val, default_value_ptr,
             sizeof(V) * ev_->ValueLen());
    }
    return Status::OK();
  }

#if GOOGLE_CUDA
  void BatchLookup(const EmbeddingVarContext<GPUDevice>& ctx,
                   const K* keys, V* output,
                   int64 num_of_keys,
                   V* default_value_ptr,
                   V* default_value_no_permission) override {
    std::vector<ValuePtr<V>*> value_ptr_list(num_of_keys, nullptr);
    ev_->BatchLookupKey(ctx, keys, value_ptr_list.data(), num_of_keys);
    std::vector<V*> embedding_ptr(num_of_keys, nullptr);
    auto do_work = [this, keys, value_ptr_list, &embedding_ptr,
                    default_value_ptr, default_value_no_permission]
        (int64 start, int64 limit) {
      for (int i = start; i < limit; i++) {
        ValuePtr<V>* value_ptr = value_ptr_list[i];
        if (value_ptr != nullptr) {
          embedding_ptr[i] =
              ev_->LookupOrCreateEmb(value_ptr, default_value_ptr);
        } else {
          embedding_ptr[i] = default_value_ptr;
        }
      }
    };
    auto worker_threads = ctx.worker_threads;
    Shard(worker_threads->num_threads,
          worker_threads->workers, num_of_keys,
          1000, do_work);
    auto stream = ctx.compute_stream;
    auto event_mgr = ctx.event_mgr;
    ev_->CopyEmbeddingsToBuffer(
        output, num_of_keys, embedding_ptr.data(),
        stream, event_mgr, ctx.gpu_device);
  }

  void BatchLookupOrCreateKey(const EmbeddingVarContext<GPUDevice>& ctx,
                              const K* keys, ValuePtr<V>** value_ptrs,
                              int64 num_of_keys) {
    int num_worker_threads = ctx.worker_threads->num_threads;
    std::vector<std::list<int64>>
        not_found_cursor_list(num_worker_threads + 1);
    ev_->BatchLookupOrCreateKey(ctx, keys, value_ptrs,
                                num_of_keys, not_found_cursor_list);
    std::vector<V*> var_ptrs(num_of_keys);
    auto do_work = [this, value_ptrs, &var_ptrs]
        (int64 start, int64 limit) {
      for (int i = start; i < limit; i++) {
        bool is_need_set_default_value = false;
        var_ptrs[i] = ev_->LookupOrCreateEmb(
            value_ptrs[i], is_need_set_default_value);
      }
    };
    auto worker_threads = ctx.worker_threads;
    Shard(worker_threads->num_threads,
          worker_threads->workers, num_of_keys,
          1000, do_work);

    ev_->SetDefaultValueOfNewFeatures(
        keys, num_of_keys,
        not_found_cursor_list[0],
        var_ptrs.data(), ctx.compute_stream,
        ctx.event_mgr, ctx.gpu_device);
  }
#endif //GOOGLE_CUDA

  void LookupOrCreate(K key, V* val, const V* default_value_ptr,
                      ValuePtr<V>** value_ptr, int count,
                      const V* default_value_no_permission) override {
    TF_CHECK_OK(ev_->LookupOrCreateKey(key, value_ptr));
    V* mem_val = ev_->LookupOrCreateEmb(*value_ptr, default_value_ptr);
    memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** val,
      bool* is_filter, int64 count) override {
    *is_filter = true;
    return ev_->LookupOrCreateKey(key, val);
  }

  int64 GetFreq(K key, ValuePtr<V>* value_ptr) override {
    if (storage_->GetLayoutType() != LayoutType::LIGHT) {
      return value_ptr->GetFreq();
    }else {
      return 0;
    }
  }

  int64 GetFreq(K key) override {
    if (storage_->GetLayoutType() != LayoutType::LIGHT) {
      ValuePtr<V>* value_ptr = nullptr;
      TF_CHECK_OK(ev_->LookupOrCreateKey(key, &value_ptr));
      return value_ptr->GetFreq();
    }else {
      return 0;
    }
  }

  Status Restore(int64 key_num, int bucket_num, int64 partition_id,
                 int64 partition_num, int64 value_len, bool is_filter,
                 bool to_dram, bool is_incr,
                 RestoreBuffer& restore_buff) override {
    K* key_buff = (K*)restore_buff.key_buffer;
    V* value_buff = (V*)restore_buff.value_buffer;
    int64* version_buff = (int64*)restore_buff.version_buffer;
    int64* freq_buff = (int64*)restore_buff.freq_buffer;
    for (auto i = 0; i < key_num; ++i) {
      // this can describe by graph(Mod + DynamicPartition),
      // but memory waste and slow
      if (*(key_buff + i) % bucket_num % partition_num != partition_id) {
        LOG(INFO) << "skip EV key:" << *(key_buff + i);
        continue;
      }
      ValuePtr<V>* value_ptr = nullptr;
      ev_->CreateKey(key_buff[i], &value_ptr, to_dram);
      if (config_.filter_freq !=0 || ev_->IsMultiLevel()
          || config_.record_freq) {
        value_ptr->SetFreq(freq_buff[i]);
      }
      if (config_.steps_to_live != 0 || config_.record_version) {
        value_ptr->SetStep(version_buff[i]);
      }
      LookupOrCreateEmbInternal(is_filter, to_dram, i, value_len,
                                value_ptr, value_buff, key_buff);
    }
    return Status::OK();
  }

  bool is_admit(K key, ValuePtr<V>* value_ptr) override {
    return true;
  }

 private:
  embedding::Storage<K, V>* storage_;
};

} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_NULLABLE_FILTER_POLICY_H_

