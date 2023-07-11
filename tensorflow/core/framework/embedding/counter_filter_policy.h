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

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_COUNTER_FILTER_POLICY_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_COUNTER_FILTER_POLICY_H_

#include "tensorflow/core/framework/embedding/embedding_config.h"
#include "tensorflow/core/framework/embedding/filter_policy.h"

namespace tensorflow {

template<typename K, typename V, typename EV>
class CounterFilterPolicy : public FilterPolicy<K, V, EV> {
 using FilterPolicy<K, V, EV>::ev_;
 using FilterPolicy<K, V, EV>::config_;
 using FilterPolicy<K, V, EV>::LookupOrCreateEmbInternal;

 public:
  CounterFilterPolicy(const EmbeddingConfig& config, EV* ev) :
      FilterPolicy<K, V, EV>(config, ev) {}

  Status Lookup(K key, V* val, const V* default_value_ptr,
      const V* default_value_no_permission) override {
    ValuePtr<V>* value_ptr = nullptr;
    Status s = ev_->LookupKey(key, &value_ptr);
    if (s.ok() && GetFreq(key, value_ptr) >= config_.filter_freq) {
      V* mem_val = ev_->LookupOrCreateEmb(value_ptr, default_value_ptr);
      memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
    } else {
      memcpy(val, default_value_no_permission, sizeof(V) * ev_->ValueLen());
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
        int64 freq = GetFreq(keys[i], value_ptr);
        if (value_ptr != nullptr && freq >= config_.filter_freq) {
          embedding_ptr[i] =
              ev_->LookupOrCreateEmb(value_ptr, default_value_ptr);
        } else {
          embedding_ptr[i] = default_value_no_permission;
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
                              const K* keys, ValuePtr<V>** value_ptrs_list,
                              int64 num_of_keys) override {
    int num_worker_threads = ctx.worker_threads->num_threads;
    std::vector<std::list<int64>>
        not_found_cursor_list(num_worker_threads + 1);
    ev_->BatchLookupOrCreateKey(ctx, keys, value_ptrs_list,
                                num_of_keys, not_found_cursor_list);
  }
#endif //GOOGLE_CUDA

  void LookupOrCreate(K key, V* val, const V* default_value_ptr,
                      ValuePtr<V>** value_ptr, int count,
                      const V* default_value_no_permission) override {
    TF_CHECK_OK(ev_->LookupOrCreateKey(key, value_ptr));
    if (GetFreq(key, *value_ptr) >= config_.filter_freq) {
      V* mem_val = ev_->LookupOrCreateEmb(*value_ptr, default_value_ptr);
      memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
    } else {
      memcpy(val, default_value_no_permission, sizeof(V) * ev_->ValueLen());
    }
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** val,
      bool* is_filter, int64 count) override {
    Status s = ev_->LookupOrCreateKey(key, val);
    *is_filter = (GetFreq(key, *val) + count) >= config_.filter_freq;
    return s;
  }

  int64 GetFreq(K key, ValuePtr<V>* value_ptr) override {
    return value_ptr->GetFreq();
  }

  int64 GetFreq(K key) override {
    ValuePtr<V>* value_ptr = nullptr;
    TF_CHECK_OK(ev_->LookupOrCreateKey(key, &value_ptr));
    return value_ptr->GetFreq();
  }

  bool is_admit(K key, ValuePtr<V>* value_ptr) override {
    return (GetFreq(key, value_ptr) >= config_.filter_freq);
  }

  Status Restore(int64 key_num, int bucket_num, int64 partition_id,
                 int64 partition_num, int64 value_len, bool is_filter,
                 bool to_dram, bool is_incr, RestoreBuffer& restore_buff) override {
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
      if (!is_filter) {
        if (freq_buff[i] >= config_.filter_freq) {
          value_ptr->SetFreq(freq_buff[i]);
        } else {
          value_ptr->SetFreq(config_.filter_freq);
        }
      } else {
        value_ptr->SetFreq(freq_buff[i]);
      }
      if (config_.steps_to_live != 0 || config_.record_version) {
        value_ptr->SetStep(version_buff[i]);
      }
      if (value_ptr->GetFreq() >= config_.filter_freq) {
        LookupOrCreateEmbInternal(is_filter, to_dram, i, value_len,
                                  value_ptr, value_buff, key_buff);
      }
    }
    return Status::OK();
  }
};

} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_COUNTER_FILTER_POLICY_H_
