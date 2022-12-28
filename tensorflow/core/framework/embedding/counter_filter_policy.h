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
namespace embedding{
template <class K, class V>
class StorageManager;
}

template<typename K, typename V, typename EV>
class CounterFilterPolicy : public FilterPolicy<K, V, EV> {
 public:
  CounterFilterPolicy(const EmbeddingConfig& config,
      EV* ev, embedding::StorageManager<K, V>* storage_manager)
       : config_(config), ev_(ev), storage_manager_(storage_manager) {
  }

  Status Lookup(EV* ev, K key, V* val, const V* default_value_ptr,
      const V* default_value_no_permission) override {
    return errors::Unimplemented(
        "Can't use counter filter in EV for inference.");
  }

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

  void CopyEmbeddingsToBuffer(
      V* val_base, int64 size,
      int64 slice_elems, int64 value_len,
      V** memcpy_address) {
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** val,
      bool* is_filter) override {
    Status s = ev_->LookupOrCreateKey(key, val);
    *is_filter = GetFreq(key, *val) >= config_.filter_freq;
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

  Status Import(RestoreBuffer& restore_buff,
                int64 key_num,
                int bucket_num,
                int64 partition_id,
                int64 partition_num,
                bool is_filter) override {
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
      ev_->CreateKey(key_buff[i], &value_ptr);
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
        if (!is_filter) {
           V* v = ev_->LookupOrCreateEmb(value_ptr,
               value_buff + i * ev_->ValueLen());
        } else {
           V* v = ev_->LookupOrCreateEmb(value_ptr,
               ev_->GetDefaultValue(key_buff[i]));
        }
      }
    }
    if (ev_->IsMultiLevel()) {
      ev_->UpdateCache(key_buff, key_num, version_buff, freq_buff);
    }
    return Status::OK();
  }

 private:
  EmbeddingConfig config_;
  embedding::StorageManager<K, V>* storage_manager_;
  EV* ev_;
};

} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_COUNTER_FILTER_POLICY_H_
