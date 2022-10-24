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

#include "tensorflow/core/framework/embedding/embedding_config.h"
#include "tensorflow/core/framework/embedding/filter_policy.h"

#if GOOGLE_CUDA
#if !TENSORFLOW_USE_GPU_EV
#include "tensorflow/core/framework/embedding/batch.h"
#endif  // TENSORFLOW_USE_GPU_EV
#endif  // GOOGLE_CUDA

namespace tensorflow {
namespace embedding{
template <class K, class V>
class StorageManager;
}

template<typename K, typename V, typename EV>
class NullableFilterPolicy : public FilterPolicy<K, V, EV> {
 public:
  NullableFilterPolicy(const EmbeddingConfig& config,
      EV* ev, embedding::StorageManager<K, V>* storage_manager)
       : config_(config), ev_(ev), storage_manager_(storage_manager) {
  }

  Status Lookup(EV* ev, K key, V* val, const V* default_value_ptr,
      const V* default_value_no_permission) override {
    ValuePtr<V>* value_ptr = nullptr;
    Status s = ev->LookupKey(key, &value_ptr);
    if (s.ok()) {
      V* mem_val = ev->LookupPrimaryEmb(value_ptr);
      memcpy(val, mem_val, sizeof(V) * ev->ValueLen());
    } else {
      memcpy(val, default_value_no_permission,
          sizeof(V) * ev->ValueLen());
    }
    return Status::OK();
  }

  void LookupOrCreate(K key, V* val, const V* default_value_ptr,
                      ValuePtr<V>** value_ptr, int count,
                      const V* default_value_no_permission) override {
    TF_CHECK_OK(ev_->LookupOrCreateKey(key, value_ptr));
    V* mem_val = ev_->LookupOrCreateEmb(*value_ptr, default_value_ptr);
    memcpy(val, mem_val, sizeof(V) * ev_->ValueLen());
  }

  void CreateGPUBatch(V* val_base, V** default_values, int64 size,
    int64 slice_elems, int64 value_len, bool* init_flags, V** memcpy_address) {
#if GOOGLE_CUDA
#if !TENSORFLOW_USE_GPU_EV
    int block_dim = 128;
    V** dev_value_address = (V**)ev_->GetBuffer1(size);
    V** dev_default_address = (V**)ev_->GetBuffer2(size);
    bool* dev_init_flags = (bool*)ev_->GetBuffer3(size);

    cudaMemcpy(dev_value_address, memcpy_address,
        sizeof(V *) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_default_address, default_values,
        sizeof(V *) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_init_flags, init_flags,
        sizeof(bool) * size, cudaMemcpyHostToDevice);

    int limit = size;
    int length = value_len;
    void* args1[] = {(void*)&dev_value_address,
                     (void*)&val_base,
                     (void*)&length,
                     (void*)&limit,
                     (void*)&dev_default_address,
                     (void*)&dev_init_flags};
    cudaLaunchKernel((void *)BatchCopy<V>,
                     (limit + block_dim - 1) / block_dim * length,
                     block_dim, args1, 0, NULL);
    cudaDeviceSynchronize();
#endif  // TENSORFLOW_USE_GPU_EV
#endif  // GOOGLE_CUDA
  }

  Status LookupOrCreateKey(K key, ValuePtr<V>** val,
      bool* is_filter) override {
    *is_filter = true;
    return ev_->LookupOrCreateKey(key, val);
  }

  int64 GetFreq(K key, ValuePtr<V>* value_ptr) override {
    if (storage_manager_->GetLayoutType() != LayoutType::LIGHT) {
      return value_ptr->GetFreq();
    }else {
      return 0;
    }
  }

  int64 GetFreq(K key) override {
    if (storage_manager_->GetLayoutType() != LayoutType::LIGHT) {
      ValuePtr<V>* value_ptr = nullptr;
      TF_CHECK_OK(ev_->LookupOrCreateKey(key, &value_ptr));
      return value_ptr->GetFreq();
    }else {
      return 0;
    }
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
      TF_CHECK_OK(ev_->LookupOrCreateKey(key_buff[i], &value_ptr));
      if (config_.filter_freq !=0 || ev_->IsMultiLevel()
          || config_.record_freq) {
        value_ptr->SetFreq(freq_buff[i]);
      }
      if (config_.steps_to_live != 0 || config_.record_version) {
        value_ptr->SetStep(version_buff[i]);
      }
      if (!is_filter) {
        V* v = ev_->LookupOrCreateEmb(value_ptr,
            value_buff + i * ev_->ValueLen());
      }else {
        V* v = ev_->LookupOrCreateEmb(value_ptr,
            ev_->GetDefaultValue(key_buff[i]));
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

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_NULLABLE_FILTER_POLICY_H_

