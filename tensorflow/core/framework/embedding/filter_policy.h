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

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FILTER_POLICY_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FILTER_POLICY_H_

#include "tensorflow/core/framework/embedding/embedding_config.h"
#include "tensorflow/core/framework/embedding/emb_file.h"

namespace tensorflow {

struct RestoreBuffer {
  char* key_buffer = nullptr;
  char* value_buffer = nullptr;
  char* version_buffer = nullptr;
  char* freq_buffer = nullptr;

  explicit RestoreBuffer(size_t buffer_size) {
    key_buffer = new char[buffer_size];
    value_buffer = new char[buffer_size];
    version_buffer = new char[buffer_size];
    freq_buffer = new char[buffer_size];
  }

  ~RestoreBuffer() {
    delete []key_buffer;
    delete []value_buffer;
    delete []version_buffer;
    delete []freq_buffer;
  }
};

template<typename K>
class RestoreSSDBuffer;

template <typename V>
class ValuePtr;

template<typename K, typename V, typename EV>
class FilterPolicy {
 public:
  FilterPolicy(const EmbeddingConfig& config, EV* ev) :
      config_(config), ev_(ev) {}

  virtual void LookupOrCreate(K key, V* val,
      const V* default_value_ptr, ValuePtr<V>** value_ptr,
      int count, const V* default_value_no_permission) = 0;

  virtual Status Lookup(K key, V* val, const V* default_value_ptr,
    const V* default_value_no_permission) = 0;

#if GOOGLE_CUDA
  virtual void BatchLookup(const EmbeddingVarContext<GPUDevice>& context,
                           const K* keys, V* output,
                           int64 num_of_keys,
                           V* default_value_ptr,
                           V* default_value_no_permission) = 0;

  virtual void BatchLookupOrCreateKey(
      const EmbeddingVarContext<GPUDevice>& ctx,
      const K* keys, ValuePtr<V>** value_ptrs_list,
      int64 num_of_keys) = 0;
#endif //GOOGLE_CUDA

  virtual Status LookupOrCreateKey(K key, ValuePtr<V>** val,
      bool* is_filter, int64 count) = 0;

  virtual int64 GetFreq(K key, ValuePtr<V>* value_ptr) = 0;

  virtual int64 GetFreq(K key) = 0;

  virtual bool is_admit(K key, ValuePtr<V>* value_ptr) = 0;

  virtual Status Restore(int64 key_num, int bucket_num, int64 partition_id,
                         int64 partition_num, int64 value_len, bool is_filter,
                         bool to_dram, bool is_incr, RestoreBuffer& restore_buff) = 0;

 protected:
  void LookupOrCreateEmbInternal(bool is_filter, bool to_dram,
                                 int i, int value_len,
                                 ValuePtr<V>* value_ptr,
                                 V* value_src, K* key_src) {
    
    if (!is_filter) {
      ev_->LookupOrCreateEmb(value_ptr, value_src + i * ev_->ValueLen());
      return;
    } else {
      if (to_dram) {
#if GOOGLE_CUDA
        std::vector<V> default_value_host;
        default_value_host.resize(config_.default_value_dim * value_len);
        cudaMemcpy(default_value_host.data(), ev_->GetDefaultValuePtr(),
                    sizeof(V) * config_.default_value_dim * value_len,
                    cudaMemcpyDeviceToHost);
        ev_->LookupOrCreateEmb(value_ptr,
                               default_value_host.data() +
                                  (key_src[i] % config_.default_value_dim)
                                  * ev_->ValueLen());
#endif
        return;
      } else {
        ev_->LookupOrCreateEmb(value_ptr, ev_->GetDefaultValue(key_src[i]));
      return;
      }
    }
  }

 protected:
  EmbeddingConfig config_;
  EV* ev_;
};
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FILTER_POLICY_H_

