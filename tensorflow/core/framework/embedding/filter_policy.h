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

namespace tensorflow {

struct RestoreBuffer {
  char* key_buffer = nullptr;
  char* value_buffer = nullptr;
  char* version_buffer = nullptr;
  char* freq_buffer = nullptr;

  ~RestoreBuffer() {
    delete key_buffer;
    delete value_buffer;
    delete version_buffer;
    delete freq_buffer;
  }
};

template<typename K, typename V, typename EV>
class FilterPolicy {
 public:
  virtual void LookupOrCreate(K key, V* val,
      const V* default_value_ptr, ValuePtr<V>** value_ptr,
      int count, const V* default_value_no_permission) = 0;

  virtual void WeightedLookupOrCreate(K key, V* val, V* sp_weights,
      const V* default_value_ptr, ValuePtr<V>** value_ptr,
      int count, const V* default_value_no_permission) = 0;

  virtual Status Lookup(EV* ev, K key, V* val, const V* default_value_ptr,
    const V* default_value_no_permission) = 0;

  virtual Status LookupOrCreateKey(K key, ValuePtr<V>** val,
      bool* is_filter) = 0;

  virtual int64 GetFreq(K key, ValuePtr<V>* value_ptr) = 0;
  virtual int64 GetFreq(K key) = 0;
  virtual Status Import(RestoreBuffer& restore_buff,
    int64 key_num,
    int bucket_num,
    int64 partition_id,
    int64 partition_num,
    bool is_filter) = 0;
  virtual Status ImportToDram(RestoreBuffer& restore_buff,
                int64 key_num,
                int bucket_num,
                int64 partition_id,
                int64 partition_num,
                bool is_filter,
                V* default_values) = 0;
};
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FILTER_POLICY_H_

