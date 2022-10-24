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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SHRINK_POLICY_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SHRINK_POLICY_H_

#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

template<typename V>
class ValuePtr;

namespace embedding {
template<typename K, typename V>
class ShrinkPolicy {
 public:
  ShrinkPolicy(KVInterface<K, V>* kv, Allocator* alloc)
      : kv_(kv), alloc_(alloc) {}

  TF_DISALLOW_COPY_AND_ASSIGN(ShrinkPolicy);

  inline Status GetSnapshot() {
    return kv_->GetSnapshot(&key_list_, &value_list_);
  }
  
  void ReleaseDeleteValues() {
    for (auto it : to_delete_) {
      (it.value_ptr)->Destroy(alloc_);
      delete it.value_ptr;
      kv_->Remove(it.key);
    }
  }

  struct KeyValuePair {
    KeyValuePair(const K& k, ValuePtr<V>* v) : key(k), value_ptr(v) {}

    K key;
    ValuePtr<V>* value_ptr;
  };

 protected: 
  std::vector<K> key_list_;
  std::vector<ValuePtr<V>*> value_list_;
  std::vector<KeyValuePair> to_delete_;

  KVInterface<K, V>* kv_;
  Allocator* alloc_;
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SHRINK_POLICY_H_
