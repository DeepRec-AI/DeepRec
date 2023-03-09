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
  ShrinkPolicy(KVInterface<K, V>* kv, Allocator* alloc, int slot_num)
      : kv_(kv), alloc_(alloc),
        slot_num_(slot_num), shrink_count_(0) {}

  TF_DISALLOW_COPY_AND_ASSIGN(ShrinkPolicy);

  inline Status GetSnapshot() {
    shrink_count_ = (shrink_count_ + 1) % slot_num_;
    return kv_->GetSnapshot(&key_list_, &value_list_);
  }
  
  void ReleaseDeleteValues() {
    if (shrink_count_ == 0) {
      for (auto it : to_delete_) {
        it->Destroy(alloc_);
        delete it;
      }
      to_delete_.clear();
    }
  }

 protected: 
  std::vector<K> key_list_;
  std::vector<ValuePtr<V>*> value_list_;
  std::vector<ValuePtr<V>*> to_delete_;

  KVInterface<K, V>* kv_;
  Allocator* alloc_;
  int slot_num_;
  int shrink_count_;
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SHRINK_POLICY_H_
