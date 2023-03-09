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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_GLOBALSTEP_SHRINK_POLICY_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_GLOBALSTEP_SHRINK_POLICY_H_

#include "tensorflow/core/framework/embedding/shrink_policy.h"

namespace tensorflow {

template<typename V>
class ValuePtr;

namespace embedding {
template<typename K, typename V>
class GlobalStepShrinkPolicy : public ShrinkPolicy<K, V> {
 public:
  GlobalStepShrinkPolicy(
      KVInterface<K, V>* kv,
      Allocator* alloc,
      int slot_num)
      : ShrinkPolicy<K, V>(kv, alloc, slot_num) {}

  TF_DISALLOW_COPY_AND_ASSIGN(GlobalStepShrinkPolicy);

  void Shrink(int64 global_step, int64 steps_to_live) {
    ShrinkPolicy<K, V>::ReleaseDeleteValues();
    ShrinkPolicy<K, V>::GetSnapshot();
    FilterToDelete(global_step, steps_to_live);
  }

 private:
  void FilterToDelete(int64 global_step, int64 steps_to_live) {
    for (int64 i = 0; i < ShrinkPolicy<K, V>::key_list_.size(); ++i) {
      int64 version = ShrinkPolicy<K, V>::value_list_[i]->GetStep();
      if (version == -1) {
        ShrinkPolicy<K, V>::value_list_[i]->SetStep(global_step);
      } else {
        if (global_step - version > steps_to_live) {
          ShrinkPolicy<K, V>::kv_->Remove(ShrinkPolicy<K, V>::key_list_[i]);
          ShrinkPolicy<K, V>::to_delete_.emplace_back(
              ShrinkPolicy<K, V>::value_list_[i]);
        }
      }
    }
    ShrinkPolicy<K, V>::key_list_.clear();
    ShrinkPolicy<K, V>::value_list_.clear();
  }
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_GLOBALSTEP_SHRINK_POLICY_H_
