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
  GlobalStepShrinkPolicy(int64 steps_to_live,
                         Allocator* alloc,
                         KVInterface<K, V>* kv)
      : steps_to_live_(steps_to_live),
        kv_(kv),
        ShrinkPolicy<K, V>(alloc) {}

  TF_DISALLOW_COPY_AND_ASSIGN(GlobalStepShrinkPolicy);

  void Shrink(std::vector<K>& key_list,
              std::vector<ValuePtr<V>*>& value_list,
              const ShrinkArgs& shrink_args) override {
    ShrinkPolicy<K, V>::ReleaseValuePtrs();
    FilterToDelete(shrink_args.global_step,
        key_list, value_list);
  }

 private:
  void FilterToDelete(int64 global_step,
                      std::vector<K>& key_list,
                      std::vector<ValuePtr<V>*>& value_list) {
    for (int64 i = 0; i < key_list.size(); ++i) {
      int64 version = value_list[i]->GetStep();
      if (version == -1) {
        value_list[i]->SetStep(global_step);
      } else {
        if (global_step - version > steps_to_live_) {
          kv_->Remove(key_list[i]);
          ShrinkPolicy<K, V>::EmplacePointer(value_list[i]);
          value_list[i] = (ValuePtr<V>*)ValuePtrStatus::IS_DELETED;
        }
      }
    }
  }

 private:
  int64 steps_to_live_;
  KVInterface<K, V>* kv_;
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_GLOBALSTEP_SHRINK_POLICY_H_
