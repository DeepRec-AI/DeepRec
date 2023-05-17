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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_L2WEIGHT_SHRINK_POLICY_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_L2WEIGHT_SHRINK_POLICY_H_

#include "tensorflow/core/framework/embedding/shrink_policy.h"

namespace tensorflow {

template<typename V>
class ValuePtr;

namespace embedding {
template<typename K, typename V>
class L2WeightShrinkPolicy : public ShrinkPolicy<K, V> {
 public:
  L2WeightShrinkPolicy(float l2_weight_threshold,
                       int64 index,
                       int64 offset,
                       Allocator* alloc,
                       KVInterface<K, V>* kv)
      : index_(index),
        offset_(offset),
        kv_(kv),
        l2_weight_threshold_(l2_weight_threshold),
        ShrinkPolicy<K, V>(alloc) {}

  TF_DISALLOW_COPY_AND_ASSIGN(L2WeightShrinkPolicy);
  
  void Shrink(const ShrinkArgs& shrink_args) override {
    ShrinkPolicy<K, V>::ReleaseValuePtrs();
    std::vector<K> key_list;
    std::vector<ValuePtr<V>*> value_list;
    kv_->GetSnapshot(&key_list, &value_list);
    FilterToDelete(shrink_args.value_len,
                   key_list, value_list);
  }

 private:
  void FilterToDelete(int64 value_len,
                      const std::vector<K>& key_list,
                      const std::vector<ValuePtr<V>*>& value_list) {
    for (int64 i = 0; i < key_list.size(); ++i) {
      V* val = value_list[i]->GetValue(index_, offset_);
      if (val != nullptr) {
        V l2_weight = (V)0.0;
        for (int64 j = 0; j < value_len; j++) {
            l2_weight += val[j] * val[j];
        }
        l2_weight *= (V)0.5;
        if (l2_weight < (V)l2_weight_threshold_) {
          kv_->Remove(key_list[i]);
          ShrinkPolicy<K, V>::EmplacePointer(value_list[i]);
        }
      }
    }
  }

 private:
  int64 index_;
  int64 offset_;
  KVInterface<K, V>* kv_;
  float l2_weight_threshold_;
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_L2WEIGHT_SHRINK_POLICY_H_
