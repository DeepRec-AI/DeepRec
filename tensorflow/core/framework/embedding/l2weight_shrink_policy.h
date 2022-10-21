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
  L2WeightShrinkPolicy(int64 primary_index, int64 primary_offset,
      KVInterface<K, V>* kv, Allocator* alloc)
      : primary_index_(primary_index), primary_offset_(primary_offset),
        ShrinkPolicy<K, V>(kv, alloc) {}

  TF_DISALLOW_COPY_AND_ASSIGN(L2WeightShrinkPolicy);
  
  void Shrink(int64 value_len, V l2_weight_threshold) {
    ShrinkPolicy<K, V>::GetSnapshot();
    FilterToDelete(value_len, l2_weight_threshold);
    ShrinkPolicy<K, V>::ReleaseDeleteValues();
  }

 private: 
  void FilterToDelete(int64 value_len, V l2_weight_threshold) {
    for (int64 i = 0; i < ShrinkPolicy<K, V>::key_list_.size(); ++i) {
      V* val = ShrinkPolicy<K, V>::value_list_[i]->GetValue(
          primary_index_, primary_offset_);
      if (val != nullptr) {
        V l2_weight = (V)0.0;
        for (int64 j = 0; j < value_len; j++) {
            l2_weight += val[j] * val[j];
        }
        l2_weight *= (V)0.5;
        if (l2_weight < l2_weight_threshold) {
          ShrinkPolicy<K, V>::to_delete_.emplace_back(
              ShrinkPolicy<K, V>::key_list_[i], ShrinkPolicy<K, V>::value_list_[i]);
        }
      }
    }
  }

 private:
  int64 primary_index_; // Shrink only handle primary slot
  int64 primary_offset_;
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_L2WEIGHT_SHRINK_POLICY_H_
