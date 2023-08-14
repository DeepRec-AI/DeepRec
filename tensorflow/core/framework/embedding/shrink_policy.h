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

#include "tensorflow/core/framework/embedding/feature_descriptor.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
class Allocator;

namespace embedding {
struct ShrinkArgs {
  ShrinkArgs(): global_step(0), value_len(0) {}

  ShrinkArgs(int64 global_step,
             int64 value_len)
      : global_step(global_step),
        value_len(value_len) {}
  int64 global_step;
  int64 value_len;
};

template<typename K, typename V>
class ShrinkPolicy {
 public:
  ShrinkPolicy(FeatureDescriptor<V>* feat_desc): feat_desc_(feat_desc) {}
  virtual ~ShrinkPolicy() {}

  TF_DISALLOW_COPY_AND_ASSIGN(ShrinkPolicy);

  virtual void Shrink(std::vector<K>& key_list,
                      std::vector<void*>& value_list,
                      const ShrinkArgs& shrink_args) = 0;

 protected:
  void EmplacePointer(void* value_ptr) {
    to_delete_.emplace_back(value_ptr);
  }

  void ReleaseValuePtrs() {
    for (auto it : to_delete_) {
      feat_desc_->Deallocate(it);
    }
    to_delete_.clear();
  }
 protected:
  std::vector<void*> to_delete_;
  FeatureDescriptor<V>* feat_desc_;
};

template<typename K, typename V>
class NonShrinkPolicy: public ShrinkPolicy<K, V> {
 public:
  NonShrinkPolicy(): ShrinkPolicy<K, V>(nullptr) {}
  TF_DISALLOW_COPY_AND_ASSIGN(NonShrinkPolicy);

  void Shrink(std::vector<K>& key_list,
              std::vector<void*>& value_list,
              const ShrinkArgs& shrink_args) override {}
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SHRINK_POLICY_H_
