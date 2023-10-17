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

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FILTER_FACTORY_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FILTER_FACTORY_H_

#include "tensorflow/core/framework/embedding/bloom_filter_policy.h"
#include "tensorflow/core/framework/embedding/counter_filter_policy.h"
#include "tensorflow/core/framework/embedding/embedding_config.h"
#include "tensorflow/core/framework/embedding/filter_policy.h"
#include "tensorflow/core/framework/embedding/nullable_filter_policy.h"

namespace tensorflow {
namespace embedding{
template <class K, class V>
class Storage;
}

class FilterFactory {
 public:
  template<typename K, typename V, typename EV>
  static FilterPolicy<K, V, EV>* CreateFilter(
      const EmbeddingConfig& config, EV* ev,
      embedding::Storage<K, V>* storage,
      embedding::FeatureDescriptor<V>* feat_desc) {
    if (config.filter_freq > 0) {
      if (config.kHashFunc != 0) {
        return new BloomFilterPolicy<K, V, EV>(
            config, ev, feat_desc);
      } else {
        return new CounterFilterPolicy<K, V, EV>(
            config, ev, feat_desc);
      }
    } else {
      return new NullableFilterPolicy<K, V, EV>(
          config, ev, storage, feat_desc);
    }
  }
};

} //namespace tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_FILTER_FACTORY_H_
