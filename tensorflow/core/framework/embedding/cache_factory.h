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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_FACTORY_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_FACTORY_H_

#include "cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"

namespace tensorflow {
namespace embedding {
class CacheFactory {
 public:
  template<typename K>
  static BatchCache<K>* Create(CacheStrategy cache_strategy, std::string name) {
    switch (cache_strategy) {
      case CacheStrategy::LRU:
        LOG(INFO) << " Use Storage::LRU in multi-tier EmbeddingVariable "
                << name;
        return new LRUCache<K>();
      case CacheStrategy::LFU:
        LOG(INFO) << " Use Storage::LFU in multi-tier EmbeddingVariable "
                << name;
        return new LFUCache<K>();
      default:
        LOG(INFO) << " Invalid Cache strategy, \
                       use LFU in multi-tier EmbeddingVariable "
                << name;
        return new LFUCache<K>();
    }
  }
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_FACTORY_H_
