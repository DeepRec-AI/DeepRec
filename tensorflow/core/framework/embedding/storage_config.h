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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_CONFIG_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_CONFIG_H_

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/embedding_config.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"
namespace tensorflow {
namespace embedding {
struct StorageConfig {
  StorageConfig() : type(StorageType::DEFAULT),
                    path(""),
                    layout_type(LayoutType::NORMAL),
                    cache_strategy(CacheStrategy::LFU) {
    size = {1<<30,1<<30,1<<30,1<<30};
  }

  StorageConfig(StorageType t,
                const std::string& p,
                const std::vector<int64>& s,
                const std::string& layout,
                const EmbeddingConfig& ec,
                const CacheStrategy cache_strategy_ = CacheStrategy::LFU)
                                      : type(t),
                                        path(p),
                                        embedding_config(ec),
                                        cache_strategy(cache_strategy_) {
    if ("normal" == layout) {
      layout_type = LayoutType::NORMAL;
    } else if ("light" == layout) {
      layout_type = LayoutType::LIGHT;
    } else if ("normal_contiguous" == layout){
      layout_type = LayoutType::NORMAL_CONTIGUOUS;
    } else if ("normal_contiguous_gpu" == layout){
      layout_type = LayoutType::NORMAL_CONTIGUOUS_GPU;
    } else if ("compact" == layout){
      layout_type = LayoutType::COMPACT;
    } else {
      LOG(WARNING) << "Unknown layout: "
        << layout << ", use LayoutType::NORMAL by default.";
      layout_type = LayoutType::NORMAL;
    }
    size = s;
  }
  StorageType type;
  LayoutType layout_type;
  std::string path;
  std::vector<int64> size;
  CacheStrategy cache_strategy;
  EmbeddingConfig embedding_config;
};
} // namespace embedding
} // namespace tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_CONFIG_H_
