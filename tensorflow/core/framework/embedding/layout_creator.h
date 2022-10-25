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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_LAYOUT_CREATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_LAYOUT_CREATOR_H_

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/storage_config.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
template <class V>
class ValuePtr;

namespace embedding {
template<typename V>
class LayoutCreator {
 public:
  virtual ValuePtr<V>* Create(Allocator* alloc, size_t size) = 0;
};

template<typename V>
class NormalLayoutCreator : public LayoutCreator<V> {
 public:
  ValuePtr<V>* Create(Allocator* alloc, size_t size) override {
    return new NormalValuePtr<V>(alloc, size);
  }
};

template<typename V>
class LightLayoutCreator : public LayoutCreator<V> {
 public:
  ValuePtr<V>* Create(Allocator* alloc, size_t size) override {
    return new LightValuePtr<V>(alloc, size);
  }
};

template<typename V>
class NormalContiguousLayoutCreator : public LayoutCreator<V> {
 public:
  ValuePtr<V>* Create(Allocator* alloc, size_t size) override {
    return new NormalContiguousValuePtr<V>(alloc, size);
  }
};

template<typename V>
class NormalContiguousGPULayoutCreator : public LayoutCreator<V> {
 public:
  ValuePtr<V>* Create(Allocator* alloc, size_t size) override {
    return new NormalGPUValuePtr<V>(alloc, size);
  }
};

class LayoutCreatorFactory {
 public:
  template<typename V>
  static LayoutCreator<V>* Create(const StorageConfig& sc) {
    switch (sc.layout_type) {
      case LayoutType::NORMAL:
        static NormalLayoutCreator<V> normal_creator;
        return &normal_creator;
      case LayoutType::LIGHT:
        static LightLayoutCreator<V> light_creator;
        return &light_creator;
      case LayoutType::NORMAL_CONTIGUOUS:
        static NormalContiguousLayoutCreator<V> normal_contiguous_creator;
        return &normal_contiguous_creator;
      case LayoutType::NORMAL_CONTIGUOUS_GPU:
        static NormalContiguousGPULayoutCreator<V>
                   normal_contiguous_gpu_creator;
        return &normal_contiguous_gpu_creator;
      default:
        static NormalLayoutCreator<V> default_creator;
        return &default_creator;
    }
  }
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_LAYOUT_CREATOR_H_
