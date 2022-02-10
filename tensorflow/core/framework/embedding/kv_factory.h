/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_KV_FACTORY_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_KV_FACTORY_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/embedding/dense_hash_map.h"
#include "tensorflow/core/framework/embedding/lockless_hash_map.h"
#include "tensorflow/core/framework/embedding/leveldb_kv.h"

namespace tensorflow {

template <class K, class V>
class KVInterface;

template <class K, class V>
class KVFactory {
 public:
  ~KVFactory() {}
  static KVInterface<K, V>* CreateKV(const std::string& kv_type,
                                     int partition_num,
                                     std::string path) {
    if ("dense_hash_map" == kv_type) {
      VLOG(2) << "Use dense_hash_map as EV data struct";
      return new DenseHashMap<K, V>();
    } else if ("lockless_hash_map" == kv_type) {
      VLOG(2) << "Use lockless_hash_map as EV data struct";
      return new LocklessHashMap<K, V>();
    } else if ("leveldb_kv" == kv_type) {
      VLOG(2) << "Use leveldb_kv as EV data struct";
      return new LevelDBKV<K, V>(path);
    }
     else {
      VLOG(2) << "Not match any hashtable_type, use default 'lockless_hash_map'";
      return new LocklessHashMap<K, V>();
    }
  }
 private:
  KVFactory() {}
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_KV_FACTORY_H_
