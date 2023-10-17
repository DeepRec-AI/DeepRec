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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_FACTORY_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_FACTORY_H_

#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/dram_leveldb_storage.h"
#include "tensorflow/core/framework/embedding/dram_pmem_storage.h"
#include "tensorflow/core/framework/embedding/dram_ssd_storage.h"
#include "tensorflow/core/framework/embedding/hbm_dram_storage.h"
#include "tensorflow/core/framework/embedding/hbm_dram_ssd_storage.h"
#include "tensorflow/core/framework/embedding/multi_tier_storage.h"
#include "tensorflow/core/framework/embedding/single_tier_storage.h"
#include "tensorflow/core/framework/embedding/storage_config.h"
#include "tensorflow/core/framework/embedding/storage.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace embedding {
class StorageFactory {
 public:
  template<typename K, typename V>
  static Storage<K, V>* Create(const StorageConfig& sc,
      Allocator* gpu_allocator, FeatureDescriptor<V>* feat_desc,
      const string& name) {
    switch (sc.type) {
      case StorageType::DRAM:
        return new DramStorage<K, V>(sc, feat_desc);
      case StorageType::PMEM_MEMKIND:
        feat_desc->SetAllocator(pmem_allocator());
        return new PmemMemkindStorage<K, V>(sc, feat_desc);
      case StorageType::PMEM_LIBPMEM:
        feat_desc->SetAllocator(
            experimental_pmem_allocator(sc.path, sc.size[0]));
        return new PmemLibpmemStorage<K, V>(sc, feat_desc);
      case StorageType::DRAM_PMEM:
        return new DramPmemStorage<K, V>(sc,
            feat_desc, name);
      case StorageType::LEVELDB:
      case StorageType::DRAM_LEVELDB:
        return new DramLevelDBStore<K, V>(sc, feat_desc, name);
      case StorageType::SSDHASH:
      case StorageType::DRAM_SSDHASH:
        return new DramSsdHashStorage<K, V>(sc, feat_desc, name);
      case StorageType::HBM:
#if GOOGLE_CUDA
        return new HbmStorage<K, V>(sc, gpu_allocator, feat_desc);
#endif  // GOOGLE_CUDA
      case StorageType::HBM_DRAM:
#if GOOGLE_CUDA
        return new HbmDramStorage<K, V>(sc, gpu_allocator, feat_desc, name);
#endif  // GOOGLE_CUDA
      case StorageType::HBM_DRAM_SSDHASH:
#if GOOGLE_CUDA
        return new HbmDramSsdStorage<K, V>(sc, gpu_allocator, feat_desc, name);
#endif  // GOOGLE_CUDA
      default:
        return new DramStorage<K, V>(sc, feat_desc);
    }
  }
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_FACTORY_H_
