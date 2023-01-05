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
#include "tensorflow/core/framework/embedding/layout_creator.h"
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
      Allocator* gpu_allocator, const string& name) {
    auto layout_creator = LayoutCreatorFactory::Create<V>(sc);

    switch (sc.type) {
      case StorageType::DRAM:
        return new DramStorage<K, V>(sc, ev_allocator(),
            layout_creator);
      case StorageType::PMEM_MEMKIND:
        return new PmemMemkindStorage<K, V>(sc, pmem_allocator(),
            layout_creator);
      case StorageType::PMEM_LIBPMEM:
        return new PmemLibpmemStorage<K, V>(sc,
            experimental_pmem_allocator(sc.path, sc.size[0]),
            layout_creator);
      case StorageType::DRAM_PMEM:
        return new DramPmemStorage<K, V>(sc, ev_allocator(),
            experimental_pmem_allocator(sc.path, sc.size[0]),
            layout_creator, name);
      case StorageType::LEVELDB:
      case StorageType::DRAM_LEVELDB:
        return new DramLevelDBStore<K, V>(sc, ev_allocator(),
            layout_creator, name);
      case StorageType::SSDHASH:
      case StorageType::DRAM_SSDHASH:
        return new DramSsdHashStorage<K, V>(sc, ev_allocator(),
            layout_creator, name);
      case StorageType::HBM:
#if GOOGLE_CUDA
        return new HbmStorage<K, V>(sc, gpu_allocator,
            layout_creator);
#endif  // GOOGLE_CUDA
      case StorageType::HBM_DRAM:
#if GOOGLE_CUDA
#if !TENSORFLOW_USE_GPU_EV
        return new HbmDramStorage<K, V>(sc, gpu_allocator,
        ev_allocator(), layout_creator, name);
#endif  // TENSORFLOW_USE_GPU_EV
#endif  // GOOGLE_CUDA
        LOG(WARNING) << "Unsupport HBM_DRAM, fallback to DRAM.";
      case StorageType::HBM_DRAM_SSDHASH:
#if GOOGLE_CUDA
#if !TENSORFLOW_USE_GPU_EV
        return new HbmDramSsdStorage<K, V>(sc, gpu_allocator,
            ev_allocator(), layout_creator, name);
#endif  // TENSORFLOW_USE_GPU_EV
#endif  // GOOGLE_CUDA
        LOG(WARNING) << "Unsupport HBM_DRAM_SSDHASH, fallback to DRAM.";
      default:
        return new DramStorage<K, V>(sc, ev_allocator(),
            layout_creator);
    }
  }
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_STORAGE_FACTORY_H_
