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

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EVICTION_MANAGER_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EVICTION_MANAGER_H_

#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

namespace embedding {
template<typename K, typename V>
class MultiTierStorage;

template<typename K, typename V>
struct StorageItem {
  volatile bool is_occupied;
  volatile bool is_deleted;

  StorageItem(bool is_occupied,
              volatile bool is_deleted) : is_occupied(is_occupied),
                                          is_deleted(is_deleted) {}
};

template<typename K, typename V>
class EvictionManager {
 public:
  EvictionManager() {
    num_of_threads_ = 1;
    TF_CHECK_OK(ReadInt64FromEnvVar("TF_MULTI_TIER_EV_EVICTION_THREADS", 1,
          &num_of_threads_));
    thread_pool_.reset(
        new thread::ThreadPool(Env::Default(), ThreadOptions(),
          "EVICTION_MANAGER", 3, /*low_latency_hint=*/false));
  }
  
  ~EvictionManager() {
  }

  TF_DISALLOW_COPY_AND_ASSIGN(EvictionManager);

  void Schedule(std::function<void()> fn) {
    thread_pool_->Schedule(std::move(fn)); 
  }

  void AddStorage(MultiTierStorage<K,V>* storage) {
    mutex_lock l(mu_);
    auto ret = storage_table_.emplace(std::make_pair(storage,
                           new StorageItem<K, V>(false, false)));
    if (ret.second && num_of_active_threads_ < num_of_threads_)
      StartThread();
  }

  void DeleteStorage(MultiTierStorage<K,V>* storage) {
    auto storage_item = storage_table_[storage];
    bool delete_flag = false;
    while (!delete_flag) {
      volatile bool* occupy_flag = &storage_item->is_occupied;
      delete_flag = __sync_bool_compare_and_swap(occupy_flag, false, true);
      if (delete_flag) {
        storage_item->is_deleted = true;
      }
      *occupy_flag = false;
    }
  }

 private:
  void StartThread() {
    while(this->flag_.test_and_set(std::memory_order_acquire));
    if (num_of_active_threads_ < num_of_threads_) {
      __sync_fetch_and_add(&num_of_active_threads_, 1);
      thread_pool_->Schedule([this]() {
        EvictionLoop();
      });
    }
    this->flag_.clear(std::memory_order_release);
  }

  bool CheckStorages() {
    mutex_lock l(mu_);
    for (auto it = storage_table_.begin(); it != storage_table_.end();) {
      if (!(it->second)->is_deleted) 
        return true;
      else 
        it = storage_table_.erase(it);
    }
    return false;
  }

  void EvictionLoop() {
    while (CheckStorages()) {
      mutex_lock l(mu_);
      int index = 0;
      for (auto it : storage_table_) {
        auto storage = it.first;
        auto storage_item = it.second;
        volatile bool* occupy_flag = &storage_item->is_occupied;
        if (__sync_bool_compare_and_swap(occupy_flag, false, true)) {
          if (storage_item->is_deleted) {
            *occupy_flag = false;
            continue; 
          }
          storage->BatchEviction();
          *occupy_flag = false;
        }
        Env::Default()->SleepForMicroseconds(1);
      }
    }
    __sync_fetch_and_sub(&num_of_active_threads_, 1);
  }

  int64 num_of_threads_;
  int64 num_of_active_threads_;
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  std::map<MultiTierStorage<K,V>*, StorageItem<K, V>*> storage_table_;
  mutex mu_;
};

class EvictionManagerCreator {
 public:
  template<typename K, typename V>
  static EvictionManager<K, V>* Create() {
    static EvictionManager<K, V> eviction_manager;
    return &eviction_manager;
  }
};

}//embedding
}//tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EVICTION_MANAGER_H_
