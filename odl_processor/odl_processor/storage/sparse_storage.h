/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef ODL_PROCESSOR_STORAGE_SPARSE_STORAGE_H_
#define ODL_PROCESSOR_STORAGE_SPARSE_STORAGE_H_

#include <atomic>
#include <thread>
#include <vector>
#include <mutex>                // std::mutex, std::unique_lock
#include <condition_variable>   // std::condition_variable

#include "concurrentqueue.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "odl_processor/storage/redis_sparse_store.h"

namespace tensorflow {
namespace processor {

const int MANAGER_MAX_THREAD_NUM = 96;
const int MANAGER_MAX_UPDATE_THREAD_NUM = 16;

// TODO: should offered later
struct SparseTask {
  int64_t num;
  void Run() {}
};

class AsyncSparseStorage;
typedef std::function<void(AsyncSparseStorage*, int, bool)> WorkFn;

using sparse_task_queue = moodycamel::ConcurrentQueue<SparseTask*>;

class AsyncSparseStorage {
 public:
  AsyncSparseStorage(
      int serving_thread_num,
      int update_thread_num,
      const std::string& type,
      WorkFn fn = nullptr);
  virtual ~AsyncSparseStorage();

  Status AddTask(SparseTask*);
  Status AddUpdateTask(SparseTask*);
  sparse_task_queue* GetSparseTaskQueue(int);
  sparse_task_queue* GetUpdateSparseTaskQueue(int);
  bool ShouldStop();

 private:
  // stop all threads
  std::atomic<bool> stop_;
  int thread_num_ = 0;
  int update_thread_num_ = 0;

  // std::atomic<uint64_t> active_thread_index_;
  uint64_t active_thread_index_ = 0;
  uint64_t active_update_thread_index_ = 0;

  // threads for serving
  std::vector<std::unique_ptr<std::thread>> threads_;
  // threads for update
  std::vector<std::unique_ptr<std::thread>> update_threads_;

  // queue for serving thread 
  std::vector<sparse_task_queue> task_queues_;
  // queue for update thread
  std::vector<sparse_task_queue> update_task_queues_;

 public:
  // for thread sync and notification
  inline std::mutex* GetMutex(int idx) {
    return mutex_[idx];
  }
  inline std::mutex* GetUpdateMutex(int idx) {
    return update_mutex_[idx];
  }
  inline std::condition_variable* GetCV(int idx) {
    return cv_[idx];
  }
  inline std::condition_variable* GetUpdateCV(int idx) {
    return update_cv_[idx];
  }
  inline std::atomic<bool>* GetReadyFlag(int idx) {
    return &(ready_[idx]);
  }
  inline std::atomic<bool>* GetUpdateReadyFlag(int idx) {
    return &(update_ready_[idx]);
  }
  inline std::atomic<bool>* GetSleepingFlag(int idx) {
    return &(sleeping_[idx]);
  }
  inline std::atomic<bool>* GetUpdateSleepingFlag(int idx) {
    return &(update_sleeping_[idx]);
  }

 private:
  // TODO: refine to std::unique_ptr
  std::vector<std::mutex*> mutex_;
  std::vector<std::mutex*> update_mutex_;
  std::vector<std::condition_variable*> cv_;
  std::vector<std::condition_variable*> update_cv_;

  std::atomic<bool> ready_[MANAGER_MAX_THREAD_NUM];
  std::atomic<bool> update_ready_[MANAGER_MAX_UPDATE_THREAD_NUM];
  std::atomic<bool> sleeping_[MANAGER_MAX_THREAD_NUM];
  std::atomic<bool> update_sleeping_[MANAGER_MAX_UPDATE_THREAD_NUM];

  std::vector<AbstractModelStore*> store_; // one connection per store
  std::vector<AbstractModelStore*> update_store_;
};

class SparseStorage {
 public:
  explicit SparseStorage(
      int serving_thread_num,
      int update_thread_num,
      const std::string& type);
  virtual ~SparseStorage();

  Status GetValues(uint64_t feature2id,
                   const char* const keys,
                   char* const values,
                   size_t bytes_per_key,
                   size_t bytes_per_values,
                   size_t N,
                   const char* default_value,
                   BatchGetCallback cb);
  Status SetValues(uint64_t feature2id,
                   const char* const keys,
                   const char* const values,
                   size_t bytes_per_key,
                   size_t bytes_per_values,
                   size_t N,
                   BatchSetCallback cb);
  Status Reset();

 private:
  int thread_num_ = 0;
  int update_thread_num_ = 0;
  std::atomic<uint64_t> active_thread_index_;
  std::atomic<uint64_t> active_update_thread_index_;
  std::mutex mutex_[MANAGER_MAX_THREAD_NUM];
  std::mutex update_mutex_[MANAGER_MAX_UPDATE_THREAD_NUM];
  std::vector<AbstractModelStore*> store_; // one connection per store
  std::vector<AbstractModelStore*> update_store_;
};

} // processor
} // tensorflow

#endif // ODL_PROCESSOR_STORAGE_SPARSE_STORAGE_MANAGER_H_
