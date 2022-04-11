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

#include "serving/processor/storage/feature_store_mgr.h"
#include "serving/processor/serving/model_config.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace processor {

namespace {
void ThreadRun(AsyncFeatureStoreMgr* mgr, int idx,
               bool is_update_thread) {
  std::mutex* mu = nullptr;
  std::condition_variable* cv = nullptr;
  sparse_task_queue* queue = nullptr;

  if (is_update_thread) {
    mu = mgr->GetUpdateMutex(idx);
    cv = mgr->GetUpdateCV(idx);
    queue = mgr->GetUpdateSparseTaskQueue(idx);
  } else {
    mu = mgr->GetMutex(idx);
    cv = mgr->GetCV(idx);
    queue = mgr->GetSparseTaskQueue(idx);
  }

  const int try_count = 64;
  int curr_try_count = 0;
  SparseTask* task = nullptr;
  bool succeeded = false;

  while ((succeeded = queue->try_dequeue(task)) ||
         !mgr->ShouldStop()) {
    if (!succeeded) {
      ++curr_try_count;
      if (curr_try_count <= try_count) {
        continue;
      }
      curr_try_count = 0;

      if (is_update_thread) {
        // Ready going to sleep
        *(mgr->GetUpdateSleepingFlag(idx)) = true;
        *(mgr->GetUpdateReadyFlag(idx)) = false;
      } else {
        // Ready going to sleep
        *(mgr->GetSleepingFlag(idx)) = true;
        *(mgr->GetReadyFlag(idx)) = false;
      }

      {
        // try to wait signal when have no elements in the queue
        std::unique_lock<std::mutex> lock(*mu);
        cv->wait(lock, [is_update_thread, mgr, idx] {
          return (is_update_thread ?
                      *(mgr->GetUpdateReadyFlag(idx)) :
                      *(mgr->GetReadyFlag(idx))) ||
                 mgr->ShouldStop();
        });
        lock.unlock();
      }

      if (is_update_thread) {
        *(mgr->GetUpdateSleepingFlag(idx)) = false;
      } else {
        *(mgr->GetSleepingFlag(idx)) = false;
      }

      continue;
    }

    // try again, maybe have some tasks left
    //if (!task) continue; // if (mgr->ShouldStop()) break;

    curr_try_count = 0;
    // run the task
    task->Run();

    task = nullptr;
  }
}

Status GetIPAndPortFromUrl(const std::string& url,
                           std::string* ip,
                           int32_t* port) {
  auto offset = url.find(":");
  if (offset == std::string::npos) {
    return tensorflow::errors::Internal(
        "Can't parse ip and port from url: ", url);
  }

  *ip = url.substr(0, offset);
  *port = atoi(url.substr(offset+1, url.length()).c_str());

  return Status::OK();
}

FeatureStore* CreateFeatureStore(ModelConfig* config) {
  if (config->feature_store_type == "redis") {
    LocalRedis::Config redis_config;
    Status s = GetIPAndPortFromUrl(config->redis_url,
                                   &redis_config.ip,
                                   &redis_config.port);
    if (!s.ok()) {
      LOG(ERROR) << "Can't parse ip and port from url: "
                 << config->redis_url;
      return nullptr;
    }
    redis_config.passwd = config->redis_password;
    redis_config.db_idx = config->redis_db_idx;

    return new LocalRedis(redis_config);
  } else {
    LOG(ERROR) << "Only LocalRedis backend now. type = "
               << config->feature_store_type;
  }

  return nullptr;
}

#define CALL_BY_UPDATE_THREAD(fn, ...)                        \
  do {                                                        \
    uint64_t index = active_update_thread_index_++;           \
    index %= update_thread_num_;                              \
    {                                                         \
      std::lock_guard<std::mutex> lock(update_mutex_[index]); \
      return update_store_[index]->fn(__VA_ARGS__);           \
    }                                                         \
  } while(0)

} // namespace

AsyncFeatureStoreMgr::AsyncFeatureStoreMgr(ModelConfig* config, WorkFn fn) :
    stop_(false),
    thread_num_(config->read_thread_num),
    update_thread_num_(config->update_thread_num),
    active_thread_index_(0),
    active_update_thread_index_(0) {
  if (thread_num_ < 1 || thread_num_ > MANAGER_MAX_THREAD_NUM) {
    LOG(FATAL) << "Invalid IO thread num, required [1, 96], get "
               << thread_num_;
  }

  if (update_thread_num_ < 1 ||
      update_thread_num_ > MANAGER_MAX_UPDATE_THREAD_NUM) {
    LOG(FATAL) << "Invalid IO thread num, required [1, 16], get "
               << update_thread_num_;
  }

  task_queues_.resize(thread_num_);
  threads_.resize(thread_num_);
  mutex_.resize(thread_num_);
  cv_.resize(thread_num_);
  store_.resize(thread_num_);
  for (int i = 0; i < thread_num_; ++i) {
    mutex_[i] = new std::mutex();
    cv_[i] = new std::condition_variable();
    ready_[i] = false;
    sleeping_[i] = false;
    store_[i] = CreateFeatureStore(config);
    threads_[i].reset(new std::thread(!fn? &ThreadRun : fn, this, i, false));
  }

  update_task_queues_.resize(update_thread_num_);
  update_threads_.resize(update_thread_num_);
  update_mutex_.resize(update_thread_num_);
  update_cv_.resize(update_thread_num_);
  update_store_.resize(update_thread_num_);
  for (int i = 0; i < update_thread_num_; ++i) {
    update_mutex_[i] = new std::mutex();
    update_cv_[i] = new std::condition_variable();
    update_ready_[i] = false;
    update_sleeping_[i] = false;
    update_store_[i] = CreateFeatureStore(config);
    update_threads_[i].reset(new std::thread(!fn ? &ThreadRun : fn, this, i, true));
  }
}

AsyncFeatureStoreMgr::~AsyncFeatureStoreMgr() {
  // stop all IO threads
  stop_ = true;

  for (int i = 0; i < thread_num_; ++i) {
    cv_[i]->notify_all();
  }

  for (int i = 0; i < update_thread_num_; ++i) {
    update_cv_[i]->notify_all();
  }

  for (int i = 0; i < thread_num_; ++i) {
    threads_[i]->join();
  }

  for (int i = 0; i < update_thread_num_; ++i) {
    update_threads_[i]->join();
  }

  for (auto store : store_) {
    delete store;
  }

  for (auto store : update_store_) {
    delete store;
  }

  for (auto mu : mutex_) {
    delete mu;
  }

  for (auto mu : update_mutex_) {
    delete mu;
  }

  for (auto cv : cv_) {
    delete cv;
  }

  for (auto cv : update_cv_) {
    delete cv;
  }
}

Status AsyncFeatureStoreMgr::AddTask(SparseTask* t) {
  // NOTE(jiankebg.pt): No need atomic add here, maybe 
  // more then one Op will call the same thread(queue).
  // TODO: Need excetly balance here ?
  //
  // uint64_t index = active_thread_index_.fetch_add(1);
  uint64_t index = active_thread_index_++;
  index %= thread_num_;
  bool ret = false;
{
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  ret = task_queues_[index].enqueue(t);
}
  // TODO: should retry ?
  if (!ret) {
    return tensorflow::errors::Internal(
        "can not enqueue task into the task_queues, index is ",
        std::to_string(index));
  }

  if (sleeping_[index]) {
    { // TODO: Need lock to promise the cv->wait(...)
      // behavior in ThreadRun function.
      std::lock_guard<std::mutex> lock(*mutex_[index]);
      ready_[index] = true;
    }

    cv_[index]->notify_all();
  }

  return Status::OK();
}

Status AsyncFeatureStoreMgr::AddUpdateTask(SparseTask* t) {
  // TODO: Need excetly balance here ?
  uint64_t index = active_update_thread_index_++;
  index %= update_thread_num_;
  bool ret = update_task_queues_[index].enqueue(t);
  if (!ret) {
    return tensorflow::errors::Internal(
        "can not enqueue task into the update task_queues, index is ",
        std::to_string(index));
  }

  if (update_sleeping_[index]) {
    {
      // TODO: Need lock to promise the cv->wait(...)
      // behavior in ThreadRun function.
      std::lock_guard<std::mutex> lock(*update_mutex_[index]);
      update_ready_[index] = true;
    }

    update_cv_[index]->notify_all();
  }

  return Status::OK();
}

sparse_task_queue* AsyncFeatureStoreMgr::GetSparseTaskQueue(int idx) {
  if (idx < 0 || idx >= thread_num_) {
    LOG(FATAL) << "Error index num: " << idx
               << ", thread_num is " << thread_num_;
  }

  return &(task_queues_[idx]);
}

sparse_task_queue* AsyncFeatureStoreMgr::GetUpdateSparseTaskQueue(int idx) {
  if (idx < 0 || idx >= update_thread_num_) {
    LOG(FATAL) << "Error index num: " << idx
               << ", update_thread_num is " << update_thread_num_;
  }

  return &(update_task_queues_[idx]);
}

bool AsyncFeatureStoreMgr::ShouldStop() {
  return stop_;
}

FeatureStoreMgr::FeatureStoreMgr(ModelConfig* config) 
  : thread_num_(config->read_thread_num),
    update_thread_num_(config->update_thread_num),
    active_thread_index_(0),
    active_update_thread_index_(0),
    storage_type_(config->feature_store_type) {
  if (thread_num_ < 1 || thread_num_ > MANAGER_MAX_THREAD_NUM) {
    LOG(FATAL) << "Invalid IO thread num, required [1, 96], get "
               << thread_num_;
  }

  if (update_thread_num_ < 1 ||
      update_thread_num_ > MANAGER_MAX_UPDATE_THREAD_NUM) {
    LOG(FATAL) << "Invalid IO thread num, required [1, 16], get "
               << update_thread_num_;
  }

  store_.resize(thread_num_);
  for (int i = 0; i < thread_num_; ++i) {
    store_[i] = CreateFeatureStore(config);
  }

  update_store_.resize(update_thread_num_);
  for (int i = 0; i < update_thread_num_; ++i) {
    update_store_[i] = CreateFeatureStore(config);
  }
}

FeatureStoreMgr::~FeatureStoreMgr() {
  for (auto store : store_) {
    delete store;
  }

  for (auto store : update_store_) {
    delete store;
  }
}

Status FeatureStoreMgr::GetValues(
    uint64_t model_version,
    uint64_t feature2id,
    const char* const keys,
    char* const values,
    size_t bytes_per_key,
    size_t bytes_per_values,
    size_t N,
    const char* default_value,
    BatchGetCallback cb) {
  uint64_t index = active_thread_index_++;
  index %= thread_num_;
  {
    std::lock_guard<std::mutex> lock(mutex_[index]);
    Status s = store_[index]->BatchGet(
        model_version, feature2id, keys, values, 
        bytes_per_key, bytes_per_values, N,
        default_value);
    if (s.ok()) {
      cb(s);
    }
    return s;
  }
}

Status FeatureStoreMgr::SetValues(
    uint64_t model_version,
    uint64_t feature2id,
    const char* const keys,
    const char* const values,
    size_t bytes_per_key,
    size_t bytes_per_values,
    size_t N,
    BatchSetCallback cb) {
  uint64_t index = active_update_thread_index_++;
  index %= update_thread_num_;
  {
    std::lock_guard<std::mutex> lock(update_mutex_[index]);
    Status s = update_store_[index]->BatchSet(
        model_version, feature2id, keys, values,
        bytes_per_key, bytes_per_values, N);
    if (s.ok()) {
      cb(s);
    }
    return s;
  }
}

Status FeatureStoreMgr::Reset() {
  uint64_t index = active_update_thread_index_++;
  index %= update_thread_num_;
  {
    std::lock_guard<std::mutex> lock(update_mutex_[index]);
    return update_store_[index]->Cleanup();
  }
}

Status FeatureStoreMgr::GetStorageMeta(StorageMeta* meta) {
  uint64_t index = active_update_thread_index_++;
  index %= update_thread_num_;
  {
    std::lock_guard<std::mutex> lock(update_mutex_[index]);
    return update_store_[index]->GetStorageMeta(meta);
  }
}

void FeatureStoreMgr::GetStorageOptions(
    StorageMeta& meta,
    StorageOptions** cur_opt,
    StorageOptions** bak_opt) {
  // Redis
  if (storage_type_.find("redis") != std::string::npos) {
    // NOTE:(jiankeng.pt) now only consider db-0 and db-1
    int cur_db, bak_db;
    bool is_init_storage = false;
    if (!meta.active[0] && !meta.active[1]) {
      cur_db = 0;
      bak_db = 1;
      is_init_storage = true;
    } else if (meta.active[0]) {
      cur_db = 0;
      bak_db = 1;
    } else {
      cur_db = 1;
      bak_db = 0;
    }
    *cur_opt = new StorageOptions(cur_db, cur_db,
                                  is_init_storage);
    *bak_opt = new StorageOptions(bak_db, bak_db,
                                  is_init_storage);
  }
}

Status FeatureStoreMgr::SetStorageActiveStatus(
    bool active) {
  CALL_BY_UPDATE_THREAD(SetActiveStatus, active);
}

Status FeatureStoreMgr::GetModelVersion(int64_t* full_version,
                                        int64_t* latest_version) {
  CALL_BY_UPDATE_THREAD(GetModelVersion, full_version,
                        latest_version);
}

Status FeatureStoreMgr::SetModelVersion(
    int64_t full_version, int64_t latest_version) {
  CALL_BY_UPDATE_THREAD(SetModelVersion, full_version,
                        latest_version);
}

Status FeatureStoreMgr::GetStorageLock(
    int value, int timeout, bool* success) {
  CALL_BY_UPDATE_THREAD(GetStorageLock, value,
                        timeout, success);
}

Status FeatureStoreMgr::ReleaseStorageLock(int value) {
  CALL_BY_UPDATE_THREAD(ReleaseStorageLock, value);
}

} // processor
} // tensorflow
