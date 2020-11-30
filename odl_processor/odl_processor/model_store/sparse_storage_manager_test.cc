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

#include <pthread.h>
#include "gtest/gtest.h"
#include "sparse_storage_manager.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace processor {

namespace {

static const int64_t NUM1 = 100001;
static const int64_t NUM2 = 10000;
static const int64_t NUM3 = 9999;

// for check
static bool global_queue_result[NUM1];

void TestThreadRun(SparseStorageManager* mgr, int idx,
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
        *(mgr->GetUpdateSleepingFlag(idx)) = true;
        *(mgr->GetUpdateReadyFlag(idx)) = false;
      } else {
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

    global_queue_result[task->num] = true;

    task = nullptr;
  }
}

void WorkThreadRun(SparseStorageManager* mgr,
                   int64_t start, int64_t end) {
  for (int64_t num = start; num <= end; ++num) {
    SparseTask* task = new SparseTask();
    task->num = num;
    mgr->AddTask(task);
  }
}

}

TEST(SparseStorageTest, SparseStorageManagerTest) {
  for (int64_t i = 0; i < NUM1; ++i) {
    global_queue_result[i] = false;
  }

  SparseStorageManager* mgr =
      new SparseStorageManager(5, 2, "Test", &TestThreadRun);

  // worker threads
  std::vector<std::unique_ptr<std::thread>> threads;
  threads.resize(10);
  for (int i = 0; i < 10; ++i) {
    threads[i].reset(new std::thread(&WorkThreadRun, mgr, i * NUM2, i * NUM2 + NUM3));
  }

  for (int i = 0; i < 10; ++i) {
    threads[i]->join();
  }

  delete mgr;

  for (int64_t i = 0; i < NUM1-1; ++i) {
    EXPECT_TRUE(global_queue_result[i]);
  }
  EXPECT_FALSE(global_queue_result[NUM1-1]);

  EXPECT_TRUE(1);
}


} // namespace processor
} // namespace tensorflow
