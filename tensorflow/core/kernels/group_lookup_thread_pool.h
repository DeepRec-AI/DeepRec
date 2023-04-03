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
=======================================================================*/

#define EIGEN_USE_THREADS

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace group_embedding {

int32_t GetWorkerThreadsCount() {
  auto atoi = [](const char* str, int32_t* num) -> bool {
    *num = std::atoi(str);
    if (*num < 1) return false;
    return true;
  };
  const auto worker_threads_cnt = std::getenv("GROUPEMBEDDING_WORKER_THREADS_CNT");
  int32_t num = 1;
  return (sok_worker_threads_cnt && atoi(sok_worker_threads_cnt, &num)) ? num : 4;
}

class GroupEmbeddingThreadPool {
 public:
  thread::ThreadPool* GetThreadPool() {
    return thread_pool_;
  }
  void Initialize() {
      auto num_threads = GetWorkerThreadsCount();
      thread_pool_ = thread::ThreadPool(Env::Default(), "GroupEmbedding", num_threads)
  }

  void Schedule(std::function<int64, int64> fn);
  
 private:
  static thread::ThreadPool* thread_pool_ {nullptr};
};

} // namespace group_embedding
} // namespace tensorflow