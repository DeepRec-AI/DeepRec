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

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_THREADPOOL_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_THREADPOOL_H_

#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

namespace embedding {
template<typename K, typename V>
class MultiTierStorage;

class CacheThreadPoolCreator {
 public:
  static thread::ThreadPool* Create() {
    int64 num_threads = 1;
    TF_CHECK_OK(ReadInt64FromEnvVar("TF_MULTI_TIER_EV_CACHE_THREADS", 1,
          &num_threads));
    static thread::ThreadPool cache_thread_pool(Env::Default(),
           ThreadOptions(),
           "MultiTier_Embedding_Cache", num_threads,
           /*low_latency_hint=*/false);
    return &cache_thread_pool;
  }
};

}//embedding
}//tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_THREADPOOL_H_
