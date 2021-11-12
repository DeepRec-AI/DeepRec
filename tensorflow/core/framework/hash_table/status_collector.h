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

#ifndef TENSORFLOW_CORE_FRAMEWORK_HASH_TABLE_STATUS_COLLECTOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_HASH_TABLE_STATUS_COLLECTOR_H_

#include <functional>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

class StatusCollector {
 public:
  StatusCollector(int size, std::function<void(Status)> done)
    : count_down_(size + 1), done_(done) {}
  void AddStatus(Status st) {
    bool run;
    {
      mutex_lock lock(mu_);
      st_.Update(st);
      run = --count_down_ == 0;
    }
    if (run) {
      done_(st_);
      delete this;
    }
  }
  void Start() {
    AddStatus(Status::OK());
  }
  std::function<void(Status)> AddStatusFunc() {
    return [this](Status st){AddStatus(st);};
  }

 private:
  mutex mu_;
  int count_down_;
  std::function<void(Status)> done_;
  Status st_;
};

template<typename Runner, typename Functor, typename Done>
void ParrellRun(
    int64 size, int64 block, const Runner& runner,
    const Functor& functor, const Done& done) {
  if (size < block * 2) {
    done(functor(0, size));
  } else {
    StatusCollector* stc = new StatusCollector(size / block, done);
    for (int64_t i = 0; i < size / block - 1; i++) {
      runner([stc, functor, block, i] {
        stc->AddStatus(functor(i * block, block));
      });
    }
    int64_t ptr = (size / block - 1) * block;
    runner([stc, functor, size, ptr] {
      stc->AddStatus(functor(ptr, size - ptr));
    });
    stc->Start();
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_HASH_TABLE_STATUS_COLLECTOR_H_
