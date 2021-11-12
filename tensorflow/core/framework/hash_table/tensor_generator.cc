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

#include "tensorflow/core/framework/hash_table/tensor_generator.h"

namespace tensorflow {

TensorGenerator::TensorGenerator(const Producer& producer)
    : producer_(producer) {
  for (int i = 0; i < kQueueSize; i++) {
    Produce();
  }
}

void TensorGenerator::GetNextTensor(
    const Consumer& done) {
  Produce();
  Deque(done);
}

void TensorGenerator::Produce() {
  Ref();
  spin_rd_lock l(&spin_mu_);
  producer_([this] (Status st, const Tensor& result) {
    Enque(st, result);
    Unref();
  });
}

void TensorGenerator::Enque(Status st, const Tensor& tensor) {
  std::function<void(Status, const Tensor&)> done = nullptr;
  {
    mutex_lock lock(mu_);
    if (waiters_.empty()) {
      results_.emplace(st, tensor);
    } else {
      done = waiters_.front();
      waiters_.pop();
    }
  }
  if (done) {
    done(st, tensor);
  }
}

void TensorGenerator::Deque(
    const std::function<void(Status, const Tensor&)>& done) {
  std::pair<Status, Tensor> result;
  bool run_done;
  {
    mutex_lock lock(mu_);
    if (results_.empty()) {
      waiters_.push(done);
      run_done = false;
    } else {
      result = results_.front();
      results_.pop();
      run_done = true;
    }
  }
  if (run_done) {
    done(result.first, result.second);
  }
}

}  // namespace tensorflow

