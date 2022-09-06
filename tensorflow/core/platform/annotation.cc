/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/annotation.h"

namespace tensorflow {
/*static*/ std::string* Annotation::ThreadAnnotation() {
  static thread_local std::string annotation;
  return &annotation;
}

namespace tracing {

thread_local std::vector<uint64_t> CallingContext::Context_;

// Moved from TraceMe
// Activity IDs: To avoid contention over a counter, the top 32 bits identify
// the originating thread, the bottom 32 bits name the event within a thread.
// IDs may be reused after 4 billion events on one thread, or 4 billion threads.
static std::atomic<uint32_t> thread_counter(1);  // avoid kUntracedActivity
uint64_t NewActivityId() {
  const thread_local static uint32_t thread_id = thread_counter.fetch_add(1);
  thread_local static uint32_t per_thread_activity_id = 0;
  return static_cast<uint64_t>(thread_id) << 32 | per_thread_activity_id++;
}

uint64_t CallingContext::GetAndPush() {
  uint64_t id = NewActivityId();
  Context_.push_back(id);
  return id;
}

}  // namespace tracing
}  // namespace tensorflow
