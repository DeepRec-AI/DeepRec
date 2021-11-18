/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "resources/cpu_resource.h"

namespace SparseOperationKit {

CpuResource::Barrier::Barrier(const size_t thread_count)
: mu_(), cond_(), thread_count_(thread_count), 
count_(thread_count), generation_(0) 
{}

void CpuResource::Barrier::wait() {
    std::unique_lock<std::mutex> lock(mu_);
    auto local_gen = generation_;
    if (!--count_) {
        generation_++;
        count_ = thread_count_;
        cond_.notify_all();
    } else {
        cond_.wait_for(lock, time_threshold_, [this, local_gen](){return local_gen != generation_; });
        if (local_gen == generation_) throw std::runtime_error("Blocking threads time out.");
    }
}

CpuResource::BlockingCallOnce::BlockingCallOnce(const size_t thread_count)
: mu_(), cond_(), thread_count_(thread_count), count_(thread_count),
generation_(0) 
{}

CpuResource::CpuResource(const size_t thread_count) 
: barrier_(std::make_shared<Barrier>(thread_count)),
blocking_call_oncer_(std::make_shared<BlockingCallOnce>(thread_count)),
mu_(), thread_pool_(new Eigen::SimpleThreadPool(thread_count))
{}

std::shared_ptr<CpuResource> CpuResource::Create(const size_t thread_count) {
    return std::shared_ptr<CpuResource>(new CpuResource(thread_count));
}

void CpuResource::sync_cpu_threads() const {
    barrier_->wait();
}

void CpuResource::sync_threadpool() const {
    while (!thread_pool_->Done()) std::this_thread::yield();
}

} // namespace SparseOperationKit