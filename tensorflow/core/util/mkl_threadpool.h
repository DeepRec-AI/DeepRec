
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

#ifndef TENSORFLOW_CORE_UTIL_MKL_THREADPOOL_H_
#define TENSORFLOW_CORE_UTIL_MKL_THREADPOOL_H_
#ifdef INTEL_MKL

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dnnl_threadpool.hpp"
#include "dnnl.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"
#define EIGEN_USE_THREADS

namespace tensorflow {

#ifdef ENABLE_DNNL_THREADPOOL
using dnnl::threadpool_interop::threadpool_iface;

// Divide 'n' units of work equally among 'teams' threads. If 'n' is not
// divisible by 'teams' and has a remainder 'r', the first 'r' teams have one
// unit of work more than the rest. Returns the range of work that belongs to
// the team 'tid'.
// Parameters
//   n        Total number of jobs.
//   team     Number of workers.
//   tid      Current thread_id.
//   n_start  start of range operated by the thread.
//   n_end    end of the range operated by the thread.

template <typename T, typename U>
inline void balance211(T n, U team, U tid, T* n_start, T* n_end) {
  if (team <= 1 || n == 0) {
    *n_start = 0;
    *n_end = n;
    return;
  }
  T min_per_team = n / team;
  T remainder = n - min_per_team * team;  // i.e., n % teams.
  *n_start = tid * min_per_team + std::min(tid, remainder);
  *n_end = *n_start + min_per_team + (tid < remainder);
}

struct MklDnnThreadPool : public threadpool_iface {
  MklDnnThreadPool() = default;

  MklDnnThreadPool(OpKernelContext* ctx)
      : eigen_interface_(ctx->device()
                             ->tensorflow_cpu_worker_threads()
                             ->workers->AsEigenThreadPool()) {
    // Set MKL intra thread pool number.
    int intra_num = 0;
    const char* intra_num_str = getenv("TF_MKL_NUM_INTRAOP");
    const int tf_intra_num = eigen_interface_->NumThreads();

    if (intra_num_str != NULL) {
      intra_num = std::stoi(intra_num_str);
    }
    intra_num_ =
        intra_num > 0 ? std::min(tf_intra_num, intra_num) : tf_intra_num;
  }

  MklDnnThreadPool(OpKernelContext* ctx, int user_intra_num)
      : eigen_interface_(ctx->device()
                             ->tensorflow_cpu_worker_threads()
                             ->workers->AsEigenThreadPool()),
        intra_num_(user_intra_num) {
    // Set MKL intra thread pool number.
    int intra_num = 0;
    const char* intra_num_str = getenv("TF_MKL_NUM_INTRAOP");

    if (intra_num_str != NULL) {
      intra_num = std::stoi(intra_num_str);
    }
    intra_num_ =
        intra_num > 0 ? std::min(user_intra_num, intra_num) : user_intra_num;
  }

  virtual int get_num_threads() const override {
    return intra_num_;
  }
  virtual bool get_in_parallel() const override {
    return (eigen_interface_->CurrentThreadId() != -1) ? true : false;
  }
  virtual uint64_t get_flags() const override { return ASYNCHRONOUS; }
  virtual void parallel_for(int n,
                            const std::function<void(int, int)>& fn) override {
    // Should never happen (handled by DNNL)
    if (n == 0) return;

    // Should never happen (handled by DNNL)
    if (n == 1) {
      fn(0, 1);
      return;
    }

    int nthr = get_num_threads();
    int njobs = std::min(n, nthr);
    bool balance = (nthr < n);
    for (int i = 0; i < njobs; i++) {
      eigen_interface_->ScheduleWithHint(
          [balance, i, n, njobs, fn]() {
            if (balance) {
              int start, end;
              balance211(n, njobs, i, &start, &end);
              for (int j = start; j < end; j++) fn(j, n);
            } else {
              fn(i, n);
            }
          },
          i, i + 1);
    }
  }
  ~MklDnnThreadPool() {}

 private:
  Eigen::ThreadPoolInterface* eigen_interface_ = nullptr;
  int intra_num_ = 0;
};

#else

// This struct was just added to enable successful OMP-based build.
struct MklDnnThreadPool {
  MklDnnThreadPool() = default;
  MklDnnThreadPool(OpKernelContext* ctx) {}
};

#endif  // ENABLE_DNNL_THREADPOOL

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_UTIL_MKL_THREADPOOL_H_
