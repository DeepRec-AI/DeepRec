/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <chrono>
#include <cstddef>
#include <deque>
#include <mutex>
#include <numeric>
#include <vector>

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset_stateful_op_whitelist.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/platform/cuda.h"
#include "tensorflow/stream_executor/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

using shape_inference::InferenceContext;

class WorkQueue : public ResourceBase {
 public:
  WorkQueue(const string& name) : name_(name), is_closed_(false) {}

  ~WorkQueue() { Close(); }

  string DebugString() const override {
    return "WorkQueue";
  }

  int64 MemoryUsed() const override {
    return static_cast<int64>(queue_.size() * DataTypeSize(DT_STRING));
  }

  Status Put(const Tensor& inputs) {
    const int64 num_puts = inputs.shape().dim_size(0);

    std::unique_lock<std::mutex> lock(mu_);
    if (TF_PREDICT_FALSE(is_closed_)) {
      lock.unlock();
      take_cv_.notify_all();
      LOG(WARNING) << "Work queue " << name_ << " reinitialized.";

      return Status::OK();
    }

    for (int64 i = 0; i < num_puts; ++i) {
      queue_.push_back(inputs.flat<string>()(i));
    }

    lock.unlock();
    take_cv_.notify_all();

    return Status::OK();
  }

  Status Take(Tensor* output) {
    std::unique_lock<std::mutex> lock(mu_);

    take_cv_.wait(lock, [this]() { return !queue_.empty() || is_closed_; });

    if (TF_PREDICT_FALSE(queue_.empty() && is_closed_)) {
      return Status(errors::OutOfRange(
          strings::StrCat("All works in work queue ", name_, " are taken.")));
    }

    output->scalar<string>().setConstant(std::move(queue_.front()));
    queue_.pop_front();

    return Status::OK();
  }

  Status GetSize(Tensor* size) {
    std::unique_lock<std::mutex> lock(mu_);
    size->scalar<int64>().setConstant(static_cast<int64>(queue_.size()));
    return Status::OK();
  }

  Status Restore(const Tensor& restorable) {
    const int64 num_works = restorable.shape().dim_size(0);

    std::unique_lock<std::mutex> lock(mu_);

    queue_.clear();
    for (int64 i = 0; i < num_works; ++i) {
      queue_.push_back(restorable.flat<string>()(i));
    }

    lock.unlock();
    take_cv_.notify_all();
    return Status::OK();
  }

  Status Save(OpKernelContext* ctx, Tensor** saveable) {
    std::unique_lock<std::mutex> lock(mu_);

    TF_RETURN_IF_ERROR(ctx->allocate_output(
        0, TensorShape({static_cast<int64>(queue_.size())}), saveable));
    for (size_t i = 0; i < queue_.size(); ++i) {
      (*saveable)->flat<string>()(i) = queue_[i];
    }

    return Status::OK();
  }

  Status Close() {
    std::unique_lock<std::mutex> lock(mu_);

    if (is_closed_) {
      return Status::OK();
    }

    is_closed_ = true;

    lock.unlock();
    take_cv_.notify_all();
    return Status::OK();
  }

  void Schedule(int64 num_threads, std::function<void()> fn) {
    std::unique_lock<std::mutex> lock(mu_);
    if (threads_) {
      lock.unlock();
      threads_->Schedule(fn);
      return;
    }

    threads_.reset(
        new thread::ThreadPool(Env::Default(), ThreadOptions(),
                               strings::StrCat("work_queue_threads_", name_),
                               num_threads, false /* low_latency_hint */));

    lock.unlock();
    threads_->Schedule(fn);
  }

 private:
  // TODO(yuanman.ym): Use memory efficient data structure, e.g. HAT-trie,
  // to implement the string queue. (See https://github.com/Tessil/hat-trie)
  std::deque<string> queue_;
  string name_;
  bool is_closed_;
  std::mutex mu_;
  std::condition_variable take_cv_;
  std::shared_ptr<thread::ThreadPool> threads_;
};

REGISTER_RESOURCE_HANDLE_KERNEL(WorkQueue);
REGISTER_KERNEL_BUILDER(Name("WorkQueueIsInitialized").Device(DEVICE_CPU),
                        IsResourceInitialized<WorkQueue>);

class WorkQueueCreateOp : public OpKernel {
 public:
  explicit WorkQueueCreateOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name_));
  }

  void Compute(OpKernelContext* ctx) override {
    WorkQueue* work_queue = new WorkQueue(shared_name_);
    Status s = CreateResource(ctx, HandleFromInput(ctx, 0), work_queue);
    if (!s.ok() && s.code() != error::ALREADY_EXISTS) {
      OP_REQUIRES(ctx, false, s);
    }
  }

 private:
  string shared_name_;
};

REGISTER_KERNEL_BUILDER(Name("WorkQueueCreate").Device(DEVICE_CPU),
                        WorkQueueCreateOp);

class WorkQueueCloseOp : public OpKernel {
 public:
  explicit WorkQueueCloseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    WorkQueue* work_queue;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &work_queue));
    OP_REQUIRES_OK(ctx, work_queue->Close());
  }
};

REGISTER_KERNEL_BUILDER(Name("WorkQueueClose").Device(DEVICE_CPU),
                        WorkQueueCloseOp);

class WorkQueueRestoreOp : public OpKernel {
 public:
  explicit WorkQueueRestoreOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    WorkQueue* work_queue;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &work_queue));
    const Tensor* works;
    OP_REQUIRES_OK(ctx, ctx->input("works", &works));
    OP_REQUIRES_OK(ctx, work_queue->Restore(*works));
  }
};

REGISTER_KERNEL_BUILDER(Name("WorkQueueRestore").Device(DEVICE_CPU),
                        WorkQueueRestoreOp);

class WorkQueueSaveOp : public OpKernel {
 public:
  explicit WorkQueueSaveOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    WorkQueue* work_queue;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &work_queue));
    Tensor* works;
    OP_REQUIRES_OK(ctx, work_queue->Save(ctx, &works));
  }
};

REGISTER_KERNEL_BUILDER(Name("WorkQueueSave").Device(DEVICE_CPU),
                        WorkQueueSaveOp);

class WorkQueueSizeOp : public OpKernel {
 public:
  explicit WorkQueueSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    WorkQueue* work_queue;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &work_queue));
    Tensor* size;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &size));
    OP_REQUIRES_OK(ctx, work_queue->GetSize(size));
  }
};

REGISTER_KERNEL_BUILDER(Name("WorkQueueSize").Device(DEVICE_CPU),
                        WorkQueueSizeOp);

class WorkQueuePutOp : public OpKernel {
 public:
  explicit WorkQueuePutOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    WorkQueue* work_queue;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &work_queue));
    const Tensor* works;
    OP_REQUIRES_OK(ctx, ctx->input("works", &works));
    OP_REQUIRES_OK(ctx, work_queue->Put(*works));
  }
};

REGISTER_KERNEL_BUILDER(Name("WorkQueuePut").Device(DEVICE_CPU),
                        WorkQueuePutOp);

class WorkQueueTakeOp : public AsyncOpKernel {
 public:
  explicit WorkQueueTakeOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_clients", &num_clients_));
  }

  void ComputeAsync(OpKernelContext* ctx,
                    AsyncOpKernel::DoneCallback done) override {
    WorkQueue* work_queue;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &work_queue));
    core::ScopedUnref scoped_list(work_queue);
    work_queue->Schedule(num_clients_, [this, ctx, done, work_queue]() {
      Tensor* work;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, TensorShape({}), &work),
                           done);
      OP_REQUIRES_OK_ASYNC(ctx, work_queue->Take(work), done);
      done();
    });
  }

 private:
  int64 num_clients_;
};

REGISTER_KERNEL_BUILDER(Name("WorkQueueTake").Device(DEVICE_CPU),
                        WorkQueueTakeOp);

class SaveLocalWorkOp : public OpKernel {
 public:
  explicit SaveLocalWorkOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("job_name", &job_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("task_index", &task_index_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("restore_works_dir", &restore_works_dir_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_;
    OP_REQUIRES_OK(ctx, ctx->input("work", &input_));
    const string& work = input_->scalar<string>()();

    Env* const env = Env::Default();
    std::vector<string> work_parts = str_util::Split(work, "?");

    OP_REQUIRES(ctx, work_parts.size() == 2,
                errors::InvalidArgument("Invalid table path format: ", work));
    OP_REQUIRES_OK(ctx, env->IsDirectory(restore_works_dir_));

    uint64 start_time = env->NowMicros();
    string range = str_util::StringReplace(work_parts[1], "=", "_", true);
    range = str_util::StringReplace(range, "&", "_", true);
    string worker_id = strings::StrCat(job_name_, "_", task_index_);
    string worker_dir = strings::StrCat(restore_works_dir_, "/", worker_id);
    if (!env->IsDirectory(worker_dir).ok()) {
      OP_REQUIRES_OK(ctx, env->CreateDir(worker_dir));
    }
    string slice_file = strings::StrCat(worker_dir, "/", range);
    std::unique_ptr<WritableFile> wfile;
    OP_REQUIRES_OK(ctx, env->NewWritableFile(slice_file, &wfile));
    OP_REQUIRES_OK(ctx, wfile->Append(work));
    OP_REQUIRES_OK(ctx, wfile->Close());
    uint64 end_time = env->NowMicros();
    float time_use = (end_time - start_time) / 1000000.0;
    LOG(INFO) << "Job_name:" << job_name_ << ", time use:" << time_use
              << ", task_index:" << task_index_ << ", work_range:" << range;
  }

 private:
  string job_name_;
  int64 task_index_;
  string restore_works_dir_;
};

REGISTER_KERNEL_BUILDER(Name("SaveLocalWork").Device(DEVICE_CPU),
                        SaveLocalWorkOp);
}  // namespace tensorflow
