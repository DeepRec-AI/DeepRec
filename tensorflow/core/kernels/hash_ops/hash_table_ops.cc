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

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/hash_table/status_collector.h"
#include "tensorflow/core/framework/hash_table/hash_table.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

using data::FunctionMetadata;
using data::InstantiatedCapturedFunction;

class HashTableInitializeOp : public OpKernel {
 public:
  explicit HashTableInitializeOp(OpKernelConstruction* context)
    : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("initialized", &initialized_));
    OP_REQUIRES_OK(context, context->GetAttr("concurrent_read", &concurrent_read_));
    OP_REQUIRES_OK(context, context->GetAttr("children", &children_));
  }

  void Compute(OpKernelContext* ctx) override {
    SessionOptions tmp;
    auto inter_op_thread_pool = ComputePool(tmp);
    auto num_threads = inter_op_thread_pool->NumThreads();
    HashTableResource* resource;
    OP_REQUIRES_OK(
        ctx,
        LookupOrCreateResource<HashTableResource>(
            ctx, HandleFromInput(ctx, 0), &resource,
            [ctx, num_threads, this](HashTableResource** ptr) {
              *ptr = new HashTableResource(
                  num_threads, concurrent_read_, children_);
              return Status::OK();
            }));
    core::ScopedUnref s(resource);
    resource->CreateInternal();
    resource->SetInitialized(initialized_);
  }

 private:
  bool initialized_;
  bool concurrent_read_;
  std::vector<string> children_;
};

class HashTableLookupOp : public AsyncOpKernel {
 public:
  explicit HashTableLookupOp(OpKernelConstruction* context)
    : AsyncOpKernel(context) {
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    HashTableResource* resource;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &resource), done);
    core::ScopedUnref s(resource);
    HashTable* table = resource->Internal();
    OP_REQUIRES_ASYNC(
        ctx, table != nullptr,
        errors::FailedPrecondition("HashTable is not initialized"), done);
    Tensor input_tensor = ctx->input(1);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(0, input_tensor.shape(), &output_tensor),
        done);
    int64* keys = reinterpret_cast<int64*>(const_cast<char*>(
          input_tensor.tensor_data().data()));
    int64* ids = reinterpret_cast<int64*>(const_cast<char*>(
          output_tensor->tensor_data().data()));
    resource->Ref();
    table->GetIds(keys, nullptr, ids, input_tensor.NumElements(), nullptr,
                  ctx->runner(),
                  [ctx, done, resource] (Status st) {
                    resource->Unref();
                    OP_REQUIRES_OK_ASYNC(ctx, st, done);
                    done();
                  });
  }
};

class HashTableLookupWithAdmitOp : public AsyncOpKernel {
 public:
  explicit HashTableLookupWithAdmitOp(OpKernelConstruction* context)
    : AsyncOpKernel(context) {
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    HashTableResource* resource;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &resource), done);
    HashTableAdmitStrategyResource* strategy;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 2), &strategy), done);
    core::ScopedUnref s(resource);
    core::ScopedUnref s2(strategy);
    HashTable* table = resource->Internal();
    OP_REQUIRES_ASYNC(
        ctx, table != nullptr,
        errors::FailedPrecondition("HashTable is not initialized"), done);
    Tensor input_tensor = ctx->input(1);
    Tensor freq_tensor = ctx->input(3);
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(0, input_tensor.shape(), &output_tensor),
        done);
    int64* keys = reinterpret_cast<int64*>(const_cast<char*>(
          input_tensor.tensor_data().data()));
    int32* freqs = reinterpret_cast<int32*>(const_cast<char*>(
          freq_tensor.tensor_data().data()));
    int64* ids = reinterpret_cast<int64*>(const_cast<char*>(
          output_tensor->tensor_data().data()));
    resource->Ref();
    strategy->Ref();
    table->GetIds(
        keys, freqs, ids, input_tensor.NumElements(),
        strategy->Internal(),
        ctx->runner(),
        //[ctx] (std::function<void()> f){
        //  ctx->device()->tensorflow_cpu_worker_threads()->workers->Schedule(f);
        //},
        [ctx, done, resource, strategy] (Status st) {
      resource->Unref();
      strategy->Unref();
      OP_REQUIRES_OK_ASYNC(ctx, st, done);
      done();
    });
  }
};

namespace {

void DoFilter(std::vector<int64>* keys, 
              std::vector<int64>* ids, 
              int64 start, 
              int64 end,
              IteratorContext* ctx,
              HashTable* hash_table,
              InstantiatedCapturedFunction* func,
              std::function<void(Status)> done) {
  Tensor input_keys(DT_INT64, TensorShape({end - start}));
  memcpy(input_keys.flat<int64>().data(), keys->data() + start, (end - start) * sizeof(int64));    
  Tensor input_ids(DT_INT64, TensorShape({end - start}));
  memcpy(input_ids.flat<int64>().data(), ids->data() + start, (end - start) * sizeof(int64));
  std::vector<Tensor> args;
  args.emplace_back(input_keys);
  args.emplace_back(input_ids);
  std::vector<Tensor>* rets = new std::vector<Tensor>();
  auto real_done = [keys, ids, start, done, rets, hash_table] (const Status& st) {
    std::unique_ptr<std::vector<Tensor>> deleter(rets);
    if (!st.ok()) {
      done(st);
      return;
    }
    if (rets->size() != 1) {
      done(errors::Internal("failed to run filter func"));
      return;
    }
    auto filter_vec = (*rets)[0].vec<bool>();
    std::vector<int64>* filtered_keys = new std::vector<int64>();
    std::vector<int64>* filtered_ids = new std::vector<int64>();
    for (int64 i = 0; i < filter_vec.size(); ++i) {
      if (filter_vec(i)) {
        filtered_keys->push_back((*keys)[i+start]);
        filtered_ids->push_back((*ids)[i+start]);
      }
    }
    auto delete_done = [done, filtered_keys, filtered_ids] (Status st) {
      delete filtered_keys;
      delete filtered_ids;
      done(st);
    };
    hash_table->DeleteKeysSimple(filtered_keys->data(),
                                 filtered_ids->data(),
                                 filtered_keys->size(),
                                 delete_done);
  };
  func->RunAsync(ctx, std::move(args), rets, real_done, "DoFilter");
}
}  // namespace

class HashTableFilterOp : public AsyncOpKernel {
 public:
  explicit HashTableFilterOp(OpKernelConstruction* ctx)
    : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("block_size", &block_size_));
    FunctionMetadata::Params params;
    OP_REQUIRES_OK(ctx, FunctionMetadata::Create(
            ctx, "f", params, &func_metadata_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    IteratorContext::Params* params = new IteratorContext::Params(ctx);
    data::FunctionHandleCache* function_handle_cache = new data::FunctionHandleCache(params->flr);
    params->function_handle_cache = function_handle_cache;
    ResourceMgr* resource_mgr = new ResourceMgr();
    params->resource_mgr = resource_mgr;
    CancellationManager* cancellation_manager = new CancellationManager();
    params->cancellation_manager = cancellation_manager;

    IteratorContext* iter_ctx = new IteratorContext(*params);
    std::unique_ptr<CapturedFunction> captured_func;
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func;
    OP_REQUIRES_OK_ASYNC(
        ctx, 
        CapturedFunction::Create(ctx, func_metadata_, "other_arguments", &captured_func), 
        done);
    OP_REQUIRES_ASYNC(
        ctx, captured_func.get() != nullptr,
        errors::FailedPrecondition("filter func is not initialized"), done);
    OP_REQUIRES_OK_ASYNC(
        ctx,
        captured_func->Instantiate(iter_ctx, &instantiated_captured_func),
        done);
    OP_REQUIRES_ASYNC(
        ctx,
        instantiated_captured_func.get() != nullptr,
        errors::FailedPrecondition("InstantiatedCapturedFunction init failed"),
        done);
    HashTableResource* resource;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &resource), done);
    core::ScopedUnref s(resource);
    HashTable* table = resource->Internal();
    OP_REQUIRES_ASYNC(
        ctx, table != nullptr,
        errors::FailedPrecondition("HashTable is not initialized"), done);
    resource->Ref();
    std::vector<int64>* keys = new std::vector<int64>();
    std::vector<int64>* ids = new std::vector<int64>();
    table->Snapshot(keys, ids);
    if (keys->empty()) {
      delete keys;
      delete ids;
      resource->Unref();
      done();
      return;
    }

    CapturedFunction* cap_func = captured_func.release();
    InstantiatedCapturedFunction* func = instantiated_captured_func.release();

    int32 block_num = (keys->size() + block_size_ - 1) / block_size_;
    int32 left = keys->size() % block_size_;
    int64 start = 0;
    auto real_done = [ctx, done, resource, keys, ids, iter_ctx, cap_func,
                      func, function_handle_cache, resource_mgr,
                      cancellation_manager, params] (Status st) {
      delete keys;
      delete ids;
      delete iter_ctx;
      delete cap_func;
      delete func;
      delete function_handle_cache;
      delete resource_mgr;
      delete cancellation_manager;
      delete params;
      resource->Unref();
      OP_REQUIRES_OK_ASYNC(ctx, st, done);
      done();
    };
    StatusCollector* stc = new StatusCollector(block_num, real_done);
    for (int i = 0; i < block_num; ++i) {
      int64 block_size = (i == block_num - 1 && left > 0) ? left : block_size_;
      int64 end = start + block_size;
      auto runner = [keys, ids,  start, end, iter_ctx, func, table, stc, i] { 
        DoFilter(keys, ids, start, end, iter_ctx, table, func,
                 [stc] (Status st) {
          stc->AddStatus(st);
        });
      };
      ctx->device()->tensorflow_cpu_worker_threads()->workers->Schedule(runner);
      start = end;
    }
    stc->Start();
  }

 private:
  int64 block_size_;
  std::shared_ptr<FunctionMetadata> func_metadata_ = nullptr;
};

class HashTableSnapshotOp : public AsyncOpKernel {
 public:
  explicit HashTableSnapshotOp(OpKernelConstruction* context)
    : AsyncOpKernel(context) {
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    HashTableResource* resource;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &resource), done);
    core::ScopedUnref s(resource);
    HashTable* table = resource->Internal();
    std::vector<int64> keys;
    std::vector<int64> ids;
    table->Snapshot(&keys, &ids);
    Tensor* output_keys = NULL;
    Tensor* output_ids = NULL;    
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(0, TensorShape({keys.size()}), &output_keys),
        done);
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(1, TensorShape({keys.size()}), &output_ids),
        done);
    int64* key_ptr = reinterpret_cast<int64*>(const_cast<char*>(
          output_keys->tensor_data().data()));
    int64* id_ptr = reinterpret_cast<int64*>(const_cast<char*>(
          output_ids->tensor_data().data()));
    memcpy(key_ptr, keys.data(), keys.size() * sizeof(int64));
    memcpy(id_ptr, ids.data(), ids.size() * sizeof(int64));
    done();
  }
};

class HashTableSizeOp : public AsyncOpKernel {
 public:
  explicit HashTableSizeOp(OpKernelConstruction* context)
    : AsyncOpKernel(context) {
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    HashTableResource* resource;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &resource), done);
    core::ScopedUnref s(resource);
    HashTable* table = resource->Internal();
    OP_REQUIRES_ASYNC(
        ctx, table != nullptr,
        errors::FailedPrecondition("HashTable is not initialized"), done);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(0, TensorShape({}), &output_tensor),
        done);
    int64* size = reinterpret_cast<int64*>(const_cast<char*>(
          output_tensor->tensor_data().data()));
    *size = table->Size();
    done();
  }
};

REGISTER_KERNEL_BUILDER(
    Name("HashTableOp").Device(DEVICE_CPU),
    ResourceHandleOp<HashTableResource>);
REGISTER_KERNEL_BUILDER(
    Name("HashTableInitializeOp").Device(DEVICE_CPU),
    HashTableInitializeOp);
REGISTER_KERNEL_BUILDER(
    Name("HashTableLookupOp").Device(DEVICE_CPU),
    HashTableLookupOp);
REGISTER_KERNEL_BUILDER(
    Name("HashTableFilterOp").Device(DEVICE_CPU),
    HashTableFilterOp);
REGISTER_KERNEL_BUILDER(
    Name("HashTableLookupWithAdmitOp").Device(DEVICE_CPU),
    HashTableLookupWithAdmitOp);
REGISTER_KERNEL_BUILDER(
    Name("HashTableSnapshotOp").Device(DEVICE_CPU),
    HashTableSnapshotOp);
REGISTER_KERNEL_BUILDER(
    Name("HashTableSizeOp").Device(DEVICE_CPU),
    HashTableSizeOp);

}  // namespace tensorflow
