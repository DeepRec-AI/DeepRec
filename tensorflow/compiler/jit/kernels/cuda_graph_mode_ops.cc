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
==============================================================================*/

#if GOOGLE_CUDA

#include "tensorflow/compiler/jit/kernels/cuda_graph_mode_ops.h"
#include <queue>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"

#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cuda_graph_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"

// OP_REQUIRES_OK_RETURN is the same as OP_REQUIRES_OK except that
// in error case, it returns RET instead of void.
#define OP_REQUIRES_OK_RETURN(CTX, RET, ...)                \
  do {                                                      \
    ::tensorflow::Status _s(__VA_ARGS__);                   \
    if (!TF_PREDICT_TRUE(_s.ok())) {                        \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s); \
      return RET;                                           \
    }                                                       \
  } while (0)

namespace tensorflow {

mutex CgmodeCompileOp::compile_mu_;

namespace {

NameAttrList FunctionAttr(OpKernelConstruction* ctx) {
  const NameAttrList* func;
  OP_REQUIRES_OK_RETURN(ctx, NameAttrList(), ctx->GetAttr("function", &func));
  return *func;
}

class CgmodeExecutableClosure {
 public:
  explicit CgmodeExecutableClosure(const NameAttrList func,
                                   cudaGraphExec_t cuda_graph_exec,
                                   std::vector<Tensor> args,
                                   std::vector<Tensor> rets,
                                   bool is_out_shape_static)
      : func_(func),
        cuda_graph_exec_(cuda_graph_exec),
        is_out_shape_static_(is_out_shape_static) {
    args_.reset(new gtl::InlinedVector<Tensor, 4>());
    rets_.reset(new gtl::InlinedVector<Tensor, 4>());
    for (auto e : args) {
      args_->emplace_back(e);
    }
    for (auto e : rets) {
      rets_->emplace_back(e);
    }
  }

  CgmodeExecutableClosure(CgmodeExecutableClosure&&) = default;
  CgmodeExecutableClosure& operator=(CgmodeExecutableClosure&&) = default;

  NameAttrList func() const { return func_; }
  bool is_out_shape_static() const { return is_out_shape_static_; }
  cudaGraphExec_t cuda_graph_exec() { return cuda_graph_exec_; }
  gtl::InlinedVector<Tensor, 4>* args() { return args_.get(); }
  gtl::InlinedVector<Tensor, 4>* rets() { return rets_.get(); }

 private:
  const NameAttrList func_;
  cudaGraphExec_t cuda_graph_exec_;
  bool is_out_shape_static_;
  std::unique_ptr<gtl::InlinedVector<Tensor, 4>> args_;
  std::unique_ptr<gtl::InlinedVector<Tensor, 4>> rets_;
  TF_DISALLOW_COPY_AND_ASSIGN(CgmodeExecutableClosure);
};

class CgmodeExecutableClosureStore {
 public:
  CgmodeExecutableClosureStore() : key_counter_(0) {}

  using KeyT = string;

  KeyT Produce(CgmodeExecutableClosure result) {
    mutex_lock l(mutex_);
    KeyT key;
    if (key_queue_.empty()) {
      key = absl::StrCat(key_counter_++);
    } else {
      key = key_queue_.front();
      key_queue_.pop();
    }
    bool insert_successful = closures_.emplace(key, std::move(result)).second;
    DCHECK(insert_successful);
    (void)insert_successful;
    return key;
  }

  CgmodeExecutableClosure Consume(const KeyT& key) {
    mutex_lock l(mutex_);
    auto it = closures_.find(key);
    DCHECK(it != closures_.end());
    CgmodeExecutableClosure value = std::move(it->second);
    closures_.erase(it);
    key_queue_.emplace(key);
    return value;
  }

  static CgmodeExecutableClosureStore* Global() {
    static CgmodeExecutableClosureStore* instance =
        new CgmodeExecutableClosureStore;
    return instance;
  }

 private:
  mutex mutex_;
  unsigned long long key_counter_ GUARDED_BY(mutex_);
  std::queue<KeyT> key_queue_;

  absl::flat_hash_map<KeyT, CgmodeExecutableClosure> closures_
      GUARDED_BY(mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(CgmodeExecutableClosureStore);
};
}  // namespace

CgmodeCompileOp::CgmodeCompileOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), function_(FunctionAttr(ctx)) {
  env_ = ctx->env();
  flib_ = ctx->function_library();
  is_compiled_ = false;
  is_out_shape_static_ = true;
}

void CgmodeCompileOp::Compile(OpKernelContext* ctx, tstring& compiled_key) {
  auto gpu_device = dynamic_cast<BaseGPUDevice*>(ctx->device());
  CudaGraphGPUBFCAllocator* device_allocator =
      reinterpret_cast<CudaGraphGPUBFCAllocator*>(
          gpu_device->GetAllocator(AllocatorAttributes()));
  OP_REQUIRES(ctx, gpu_device, errors::Internal("BaseGPUDevice not found"));
  se::Stream* default_stream = gpu_device->GetDefaultTFStream();
  stream_executor::gpu::GpuContext* gpu_ctx =
      reinterpret_cast<stream_executor::gpu::GpuContext*>(
          default_stream->parent()->implementation()->GpuContextHack());
  cudaStream_t cu_stream;
  cu_stream = *(static_cast<cudaStream_t*>(gpu_device->GetStream()));
  // enable cuda graph mode
  gpu_device->SetSingleStreamMode();
  gpu_ctx->enable_single_stream_mode();

  FunctionLibraryRuntime::Handle handle;
  flib_->Instantiate(function_.name(), AttrSlice(&function_.attr()), &handle);
  const FunctionBody* fbody;
  fbody = flib_->GetFunctionBody(handle);
  VLOG(1) << "CgmodeCompileOp function def: " << DebugString(fbody->fdef);

  FunctionLibraryRuntime::Options opts;
  std::vector<Tensor> in;
  AllocatorAttributes attr;
  attr.set_gpu_compatible(true);
  for (int i = 0; i < ctx->num_inputs(); i++) {
    PersistentTensor arg_persistent;
    Tensor* out_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_persistent(
                            ctx->input(i).dtype(), ctx->input(i).shape(),
                            &arg_persistent, &out_tensor, attr));
    persist_args_.emplace_back(arg_persistent);
    in.emplace_back(*out_tensor);
  }

  Status s_dry_run;
  Notification done_dry_run;
  std::vector<Tensor> out_dry_run(fbody->ret_types.size());
  device_allocator->EnableCudaGraphModeMem();
  flib_->Run(opts, handle, in, &out_dry_run,
             [&s_dry_run, &done_dry_run](const Status& s) {
               s_dry_run = s;
               done_dry_run.Notify();
             });
  done_dry_run.WaitForNotification();
  OP_REQUIRES(ctx, s_dry_run.ok(),
              errors::Internal("CgmodeCompile failed" + s_dry_run.ToString()));
  device_allocator->DisableCudaGraphModeMem();
  Status s_compile;
  Notification done_compile;
  std::vector<Tensor> out(fbody->ret_types.size());

  cudaError_t e_cu_capture_begin =
      cudaStreamBeginCapture(cu_stream, cudaStreamCaptureModeThreadLocal);

  OP_REQUIRES(ctx, e_cu_capture_begin == cudaSuccess,
              errors::Internal(std::string("cuda graph begin capture failed ") +
                               cudaGetErrorString(e_cu_capture_begin)));

  flib_->Run(opts, handle, in, &out,
             [&s_compile, &done_compile](const Status& s) {
               s_compile = s;
               done_compile.Notify();
             });
  done_compile.WaitForNotification();
  OP_REQUIRES(ctx, s_compile.ok(),
              errors::Internal("CgmodeCompile failed" + s_compile.ToString()));

  cudaError_t e_cu_capture_end =
      cudaStreamEndCapture(cu_stream, &cuda_graph_obj_);
  OP_REQUIRES(ctx, e_cu_capture_end == cudaSuccess,
              errors::Internal(std::string("cuda graph end capture failed ") +
                               cudaGetErrorString(e_cu_capture_end)));

  if (cuda_graph_exec_ != nullptr) {
    cudaError_t e_cu_destroy_graph_exec =
        cudaGraphExecDestroy(cuda_graph_exec_);
    if (e_cu_destroy_graph_exec != cudaSuccess) {
      LOG(ERROR) << std::string("destroy cuda graph exec failed ")
                 << cudaGetErrorString(e_cu_destroy_graph_exec);
    }
    cuda_graph_exec_ = nullptr;
  }
  cudaError_t e_cu_instantiate =
      cudaGraphInstantiate(&cuda_graph_exec_, cuda_graph_obj_, NULL, NULL, 0);
  OP_REQUIRES(ctx, e_cu_instantiate == cudaSuccess,
              errors::Internal(
                  std::string("cuda graph create execute instance failed ") +
                  cudaGetErrorString(e_cu_instantiate)));

  gpu_device->ResetStreamMode();
  gpu_ctx->disable_single_stream_mode();

  for (int i = 0; i < out.size(); i++) {
    if (out[i].shape() != out_dry_run[i].shape()) {
      is_out_shape_static_ = false;
    }
    persist_rets_.emplace_back(PersistentTensor(out[i]));
  }

  std::vector<Tensor> cuda_graph_args;
  std::vector<Tensor> cuda_graph_rets;

  for (auto t : persist_args_) {
    cuda_graph_args.emplace_back(*(t.AccessTensor(ctx)));
  }
  for (auto t : persist_rets_) {
    cuda_graph_rets.emplace_back(*(t.AccessTensor(ctx)));
  }

  compiled_key = CgmodeExecutableClosureStore::Global()->Produce(
      CgmodeExecutableClosure(function_, cuda_graph_exec_, cuda_graph_args,
                              cuda_graph_rets, is_out_shape_static_));
  cudaError_t e_cu_destroy_graph = cudaGraphDestroy(cuda_graph_obj_);
  OP_REQUIRES(ctx, e_cu_destroy_graph == cudaSuccess,
              errors::Internal(std::string("destroy cuda graph failed ") +
                               cudaGetErrorString(e_cu_destroy_graph)));
  is_compiled_ = true;
}

void CgmodeCompileOp::Compute(OpKernelContext* ctx) {
  tf_shared_lock lock(CgmodeCompileOp::compile_mu_);
  bool do_recompile = false;
  if (persist_args_.size() > 0 && persist_args_.size() == ctx->num_inputs()) {
    for (int i = 0; i < persist_args_.size(); i++) {
      if (persist_args_[i].AccessTensor(ctx)->shape() !=
          ctx->input(i).shape()) {
        do_recompile = true;
        VLOG(2) << "A recompile is triggered";
        break;
      }
    }
  }

  CgmodeExecutableClosureStore::KeyT key;
  if (!is_compiled_ || do_recompile) {
    if (do_recompile) {
      persist_args_.clear();
      persist_rets_.clear();
    }
    Compile(ctx, key);
  } else {
    std::vector<Tensor> cuda_graph_args;
    std::vector<Tensor> cuda_graph_rets;

    for (auto t : persist_args_) {
      cuda_graph_args.emplace_back(*(t.AccessTensor(ctx)));
    }
    for (auto t : persist_rets_) {
      cuda_graph_rets.emplace_back(*(t.AccessTensor(ctx)));
    }

    key = CgmodeExecutableClosureStore::Global()->Produce(
        CgmodeExecutableClosure(function_, cuda_graph_exec_, cuda_graph_args,
                                cuda_graph_rets, is_out_shape_static_));
  }

  AllocatorAttributes host_alloc_attrs;
  host_alloc_attrs.set_gpu_compatible(true);
  host_alloc_attrs.set_on_host(true);
  Allocator* cpu_allocator = ctx->device()->GetAllocator(host_alloc_attrs);

  Tensor compilation_key(cpu_allocator, DT_STRING, TensorShape({}));
  compilation_key.flat<tstring>()(0) = key;

  Tensor compilation_successful(cpu_allocator, DT_BOOL, TensorShape({}));
  compilation_successful.flat<bool>()(0) = true;

  ctx->set_output(0, compilation_key);
  ctx->set_output(1, compilation_successful);
}

CgmodeRunOp::CgmodeRunOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  env_ = ctx->env();
  flib_ = ctx->function_library();
}

CgmodeRunOp::~CgmodeRunOp() {
  if (cuda_graph_exec_ != nullptr) {
    cudaError_t e_cu_destroy_graph_exec =
        cudaGraphExecDestroy(cuda_graph_exec_);
    if (e_cu_destroy_graph_exec != cudaSuccess) {
      LOG(ERROR) << std::string("destroy cuda graph exec failed ")
                 << cudaGetErrorString(e_cu_destroy_graph_exec);
    }
  }
}

void CgmodeRunOp::Compute(OpKernelContext* ctx) {
  mutex_lock l(mutex_);
  Tensor key_tensor = ctx->input(ctx->num_inputs() - 1);
  const CgmodeExecutableClosureStore::KeyT& key = key_tensor.flat<tstring>()(0);
  CgmodeExecutableClosure closure =
      CgmodeExecutableClosureStore::Global()->Consume(key);
  gtl::InlinedVector<Tensor, 4>* cuda_graph_args = closure.args();
  gtl::InlinedVector<Tensor, 4>* cuda_graph_rets = closure.rets();
  cuda_graph_exec_ = closure.cuda_graph_exec();
  const NameAttrList func = closure.func();
  bool is_out_shape_static = closure.is_out_shape_static();
  if (!is_out_shape_static) {
    FallbackToTF(ctx, func);
    return;
  }

  auto gpu_device = dynamic_cast<BaseGPUDevice*>(ctx->device());
  OP_REQUIRES(ctx, gpu_device, errors::Internal("BaseGPUDevice not found"));
  cudaStream_t cu_stream =
      *(static_cast<cudaStream_t*>(gpu_device->GetStream()));

  for (int i = 0; i < ctx->num_inputs() - 1; i++) {
    cudaError_t e_cu_copy = cudaMemcpyAsync(
        cuda_graph_args->at(i).data(), ctx->input(i).data(),
        ctx->input(i).TotalBytes(), cudaMemcpyDefault, cu_stream);
    OP_REQUIRES(ctx, e_cu_copy == cudaSuccess,
                errors::Internal(std::string("async copy args failed ") +
                                 cudaGetErrorString(e_cu_copy)));
  }

  cudaError_t e_cu_launch_graph =
      cudaGraphLaunch(closure.cuda_graph_exec(), cu_stream);
  if (e_cu_launch_graph == cudaSuccess) {
    for (int i = 0; i < ctx->num_outputs(); i++) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(
                              i, cuda_graph_rets->at(i).shape(), &output));
      cudaMemcpyAsync(output->data(), cuda_graph_rets->at(i).data(),
                      cuda_graph_rets->at(i).TotalBytes(), cudaMemcpyDefault,
                      cu_stream);
    }
    cudaStreamSynchronize(cu_stream);
  } else {
    FallbackToTF(ctx, func);
  }
}

void CgmodeRunOp::FallbackToTF(OpKernelContext* ctx, const NameAttrList func) {
  VLOG(2) << "Fallback to run TF subgraph when CUDA Graph execution fails";
  FunctionLibraryRuntime::Handle handle;
  flib_->Instantiate(func.name(), AttrSlice(&func.attr()), &handle);
  const FunctionBody* fbody;
  fbody = flib_->GetFunctionBody(handle);
  FunctionLibraryRuntime::Options opts;
  std::vector<Tensor> in;
  for (int i = 0; i < ctx->num_inputs() - 1; i++) {
    in.emplace_back(ctx->input(i));
  }
  Status s_tf_run;
  Notification done_tf_run;
  std::vector<Tensor> out(fbody->ret_types.size());
  flib_->Run(opts, handle, in, &out,
             [&s_tf_run, &done_tf_run](const Status& s) {
               s_tf_run = s;
               done_tf_run.Notify();
             });
  done_tf_run.WaitForNotification();
  OP_REQUIRES(ctx, s_tf_run.ok(), errors::Internal(s_tf_run.ToString()));
  for (int i = 0; i < out.size(); i++) {
    ctx->set_output(i, out[i]);
  }
}
REGISTER_KERNEL_BUILDER(Name("_CgmodeCompile")
                            .Device(DEVICE_GPU)
                            .HostMemory("key")
                            .HostMemory("compilation_successful"),
                        CgmodeCompileOp);

REGISTER_KERNEL_BUILDER(Name("_CgmodeRun").Device(DEVICE_GPU).HostMemory("key"),
                        CgmodeRunOp);

}  // namespace tensorflow
#endif  // GOOGLE_CUDA
