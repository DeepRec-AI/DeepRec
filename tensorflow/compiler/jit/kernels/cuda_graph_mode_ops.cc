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
                                   std::vector<Tensor> rets)
      : func_(func),
        cuda_graph_exec_(cuda_graph_exec) {
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
  cudaGraphExec_t cuda_graph_exec() { return cuda_graph_exec_; }
  gtl::InlinedVector<Tensor, 4>* args() { return args_.get(); }
  gtl::InlinedVector<Tensor, 4>* rets() { return rets_.get(); }

 private:
  const NameAttrList func_;
  cudaGraphExec_t cuda_graph_exec_;
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
}

Status CgmodeCompileOp::Compile(OpKernelContext* ctx, tstring& compiled_key) {
  auto gpu_device = dynamic_cast<BaseGPUDevice*>(ctx->device());
  CudaGraphGPUBFCAllocator* device_allocator =
      reinterpret_cast<CudaGraphGPUBFCAllocator*>(
          gpu_device->GetAllocator(AllocatorAttributes()));
  if (gpu_device == nullptr) {
    return errors::Internal("BaseGPUDevice not found");
  } 
  se::Stream* default_stream = gpu_device->GetDefaultTFStream();
  cudaStream_t cu_stream;
  cu_stream = *(static_cast<cudaStream_t*>(gpu_device->GetStream()));

  FunctionLibraryRuntime::Handle handle;
  Status s_flib_instantiate = flib_->Instantiate(
    function_.name(), AttrSlice(&function_.attr()), &handle);
  if (!s_flib_instantiate.ok()) {
    return s_flib_instantiate;
  }

  const FunctionBody* fbody;
  fbody = flib_->GetFunctionBody(handle);
  VLOG(1) << "CgmodeCompileOp function def: " << DebugString(fbody->fdef);

  FunctionLibraryRuntime::Options opts;
  std::vector<Tensor> in;
  AllocatorAttributes attr;
  attr.set_gpu_compatible(true);
  for (int i = 0; i < ctx->num_inputs(); i++) {
    // validate the original input is on device
    PersistentTensor arg_persistent;
    Tensor* out_tensor = nullptr;
    OP_REQUIRES_OK_RETURN(ctx, errors::Internal("Failed to allocate persistent memory"),
                          ctx->allocate_persistent(ctx->input(i).dtype(), ctx->input(i).shape(),
                            &arg_persistent, &out_tensor, attr));
    persist_args_.emplace_back(arg_persistent);
    cudaError_t e_cu_copy = cudaMemcpyAsync(
        out_tensor->data(), ctx->input(i).data(),
        ctx->input(i).TotalBytes(), cudaMemcpyDefault, cu_stream);
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
  cudaStreamSynchronize(cu_stream);
  if (!s_dry_run.ok()) {
    return errors::Internal("CgmodeCompile failed" + s_dry_run.ToString());
  }

  device_allocator->DisableCudaGraphModeMem();
  Status s_compile;
  Notification done_compile;
  std::vector<Tensor> out(fbody->ret_types.size());

  cudaError_t e_cu_capture_begin =
      cudaStreamBeginCapture(cu_stream, cudaStreamCaptureModeThreadLocal);

  if (e_cu_capture_begin != cudaSuccess) {
    return errors::Internal(std::string("cuda graph begin capture failed ") +
                               cudaGetErrorString(e_cu_capture_begin));
  }

  flib_->Run(opts, handle, in, &out,
             [&s_compile, &done_compile](const Status& s) {
               s_compile = s;
               done_compile.Notify();
             });
  done_compile.WaitForNotification();
  if (!s_compile.ok()) {
    return errors::Internal("CgmodeCompile failed" + s_compile.ToString());
  }

  cudaError_t e_cu_capture_end =
      cudaStreamEndCapture(cu_stream, &cuda_graph_obj_);
  if (e_cu_capture_end != cudaSuccess) {
    return errors::Internal(std::string("cuda graph end capture failed ") +
                               cudaGetErrorString(e_cu_capture_end));
  }

  if (cuda_graph_exec_ != nullptr) {
    cudaError_t e_cu_destroy_graph_exec =
        cudaGraphExecDestroy(cuda_graph_exec_);
    if (e_cu_destroy_graph_exec != cudaSuccess) {
      return errors::Internal(std::string("destroy cuda graph exec failed ") +
                                cudaGetErrorString(e_cu_destroy_graph_exec));
    }
    cuda_graph_exec_ = nullptr;
  }
  cudaError_t e_cu_instantiate =
      cudaGraphInstantiate(&cuda_graph_exec_, cuda_graph_obj_, NULL, NULL, 0);
  if (e_cu_instantiate != cudaSuccess) {
    return errors::Internal(
                  std::string("cuda graph create execute instance failed ") +
                  cudaGetErrorString(e_cu_instantiate));
  }

  for (int i = 0; i < out.size(); i++) {
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
                              cuda_graph_rets));
  cudaError_t e_cu_destroy_graph = cudaGraphDestroy(cuda_graph_obj_);
  if (e_cu_destroy_graph != cudaSuccess) {
    return errors::Internal(std::string("destroy cuda graph failed ") +
                               cudaGetErrorString(e_cu_destroy_graph));
  }
  is_compiled_ = true;
  return Status::OK();
}

void CgmodeCompileOp::Compute(OpKernelContext* ctx) {
  tf_shared_lock lock(CgmodeCompileOp::compile_mu_);
  bool has_dynamic_input_shape = false;
  if (persist_args_.size() > 0 && persist_args_.size() == ctx->num_inputs()) {
    for (int i = 0; i < persist_args_.size(); i++) {
      if (persist_args_[i].AccessTensor(ctx)->shape() !=
          ctx->input(i).shape()) {
        has_dynamic_input_shape = true;
        VLOG(2) << "Detect a dynamic input shape";
        break;
      }
    }
  }

  CgmodeExecutableClosureStore::KeyT key;
  Status compile_succeed;
  if (!is_compiled_ && !has_dynamic_input_shape) {
    compile_succeed = Compile(ctx, key);
  } else if (!has_dynamic_input_shape) {
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
                                cuda_graph_rets));
    compile_succeed = Status::OK();
  } else {
    compile_succeed = errors::Internal("dynamic input shape");
  }

  if (!compile_succeed.ok()) {
    LOG(WARNING) << std::string("CUDA Graph compilation failed because of ")
      + compile_succeed.ToString() + std::string(" fallback to TF execution");
  }

  AllocatorAttributes host_alloc_attrs;
  host_alloc_attrs.set_gpu_compatible(true);
  host_alloc_attrs.set_on_host(true);
  Allocator* cpu_allocator = ctx->device()->GetAllocator(host_alloc_attrs);

  Tensor compilation_key(cpu_allocator, DT_STRING, TensorShape({}));
  compilation_key.flat<tstring>()(0) = key;

  Tensor compilation_successful(cpu_allocator, DT_BOOL, TensorShape({}));
  compilation_successful.flat<bool>()(0) = compile_succeed.ok();

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

  auto gpu_device = dynamic_cast<BaseGPUDevice*>(ctx->device());
  cudaStream_t cu_stream =
      *(static_cast<cudaStream_t*>(gpu_device->GetStream()));

  for (int i = 0; i < ctx->num_inputs() - 1; i++) {
    cudaError_t e_cu_copy = cudaMemcpyAsync(
        cuda_graph_args->at(i).data(), ctx->input(i).data(),
        ctx->input(i).TotalBytes(), cudaMemcpyDefault, cu_stream);
    if (e_cu_copy != cudaSuccess) {
      LOG(WARNING) << std::string("async copy args failed ") + 
                      cudaGetErrorString(e_cu_copy) +
                      std::string(" fallback to TF execution");
      FallbackToTF(ctx, func);
      return;
    }
  }

  cudaError_t e_cu_launch_graph =
      cudaGraphLaunch(closure.cuda_graph_exec(), cu_stream);
  if (e_cu_launch_graph != cudaSuccess) {
    LOG(WARNING) << std::string("CUDA Graph execution failed ") +
                    cudaGetErrorString(e_cu_launch_graph) +
                    std::string(" fallback to TF execution");
    FallbackToTF(ctx, func);
    return;
  }

  for (int i = 0; i < ctx->num_outputs(); i++) {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            i, cuda_graph_rets->at(i).shape(), &output));
    cudaMemcpyAsync(output->data(), cuda_graph_rets->at(i).data(),
                    cuda_graph_rets->at(i).TotalBytes(), cudaMemcpyDefault,
                    cu_stream);
  }
  cudaStreamSynchronize(cu_stream);
}

void CgmodeRunOp::FallbackToTF(OpKernelContext* ctx, const NameAttrList func) {
  VLOG(1) << "Fallback to run TF subgraph when CUDA Graph execution fails";
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
