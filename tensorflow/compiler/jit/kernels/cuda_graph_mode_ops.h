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

#ifndef TENSORFLOW_COMPILER_JIT_KERNELS_CUDA_GRAPH_MODE_OPS_H_
#define TENSORFLOW_COMPILER_JIT_KERNELS_CUDA_GRAPH_MODE_OPS_H_

#include <cuda_runtime.h>
#include <atomic>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"

namespace tensorflow {

class CgmodeCompileOp : public OpKernel {
 public:
  explicit CgmodeCompileOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 private:
  void Compile(OpKernelContext* ctx, tstring& compiled_key);

  const NameAttrList function_;

  Env* env_;

  FunctionLibraryRuntime* flib_;
  std::vector<PersistentTensor> persist_args_;
  std::vector<PersistentTensor> persist_rets_;
  cudaGraph_t cuda_graph_obj_ = nullptr;
  cudaGraphExec_t cuda_graph_exec_ = nullptr;

  bool is_compiled_;
  bool is_out_shape_static_;
  static mutex compile_mu_;
};

class CgmodeRunOp : public OpKernel {
 public:
  explicit CgmodeRunOp(OpKernelConstruction* ctx);
  ~CgmodeRunOp();

  void Compute(OpKernelContext* ctx) override;

 private:
  void FallbackToTF(OpKernelContext* ctx, const NameAttrList func);
  Env* env_;
  FunctionLibraryRuntime* flib_;
  cudaGraphExec_t cuda_graph_exec_ = nullptr;
  mutex mutex_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_KERNELS_CUDA_GRAPH_MODE_OPS_H_
#endif  // GOOGLE_CUDA
