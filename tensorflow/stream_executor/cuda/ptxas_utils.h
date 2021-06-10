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

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_PTXAS_UTILS_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_PTXAS_UTILS_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace stream_executor {
namespace cuda {

// Compilation options for compiling ptxas.
struct PtxCompilationOptions {
  bool disable_ptxas_optimizations;

  // CUDA directory which would be searched first.
  std::string preferred_cuda_dir;

  std::vector<std::string> extra_flags;

  explicit PtxCompilationOptions(bool disable_ptxas_optimizations = false,
                                 absl::string_view preferred_cuda_dir = "",
				 absl::Span<const std::string> extra_flags = {})
      : disable_ptxas_optimizations(disable_ptxas_optimizations),
        preferred_cuda_dir(preferred_cuda_dir),
        extra_flags(extra_flags.begin(), extra_flags.end()) {}

  using PtxOptionsTuple =
      std::tuple<bool, std::string, std::vector<std::string>>;

  PtxOptionsTuple ToTuple() {
    return std::make_tuple(disable_ptxas_optimizations, preferred_cuda_dir, extra_flags);
  }
};

// Compiles the given PTX string using ptxas and returns the resulting machine
// code (i.e. a cubin) as a byte array.
//
// compile_ptx_options is used to query for the CUDA location in case it is
// customized in a passed flag, and for controlling ptxas optimizations.
port::StatusOr<std::vector<uint8>> CompilePtx(int device_ordinal,
                                              const char* ptx_contents,
                                              PtxCompilationOptions options);

// Same as CompilePtx, but caches the result, and returns unowned view of
// the compiled binary.
//
// A copy of the string provided in ptx will be made.
port::StatusOr<absl::Span<const uint8>> CompilePtxOrGetCached(
    int device_ordinal, const char* ptx,
    PtxCompilationOptions compilation_options);

template <typename... Args>
port::StatusOr<std::shared_ptr<TypedKernel<Args...>>> LoadKernelOrGetPtr(
    StreamExecutor* executor, absl::string_view kernel_name,
    absl::string_view ptx, absl::Span<const uint8> cubin_data) {
  using KernelPtrCacheKey =
      std::tuple<CUcontext, absl::string_view, absl::string_view>;

  static tensorflow::mutex kernel_ptr_cache_mutex(
      tensorflow::LINKER_INITIALIZED);
  static auto& kernel_ptr_cache GUARDED_BY(kernel_ptr_cache_mutex) =
      *new absl::flat_hash_map<KernelPtrCacheKey,
                               std::shared_ptr<TypedKernel<Args...>>>();
  CUcontext current_context = CurrentContextOrDie();

  KernelPtrCacheKey kernel_ptr_cache_key{current_context, kernel_name, ptx};

  tensorflow::mutex_lock lock(kernel_ptr_cache_mutex);

  auto it = kernel_ptr_cache.find(kernel_ptr_cache_key);
  if (it == kernel_ptr_cache.end()) {
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<TypedKernel<Args...>> loaded,
        executor->CreateTypedKernel<Args...>(kernel_name, ptx, cubin_data));
    it =
        kernel_ptr_cache.emplace(kernel_ptr_cache_key, std::move(loaded)).first;
  }

  CHECK(it != kernel_ptr_cache.end());
  return it->second;
}

char* GetTFExtraPTXOptions();

}  // namespace cuda
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_PTXAS_UTILS_H_
