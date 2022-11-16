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

#ifndef TENSORFLOW_CORE_FRAMEWORK_GPU_CUDA_GRAPH_MODE_SESSION_H_
#define TENSORFLOW_CORE_FRAMEWORK_GPU_CUDA_GRAPH_MODE_SESSION_H_

#include <cuda_runtime.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stack>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cuda_graph_mode_context.h"

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

class CudaGraphModeSessionFactory;

class CudaGraphModeSession : public Session {
 public:
  typedef std::vector<std::pair<string, Tensor>> NamedTensorList;
  typedef std::unordered_map<StringPiece, Node*, StringPieceHasher> NameNodeMap;

  CudaGraphModeSession(const SessionOptions& options, const DeviceMgr* device_mgr,
                   CudaGraphModeSessionFactory* factory);

  ~CudaGraphModeSession() override;

  void Clean();

  Status Create(const GraphDef& graph) override;

  Status Run(const NamedTensorList& inputs,
             const std::vector<string>& output_names,
             const std::vector<string>& target_nodes,
             std::vector<Tensor>* outputs) override;

  // NOTE: Experimental and subject to change.
  Status Run(const ::tensorflow::RunOptions& run_options,
             const NamedTensorList& inputs,
             const std::vector<string>& output_names,
             const std::vector<string>& target_nodes,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata) override;

  Allocator* get_gpu_allocator() { return ctx_.device_allocator(); }

  Status Extend(const GraphDef& graph) override {
    return errors::Unimplemented(
        "Extend(const RunOptions& run_options, const GraphDef& graph) is not "
        "supported for this session.");
  }

  Status Close() override;

  Status ListDevices(std::vector<DeviceAttributes>* response) override {
    session_->ListDevices(response);
  }

  DirectSession* session() { return session_.get(); }
  void DisableFallBack() { ctx_.disable_fallback(); }

 private:
  CudaGraphModeContext ctx_;
  std::unique_ptr<DirectSession> session_;
  const std::unique_ptr<const DeviceMgr> device_mgr_;
  bool session_cleaned_ = false;
  mutex closed_lock_;
  bool closed_ GUARDED_BY(closed_lock_) = false;
  int batch_size_;
  bool allow_fallback_;
  std::vector<std::string> output_names_;
  CudaGraphModeSessionFactory* const factory_;  // not owned
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_GPU_CUDA_GRAPH_MODE_SESSION_H_
#endif  // GOOGLE_CUDA
