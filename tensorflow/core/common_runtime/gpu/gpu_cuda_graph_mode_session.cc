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
#include "tensorflow/core/common_runtime/gpu/gpu_cuda_graph_mode_session.h"

namespace tensorflow {

typedef std::vector<std::pair<string, Tensor>> NamedTensorList;

class CudaGraphModeSessionFactory : public SessionFactory {
 public:
  CudaGraphModeSessionFactory() {}

  bool AcceptsOptions(const SessionOptions& options) override {
    return options.target == CUDA_GRAPH_MODE_TARGET_NAME;
  }

  Status NewSession(const SessionOptions& options,
                    Session** out_session) override {
    std::vector<std::unique_ptr<Device>> devices;
    TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices));
    CudaGraphModeSession* session =
        new CudaGraphModeSession(options, new DeviceMgr(std::move(devices)), this);
    {
      mutex_lock l(sessions_lock_);
      sessions_.emplace_back(session);
    }
    *out_session = session;
    return Status::OK();
  }

  void Deregister(const CudaGraphModeSession* session) {
    mutex_lock l(sessions_lock_);
    sessions_.erase(std::remove(sessions_.begin(), sessions_.end(), session),
                    sessions_.end());
  }

  Status NewSessionGroup(const SessionOptions& options,
                         SessionGroup** out_session_group,
                         int session_num = 1) override {
    return errors::Internal(
        "NewSessionGroup method not implemented in GraphModeSessionFactory.");
  }

 private:
  mutex sessions_lock_;
  std::vector<CudaGraphModeSession*> sessions_ GUARDED_BY(sessions_lock_);
};

class CudaGraphModeSessionRegistrar {
 public:
  CudaGraphModeSessionRegistrar() {
    SessionFactory::Register("CUDA_GRAPH_MODE_SESSION",
                             new CudaGraphModeSessionFactory());
  }
};

static CudaGraphModeSessionRegistrar registrar;

CudaGraphModeSession::CudaGraphModeSession(const SessionOptions& options,
                                   const DeviceMgr* device_mgr,
                                   CudaGraphModeSessionFactory* const factory)
    : device_mgr_(device_mgr), factory_(factory) {
  batch_size_ = options.config.cuda_graph_mode_options().batch_size();
  allow_fallback_ = options.config.cuda_graph_mode_options().allow_fallback();
  output_names_ = std::vector<std::string>(
      options.config.cuda_graph_mode_options().output_names().begin(),
      options.config.cuda_graph_mode_options().output_names().end());
  SessionOptions new_options = options;
  new_options.target = "";
  new_options.config.mutable_gpu_options()->set_allow_growth(false);
  session_.reset(dynamic_cast<DirectSession*>(NewSession(new_options)));
  assert(session_);
}

void CudaGraphModeSession::Clean() {
  if (ctx_.sess_feed_and_fetch()) {
    session_->ReleaseCallable(ctx_.sess_feed_and_fetch());
    ctx_.reset_sess_feed_and_fetch();
  }
  ctx_.Clean();
  session_cleaned_ = true;
}

Status CudaGraphModeSession::Close() {
  session_->Close();
  {
    mutex_lock l(closed_lock_);
    if (closed_) return Status::OK();
    closed_ = true;
  }
  if (factory_ != nullptr) factory_->Deregister(this);
  if (!session_cleaned_) {
    Clean();
  }
  return Status::OK();
}

CudaGraphModeSession::~CudaGraphModeSession() {
  if (!closed_) Close().IgnoreError();
  if (device_mgr_) {
    for (auto d : device_mgr_->ListDevices()) {
      d->ClearResourceMgr();
    }
  }
}

Status CudaGraphModeSession::Create(const GraphDef& graph) {
  TF_RETURN_IF_ERROR(ctx_.InitDevices(device_mgr_.get(), session_.get()));
  TF_RETURN_IF_ERROR(ctx_.InitInputTensors(graph, batch_size_));
  TF_RETURN_IF_ERROR(ctx_.InitOutputTensors(graph, output_names_));
  TF_RETURN_IF_ERROR(ctx_.BuildGraph(graph, session_.get()));
  TF_RETURN_IF_ERROR(ctx_.InitCallableOptions());
  TF_RETURN_IF_ERROR(ctx_.MakeCallable(session_.get()));
  if (!ctx_.has_invalid_graph()) {
    TF_RETURN_IF_ERROR(ctx_.CaptureCudaGraph(session_.get()));
  }
  session_cleaned_ = false;
  if (!allow_fallback_) {
    ctx_.disable_fallback();
  }
  return Status::OK();
}

Status CudaGraphModeSession::Run(const ::tensorflow::RunOptions& run_options,
                             const NamedTensorList& inputs,
                             const std::vector<string>& output_names,
                             const std::vector<string>& target_nodes,
                             std::vector<Tensor>* outputs,
                             RunMetadata* run_metadata) {
  TF_CHECK_CUDA_CALL(cudaStreamSynchronize(ctx_.stream()),
                     "synchronize capturing stream before run failed");
  return Run(inputs, output_names, target_nodes, outputs);
}

Status CudaGraphModeSession::Run(const NamedTensorList& inputs,
                             const std::vector<string>& output_names,
                             const std::vector<string>& target_nodes,
                             std::vector<Tensor>* outputs) {
  if (ctx_.has_invalid_graph() && ctx_.enable_fallback()) {
    VLOG(2) << "Run TF graph instead of graph mode";
    return ctx_.RunTFGraph(inputs, output_names, outputs, session_.get());
  }
  Status s = ctx_.RunCudaGraph(inputs, output_names, outputs);
  if (!s.ok()) {
    // fallback to normal run
    if (ctx_.enable_fallback()) {
      if (s.code() != tensorflow::error::OUT_OF_RANGE) {
        VLOG(2) << "CUDA Graph mode run fallback because: " << s.ToString();
      }
      return ctx_.RunTFGraph(inputs, output_names, outputs, session_.get());
    }
  }
  return s;
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
