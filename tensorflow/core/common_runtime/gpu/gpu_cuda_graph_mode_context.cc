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
#include "tensorflow/core/common_runtime/gpu/gpu_cuda_graph_mode_context.h"

#include <cstdlib>
#include <cstring>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"

namespace tensorflow {

CudaGraphModeContext::CudaGraphModeContext() {}
CudaGraphModeContext::~CudaGraphModeContext() {
  if (!ctx_cleaned_) {
    Clean();
  }
}

void CudaGraphModeContext::Clean() {
  if (cuda_graph_instantiated_) {
    TF_CHECK_CUDA_CALL(cudaGraphExecDestroy(cuda_graph_exec_),
                       "destroy cuda graph exec failed");
    cuda_graph_instantiated_ = false;
  }
  if (cuda_graph_created_) {
    TF_CHECK_CUDA_CALL(cudaGraphDestroy(cuda_graph_),
                       "destroy cuda graph failed");
    cuda_graph_created_ = false;
  }
  if (cuda_graph_compute_stream_created_) {
    TF_CHECK_CUDA_CALL(cudaStreamDestroy(compute_stream_),
                       "destroy cuda graph stream failed");
    cuda_graph_compute_stream_created_ = false;
  }
  if (cublas_workspace_) {
    TF_CHECK_CUDA_CALL(cudaFree(cublas_workspace_),
                       "free cuBLAS workspace failed");
  }
  ctx_cleaned_ = true;
}

Status CudaGraphModeContext::InitDevices(const DeviceMgr* device_mgr,
                                         Session* sess) {
  sess->LocalDeviceManager(&device_mgr);
  std::vector<Device*> devices = device_mgr->ListDevices();
  bool find_gpu_dev = false;
  for (auto* d : devices) {
    if (!find_gpu_dev &&
        (d->device_type() == "GPU" || d->device_type() == "gpu")) {
      auto gpu = dynamic_cast<BaseGPUDevice*>(d);
      if (!gpu) {
        return errors::Internal("cast gpu device failed");
      }
      tf_stream_ = gpu->GetDefaultTFStream();
      stream_ = *(static_cast<cudaStream_t*>(gpu->GetStream()));
      device_allocator_ = reinterpret_cast<CudaGraphGPUBFCAllocator*>(
          gpu->GetAllocator(AllocatorAttributes()));
      host_allocator_ = GPUProcessState::singleton()->GetGpuHostAllocator(0);
      gpu_device_name_ = gpu->name();
      cur_gpu_device_ = gpu;
      find_gpu_dev = true;
    } else {
      cpu_device_name_ = d->name();
    }
  }
  if (!device_allocator_) {
    return errors::Internal("Fail to obtain gpu device allocator");
  }
  if (!host_allocator_) {
    return errors::Internal("Fail to obtain gpu host allocator");
  }
  return Status::OK();
}

Status CudaGraphModeContext::InitInputTensors(const GraphDef& graph_def,
                                              const int batch_size) {
  for (auto& node : graph_def.node()) {
    if (node.op() == "Placeholder") {
      PartialTensorShape shape = node.attr().at("shape").shape();
      DataType dtype = node.attr().at("dtype").type();
      if (shape.unknown_rank()) {
        return errors::Internal("Unknown shape of node: ", node.name(),
                                " is not allowed in CUDA Graph");
      }
      inputs_from_def_[node.name()] = std::make_pair(shape, dtype);
      input_node_names_.emplace_back(node.name());
      if (node.device().find("GPU") != std::string::npos) {
        input_node_devices_[node.name()] = gpu_device_name_;
      } else {
        input_node_devices_[node.name()] = cpu_device_name_;
      }
      inputs_from_def_idx_[node.name()] = input_node_names_.size() - 1;
    }
  }
  input_tensors_.clear();
  input_tensors_.resize(inputs_from_def_.size());
  for (auto& it : inputs_from_def_) {
    auto dtype = it.second.second;
    TensorShape new_shape;
    if (it.second.first.dim_size(0) < 0) {
      it.second.first.set_dim(0, batch_size);
    }
    if (!it.second.first.AsTensorShape(&new_shape)) {
      return errors::Internal("part shape convert to tensor shape failed.");
    }
    if (input_node_devices_[it.first] == gpu_device_name_) {
      auto input = Tensor(device_allocator_, dtype, new_shape);
      input_tensors_[inputs_from_def_idx_[it.first]] = input;
    } else if (input_node_devices_[it.first] == cpu_device_name_) {
      auto input = Tensor(host_allocator_, dtype, new_shape);
      input_tensors_[inputs_from_def_idx_[it.first]] = input;
    } else {
      return errors::Internal("device type", it.first,
                              " does not own an allocator");
    }
  }
  return Status::OK();
}

Status CudaGraphModeContext::InitOutputTensors(
    const GraphDef& graph_def, const std::vector<std::string>& output_names) {
  for (auto& node : graph_def.node()) {
    auto it = std::find(output_names.begin(), output_names.end(), node.name());
    if (it != output_names.end()) {
      output_node_names_.emplace_back(node.name());
      if (node.device().find("GPU") != std::string::npos) {
        output_node_devices_[node.name()] = gpu_device_name_;
      } else {
        output_node_devices_[node.name()] = cpu_device_name_;
      }
    }
  }
  if (output_node_names_.size() != output_names.size()) {
    return errors::Internal(
        "Inconsist number of output names: ", output_node_names_.size(),
        " and ", output_names.size());
  }
  outputs_idx_.clear();
  for (size_t i = 0; i < output_node_names_.size(); ++i) {
    outputs_idx_[output_node_names_[i]] = i;
  }
  return Status::OK();
}

Status CudaGraphModeContext::BuildGraph(const GraphDef& graph_def,
                                        Session* sess) {
  TF_RETURN_IF_ERROR(sess->Create(graph_def));
  return Status::OK();
}

Status CudaGraphModeContext::InitCallableOptions() {
  for (int i = 0; i < input_node_names_.size(); i++) {
    callable_opts_.add_feed(input_node_names_[i]);
    callable_opts_.mutable_feed_devices()->insert(
        {input_node_names_[i], input_node_devices_[input_node_names_[i]]});
  }
  for (int i = 0; i < output_node_names_.size(); i++) {
    callable_opts_.add_fetch(output_node_names_[i]);
    callable_opts_.mutable_fetch_devices()->insert(
        {output_node_names_[i], output_node_devices_[output_node_names_[i]]});
  }
  // on cuda graph capture mode, sync with cuda call is not supported.
  callable_opts_.set_fetch_skip_sync(true);
  return Status::OK();
}

Status CudaGraphModeContext::MakeCallable(Session* sess) {
  // create a handle
  {
    tf_shared_lock lock(DirectSession::capture_run_mu_);
    TF_RETURN_IF_ERROR(
        sess->MakeCallable(callable_opts_, &sess_feed_and_fetch_));
  }
  has_invalid_graph_ = reinterpret_cast<DirectSession*>(sess)
                           ->options()
                           .config.cuda_graph_mode_options()
                           .has_invalid_graph();
  return Status::OK();
}

Status CudaGraphModeContext::CaptureCudaGraph(Session* sess) {
  // init run (bfcallocator allocates mem chunks)
  {
    tf_shared_lock lock(DirectSession::capture_run_mu_);
    if (cur_gpu_device_) {
      cur_gpu_device_->SetSingleStreamMode();
    }
    device_allocator_->EnableCudaGraphModeMem();
    TF_RETURN_IF_ERROR(sess->RunCallable(sess_feed_and_fetch_, input_tensors_,
                                         &(output_tensors_), nullptr));
    device_allocator_->DisableCudaGraphModeMem();
  }
  {
    tf_shared_lock lock(DirectSession::capture_run_mu_);
    DirectSession* direct_sess = reinterpret_cast<DirectSession*>(sess);
    bool sync_on_finish = direct_sess->sync_on_finish();
    direct_sess->set_sync_on_finish(false);

    // for disabling stream event polling
    stream_executor::gpu::GpuContext* gpu_ctx =
        reinterpret_cast<stream_executor::gpu::GpuContext*>(
            tf_stream_->parent()->implementation()->GpuContextHack());
    if (cur_gpu_device_) {
      gpu_ctx->enable_single_stream_mode();
    }

    // for cublas
    int num_bytes = 1 << 22;
    void* workspace = 0;
    TF_CHECK_CUDA_CALL(cudaMalloc(&workspace, num_bytes),
                       "malloc buffer for cublas workspace failed.");
    cublas_workspace_ = workspace;
    // TODO: provide set blas workspace
    // tf_stream_->SetBlasWorkspace(workspace, num_bytes);
    tf_stream_->parent()->AsBlas()->SetWorkspace(tf_stream_, workspace,
                                                 num_bytes);
    VLOG(2) << "cuda graph capture : set workspace for cublas";

    // start capturing cuda graph on stream
    TF_CHECK_CUDA_CALL(
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal),
        "cuda graph begin capture failed.");
    Status s = sess->RunCallable(sess_feed_and_fetch_, input_tensors_,
                                 &(output_tensors_), nullptr);
    if (!s.ok()) {
      return errors::Internal("cuda graph capture run failed:", s.ToString());
    }

    // finish capturing cuda graph on stream
    TF_CHECK_CUDA_CALL(cudaStreamEndCapture(stream_, &(cuda_graph_)),
                       "cuda graph end capture failed");
    cuda_graph_created_ = true;
    // create excutable instance of cuda graph
    TF_CHECK_CUDA_CALL(
        cudaGraphInstantiate(&(cuda_graph_exec_), cuda_graph_, NULL, NULL, 0),
        "cuda graph create execute instance failed.");
    cuda_graph_instantiated_ = true;
    TF_CHECK_CUDA_CALL(cudaStreamCreate(&(compute_stream_)),
                       "cuda stream create error");
    cuda_graph_compute_stream_created_ = true;
    direct_sess->set_sync_on_finish(sync_on_finish);
    // enabling event polling
    if (cur_gpu_device_) {
      gpu_ctx->disable_single_stream_mode();
    }

    if (cur_gpu_device_) {
      cur_gpu_device_->ResetStreamMode();
    }
  }
  return Status::OK();
}

bool CudaGraphModeContext::IsTensorOnDevice(const Tensor* t) {
  if (t->TotalBytes() == 0) return false;
  cudaPointerAttributes attributes;
  cudaError_t err =
      cudaPointerGetAttributes(&attributes, t->tensor_data().data());
  if (err == cudaErrorInvalidValue) return false;
  CHECK_EQ(cudaSuccess, err) << cudaGetErrorString(err);
  return (attributes.type == cudaMemoryTypeDevice);
}

Status CudaGraphModeContext::SyncData(const Tensor* from, Tensor* to,
                                      size_t size, cudaStream_t stream) {
  if (from->dtype() != to->dtype()) {
    return errors::Internal("cuda input type not consist with input");
  }
  auto fromShape = from->shape();
  auto toShape = to->shape();
  if (size > from->TotalBytes() || size > to->TotalBytes()) {
    return errors::Internal("copy size large than to tensor size");
  }
  cudaMemcpyKind copyKind = cudaMemcpyDefault;
  TF_CHECK_CUDA_CALL(
      cudaMemcpyAsync(to->data(), from->data(), size, copyKind, stream),
      "async copy output failed");
  return Status::OK();
}

Status CudaGraphModeContext::CheckShape(const PartialTensorShape& fromShape,
                                        const TensorShape& toShape) {
  if (fromShape.dims() != toShape.dims()) {
    return errors::Internal("cuda input dim size not consist with input:",
                            fromShape.dims(), ":", toShape.dims());
  }
  for (int d = 0; d < toShape.dims(); d++) {
    if (fromShape.dim_size(d) != toShape.dim_size(d)) {
      return errors::Internal("cuda input dim not consist with input at ", d,
                              " which is ", fromShape.dim_size(d), " and ",
                              toShape.dim_size(d));
    }
  }
  return Status::OK();
}

Status CudaGraphModeContext::CheckInputsInfo(
    const std::vector<std::pair<string, Tensor> >& inputs) {
  std::vector<bool> visited(inputs_from_def_idx_.size(), false);
  for (auto& input : inputs) {
    const Tensor& input_tensor = input.second;
    auto input_info_itr = inputs_from_def_.find(input.first);
    if (input_info_itr == inputs_from_def_.end()) {
      return errors::Internal("do not need input name: ", input.first);
    }
    auto idx_itr = inputs_from_def_idx_.find(input.first);
    if (idx_itr == inputs_from_def_idx_.end() ||
        idx_itr->second >= visited.size()) {
      return errors::Internal("find input idx is not right: ", input.first);
    }
    visited[idx_itr->second] = true;
    if (input_tensor.dtype() != input_info_itr->second.second) {
      return errors::Internal("data type mismatch for ", input.first);
    }
    const PartialTensorShape& capture_shape = input_info_itr->second.first;
    TF_RETURN_IF_ERROR(CheckShape(capture_shape, input_tensor.shape()));
    int capture_dim_0 = capture_shape.dim_size(0);
    int input_dim_0 = input_tensor.shape().dim_size(0);
    if (input_dim_0 < 0) {
      return errors::Internal("input dim 0 size < 0");
    }
    if (capture_dim_0 > 0 && input_dim_0 != capture_dim_0) {
      return errors::Internal("capture dim 0 not consist with input");
    }
  }
  for (size_t i = 0; i < visited.size(); ++i) {
    if (!visited[i]) {
      return errors::Internal("lack inputs num:", i);
    }
  }
  return Status::OK();
}

Status CudaGraphModeContext::InitCallableInputs(
    const std::vector<std::pair<string, Tensor> >& inputs,
    std::vector<Tensor>& callable_inputs) {
  if (inputs.size() != inputs_from_def_idx_.size()) {
    return errors::Internal("input size is not equal to require.");
  }
  callable_inputs.clear();
  callable_inputs.resize(inputs.size());

  for (int i = 0; i < inputs.size(); ++i) {
    auto it = inputs_from_def_idx_.find(inputs[i].first);
    if (it == inputs_from_def_idx_.end()) {
      return errors::Internal("can not find input idx: ", inputs[i].first);
    }
    callable_inputs[it->second] = inputs[i].second;
  }
  TF_CHECK_CUDA_CALL(cudaStreamSynchronize(stream_),
                     "synchronize with capturing stream failed when init");
  return Status::OK();
}

Status CudaGraphModeContext::RunTFGraph(
    const std::vector<std::pair<string, Tensor> >& inputs,
    const std::vector<std::string>& output_node_names,
    std::vector<Tensor>* outputs, Session* sess) {
  std::vector<Tensor> callable_inputs;
  TF_RETURN_IF_ERROR(InitCallableInputs(inputs, callable_inputs));
  std::vector<Tensor> tmp_outputs;
  tf_shared_lock lock(DirectSession::capture_run_mu_);
  TF_RETURN_IF_ERROR(sess->RunCallable(sess_feed_and_fetch_, callable_inputs,
                                       &tmp_outputs, nullptr));
  for (auto& tensor : tmp_outputs) {
    outputs->emplace_back(tensor);
  }
  TF_CHECK_CUDA_CALL(cudaStreamSynchronize(stream_),
                     "synchronize capturing stream after run failed");
  return Status::OK();
}

Status CudaGraphModeContext::RunCudaGraph(
    const std::vector<std::pair<string, Tensor> >& inputs,
    const std::vector<std::string>& output_node_names,
    std::vector<Tensor>* outputs) {
  TF_CHECK_CUDA_CALL(cudaStreamSynchronize(stream_),
                     "synchronize capturing stream before run failed");
  TF_RETURN_IF_ERROR(CheckInputsInfo(inputs));
  // copy and validate data from inputs to cuda graph's inputs
  for (auto& input : inputs) {
    Tensor t = input.second;
    auto it = inputs_from_def_idx_.find(input.first);
    if (it == inputs_from_def_idx_.end()) {
      return errors::Internal("input name not consisit ", input.first);
    } else {
      assert(it->second < input_tensors_.size());
      Tensor dst = input_tensors_[it->second];
      auto num_bytes = dst.TotalBytes();
      TF_RETURN_IF_ERROR(SyncData(&t, &dst, t.TotalBytes(), compute_stream_));
    }
  }
  TF_CHECK_CUDA_CALL(cudaStreamSynchronize(compute_stream_),
                     "synchronize with compute stream failed when init");
  TF_CHECK_CUDA_CALL(cudaGraphLaunch(cuda_graph_exec_, compute_stream_),
                     "cuda graph launch failed");
  // copy data back from cuda graph's outputs to outputs
  outputs->clear();
  for (auto& name : output_node_names_) {
    auto it = outputs_idx_.find(name);
    if (it == outputs_idx_.end()) {
      return errors::Internal("not found output: ", name);
    }
    assert(it->second < output_tensors_.size());
    Tensor cuda_output = output_tensors_[it->second];
    outputs->emplace_back(cuda_output);
  }
  TF_CHECK_CUDA_CALL(cudaStreamSynchronize(compute_stream_),
                     "synchronize cuda graph stream on copy output failed");
  return Status::OK();
}
}  // namespace tensorflow
#endif  // GOOGLE_CUDA
