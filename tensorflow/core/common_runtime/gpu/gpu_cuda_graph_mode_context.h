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
#ifndef TENSORFLOW_CORE_FRAMEWORK_GPU_CUDA_GRAPH_MODE_CONTEXT_H_
#define TENSORFLOW_CORE_FRAMEWORK_GPU_CUDA_GRAPH_MODE_CONTEXT_H_

#include <cuda_runtime.h>
#include <atomic>
#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cuda_graph_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/public/session.h"

#define TF_CHECK_CUDA_CALL(x, error_msg)                                       \
  do {                                                                         \
    cudaError_t retval = (x);                                                  \
    if (retval != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("Runtime error: ") +                \
                               (cudaGetErrorString(retval)) + " " + __FILE__ + \
                               ":" + std::to_string(__LINE__) + ":" +          \
                               std::string(error_msg) + " \n");                \
    }                                                                          \
  } while (0)

namespace Eigen {
struct ThreadPoolDevice;
struct GpuDevice;
struct SyclDevice;
}  // end namespace Eigen

namespace tensorflow {
// add declaration of other classes
class CudaGraphModeContext {
  typedef int64 CallableHandle;

 public:
  CudaGraphModeContext();
  ~CudaGraphModeContext();

  void Create(const GraphDef& graph);
  void Clean();
  Status Run();
  Status InitDevices(const DeviceMgr* device_mgr, Session* sess);
  Status InitCallableOptions();
  Status InitInputTensors(const GraphDef& graph_def, const int batch_size);
  Status InitOutputTensors(const GraphDef& graph_def,
                           const std::vector<std::string>& output_names);
  Status BuildGraph(const GraphDef& graph_def, Session* sess);
  Status MakeCallable(Session* sess);
  Status CaptureCudaGraph(Session* sess);
  Status RunCudaGraph(const std::vector<std::pair<string, Tensor>>& inputs,
                      const std::vector<std::string>& output_node_names,
                      std::vector<Tensor>* outputs);
  Status RunTFGraph(const std::vector<std::pair<string, Tensor>>& inputs,
                    const std::vector<std::string>& output_node_names,
                    std::vector<Tensor>* outputs, Session* sess);

  CudaGraphGPUBFCAllocator* device_allocator() { return device_allocator_; }
  Allocator* host_allocator() { return host_allocator_; }
  cudaStream_t stream() { return stream_; }
  cudaStream_t compute_stream() { return compute_stream_; }
  se::Stream* tf_stream() { return tf_stream_; }
  CallableHandle sess_feed_and_fetch() { return sess_feed_and_fetch_; }
  void reset_sess_feed_and_fetch() { sess_feed_and_fetch_ = 0; }
  bool enable_fallback() { return enable_fallback_; }
  void disable_fallback() { enable_fallback_ = false; }
  bool has_invalid_graph() { return has_invalid_graph_; }

 private:
  bool IsTensorOnDevice(const Tensor* t);
  Status SyncData(const Tensor* from, Tensor* to, size_t size,
                  cudaStream_t stream);
  Status CheckShape(const PartialTensorShape& fromShape,
                    const TensorShape& toShape);
  Status CheckInputsInfo(const std::vector<std::pair<string, Tensor>>& inputs);
  Status InitCallableInputs(
      const std::vector<std::pair<string, Tensor>>& inputs,
      std::vector<Tensor>& callable_inputs);

  cudaGraph_t cuda_graph_;
  cudaGraphExec_t cuda_graph_exec_;
  std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;

  std::map<std::string, std::pair<PartialTensorShape, DataType>>
      inputs_from_def_;
  std::map<std::string, int> inputs_from_def_idx_;
  std::map<std::string, int> outputs_idx_;

  std::vector<std::string> input_node_names_;
  std::map<std::string, std::string> input_node_devices_;
  std::vector<std::string> output_node_names_;
  std::map<std::string, std::string> output_node_devices_;

  BaseGPUDevice* cur_gpu_device_ = nullptr;
  CallableHandle sess_feed_and_fetch_ = 0;
  se::Stream* tf_stream_ = nullptr;
  cudaStream_t stream_;
  cudaStream_t compute_stream_;
  Allocator* host_allocator_ = nullptr;
  CudaGraphGPUBFCAllocator* device_allocator_ = nullptr;
  CallableOptions callable_opts_;
  std::string gpu_device_name_;
  std::string cpu_device_name_;
  bool ctx_cleaned_ = false;
  bool cuda_graph_created_ = false;
  bool cuda_graph_instantiated_ = false;
  bool cuda_graph_compute_stream_created_ = false;
  bool enable_fallback_ = true;
  bool has_invalid_graph_ = false;
  void* cublas_workspace_ = nullptr;
};
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_FRAMEWORK_GPU_CUDA_GRAPH_MODE_CONTEXT_H_
#endif // GOOGLE_CUDA
