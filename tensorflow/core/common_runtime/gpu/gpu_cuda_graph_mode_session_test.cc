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

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cuda/include/cuda.h"
#include "cuda/include/cuda_runtime_api.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

class CudaGraphModeSessionTest : public ::testing::Test {
 public:
  void init(const SessionOptions& options, std::unique_ptr<Session>& session) {
    SessionOptions new_options = options;
    new_options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
    new_options.config.mutable_gpu_options()->set_allow_growth(false);
    new_options.config.mutable_gpu_options()
        ->set_per_process_gpu_memory_fraction(0.9);
    new_options.config.mutable_gpu_options()->set_cuda_graph_mode_compatible(true);
    new_options.target = CUDA_GRAPH_MODE_TARGET_NAME;
    session.reset(NewSession(new_options));
  }

  Tensor copyFromGPU(const Tensor& t) {
    Tensor cpu_tensor(t.dtype(), t.shape());
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(cpu_tensor.data(), t.data(), t.TotalBytes(),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return cpu_tensor;
  }

  Tensor copyToGPU(const Tensor& t, Allocator* gpu_allocator_) {
    Tensor gpu_tensor(gpu_allocator_, t.dtype(), t.shape());
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(gpu_tensor.data(), t.data(), t.TotalBytes(),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return gpu_tensor;
  }
};

void TFRun(Session* sess, int infer_num,
           std::vector<std::pair<std::string, Tensor>>* inputs,
           std::vector<std::string>* output_names,
           std::vector<Tensor>* output_tensors) {
  for (int i = 0; i < infer_num; i++) {
    TF_CHECK_OK(sess->Run(*inputs, *output_names, {}, output_tensors));
  }
}

void CudaGraphModeRun(Session* sess, int infer_num,
                  std::vector<std::pair<std::string, Tensor>>* inputs,
                  std::vector<std::string>* output_names,
                  std::vector<Tensor>* output_tensors) {
  for (int i = 0; i < infer_num; i++) {
    TF_CHECK_OK(sess->Run(*inputs, *output_names, {}, output_tensors));
  }
}

void RandomInitialize(Tensor& t) {
  int num_elements = t.NumElements();
  if (t.dtype() == DT_FLOAT) {
    float* data = t.flat<float>().data();
    for (int i = 0; i < num_elements; i++) {
      float value1 = static_cast<float>(rand() % 101 - 50) / 100.0f;
      data[i] = value1;
    }
  } else if (t.dtype() == DT_INT32) {
    int* data = t.flat<int>().data();
    for (int i = 0; i < num_elements; i++) {
      int value = static_cast<int>(rand() % 10);
      data[i] = value;
    }
  } else if (t.dtype() == DT_INT64) {
    int64* data = t.flat<int64>().data();
    for (int i = 0; i < num_elements; i++) {
      int64 value = static_cast<int64>(rand() % 10);
      data[i] = value;
    }
  } else {
    VLOG(1) << t.dtype();
    VLOG(1) << "Random init: unsupported data type.";
  }
}

// y = tf.square(x)
GraphDef CreateGraphForYEqualsXSquared() {
  GraphDef graph_def;
  const char* text_proto = R"EOF(
node {
  name: "x"
  op: "Placeholder"
  device: "/device:GPU:0"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr { key: "shape" value { shape { dim { size: -1 } } } }
}
node {
  name: "y"
  op: "Square"
  device: "/device:GPU:0"
  input: "x"
  attr { key: "T" value { type: DT_FLOAT } }
}
versions {
  producer: 26
}
  )EOF";

  QCHECK(protobuf::TextFormat::ParseFromString(text_proto, &graph_def));
  return graph_def;
}

// result1 = x + y
// result2 = result1 * z
GraphDef CreateGraphForXPlusYThenMultiplyByZ() {
  GraphDef graph_def;
  const char* text_proto = R"EOF(
node {
  name: "x"
  op: "Placeholder"
  device: "/device:CPU:0"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr { key: "shape" value { shape { dim { size: -1 } } } }
}
node {
  name: "y"
  op: "Placeholder"
  device: "/device:CPU:0"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr { key: "shape" value { shape { dim { size: -1 } } } }
}
node {
  name: "result1"
  op: "Add"
  input: "x"
  input: "y"
  device: "/device:GPU:0"
  attr { key: "T" value { type: DT_FLOAT } }
}
node {
  name: "const1"
  op: "Const"
  device: "/device:GPU:0"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr { key: "value" value { tensor { dtype: DT_FLOAT tensor_shape { dim { size: 1 } } float_val: 1 } } }
}
node {
  name: "result"
  op: "Mul"
  input: "result1"
  input: "const1"
  device: "/device:GPU:0"
  attr { key: "T" value { type: DT_FLOAT } }
}

node {
  name: "z"
  op: "Placeholder"
  device: "/device:CPU:0"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr { key: "shape" value { shape { dim { size: -1 } } } }
}
node {
  name: "result2"
  op: "Mul"
  input: "result"
  input: "z"
  device: "/device:GPU:0"
  attr { key: "T" value { type: DT_FLOAT } }
}
versions {
  producer: 26
}
  )EOF";

  QCHECK(protobuf::TextFormat::ParseFromString(text_proto, &graph_def));
  return graph_def;
}

GraphDef CreateGraphForYEqualsXSquaredUnknownRank() {
  GraphDef graph_def;
  const char* text_proto = R"EOF(
node {
  name: "x"
  op: "Placeholder"
  device: "/device:GPU:0"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr { key: "shape" value { shape { unknown_rank: true } } }
}
node {
  name: "y"
  op: "Square"
  input: "x"
  device: "/device:GPU:0"
  attr { key: "T" value { type: DT_FLOAT } }
}
versions {
  producer: 26
}
  )EOF";

  QCHECK(protobuf::TextFormat::ParseFromString(text_proto, &graph_def));
  return graph_def;
}

GraphDef CreateGraphForCublasGemm() {
  GraphDef graph_def;
  const char* text_proto = R"EOF(
node {
  name: "x"
  op: "Placeholder"
  device: "/device:GPU:0"
  attr { key: "dtype" value { type: DT_HALF } }
  attr { key: "shape" value { shape {
        dim { size: -1 }
        dim { size: 328 } } } } 
}
node {
  name: "y"
  op: "Placeholder"
  device: "/device:GPU:0"
  attr { key: "dtype" value { type: DT_HALF } }
  attr { key: "shape" value { shape {
        dim { size: 328 }
        dim { size: 256 } } } }
} 
node {
  name: "matmul"
  op: "MatMul"
  input: "x"
  input: "y"
  device: "/device:GPU:0"
  attr { key: "T" value { type: DT_HALF } }
  attr { key: "_output_shapes" value { list { shape {
          dim { size: -1 }
          dim { size: 256 } } } } }
  attr { key: "transpose_a" value { b: false } }
  attr { key: "transpose_b" value { b: false } }
}

versions {
  producer: 26
}
  )EOF";

  QCHECK(protobuf::TextFormat::ParseFromString(text_proto, &graph_def));
  return graph_def;
}

GraphDef CreateGraphForDLRMBenchmark() {
  GraphDef graph_def;
  const char* text_proto = R"EOF(
node {
  name: "emb_input"
  op: "Placeholder"
  device: "/device:CPU:0"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr { key: "shape" value { shape { dim { size: -1 }, dim {size: 64 } } } }
}
node {
  name: "dlrm/add/y"
  op: "Const"
  device: "/device:GPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "dlrm/add/y_1"
  op: "Const"
  device: "/device:GPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "dlrm/add/y_2"
  op: "Const"
  device: "/device:GPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "dlrm/add/y_3"
  op: "Const"
  device: "/device:GPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "dlrm/add/y_4"
  op: "Const"
  device: "/device:GPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "dlrm/add/y_5"
  op: "Const"
  device: "/device:GPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "dlrm/add/y_6"
  op: "Const"
  device: "/device:GPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "dlrm/add/y_7"
  op: "Const"
  device: "/device:GPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "dlrm/add/y_8"
  op: "Const"
  device: "/device:GPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "dlrm/add"
  op: "AddV2"
  input: "emb_input"
  input: "dlrm/add/y"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dlrm/add_1"
  op: "AddV2"
  input: "dlrm/add"
  input: "dlrm/add/y_1"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dlrm/add_2"
  op: "AddV2"
  input: "dlrm/add_1"
  input: "dlrm/add/y_2"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dlrm/add_3"
  op: "AddV2"
  input: "dlrm/add_2"
  input: "dlrm/add/y_3"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dlrm/add_4"
  op: "AddV2"
  input: "dlrm/add_3"
  input: "dlrm/add/y_4"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dlrm/add_5"
  op: "AddV2"
  input: "dlrm/add_4"
  input: "dlrm/add/y_5"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dlrm/add_6"
  op: "AddV2"
  input: "dlrm/add_5"
  input: "dlrm/add/y_6"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dlrm/add_7"
  op: "AddV2"
  input: "dlrm/add_6"
  input: "dlrm/add/y_7"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dlrm/add_8"
  op: "AddV2"
  input: "dlrm/add_7"
  input: "dlrm/add/y_8"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dlrm/Log"
  op: "Log"
  input: "dlrm/add_8"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "dlrm/bot_mlp_0/kernel/Initializer/truncated_normal/mean"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dlrm/bot_mlp_0/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "dlrm/bot_mlp_0/kernel/Initializer/truncated_normal/stddev"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dlrm/bot_mlp_0/kernel"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          },
          dim {
            size: 64
          }
        }
        float_val: 0.07016773521900177
      }
    }
  }
}
node {
  name: "dlrm/bot_mlp_0/kernel/Initializer/truncated_normal"
  op: "Add"
  input: "dlrm/bot_mlp_0/kernel/Initializer/truncated_normal/stddev"
  input: "dlrm/bot_mlp_0/kernel/Initializer/truncated_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dlrm/bot_mlp_0/kernel"
      }
    }
  }
}
node {
  name: "dlrm/bot_mlp_0/kernel"
  op: "VariableV2"
  device: "/device:GPU:0"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dlrm/bot_mlp_0/kernel"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "dlrm/bot_mlp_0/kernel/Assign"
  op: "Assign"
  input: "dlrm/bot_mlp_0/kernel"
  input: "dlrm/bot_mlp_0/kernel/Initializer/truncated_normal"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dlrm/bot_mlp_0/kernel"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "dlrm/bot_mlp_0/kernel/read"
  op: "Identity"
  input: "dlrm/bot_mlp_0/kernel/Assign"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dlrm/bot_mlp_0/kernel"
      }
    }
  }
}
node {
  name: "dlrm/bot_mlp_0/bias/Initializer/random_normal/mean"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dlrm/bot_mlp_0/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "dlrm/bot_mlp_0/bias/Initializer/random_normal/stddev"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dlrm/bot_mlp_0/bias"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.04419417306780815
      }
    }
  }
}
node {
  name: "dlrm/bot_mlp_0/bias/Initializer/random_normal"
  op: "Add"
  input: "dlrm/bot_mlp_0/bias/Initializer/random_normal/stddev"
  input: "dlrm/bot_mlp_0/bias/Initializer/random_normal/mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@dlrm/bot_mlp_0/bias"
      }
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul"
  op: "MatMul"
  input: "dlrm/Log"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_1"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_2"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_1"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_3"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_2"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_4"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_3"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_5"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_4"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_6"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_5"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_7"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_6"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_8"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_7"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_9"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_8"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_10"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_9"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_11"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_10"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_12"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_11"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_13"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_12"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_14"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_13"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_15"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_14"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_16"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_15"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_17"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_16"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_18"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_17"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_19"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_18"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/MatMul_20"
  op: "MatMul"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_19"
  input: "dlrm/bot_mlp_0/kernel/read"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "transpose_a"
    value {
      b: false
    }
  }
  attr {
    key: "transpose_b"
    value {
      b: false
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/BiasAdd"
  op: "BiasAdd"
  input: "dlrm/bot_mlp/bot_mlp_0/MatMul_20"
  input: "dlrm/bot_mlp_0/bias/Initializer/random_normal"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_1"
  op: "BiasAdd"
  input: "dlrm/bot_mlp/bot_mlp_0/BiasAdd"
  input: "dlrm/bot_mlp_0/bias/Initializer/random_normal"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_2"
  op: "BiasAdd"
  input: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_1"
  input: "dlrm/bot_mlp_0/bias/Initializer/random_normal"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_3"
  op: "BiasAdd"
  input: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_2"
  input: "dlrm/bot_mlp_0/bias/Initializer/random_normal"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_4"
  op: "BiasAdd"
  input: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_3"
  input: "dlrm/bot_mlp_0/bias/Initializer/random_normal"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_5"
  op: "BiasAdd"
  input: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_4"
  input: "dlrm/bot_mlp_0/bias/Initializer/random_normal"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_6"
  op: "BiasAdd"
  input: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_5"
  input: "dlrm/bot_mlp_0/bias/Initializer/random_normal"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_7"
  op: "BiasAdd"
  input: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_6"
  input: "dlrm/bot_mlp_0/bias/Initializer/random_normal"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_8"
  op: "BiasAdd"
  input: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_7"
  input: "dlrm/bot_mlp_0/bias/Initializer/random_normal"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_9"
  op: "BiasAdd"
  input: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_8"
  input: "dlrm/bot_mlp_0/bias/Initializer/random_normal"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_10"
  op: "BiasAdd"
  input: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_9"
  input: "dlrm/bot_mlp_0/bias/Initializer/random_normal"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
node {
  name: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_11"
  op: "BiasAdd"
  input: "dlrm/bot_mlp/bot_mlp_0/BiasAdd_10"
  input: "dlrm/bot_mlp_0/bias/Initializer/random_normal"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
versions {
  producer: 26
}
  )EOF";
  QCHECK(protobuf::TextFormat::ParseFromString(text_proto, &graph_def));
  return graph_def;
}

void ModifyType(GraphDef& graph, DataType type) {
  for (size_t i = 0; i < graph.node_size(); ++i) {
    auto* node = graph.mutable_node(i);
    if (node->mutable_attr()->count("T") > 0) {
      (*node->mutable_attr())["T"].set_type(type);
    }
    if (node->mutable_attr()->count("dtype") > 0) {
      (*node->mutable_attr())["dtype"].set_type(type);
    }
  }
}

void ModifyNodeDevice(GraphDef& graph, const std::string& device) {
  for (size_t i = 0; i < graph.node_size(); ++i) {
    auto* node = graph.mutable_node(i);
    node->set_device(device);
  }
}

TEST_F(CudaGraphModeSessionTest, TestCublas) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  std::unique_ptr<Session> cuda_graph_mode_session;
  SessionOptions options;
  options.config.mutable_cuda_graph_mode_options()->set_batch_size(3);
  options.config.mutable_cuda_graph_mode_options()->set_allow_fallback(true);
  options.config.mutable_cuda_graph_mode_options()->add_output_names("matmul");
  init(options, cuda_graph_mode_session);
  TF_CHECK_OK(cuda_graph_mode_session->Create(CreateGraphForCublasGemm()));
  cuda_graph_mode_session.reset();
  for (int i = 0; i < 10; ++i) {
    usleep(1000);
    std::unique_ptr<Session> new_cuda_graph_mode_session;
    init(options, new_cuda_graph_mode_session);
    TF_CHECK_OK(new_cuda_graph_mode_session->Create(CreateGraphForCublasGemm()));
    new_cuda_graph_mode_session.reset();
  }
  unsetenv("CUDA_VISIBLE_DEVICES");
}

TEST_F(CudaGraphModeSessionTest, TestSimple) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  std::unique_ptr<Session> cuda_graph_mode_session;
  SessionOptions options;
  options.config.mutable_cuda_graph_mode_options()->set_batch_size(3);
  options.config.mutable_cuda_graph_mode_options()->set_allow_fallback(true);
  options.config.mutable_cuda_graph_mode_options()->add_output_names("y");
  init(options, cuda_graph_mode_session);
  TF_CHECK_OK(cuda_graph_mode_session->Create(CreateGraphForYEqualsXSquared()));
  Tensor input(DT_FLOAT, TensorShape({3}));
  std::vector<Tensor> outputs;
  float* data = input.flat<float>().data();
  data[0] = 1.0f;
  data[1] = 2.2f;
  data[2] = 3.7f;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", input));
  TF_CHECK_OK(cuda_graph_mode_session->Run(inputs_map, {"y"}, {}, &outputs));
  ASSERT_EQ(1, outputs.size());
  ASSERT_EQ(3, outputs[0].NumElements());
  auto result = copyFromGPU(outputs[0]);
  data = result.flat<float>().data();
  EXPECT_FLOAT_EQ(1.0, data[0]);
  EXPECT_FLOAT_EQ(4.84, data[1]);
  EXPECT_FLOAT_EQ(13.69, data[2]);
  unsetenv("CUDA_VISIBLE_DEVICES");
}

TEST_F(CudaGraphModeSessionTest, TestGPUInput) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  std::unique_ptr<Session> cuda_graph_mode_session;
  SessionOptions options;
  options.config.mutable_cuda_graph_mode_options()->set_batch_size(3);
  options.config.mutable_cuda_graph_mode_options()->set_allow_fallback(true);
  options.config.mutable_cuda_graph_mode_options()->add_output_names("y");
  init(options, cuda_graph_mode_session);
  TF_CHECK_OK(cuda_graph_mode_session->Create(CreateGraphForYEqualsXSquared()));
  Tensor input(DT_FLOAT, TensorShape({3}));
  std::vector<Tensor> outputs;
  float* data = input.flat<float>().data();
  data[0] = 1.0f;
  data[1] = 2.2f;
  data[2] = 3.7f;
  auto gpu_input = copyToGPU(
      input, reinterpret_cast<CudaGraphModeSession*>(cuda_graph_mode_session.get())
                 ->get_gpu_allocator());
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", gpu_input));
  TF_CHECK_OK(cuda_graph_mode_session->Run(inputs_map, {"y"}, {}, &outputs));
  ASSERT_EQ(1, outputs.size());
  ASSERT_EQ(3, outputs[0].NumElements());
  auto result = copyFromGPU(outputs[0]);
  data = result.flat<float>().data();
  EXPECT_FLOAT_EQ(1.0, data[0]);
  EXPECT_FLOAT_EQ(4.84, data[1]);
  EXPECT_FLOAT_EQ(13.69, data[2]);
  unsetenv("CUDA_VISIBLE_DEVICES");
}

TEST_F(CudaGraphModeSessionTest, TestMultiInputAndOutput) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  std::unique_ptr<Session> cuda_graph_mode_session;
  SessionOptions options;
  options.config.mutable_cuda_graph_mode_options()->set_batch_size(2);
  options.config.mutable_cuda_graph_mode_options()->set_allow_fallback(true);
  options.config.mutable_cuda_graph_mode_options()->add_output_names("result1");
  options.config.mutable_cuda_graph_mode_options()->add_output_names("result2");
  init(options, cuda_graph_mode_session);
  TF_CHECK_OK(
      cuda_graph_mode_session->Create(CreateGraphForXPlusYThenMultiplyByZ()));
  Tensor x(DT_FLOAT, TensorShape({2}));
  Tensor y(DT_FLOAT, TensorShape({2}));
  Tensor z(DT_FLOAT, TensorShape({2}));
  std::vector<Tensor> outputs;
  float* dx = x.flat<float>().data();
  float* dy = y.flat<float>().data();
  float* dz = z.flat<float>().data();
  dx[0] = 1.0f;
  dx[1] = 2.0f;
  dy[0] = 3.0f;
  dy[1] = 4.0f;
  dz[0] = 5.0f;
  dz[1] = 6.0f;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", x));
  inputs_map.push_back(std::make_pair("y", y));
  inputs_map.push_back(std::make_pair("z", z));
  TF_CHECK_OK(cuda_graph_mode_session->Run(inputs_map, {"result1", "result2"}, {},
                                      &outputs));
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(2, outputs[0].NumElements());
  ASSERT_EQ(2, outputs[1].NumElements());
  auto out1 = copyFromGPU(outputs[0]);
  auto out2 = copyFromGPU(outputs[1]);
  float* result1 = out1.flat<float>().data();
  float* result2 = out2.flat<float>().data();
  EXPECT_FLOAT_EQ(4.0, result1[0]);
  EXPECT_FLOAT_EQ(20.0, result2[0]);
  EXPECT_FLOAT_EQ(6.0, result1[1]);
  EXPECT_FLOAT_EQ(36.0, result2[1]);
  unsetenv("CUDA_VISIBLE_DEVICES");
}

TEST_F(CudaGraphModeSessionTest, TestInt32TypeFailed) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  std::unique_ptr<Session> cuda_graph_mode_session;
  SessionOptions options;
  options.config.mutable_cuda_graph_mode_options()->set_batch_size(3);
  options.config.mutable_cuda_graph_mode_options()->set_allow_fallback(true);
  options.config.mutable_cuda_graph_mode_options()->add_output_names("y");
  GraphDef graph = CreateGraphForYEqualsXSquared();
  ModifyType(graph, DT_INT32);
  init(options, cuda_graph_mode_session);
  ASSERT_FALSE(cuda_graph_mode_session->Create(graph).ok());
  unsetenv("CUDA_VISIBLE_DEVICES");
}

TEST_F(CudaGraphModeSessionTest, TestInt64Type) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  std::unique_ptr<Session> cuda_graph_mode_session;
  SessionOptions options;
  options.config.mutable_cuda_graph_mode_options()->set_batch_size(3);
  options.config.mutable_cuda_graph_mode_options()->set_allow_fallback(true);
  options.config.mutable_cuda_graph_mode_options()->add_output_names("y");
  GraphDef graph = CreateGraphForYEqualsXSquared();
  ModifyType(graph, DT_INT64);
  init(options, cuda_graph_mode_session);
  TF_CHECK_OK(cuda_graph_mode_session->Create(graph));
  Tensor input(DT_INT64, TensorShape({3}));
  std::vector<Tensor> outputs;
  int64* data = input.flat<int64>().data();
  data[0] = 1;
  data[1] = 2;
  data[2] = 3;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", input));
  TF_CHECK_OK(cuda_graph_mode_session->Run(inputs_map, {"y"}, {}, &outputs));
  ASSERT_EQ(1, outputs.size());
  ASSERT_EQ(3, outputs[0].NumElements());
  auto result = copyFromGPU(outputs[0]);
  data = result.flat<int64>().data();
  EXPECT_EQ(1, data[0]);
  EXPECT_EQ(4, data[1]);
  EXPECT_EQ(9, data[2]);
  unsetenv("CUDA_VISIBLE_DEVICES");
}

TEST_F(CudaGraphModeSessionTest, TestShapeUnknownRank) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  std::unique_ptr<Session> cuda_graph_mode_session;
  SessionOptions options;
  options.config.mutable_cuda_graph_mode_options()->set_batch_size(3);
  options.config.mutable_cuda_graph_mode_options()->set_allow_fallback(true);
  options.config.mutable_cuda_graph_mode_options()->add_output_names("y");
  GraphDef graph = CreateGraphForYEqualsXSquaredUnknownRank();
  init(options, cuda_graph_mode_session);
  ASSERT_FALSE(cuda_graph_mode_session->Create(graph).ok());
  unsetenv("CUDA_VISIBLE_DEVICES");
}

TEST_F(CudaGraphModeSessionTest, TestRunAndCudaGraphRunParallel) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  SessionOptions options;
  options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  std::unique_ptr<Session> session(NewSession(options));

  std::unique_ptr<Session> cuda_graph_mode_session;
  SessionOptions graph_options;
  graph_options.config.mutable_cuda_graph_mode_options()->set_batch_size(3);
  graph_options.config.mutable_cuda_graph_mode_options()->set_allow_fallback(true);
  graph_options.config.mutable_cuda_graph_mode_options()->add_output_names("y");
  init(graph_options, cuda_graph_mode_session);

  std::vector<std::string> output_names({"y"});
  GraphDef graph = CreateGraphForYEqualsXSquared();
  TF_CHECK_OK(cuda_graph_mode_session->Create(graph));
  TF_CHECK_OK(session->Create(graph));
  Tensor input(DT_FLOAT, TensorShape({3}));
  float* data = input.flat<float>().data();
  data[0] = 1.0;
  data[1] = 2.0;
  data[2] = 3.0;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", input));
  std::vector<std::thread> threads;
  int num_threads = 100;
  std::vector<Tensor> outputs1[num_threads];
  std::vector<Tensor> outputs2[num_threads];
  std::vector<Tensor> outputs3[num_threads];
  const int num_infers_per_thread = 10;

  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(TFRun, session.get(), num_infers_per_thread,
                                  &inputs_map, &output_names, &outputs1[i]));
  }

  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(CudaGraphModeRun, cuda_graph_mode_session.get(),
                                  num_infers_per_thread, &inputs_map,
                                  &output_names, &outputs2[i]));
  }

  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(TFRun, session.get(), num_infers_per_thread,
                                  &inputs_map, &output_names, &outputs3[i]));
  }

  for (auto& thread : threads) {
    thread.join();
  }
  unsetenv("CUDA_VISIBLE_DEVICES");
}

TEST_F(CudaGraphModeSessionTest, TestInputsSizeNotConsist) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  std::unique_ptr<Session> cuda_graph_mode_session;
  SessionOptions options;
  options.config.mutable_cuda_graph_mode_options()->set_batch_size(2);
  options.config.mutable_cuda_graph_mode_options()->set_allow_fallback(false);
  options.config.mutable_cuda_graph_mode_options()->add_output_names("result1");
  options.config.mutable_cuda_graph_mode_options()->add_output_names("result2");
  init(options, cuda_graph_mode_session);
  TF_CHECK_OK(
      cuda_graph_mode_session->Create(CreateGraphForXPlusYThenMultiplyByZ()));

  Tensor x(DT_FLOAT, TensorShape({2}));
  Tensor y(DT_FLOAT, TensorShape({1}));
  Tensor z(DT_FLOAT, TensorShape({1}));
  std::vector<Tensor> outputs;
  float* dx = x.flat<float>().data();
  float* dy = y.flat<float>().data();
  float* dz = z.flat<float>().data();
  dx[0] = 1.0f;
  dx[1] = 2.0f;
  dy[0] = 3.0f;
  dz[0] = 5.0f;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", x));
  inputs_map.push_back(std::make_pair("y", y));
  inputs_map.push_back(std::make_pair("z", y));
  ASSERT_FALSE(
      cuda_graph_mode_session->Run(inputs_map, {"result1", "result2"}, {}, &outputs)
          .ok());
  unsetenv("CUDA_VISIBLE_DEVICES");
}

TEST_F(CudaGraphModeSessionTest, TestLackInput) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  std::unique_ptr<Session> cuda_graph_mode_session;
  SessionOptions options;
  options.config.mutable_cuda_graph_mode_options()->set_batch_size(2);
  options.config.mutable_cuda_graph_mode_options()->set_allow_fallback(false);
  options.config.mutable_cuda_graph_mode_options()->add_output_names("result1");
  options.config.mutable_cuda_graph_mode_options()->add_output_names("result2");
  init(options, cuda_graph_mode_session);
  TF_CHECK_OK(
      cuda_graph_mode_session->Create(CreateGraphForXPlusYThenMultiplyByZ()));
  Tensor x(DT_FLOAT, TensorShape({2}));
  std::vector<Tensor> outputs;
  float* dx = x.flat<float>().data();
  dx[0] = 1.0f;
  dx[1] = 2.0f;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", x));
  ASSERT_FALSE(
      cuda_graph_mode_session->Run(inputs_map, {"result1", "result2"}, {}, &outputs)
          .ok());
  unsetenv("CUDA_VISIBLE_DEVICES");
}

TEST_F(CudaGraphModeSessionTest, TestInputTypeNotConsist) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  std::unique_ptr<Session> cuda_graph_mode_session;
  SessionOptions options;
  options.config.mutable_cuda_graph_mode_options()->set_batch_size(2);
  options.config.mutable_cuda_graph_mode_options()->set_allow_fallback(false);
  options.config.mutable_cuda_graph_mode_options()->add_output_names("result1");
  options.config.mutable_cuda_graph_mode_options()->add_output_names("result2");
  init(options, cuda_graph_mode_session);
  TF_CHECK_OK(
      cuda_graph_mode_session->Create(CreateGraphForXPlusYThenMultiplyByZ()));
  Tensor x(DT_FLOAT, TensorShape({2}));
  Tensor y(DT_INT64, TensorShape({2}));
  Tensor z(DT_FLOAT, TensorShape({2}));
  std::vector<Tensor> outputs;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", x));
  inputs_map.push_back(std::make_pair("y", y));
  inputs_map.push_back(std::make_pair("z", y));
  ASSERT_FALSE(
      cuda_graph_mode_session->Run(inputs_map, {"result1", "result2"}, {}, &outputs)
          .ok());
  unsetenv("CUDA_VISIBLE_DEVICES");
}

TEST_F(CudaGraphModeSessionTest, TestOutputOnCPU) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  std::unique_ptr<Session> cuda_graph_mode_session;
  SessionOptions options;
  options.config.mutable_cuda_graph_mode_options()->set_batch_size(3);
  options.config.mutable_cuda_graph_mode_options()->set_allow_fallback(true);
  options.config.mutable_cuda_graph_mode_options()->add_output_names("y");
  init(options, cuda_graph_mode_session);
  TF_CHECK_OK(cuda_graph_mode_session->Create(CreateGraphForYEqualsXSquared()));
  Tensor input(DT_FLOAT, TensorShape({3}));
  std::vector<Tensor> outputs;
  float* data = input.flat<float>().data();
  data[0] = 1.0f;
  data[1] = 2.2f;
  data[2] = 3.7f;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", input));
  TF_CHECK_OK(cuda_graph_mode_session->Run(inputs_map, {"y"}, {}, &outputs));
  ASSERT_EQ(1, outputs.size());
  ASSERT_EQ(3, outputs[0].NumElements());
  auto result = copyFromGPU(outputs[0]);
  data = result.flat<float>().data();
  EXPECT_FLOAT_EQ(1.0, data[0]);
  EXPECT_FLOAT_EQ(4.84, data[1]);
  EXPECT_FLOAT_EQ(13.69, data[2]);
  unsetenv("CUDA_VISIBLE_DEVICES");
}

TEST_F(CudaGraphModeSessionTest, TestJitSimple) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  std::unique_ptr<Session> session;
  SessionOptions options;
  options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  options.config.mutable_gpu_options()->set_allow_growth(false);
  options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(
      0.9);
  options.config.mutable_gpu_options()->set_cuda_graph_enable_jit(true);
  session.reset(NewSession(options));
  TF_CHECK_OK(session->Create(CreateGraphForYEqualsXSquared()));
  Tensor input(DT_FLOAT, TensorShape({3}));
  std::vector<Tensor> outputs;
  float* data = input.flat<float>().data();
  data[0] = 1.0f;
  data[1] = 2.2f;
  data[2] = 3.7f;

  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", input));
  TF_CHECK_OK(session->Run(inputs_map, {"y"}, {}, &outputs));
  ASSERT_EQ(1, outputs.size());
  ASSERT_EQ(3, outputs[0].NumElements());
  auto result = copyFromGPU(outputs[0]);
  data = result.flat<float>().data();
  EXPECT_FLOAT_EQ(1.0, data[0]);
  EXPECT_FLOAT_EQ(4.84, data[1]);
  EXPECT_FLOAT_EQ(13.69, data[2]);
  unsetenv("CUDA_VISIBLE_DEVICES");
}

TEST_F(CudaGraphModeSessionTest, TestJitMultiInputAndOutput) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  std::unique_ptr<Session> session;
  SessionOptions options;
  options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  options.config.mutable_gpu_options()->set_allow_growth(false);
  options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(
      0.9);
  options.config.mutable_gpu_options()->set_cuda_graph_enable_jit(true);
  session.reset(NewSession(options));
  TF_CHECK_OK(session->Create(CreateGraphForXPlusYThenMultiplyByZ()));
  Tensor x(DT_FLOAT, TensorShape({2}));
  Tensor y(DT_FLOAT, TensorShape({2}));
  Tensor z(DT_FLOAT, TensorShape({2}));
  std::vector<Tensor> outputs;
  float* dx = x.flat<float>().data();
  float* dy = y.flat<float>().data();
  float* dz = z.flat<float>().data();
  dx[0] = 1.0f;
  dx[1] = 2.0f;
  dy[0] = 3.0f;
  dy[1] = 4.0f;
  dz[0] = 5.0f;
  dz[1] = 6.0f;
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", x));
  inputs_map.push_back(std::make_pair("y", y));
  inputs_map.push_back(std::make_pair("z", z));
  TF_CHECK_OK(session->Run(inputs_map, {"result1", "result2"}, {},
                                      &outputs));
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(2, outputs[0].NumElements());
  ASSERT_EQ(2, outputs[1].NumElements());
  auto out1 = copyFromGPU(outputs[0]);
  auto out2 = copyFromGPU(outputs[1]);
  float* result1 = out1.flat<float>().data();
  float* result2 = out2.flat<float>().data();
  EXPECT_FLOAT_EQ(4.0, result1[0]);
  EXPECT_FLOAT_EQ(20.0, result2[0]);
  EXPECT_FLOAT_EQ(6.0, result1[1]);
  EXPECT_FLOAT_EQ(36.0, result2[1]);
  unsetenv("CUDA_VISIBLE_DEVICES");
}

// cuda graph benchmark
static void CudaGraphSessRunBenchSimple(int iters, int batch_size, int expect_iters) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  testing::StopTiming();
  std::unique_ptr<Session> cuda_graph_mode_session;
  SessionOptions options;
  options.config.mutable_cuda_graph_mode_options()->set_batch_size(batch_size);
  options.config.mutable_cuda_graph_mode_options()->set_allow_fallback(true);
  options.config.mutable_cuda_graph_mode_options()->add_output_names("result2");

  SessionOptions new_options = options;
  new_options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  new_options.config.mutable_gpu_options()->set_allow_growth(false);
  new_options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(
      0.9);
  new_options.config.mutable_gpu_options()->set_cuda_graph_mode_compatible(true);
  new_options.target = CUDA_GRAPH_MODE_TARGET_NAME;
  cuda_graph_mode_session.reset(NewSession(new_options));
  Tensor x(DT_FLOAT, TensorShape({batch_size}));
  Tensor y(DT_FLOAT, TensorShape({batch_size}));
  Tensor z(DT_FLOAT, TensorShape({batch_size}));
  std::vector<Tensor> outputs;
  RandomInitialize(x);
  RandomInitialize(y);
  RandomInitialize(z);
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", x));
  inputs_map.push_back(std::make_pair("y", y));
  inputs_map.push_back(std::make_pair("z", z));
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters) * (batch_size));
  testing::StartTiming();
  TF_CHECK_OK(
      cuda_graph_mode_session->Create(CreateGraphForXPlusYThenMultiplyByZ()));
  for (int i = 0; i < iters; ++i) {
    TF_CHECK_OK(cuda_graph_mode_session->Run(inputs_map, {"result2"}, {}, &outputs));
  }
  testing::StopTiming();
  unsetenv("CUDA_VISIBLE_DEVICES");
}

static void TFRunBenchSimple(int iters, int batch_size, int expect_iters) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  testing::StopTiming();
  std::unique_ptr<Session> session;
  SessionOptions options;
  options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  options.config.mutable_gpu_options()->set_allow_growth(false);
  options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(
      0.9);
  options.config.mutable_gpu_options()->set_cuda_graph_mode_compatible(true);
  session.reset(NewSession(options));
  Tensor x(DT_FLOAT, TensorShape({batch_size}));
  Tensor y(DT_FLOAT, TensorShape({batch_size}));
  Tensor z(DT_FLOAT, TensorShape({batch_size}));
  std::vector<Tensor> outputs;
  RandomInitialize(x);
  RandomInitialize(y);
  RandomInitialize(z);
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("x", x));
  inputs_map.push_back(std::make_pair("y", y));
  inputs_map.push_back(std::make_pair("z", z));
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters) * (batch_size));
  testing::StartTiming();
  TF_CHECK_OK(session->Create(CreateGraphForXPlusYThenMultiplyByZ()));
  for (int i = 0; i < iters; ++i) {
    TF_CHECK_OK(session->Run(inputs_map, {"result2"}, {}, &outputs));
  }
  testing::StopTiming();
  unsetenv("CUDA_VISIBLE_DEVICES");
}

static void CudaGraphJitRunBenchDLRM(int iters, int batch_size, int expect_iters) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  testing::StopTiming();
  std::unique_ptr<Session> session;
  SessionOptions options;
  options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  options.config.mutable_gpu_options()->set_allow_growth(false);
  options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(
      0.9);
  options.config.mutable_gpu_options()->set_cuda_graph_enable_jit(true);
  session.reset(NewSession(options));
  Tensor emb_input(DT_FLOAT, TensorShape({batch_size, 64}));
  std::vector<Tensor> outputs;
  RandomInitialize(emb_input);
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("emb_input", emb_input));
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters) * (batch_size));
  testing::StartTiming();
  TF_CHECK_OK(session->Create(CreateGraphForDLRMBenchmark()));
  for (int i = 0; i < iters; ++i) {
    TF_CHECK_OK(session->Run(inputs_map, {"dlrm/bot_mlp/bot_mlp_0/BiasAdd_11"}, {},
                             &outputs));
  }
  testing::StopTiming();
  unsetenv("CUDA_VISIBLE_DEVICES");
}

static void TFRunBenchDLRM(int iters, int batch_size, int expect_iters) {
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  testing::StopTiming();
  std::unique_ptr<Session> session;
  SessionOptions options;
  options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
  options.config.mutable_gpu_options()->set_allow_growth(false);
  options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(
      0.9);
  options.config.mutable_gpu_options()->set_cuda_graph_mode_compatible(true);
  session.reset(NewSession(options));
  Tensor emb_input(DT_FLOAT, TensorShape({batch_size, 64}));
  std::vector<Tensor> outputs;
  RandomInitialize(emb_input);
  std::vector<std::pair<std::string, Tensor>> inputs_map;
  inputs_map.push_back(std::make_pair("emb_input", emb_input));
  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters) * (batch_size));
  testing::StartTiming();
  TF_CHECK_OK(session->Create(CreateGraphForDLRMBenchmark()));
  for (int i = 0; i < iters; ++i) {
    TF_CHECK_OK(session->Run(inputs_map, {"dlrm/bot_mlp/bot_mlp_0/BiasAdd_11"}, {},
                             &outputs));
  }
  testing::StopTiming();
  unsetenv("CUDA_VISIBLE_DEVICES");
}

BENCHMARK(TFRunBenchSimple)->ArgPair(100, 10000);
BENCHMARK(CudaGraphSessRunBenchSimple)->ArgPair(100, 10000);

BENCHMARK(TFRunBenchDLRM)->ArgPair(100, 10000);
BENCHMARK(CudaGraphJitRunBenchDLRM)->ArgPair(100, 10000);

}  // namespace
}  // namespace tensorflow
#endif  // GOOGLE_CUDA
