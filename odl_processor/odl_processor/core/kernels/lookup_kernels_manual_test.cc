/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "odl_processor/core/util/utils.h"
#include "odl_processor/core/graph_optimizer.h"
#include "odl_processor/core/util/utils.h"
#include "odl_processor/model_store/model_store_factory.h"
#include "odl_processor/model_store/sparse_storage_manager.h"


using namespace tensorflow;

int main() {
 processor::SimpleSparseStorageManager manager(4, 2, "local_redis");
 {
  NodeDef version_def;
  Tensor version_value(DT_STRING, TensorShape({1}));
  version_value.flat<std::string>()(0) = "v1";
  TF_CHECK_OK(NodeDefBuilder("version", "Const")
                  .Attr("dtype", DT_STRING)
                  .Attr("value", version_value)
                  .Finalize(&version_def));

  NodeDef prefix_def;
  Tensor prefix_value(DT_STRING, TensorShape({1}));
  prefix_value.flat<std::string>()(0) =
      // Local test
      //"/local/path/workspace/tmp/DeepFM/ev/1598442950_pai/variables/variables";
      // OSS test
      //"oss://bucket-name\x01id=id\x02key=key\x02host=host/jktest/mm/odl_test_files/saved_model/variables";
  TF_CHECK_OK(NodeDefBuilder("prefix", "Const")
                  .Attr("dtype", DT_STRING)
                  .Attr("value", prefix_value)
                  .Finalize(&prefix_def));

  NodeDef tensor_name_def;
  Tensor tensor_name_value(DT_STRING, TensorShape({1}));
  tensor_name_value.flat<std::string>()(0) =
      "input_from_feature_columns/fm_10169_embedding/weights";
  TF_CHECK_OK(NodeDefBuilder("tensor_name", "Const")
                  .Attr("dtype", DT_STRING)
                  .Attr("value", tensor_name_value)
                  .Finalize(&tensor_name_def));

  NodeDef pointer_value_def;
  Tensor pointer_value(DT_UINT64, TensorShape({1}));
  pointer_value.scalar<tensorflow::uint64>()() = (uint64_t)(&manager);
  TF_CHECK_OK(NodeDefBuilder("pointer_value", "Const")
                  .Attr("dtype", DT_UINT64)
                  .Attr("value", pointer_value)
                  .Finalize(&pointer_value_def));

  NodeDef kv_import_def;
  TF_CHECK_OK(NodeDefBuilder("kv_import", "KvImport")
                  .Input("version", 0, DT_STRING)
                  .Input("prefix", 1, DT_STRING)
                  .Input("tensor_name", 2, DT_STRING)
                  .Input("storage_pointer_value", 3, DT_UINT64)
                  .Attr("var_name", "XXX")
                  .Attr("dim_len", 1)
                  .Attr("Tkeys", DT_INT64)
                  .Attr("dtype", DT_FLOAT)
                  .Finalize(&kv_import_def));

  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:localhost/replica:0/task:0"));

  Status status;
  std::unique_ptr<OpKernel> kv_import_op(
      CreateOpKernel(DEVICE_CPU, device.get(), cpu_allocator(),
                     kv_import_def, TF_GRAPH_DEF_VERSION, &status));
  TF_CHECK_OK(status);
  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  // Create inputs
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back({nullptr, &version_value});
  inputs.push_back({nullptr, &prefix_value});
  inputs.push_back({nullptr, &tensor_name_value});
  inputs.push_back({nullptr, &pointer_value});
  params.inputs = &inputs;
  params.op_kernel = kv_import_op.get();

  std::unique_ptr<OpKernelContext> kv_import_context(
      new OpKernelContext(&params, 0));

  auto done = []() {
    LOG(INFO) << "I'm import done.";
  };

  AsyncOpKernel* real_kv_import_op = (AsyncOpKernel*)(kv_import_op.get());
  real_kv_import_op->ComputeAsync(kv_import_context.get(), std::move(done));
  TF_CHECK_OK(kv_import_context->status());
 }
 {
  NodeDef version_def;
  Tensor version_value(DT_STRING, TensorShape({1}));
  version_value.flat<std::string>()(0) = "v1";
  TF_CHECK_OK(NodeDefBuilder("version", "Const")
                  .Attr("dtype", DT_STRING)
                  .Attr("value", version_value)
                  .Finalize(&version_def));

  NodeDef indices_def;
  Tensor indices_value(DT_INT64, TensorShape({4}));
  indices_value.flat<int64>()(0) = 4672086491694744000;
  indices_value.flat<int64>()(1) = 3;
  indices_value.flat<int64>()(2) = 8913915434458630000;
  indices_value.flat<int64>()(3) = 1;
  TF_CHECK_OK(NodeDefBuilder("indices", "Const")
                  .Attr("dtype", DT_INT64)
                  .Attr("value", indices_value)
                  .Finalize(&indices_def));

  NodeDef default_value_def;
  Tensor default_value(DT_FLOAT, TensorShape({1}));
  default_value.flat<float>()(0) = 4.2;
  TF_CHECK_OK(NodeDefBuilder("default_value", "Const")
                  .Attr("dtype", DT_FLOAT)
                  .Attr("value", default_value)
                  .Finalize(&default_value_def));

  NodeDef pointer_value_def;
  Tensor pointer_value(DT_UINT64, TensorShape({1}));
  pointer_value.scalar<uint64_t>()() = (uint64_t)(&manager);
  TF_CHECK_OK(NodeDefBuilder("pointer_value", "Const")
                  .Attr("dtype", DT_UINT64)
                  .Attr("value", pointer_value)
                  .Finalize(&pointer_value_def));

  NodeDef kv_lookup_def;
  TF_CHECK_OK(NodeDefBuilder("kv_lookup", "KvLookup")
                  .Input("version", 0, DT_STRING)
                  .Input("indices", 1, DT_INT64)
                  .Input("default_value", 2, DT_FLOAT)
                  .Input("storage_pointer_value", 3, DT_UINT64)
                  .Attr("var_name", "XXX")
                  .Attr("dim_len", 1)
                  .Attr("Tkeys", DT_INT64)
                  .Attr("dtype", DT_FLOAT)
                  .Finalize(&kv_lookup_def));

  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:localhost/replica:0/task:0"));

  Status status;
  std::unique_ptr<OpKernel> kv_lookup_op(
      CreateOpKernel(DEVICE_CPU, device.get(), cpu_allocator(),
                     kv_lookup_def, TF_GRAPH_DEF_VERSION, &status));
  TF_CHECK_OK(status);
  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  // Create inputs
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back({nullptr, &version_value});
  inputs.push_back({nullptr, &indices_value});
  inputs.push_back({nullptr, &default_value});
  inputs.push_back({nullptr, &pointer_value});
  params.inputs = &inputs;
  //params.inputs = &inputs;
  params.op_kernel = kv_lookup_op.get();
  AllocatorAttributes alloc_attrs;
  params.output_attr_array = &alloc_attrs;

  std::unique_ptr<OpKernelContext> kv_lookup_context(
      new OpKernelContext(&params, 1));

  auto done = []() {
    LOG(INFO) << "I'm lookup done.";
  };

  AsyncOpKernel* real_kv_lookup_op = (AsyncOpKernel*)(kv_lookup_op.get());
  real_kv_lookup_op->ComputeAsync(kv_lookup_context.get(), std::move(done));
  TF_CHECK_OK(kv_lookup_context->status());
 }
 sleep(5);
 // cleanup redis
 TF_CHECK_OK(manager.Reset());
}
