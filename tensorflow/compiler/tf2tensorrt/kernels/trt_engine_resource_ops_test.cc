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

#include <dirent.h>
#include <string.h>

#include <fstream>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_engine_instance.pb.h"  // NOLINT
#include "tensorflow/compiler/tf2tensorrt/utils/trt_logger.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

class TRTEngineResourceOpsTest : public OpsTestBase {
 protected:
  void Reset() {
    inputs_.clear();
    gtl::STLDeleteElements(&tensors_);
    gtl::STLDeleteElements(&managed_outputs_);
  }

  TrtUniquePtrType<nvinfer1::ICudaEngine> CreateTRTEngine() {
    TrtUniquePtrType<nvinfer1::IBuilder> builder(
        nvinfer1::createInferBuilder(logger_));
    TrtUniquePtrType<nvinfer1::INetworkDefinition> network;
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
    const uint32_t flags = 0U;
    network =
        TrtUniquePtrType<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
#else
    network = TrtUniquePtrType<nvinfer1::INetworkDefinition>(
        builder->createNetwork());
#endif
    // Add the input.
    nvinfer1::Dims dims;
    dims.nbDims = 1;
    dims.d[0] = 1;
    ITensorProxyPtr input =
        network->addInput("input", nvinfer1::DataType::kFLOAT, dims);
    EXPECT_NE(nullptr, input->trt_tensor());

    // Add a unary layer.
    nvinfer1::IUnaryLayer* layer =
        network->addUnary(*input->trt_tensor(), nvinfer1::UnaryOperation::kEXP);
    EXPECT_NE(nullptr, layer);

    // Mark the output.
    ITensorProxyPtr output = layer->getOutput(0);
    output->setName("output");
    network->markOutput(*output->trt_tensor());

    // Build the engine
    builder->setMaxBatchSize(1);
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  TrtUniquePtrType<nvinfer1::IBuilderConfig> builder_config(builder->createBuilderConfig());
  builder_config->setMaxWorkspaceSize(1 << 10);
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine(
      builder->buildEngineWithConfig(*network, *builder_config));
#else
    builder->setMaxWorkspaceSize(1 << 10);
    TrtUniquePtrType<nvinfer1::ICudaEngine> engine(
        builder->buildCudaEngine(*network));
#endif
    EXPECT_NE(nullptr, engine);
    return engine;
  }
  Logger& logger_ = *Logger::GetLogger();
};

TEST_F(TRTEngineResourceOpsTest, Basic) {
  // Create the GPU device.
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("GPU", {}, "/job:worker/replica:0/task:0"));
  ResourceMgr* rm = device->resource_manager();
  SetDevice(DEVICE_GPU, std::move(device));

  // Create the resource handle.
  const string container(kTfTrtContainerName);
  const string resource_name = "myresource";
  Reset();
  TF_ASSERT_OK(NodeDefBuilder("op", "CreateTRTResourceHandle")
                   .Attr("resource_name", resource_name)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  TF_ASSERT_OK(RunOpKernel());
  ResourceHandle handle =
      context_->mutable_output(0)->scalar<ResourceHandle>()();

  TRTEngineCacheResource* resource = nullptr;
  EXPECT_TRUE(
      errors::IsNotFound(rm->Lookup(container, resource_name, &resource)));

  // Create the resouce using an empty file with InitializeTRTResource.
  Reset();
  Env* env = Env::Default();
  const string filename = io::JoinPath(testing::TmpDir(), "trt_engine_file");
  {
    std::unique_ptr<WritableFile> file;
    TF_ASSERT_OK(env->NewWritableFile(filename, &file));
  }
  TF_ASSERT_OK(NodeDefBuilder("op", "InitializeTRTResource")
                   .Input(FakeInput(DT_RESOURCE))
                   .Input(FakeInput(DT_STRING))
                   .Attr("max_cached_engines_count", 1)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<ResourceHandle>(TensorShape({}), {handle});
  AddInputFromArray<string>(TensorShape({}), {filename});
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_TRUE(rm->Lookup(container, resource_name, &resource).ok());
  EXPECT_EQ(0, resource->cache_.size());

  // Create a serialized TRT engine file.
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine = CreateTRTEngine();
  TrtUniquePtrType<nvinfer1::IExecutionContext> context(
      engine->createExecutionContext());
  resource->cache_.emplace(
      std::vector<TensorShape>{TensorShape({1, 1})},
      absl::make_unique<EngineContext>(std::move(engine), std::move(context)));
  resource->Unref();

  // Serialize the engine using SerializeTRTResource op.
  Reset();
  TF_ASSERT_OK(NodeDefBuilder("op", "SerializeTRTResource")
                   .Attr("delete_resource", true)
                   .Input(FakeInput(DT_STRING))
                   .Input(FakeInput(DT_STRING))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<tstring>(TensorShape({}), {resource_name});
  AddInputFromArray<tstring>(TensorShape({}), {filename});
  TF_ASSERT_OK(RunOpKernel());

  // Make sure the cache is deleted.
  Reset();
  TF_ASSERT_OK(NodeDefBuilder("op", "DestroyResourceOp")
                   .Attr("ignore_lookup_error", false)
                   .Input(FakeInput(DT_RESOURCE))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<ResourceHandle>(TensorShape({}), {handle});
  EXPECT_TRUE(errors::IsNotFound(RunOpKernel()));

  // Verify the serialized engine file.
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(filename, &file));
  auto reader = absl::make_unique<io::RecordReader>(file.get());
  uint64 offset = 0;
  string record;
  TF_ASSERT_OK(reader->ReadRecord(&offset, &record));
  TRTEngineInstance engine_instance;
  engine_instance.ParseFromString(record);
  EXPECT_EQ(1, engine_instance.input_shapes_size());
  EXPECT_EQ(2, engine_instance.input_shapes(0).dim_size());
  EXPECT_EQ(1, engine_instance.input_shapes(0).dim(0).size());
  EXPECT_EQ(1, engine_instance.input_shapes(0).dim(1).size());
  EXPECT_TRUE(errors::IsOutOfRange(reader->ReadRecord(&offset, &record)));

  // Recreate the cache resource.
  Reset();
  TF_ASSERT_OK(NodeDefBuilder("op", "InitializeTRTResource")
                   .Input(FakeInput(DT_RESOURCE))
                   .Input(FakeInput(DT_STRING))
                   .Attr("max_cached_engines_count", 1)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<ResourceHandle>(TensorShape({}), {handle});
  AddInputFromArray<tstring>(TensorShape({}), {filename});
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_TRUE(rm->Lookup(container, resource_name, &resource).ok());
  EXPECT_EQ(1, resource->cache_.size());
  resource->Unref();

  // Destroy the engine cache again.
  Reset();
  TF_ASSERT_OK(NodeDefBuilder("op", "DestroyResourceOp")
                   .Attr("ignore_lookup_error", false)
                   .Input(FakeInput(DT_RESOURCE))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<ResourceHandle>(TensorShape({}), {handle});
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_TRUE(errors::IsNotFound(RunOpKernel()));
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
