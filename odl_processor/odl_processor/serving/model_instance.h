#ifndef TENSORFLOW_SERVING_MODEL_INSTANCE_H
#define TENSORFLOW_SERVING_MODEL_INSTANCE_H

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "odl_processor/framework/model_version.h"
#include "odl_processor/serving/model_message.h"

#include <thread>
#include <atomic>

namespace tensorflow {
class SessionOptions;
class RunOptions;
class Tensor;
class TensorInfo;

namespace processor {
class SavedModelOptimizer;
class ModelConfig;
class ModelStorage;
class ModelSessionMgr;
class SparseStorage;

class ModelInstance {
 public:
  ModelInstance(SessionOptions* sess_options, RunOptions* run_options);
  Status Init(ModelConfig* config,
      ModelStorage* model_storage);

  Status Predict(Request& req, Response& resp);

  Status FullModelUpdate(const Version& version);
  Status DeltaModelUpdate(const Version& version);
  Status Warmup();

  Version GetVersion() { return version_; }
  std::string DebugString();

 private:
  Status ReadModelSignature(ModelConfig* model_config);

  Status RecursionCreateSession(const Version& version,
      SparseStorage* sparse_storge);

  Tensor CreateTensor(const TensorInfo& tensor_info);
  Call CreateWarmupParams();
  bool ShouldWarmup();

 private:
  MetaGraphDef meta_graph_def_;
  std::pair<std::string, SignatureDef> model_signature_;

  ModelSessionMgr* session_mgr_ = nullptr;
  SessionOptions* session_options_ = nullptr;
  RunOptions* run_options_ = nullptr;
  SavedModelOptimizer* optimizer_ = nullptr;

  SparseStorage* serving_storage_ = nullptr;
  SparseStorage* backup_storage_ = nullptr; 

  Version version_;
};

class ModelInstanceMgr {
 public:
  ModelInstanceMgr(ModelConfig* config);
  ~ModelInstanceMgr();

  Status Init();
  Status Predict(Request& req, Response& resp);

  Status Rollback();
  std::string DebugString();

  void WorkLoop();

 private:
  Status CreateInstances();
  Status FullModelUpdate(const Version& version);
  Status DeltaModelUpdate(const Version& version);
  
  Status ModelUpdate(const Version& version);

 private:
  volatile bool is_stop_ = false;

  Status status_;
  std::thread* thread_ = nullptr;

  //mutex mu_;
  ModelInstance* cur_instance_ = nullptr;
  ModelInstance* base_instance_ = nullptr;

  ModelStorage* model_storage_ = nullptr;
  ModelConfig* model_config_ = nullptr; // not owned

  SessionOptions* session_options_ = nullptr;
  RunOptions* run_options_ = nullptr;
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_INSTANCE_H

