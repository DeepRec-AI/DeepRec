#ifndef TENSORFLOW_SERVING_MODEL_INSTANCE_H
#define TENSORFLOW_SERVING_MODEL_INSTANCE_H

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "odl_processor/framework/model_version.h"
#include "odl_processor/serving/model_message.h"
#include "odl_processor/serving/model_config.h"
#include <thread>
#include <atomic>

namespace tensorflow {
class SessionOptions;
class RunOptions;
class Tensor;
class TensorInfo;
class Session;
namespace processor {
class SavedModelOptimizer;
class ModelStore;
class ModelSessionMgr;
class IFeatureStoreMgr;

class SingleSessionInstance {
 public:
  SingleSessionInstance(SessionOptions* sess_options,
      RunOptions* run_options);

  Status Init(ModelConfig* config,
      ModelStore* model_store);

  Status Predict(Request& req, Response& resp);
  Status Warmup();
  Version GetVersion() { return version_; }
  std::string DebugString();

 private:
  Status ReadModelSignature(ModelConfig* model_config);
  Status LoadSavedModel(const std::string& export_dir);

 private: 
  MetaGraphDef meta_graph_def_;
  std::pair<std::string, SignatureDef> model_signature_;

  Session* session_ = nullptr;
  SessionOptions* session_options_ = nullptr;
  RunOptions* run_options_ = nullptr;
  SavedModelOptimizer* optimizer_ = nullptr;
  
  Version version_;
};

class MultipleSessionInstance {
 public:
  MultipleSessionInstance(SessionOptions* sess_options,
      RunOptions* run_options);
  Status Init(ModelConfig* config,
      ModelStore* model_store);

  Status Predict(Request& req, Response& resp);

  Status FullModelUpdate(const Version& version);
  Status DeltaModelUpdate(const Version& version);
  Status Warmup();

  Version GetVersion() { return version_; }
  std::string DebugString();

 private:
  Status ReadModelSignature(ModelConfig* model_config);

  Status RecursionCreateSession(const Version& version,
      IFeatureStoreMgr* sparse_storge);

 private:
  MetaGraphDef meta_graph_def_;
  std::pair<std::string, SignatureDef> model_signature_;

  ModelSessionMgr* session_mgr_ = nullptr;
  SessionOptions* session_options_ = nullptr;
  RunOptions* run_options_ = nullptr;
  SavedModelOptimizer* optimizer_ = nullptr;

  IFeatureStoreMgr* serving_storage_ = nullptr;
  IFeatureStoreMgr* backup_storage_ = nullptr; 

  Version version_;
};

class IModelInstanceMgr {
 public:
  virtual ~IModelInstanceMgr() {}

  virtual Status Init() = 0;
  virtual Status Predict(Request& req, Response& resp) = 0;
  virtual Status Rollback() = 0;

  virtual std::string DebugString() = 0;
};

class TFInstanceMgr : public IModelInstanceMgr{
 public:
  TFInstanceMgr(ModelConfig* config);
  ~TFInstanceMgr() override;

  Status Init() override;
  Status Predict(Request& req, Response& resp) override;
  Status Rollback() override;

  std::string DebugString() override;

 protected:
  SingleSessionInstance* instance_ = nullptr;
  ModelStore* model_store_ = nullptr;
  ModelConfig* model_config_ = nullptr; // not owned

  SessionOptions* session_options_ = nullptr;
  RunOptions* run_options_ = nullptr;
};

class ODLInstanceMgr : public IModelInstanceMgr {
 public:
  ODLInstanceMgr(ModelConfig* config);
  virtual ~ODLInstanceMgr();

  Status Init() override;
  Status Predict(Request& req, Response& resp) override;

  Status Rollback() override;
  std::string DebugString() override;

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
  MultipleSessionInstance* cur_instance_ = nullptr;
  MultipleSessionInstance* base_instance_ = nullptr;

  ModelStore* model_storage_ = nullptr;
  ModelConfig* model_config_ = nullptr; // not owned

  SessionOptions* session_options_ = nullptr;
  RunOptions* run_options_ = nullptr;
};

class ModelInstanceMgrFactory {
 public:
  static IModelInstanceMgr* Create(ModelConfig* config) {
    if (config->processor_type == "odl") {
      return new ODLInstanceMgr(config);
    } else if (config->processor_type == "tf") {
      return new TFInstanceMgr(config);
    } else {
      return nullptr;
    }
  }
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_INSTANCE_H

