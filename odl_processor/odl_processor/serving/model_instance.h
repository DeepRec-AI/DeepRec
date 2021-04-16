#ifndef TENSORFLOW_SERVING_MODEL_INSTANCE_H
#define TENSORFLOW_SERVING_MODEL_INSTANCE_H

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "odl_processor/framework/model_version.h"
#include "odl_processor/serving/model_message.h"
#include "odl_processor/serving/model_config.h"
#include "odl_processor/storage/feature_store.h"
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
class ModelSession;
class ModelSessionMgr;
class IFeatureStoreMgr;

class LocalSessionInstance {
 public:
  LocalSessionInstance(SessionOptions* sess_options,
      RunOptions* run_options);

  Status Init(ModelConfig* config,
      ModelStore* model_store);

  Status Predict(Request& req, Response& resp);
  Status Warmup(ModelSession* warmup_session = nullptr);
  Version GetVersion() { return version_; }
  void UpdateVersion(const Version& v) { version_ = v; }
  std::string DebugString();

  Status FullModelUpdate(const Version& version,
                         ModelConfig* model_config);
  Status DeltaModelUpdate(const Version& version,
                          ModelConfig* model_config);
 
 private:
  Status ReadModelSignature(ModelConfig* model_config);

 private: 
  MetaGraphDef meta_graph_def_;
  std::pair<std::string, SignatureDef> model_signature_;
  std::string warmup_file_name_;

  ModelSessionMgr* session_mgr_ = nullptr;
  SessionOptions* session_options_ = nullptr;
  RunOptions* run_options_ = nullptr;
  SavedModelOptimizer* optimizer_ = nullptr;
  
  Version version_;
};

class RemoteSessionInstance {
 public:
  RemoteSessionInstance(SessionOptions* sess_options,
                          RunOptions* run_options,
                          StorageOptions* storage_options);
  Status Init(ModelConfig* config,
      ModelStore* model_store, bool active);

  Status Predict(Request& req, Response& resp);

  Status FullModelUpdate(const Version& version,
                         ModelConfig* model_config);
  Status DeltaModelUpdate(const Version& version,
                          ModelConfig* model_config);
  Status Warmup(ModelSession* warmup_session = nullptr);

  Version GetVersion() { return version_; }
  void UpdateVersion(const Version& v) { version_ = v; }
  std::string DebugString();

 private:
  Status ReadModelSignature(ModelConfig* model_config);

  Status RecursionCreateSession(const Version& version,
      IFeatureStoreMgr* sparse_storge,
      ModelConfig* model_config);

 private:
  MetaGraphDef meta_graph_def_;
  std::pair<std::string, SignatureDef> model_signature_;
  std::string warmup_file_name_;

  ModelSessionMgr* session_mgr_ = nullptr;
  SessionOptions* session_options_ = nullptr;
  RunOptions* run_options_ = nullptr;
  SavedModelOptimizer* optimizer_ = nullptr;

  IFeatureStoreMgr* serving_storage_ = nullptr;
  IFeatureStoreMgr* backup_storage_ = nullptr; 
  StorageOptions* storage_options_ = nullptr;

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

class ModelUpdater {
 public:
  ModelUpdater(ModelConfig* config);
  virtual ~ModelUpdater();
  void WorkLoop();

 protected:
  virtual Status FullModelUpdate(const Version& version,
                                 ModelConfig* model_config) = 0;
  virtual Status DeltaModelUpdate(const Version& version,
                                  ModelConfig* model_config) = 0;
  virtual Version GetVersion() = 0;

  Status ModelUpdate(const Version& version,
                     ModelConfig* model_config);

 protected:
  ModelStore* model_store_ = nullptr;
  ModelConfig* model_config_ = nullptr; // not owned
  volatile bool is_stop_ = false;
  std::thread* thread_ = nullptr;
};

class LocalSessionInstanceMgr : public ModelUpdater, public IModelInstanceMgr {
 public:
  LocalSessionInstanceMgr(ModelConfig* config);
  ~LocalSessionInstanceMgr() override;

  Status Init() override;
  Status Predict(Request& req, Response& resp) override;
  Status Rollback() override;

  std::string DebugString() override;

 protected:
  Status FullModelUpdate(const Version& version,
                         ModelConfig* model_config) override;
  Status DeltaModelUpdate(const Version& version,
                          ModelConfig* model_config) override;
  Version GetVersion() override;

 protected:
  LocalSessionInstance* instance_ = nullptr;

  SessionOptions* session_options_ = nullptr;
  RunOptions* run_options_ = nullptr;
};

class RemoteSessionInstanceMgr : public ModelUpdater, public IModelInstanceMgr {
 public:
  RemoteSessionInstanceMgr(ModelConfig* config);
  virtual ~RemoteSessionInstanceMgr();

  Status Init() override;
  Status Predict(Request& req, Response& resp) override;

  Status Rollback() override;
  std::string DebugString() override;

 protected:
  Status FullModelUpdate(const Version& version,
                         ModelConfig* model_config) override;
  Status DeltaModelUpdate(const Version& version,
                          ModelConfig* model_config) override;
  Version GetVersion() override;

 private:
  Status CreateInstances();

 private:
  Status status_;

  //mutex mu_;
  RemoteSessionInstance* cur_instance_ = nullptr;
  RemoteSessionInstance* base_instance_ = nullptr;

  SessionOptions* session_options_ = nullptr;
  RunOptions* run_options_ = nullptr;
  StorageOptions* cur_inst_storage_options_ = nullptr;
  StorageOptions* base_inst_storage_options_ = nullptr;
};

class ModelInstanceMgrFactory {
 public:
  static IModelInstanceMgr* Create(ModelConfig* config) {
    if (config->feature_store_type == "redis" ||
        config->feature_store_type == "cluster_redis") {
      return new RemoteSessionInstanceMgr(config);
    } else if (config->feature_store_type == "memory") {
      return new LocalSessionInstanceMgr(config);
    } else {
      return nullptr;
    }
  }
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_INSTANCE_H

