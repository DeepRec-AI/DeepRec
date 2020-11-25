#ifndef TENSORFLOW_SERVING_MODEL_INSTANCE_H
#define TENSORFLOW_SERVING_MODEL_INSTANCE_H

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

#include <thread>
#include <atomic>

class RunRequest;
class RunResponse;
namespace tensorflow {
namespace eas {
  class PredictRequest;
  class PredictResponse;
}
class SavedModelBundle;
class SessionOptions;
class RunOptions;
class Session;

namespace processor {
class SavedModelOptimizer;
class ModelConfig;
class ModelStorage;
class SparseStorage;

struct Version {
  std::string full_model_version;
  std::string full_model_name;

  std::string delta_model_version;
  std::string delta_model_name;

  Version() = default;
  ~Version() = default;
  Version(const Version&) = default;
  Version& operator=(const Version&) = default;

  bool IsFullModel() const {
    return delta_model_name.empty();
  }

  friend bool operator ==(const Version& lhs, const Version& rhs) {
    return lhs.full_model_version == rhs.full_model_version
        && lhs.delta_model_version == rhs.delta_model_version;
  }

  friend bool operator !=(const Version& lhs, const Version& rhs) {
    return !(lhs == rhs);
  }

  bool IsSameFullModel(const Version& other) const {
    return full_model_version == other.full_model_version;
  }
};

struct ModelSession {
  ModelSession(Session* s, const Version& version,
      SparseStorage* sparse_storage) : session_(s), counter_(0),
    version_(version) {}

  Session* session_ = nullptr;
  SparseStorage* sparse_storage_ = nullptr;
  std::atomic<int64> counter_;

  Version version_;
};

class ModelSessionMgr {
 public:
  ModelSessionMgr(const MetaGraphDef& meta_graph_def,
      SessionOptions* session_options, RunOptions* run_options);

  Status CreateDeltaModelSession(const Version& version,
      const char* model_dir, SparseStorage* sparse_storage);
  Status CreateFullModelSession(const Version& version,
      const char* model_dir, SparseStorage* sparse_storage);

 private:
  Status CreateSession(Session** sess);
  Status RunRestoreOps(const char* model_dir, Session* session,
      SparseStorage* sparse_storage);
  void ResetServingSession(Session* session, const Version& version,
      SparseStorage* sparse_storage);

 private:
  ModelSession* serving_session_ = nullptr;
  std::vector<ModelSession*> sessions_;

  MetaGraphDef meta_graph_def_;
  SessionOptions* session_options_;
  RunOptions* run_options_;
  std::vector<AssetFileDef> asset_file_defs_;
};

class ModelInstance {
 public:
  ModelInstance(SessionOptions* sess_options, RunOptions* run_options);
  Status Init(const Version& version, ModelConfig* config,
      ModelStorage* model_storage);

  Status Predict(const eas::PredictRequest& req, eas::PredictResponse* resp);
  Status Predict(const RunRequest& req, RunResponse* resp);

  Status FullModelUpdate(const Version& version);
  Status DeltaModelUpdate(const Version& version);

  Version GetVersion() { return version_; }
  std::string DebugString();

 private:
  Status Warmup();
  Status ReadModelSignature(ModelConfig* model_config);

  Status RecursionCreateSession(const Version& version,
      SparseStorage* sparse_storge);

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
  ModelInstanceMgr(const char* root_dir, ModelConfig* config);
  ~ModelInstanceMgr();

  Status Init(SessionOptions* sess_options, RunOptions* run_options);
  Status Predict(const eas::PredictRequest& req, eas::PredictResponse* resp);
  Status Rollback();
  std::string DebugString();

  void WorkLoop();

 private:
  Status CreateInstances(const Version& version);
  Status FullModelUpdate(const Version& version);
  Status DeltaModelUpdate(const Version& version);
  
  Status ModelUpdate(const Version& version);

 private:
  bool is_stop = false;

  Status status_;
  std::thread* thread_ = nullptr;

  //mutex mu_;
  ModelInstance* cur_instance_ = nullptr;
  ModelInstance* base_instance_ = nullptr;

  ModelStorage* model_storage_ = nullptr;
  ModelConfig* model_config_ = nullptr;
  SessionOptions* session_options_ = nullptr;
  RunOptions* run_options_ = nullptr;
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_INSTANCE_H

