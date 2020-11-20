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
  ModelSession(Session* s) : session_(s), counter_(0) {}

  Session* session_ = nullptr;
  std::atomic<int64> counter_;
};

class ModelSessionMgr {
 public:
  Status CreateModelSession(const MetaGraphDef& meta_graph_def,
      SessionOptions* session_options, RunOptions* run_options,
      const char* model_dir);

 private:
  ModelSession* serving_session_ = nullptr;
  std::vector<ModelSession*> sessions_;
};

class ModelInstance {
 public:
  ModelInstance(SessionOptions* sess_options, RunOptions* run_options);
  Status Init(const Version& version, ModelConfig* config,
      SparseStorage* sparse_storage);

  Status Predict(const eas::PredictRequest& req, eas::PredictResponse* resp);
  Status Predict(const RunRequest& req, RunResponse* resp);

  void FullModelUpdate(const Version& version);
  void DeltaModelUpdate(const Version& version);

  Version GetVersion() { return version_; }
  std::string DebugString();

 private:
  Status Warmup();
  Status ReadModelSignature(ModelConfig* model_config);

  Status CreateSession(const char* model_dir);

 private:
  MetaGraphDef meta_graph_def_;
  //std::unique_ptr<Session> session_;
  ModelSessionMgr* session_mgr_;

  std::pair<std::string, SignatureDef> model_signature_;

  SessionOptions* session_options_ = nullptr;
  RunOptions* run_options_ = nullptr;
  SavedModelOptimizer* optimizer_ = nullptr;
  SparseStorage* sparse_storage_ = nullptr;

  Version version_;
};

class ModelInstanceMgr {
 public:
  ModelInstanceMgr(const char* root_dir, ModelConfig* config);
  ~ModelInstanceMgr();

  Status Init(SessionOptions* sess_options, RunOptions* run_options);
  Status Predict(const eas::PredictRequest& req, eas::PredictResponse* resp);

  void WorkLoop();
  std::string DebugString();

 private:
  void CreateInstances(const Version& version);
  void FullModelUpdate(const Version& version);
  void DeltaModelUpdate(const Version& version);
  
  void ModelUpdate(const Version& version);

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

