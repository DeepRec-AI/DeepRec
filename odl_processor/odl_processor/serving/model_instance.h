#ifndef TENSORFLOW_SERVING_MODEL_INSTANCE_H
#define TENSORFLOW_SERVING_MODEL_INSTANCE_H

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

#include <thread>

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

namespace processor {
class SavedModelOptimizer;
class ModelConfig;
class ModelStorage;

struct Version {
  std::string full_model_version;
  std::string full_model_name;

  std::string delta_model_version;
  std::string delta_model_name;

  bool IsBaseCheckpoint() const {
    return delta_model_name.empty();
  }

  friend bool operator ==(const Version& lhs, const Version& rhs) {
    return lhs.full_model_version == rhs.full_model_version
        && lhs.delta_model_version == rhs.delta_model_version;
  }
};

class ModelInstance {
 public:
  ModelInstance(SessionOptions* sess_options, RunOptions* run_options);
  Status Load(const Version& version, ModelConfig* config);

  Status Predict(const eas::PredictRequest& req, eas::PredictResponse* resp);
  Status Predict(const RunRequest& req, RunResponse* resp);

  Version GetVersion() { return version_; }
  std::string DebugString();

 private:
  Status Warmup();

 private:
  SavedModelBundle* saved_model_bundle_ = nullptr;
  std::pair<std::string, SignatureDef> model_signature_;

  SessionOptions* session_options_ = nullptr;
  RunOptions* run_options_ = nullptr;

  SavedModelOptimizer* optimizer_;

  Version version_;
};

class ModelInstanceMgr {
 public:
  ModelInstanceMgr(const char* root_dir, ModelConfig* config);
  ~ModelInstanceMgr();

  Status Init(SessionOptions* sess_options, RunOptions* run_options);

  Status Predict(const eas::PredictRequest& req, eas::PredictResponse* resp);

  void Update(const Version& version);
  void WorkLoop();

  std::string DebugString();

 private:
  void CreateInstances(const Version& version);
  void UpdateBaseInstance(const Version& version);
  void UpdateIncrementalInstance(const Version& version);

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

