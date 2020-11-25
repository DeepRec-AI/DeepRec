#ifndef TENSORFLOW_SERVING_MODEL_IMPL_H
#define TENSORFLOW_SERVING_MODEL_IMPL_H

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

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
class ModelInstanceMgr;

class ModelImpl {
 public:
  virtual ~ModelImpl() {}
  virtual Status Init(const char* root_dir) = 0;
  virtual Status Predict(const eas::PredictRequest& req,
                         eas::PredictResponse* resp) = 0;

  virtual Status Rollback() = 0;
  virtual std::string DebugString() = 0;
};

class FreezeSavedModelImpl : public ModelImpl {
 public:
  ~FreezeSavedModelImpl() override {}
  
  Status Init(const char* root_dir) override {
    return Status::OK();
  }

  Status Predict(const eas::PredictRequest& req,
                 eas::PredictResponse* resp) override {
    return Status::OK();
  }

  Status Rollback() override {
    return Status::OK();
  }

  std::string DebugString() override {
    return std::string();
  }
};

class SavedModelImpl : public ModelImpl {
 public:
  explicit SavedModelImpl(ModelConfig* config);
  ~SavedModelImpl() override;

  Status Init(const char* root_dir) override;
  Status Predict(const eas::PredictRequest& req,
                 eas::PredictResponse* resp) override;

  Status Rollback() override;
  std::string DebugString() override;
 
 private:
  ModelConfig* model_config_;
  ModelInstanceMgr* instance_mgr_ = nullptr;
  //SavedModelBundle* saved_model_bundle_;
  //std::pair<std::string, SignatureDef> model_signature_;

  SavedModelOptimizer* optimizer_ = nullptr;
  
  SessionOptions* session_options_ = nullptr;
  RunOptions* run_options_ = nullptr;
};

class ModelImplFactory {
 public:
  static ModelImpl* Create(ModelConfig* config) {
    return new SavedModelImpl(config);
  }
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_IMPL_H

