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

class ModelImpl {
 public:
  virtual ~ModelImpl() {}
  virtual Status Load(const char* model_dir) = 0;
  virtual Status Predict(const eas::PredictRequest& req,
                         eas::PredictResponse* resp) = 0;
  virtual Status Predict(const RunRequest& req, RunResponse* resp) = 0;

  virtual Status Warmup() = 0;
  virtual std::string DebugString() = 0;
};

class FreezeSavedModelImpl : public ModelImpl {
 public:
  ~FreezeSavedModelImpl() override {}
  
  Status Load(const char* model_dir) override {
    return Status::OK();
  }

  Status Predict(const eas::PredictRequest& req,
                 eas::PredictResponse* resp) override {
    return Status::OK();
  }

  Status Predict(const RunRequest& req, RunResponse* resp) override {
    return Status::OK();
  }
  
  Status Warmup() override {
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

  Status Load(const char* model_dir) override;
  Status Predict(const eas::PredictRequest& req,
                 eas::PredictResponse* resp) override;

 private:
  Status Predict(const RunRequest& req, RunResponse* resp) override;
  Status Warmup() override;
  std::string DebugString() override;
 
 private:
  ModelConfig* model_config_;
  SavedModelBundle* saved_model_bundle_;

  SavedModelOptimizer* optimizer_;
  std::pair<std::string, SignatureDef> model_signature_;
  
  SessionOptions* session_options_;
  RunOptions* run_options_;
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

