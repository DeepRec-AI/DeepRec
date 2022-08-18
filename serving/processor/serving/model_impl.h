#ifndef SERVING_PROCESSOR_SERVING_MODEL_IMPL_H
#define SERVING_PROCESSOR_SERVING_MODEL_IMPL_H

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
class SavedModelBundle;
class Tensor;

namespace processor {
class SavedModelOptimizer;
class ModelConfig;
class IModelInstanceMgr;
class Request;
class Response;
class ServingModelInfo;
class SignatureInfo;

class ModelImpl {
 public:
  virtual ~ModelImpl() {}
  virtual Status Init() = 0;
  virtual Status Predict(Request& req, Response& resp) = 0;
  virtual Status GetServingModelInfo(ServingModelInfo& model_info) = 0;
  virtual Status Rollback() = 0;
  virtual std::string DebugString() = 0;
  virtual SignatureDef GetServingSignatureDef() = 0;
  virtual const SignatureInfo* GetSignatureInfo() = 0;
};

class FreezeSavedModelImpl : public ModelImpl {
 public:
  ~FreezeSavedModelImpl() override {}
  
  Status Init() override {
    return Status::OK();
  }
  
  Status Predict(Request& req, Response& resp) override {
    return Status::OK();
  }

  Status GetServingModelInfo(ServingModelInfo& model_info) override {
    return Status::OK();
  }

  Status Rollback() override {
    return Status::OK();
  }

  std::string DebugString() override {
    return std::string();
  }

  SignatureDef GetServingSignatureDef() override {
    SignatureDef def;
    return def;
  }

  const SignatureInfo* GetSignatureInfo() override {
    return nullptr;
  }
};

class SavedModelImpl : public ModelImpl {
 public:
  explicit SavedModelImpl(ModelConfig* config);
  ~SavedModelImpl() override;

  Status Init() override;
  Status Predict(Request& req, Response& resp) override;
  Status GetServingModelInfo(ServingModelInfo& model_info) override;
  Status Rollback() override;
  std::string DebugString() override;
  SignatureDef GetServingSignatureDef() override;
  const SignatureInfo* GetSignatureInfo() override;

 private:
  ModelConfig* model_config_;
  IModelInstanceMgr* instance_mgr_ = nullptr;
};

class ModelImplFactory {
 public:
  static ModelImpl* Create(ModelConfig* config) {
    return new SavedModelImpl(config);
  }
};

} // processor
} // tensorflow

#endif // SERVING_PROCESSOR_SERVING_MODEL_IMPL_H

