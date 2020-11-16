#include "odl_processor/serving/tf_predict.pb.h"
#include "model_serving.h"

namespace tensorflow {
namespace processor {

class ModelImpl {
 public:
  virtual Status Load(const char* model_dir) = 0;
  virtual Status Predict(const eas::PredictRequest& req,
                         const eas::PredictResponse& resp) = 0;
};

class FreezeSavedModelImpl : public ModelImpl {
 public:
  Status Load(const char* model_dir) override {
    return Status::OK();
  }

  Status Predict(const eas::PredictRequest& req,
                 const eas::PredictResponse& resp) override {
    return Status::OK();
  }
};

class SavedModelImpl : public ModelImpl {
 public:
  Status Load(const char* model_dir) override {
    return Status::OK();
  }

  Status Predict(const eas::PredictRequest& req,
                 const eas::PredictResponse& resp) override {
    return Status::OK();
  }
};

class ModelImplFactory {
 public:
  ModelImpl* Create() {
    return new SavedModelImpl();
  }
};

Model::Model(const ModelConfig& config) {
  model_impl_ = ModelImplFactory::Create();
}

Status Model::Load(const char* model_dir) {
  return model_impl_->Load(model_dir);
}

Status Model::Predict(const eas::PredictRequest& req,
                      const eas::PredictResponse& resp) {
  return model_impl_->Predict(req, resp);
}

} // processor
} // tensorflow
