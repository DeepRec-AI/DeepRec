#include "odl_processor/core/graph_optimizer.h"
#include "odl_processor/serving/tf_predict.pb.h"
#include "model_serving.h"

namespace tensorflow {
namespace processor {

class ModelImpl {
 public:
  virtual ~ModelImpl() = 0;
  virtual Status Load(const char* model_dir) = 0;
  virtual Status Predict(const eas::PredictRequest& req,
                         const eas::PredictResponse& resp) = 0;
  virtual void GraphOptimize() = 0;
};

class FreezeSavedModelImpl : public ModelImpl {
 public:
  ~FreezeSavedModelImpl() override {
    // TODO
  }

  Status Load(const char* model_dir) override {
    return Status::OK();
  }

  Status Predict(const eas::PredictRequest& req,
                 const eas::PredictResponse& resp) override {
    return Status::OK();
  }

  void GraphOptimize() {
    // TODO
  }
};

class SavedModelImpl : public ModelImpl {
 public:
  SavedModelImpl()
    : saved_model_bundle_(nullptr),
      optimizer_(nullptr) {
  }

  ~SavedModelImpl() override {
    delete saved_model_bundle_;
    delete optimizer_;
  }

  Status Load(const char* model_dir) override {
    // TODO: create SavedModelBundle
    return Status::OK();
  }

  Status Predict(const eas::PredictRequest& req,
                 const eas::PredictResponse& resp) override {
    return Status::OK();
  }

  void GraphOptimize() {
    GraphOptimizerOptions opts;
    // TODO: please set flag 'cache_sparse_locally',
    // can get the value from config, default is false.
    optimizer_ = new SavedModelOptimizer(saved_model_bundle_, opts);
    optimizer_->Optimize();
  }

 private:
  SavedModelBundle* saved_model_bundle_;
  SavedModelOptimizer* optimizer_;
};

class ModelImplFactory {
 public:
  static ModelImpl* Create() {
    return new SavedModelImpl();
  }
};

Model::Model(const ModelConfig& config) {
  impl_ = ModelImplFactory::Create();
}

Status Model::Load(const char* model_dir) {
  return impl_->Load(model_dir);
}

Status Model::Predict(const eas::PredictRequest& req,
                      const eas::PredictResponse& resp) {
  return impl_->Predict(req, resp);
}

} // processor
} // tensorflow
