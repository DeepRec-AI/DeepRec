#include "odl_processor/serving/tf_predict.pb.h"
#include "model_impl.h"
#include "model_serving.h"
#include "model_config.h"

namespace tensorflow {
namespace processor {

Model::~Model() {
  delete impl_;
}

Status Model::Load(const char* model_config, const char* model_dir) {
  ModelConfig* config = nullptr;
  auto status = ModelConfigFactory::Create(model_config, &config);
  if (!status.ok()) {
    return status;
  }

  impl_ = ModelImplFactory::Create(config);
  return impl_->Load(model_dir);
}

Status Model::Predict(const eas::PredictRequest& req,
                      eas::PredictResponse* resp) {
  return impl_->Predict(req, resp);
}

Status Model::Predict(const RunRequest& req,
                      RunResponse* resp) {
  return Status::OK();
}

Status Model::Warmup() {
  return impl_->Warmup();
}

std::string Model::DebugString() {
  return impl_->DebugString();
}

} // processor
} // tensorflow
