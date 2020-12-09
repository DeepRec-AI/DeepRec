#include "odl_processor/serving/model_impl.h"
#include "odl_processor/serving/model_serving.h"
#include "odl_processor/serving/model_config.h"
#include "odl_processor/serving/model_message.h"

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace processor {

Model::~Model() {
  delete impl_;
}

Status Model::Init(const char* model_config, const char* model_dir) {
  ModelConfig* config = nullptr;
  auto status = ModelConfigFactory::Create(model_config, &config);
  if (!status.ok()) {
    return status;
  }

  impl_ = ModelImplFactory::Create(config);
  return impl_->Init(model_dir);
}

Status Model::Predict(Request& req, Response& resp) {
  return impl_->Predict(req, resp);
}

Status Model::Rollback() {
  return impl_->Rollback();
}

std::string Model::DebugString() {
  return impl_->DebugString();
}

} // processor
} // tensorflow
