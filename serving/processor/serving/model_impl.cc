#include "serving/processor/serving/model_impl.h"
#include "serving/processor/serving/model_config.h"
#include "serving/processor/serving/model_instance.h"
#include "serving/processor/serving/model_message.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace processor {
SavedModelImpl::SavedModelImpl(ModelConfig* config) :
  model_config_(config) {
}

SavedModelImpl::~SavedModelImpl() {
  delete instance_mgr_;
  delete model_config_;
}

Status SavedModelImpl::Init() {
  instance_mgr_ = ModelInstanceMgrFactory::Create(model_config_);
  return instance_mgr_->Init();
}

Status SavedModelImpl::Predict(Request& req, Response& resp) {
  return instance_mgr_->Predict(req, resp);
}

Status SavedModelImpl::GetServingModelInfo(ServingModelInfo& model_info) {
  return instance_mgr_->GetServingModelInfo(model_info);
}

Status SavedModelImpl::Rollback() {
  return instance_mgr_->Rollback();
}
 
std::string SavedModelImpl::DebugString() {
  return instance_mgr_->DebugString();
}

SignatureDef SavedModelImpl::GetServingSignatureDef() {
  return instance_mgr_->GetServingSignatureDef();
}

const SignatureInfo* SavedModelImpl::GetSignatureInfo() {
  return instance_mgr_->GetSignatureInfo();
}

} // processor
} // tensorflow
