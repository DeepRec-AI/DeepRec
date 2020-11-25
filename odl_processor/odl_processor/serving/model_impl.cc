#include "odl_processor/serving/model_impl.h"
#include "odl_processor/serving/model_config.h"
#include "odl_processor/serving/model_instance.h"

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

Status SavedModelImpl::Init(const char* root_dir) {
  instance_mgr_ = new ModelInstanceMgr(root_dir, model_config_);
  return instance_mgr_->Init(session_options_, run_options_);
}

Status SavedModelImpl::Predict(
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_tensor_names,
    std::vector<Tensor>* outputs) {
  return instance_mgr_->Predict(inputs, output_tensor_names, outputs);
}

Status SavedModelImpl::Rollback() {
  return instance_mgr_->Rollback();
}
 
std::string SavedModelImpl::DebugString() {
  return instance_mgr_->DebugString();
}

} // processor
} // tensorflow
