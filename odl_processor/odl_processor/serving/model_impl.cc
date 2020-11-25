#include "odl_processor/core/graph_optimizer.h"
#include "odl_processor/serving/tf_predict.pb.h"
#include "odl_processor/serving/model_impl.h"
#include "odl_processor/serving/model_config.h"
#include "odl_processor/serving/run_predict.h"
#include "odl_processor/serving/model_instance.h"

namespace tensorflow {
namespace processor {
SavedModelImpl::SavedModelImpl(ModelConfig* config) :
  model_config_(config) {
}

SavedModelImpl::~SavedModelImpl() {
  delete optimizer_;
  delete instance_mgr_;
  delete model_config_;
}

Status SavedModelImpl::Init(const char* root_dir) {
  instance_mgr_ = new ModelInstanceMgr(root_dir, model_config_);
  return instance_mgr_->Init(session_options_, run_options_);
}

Status SavedModelImpl::Predict(const eas::PredictRequest& req,
               eas::PredictResponse* resp) {
  return instance_mgr_->Predict(req, resp);
}
 
std::string SavedModelImpl::DebugString() {
  return instance_mgr_->DebugString();
}

} // processor
} // tensorflow
