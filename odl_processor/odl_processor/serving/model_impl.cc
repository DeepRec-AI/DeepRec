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
  //delete saved_model_bundle_;
  delete optimizer_;
  delete instance_mgr_;
  delete model_config_;
}

// 1 meta_graph_def = ReadMetaGraphDefFromSavedModel
// 2 bundle->meta_graph_def = OptimizeGraph(signature_name, meta_graph_def)
// 3 bundle->session = LoadMetaGraphIntoSession
// 4 RunRestore(bundle)
// 5 UpdateData(oss, embedding_service)
//
// 2 bundle object --> VersionManager
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
