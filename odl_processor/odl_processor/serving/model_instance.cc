#include <fstream>
#include "odl_processor/serving/model_instance.h"
#include "odl_processor/serving/model_session.h"
#include "odl_processor/serving/tf_predict.pb.h"
#include "odl_processor/serving/util.h"
#include "odl_processor/storage/model_store.h"
#include "odl_processor/storage/feature_store_mgr.h"
#include "odl_processor/framework/graph_optimizer.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/platform/protobuf_internal.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"

namespace tensorflow {
namespace processor {
namespace {
constexpr int _60_Seconds = 60;

Tensor CreateTensor(const TensorInfo& tensor_info) {
  Tensor tensor(tensor_info.dtype(),
      TensorShape(tensor_info.tensor_shape())); 

  switch(tensor.dtype()) {
    case DT_FLOAT: {
      auto flat = tensor.flat<float>();
      for (int i = 0; i < flat.size(); ++i) flat(i) = 1;
      break;
    }
    case DT_DOUBLE: {
      auto flat = tensor.flat<double>();
      for (int i = 0; i < flat.size(); ++i) flat(i) = 1;
      break;
    }
    case DT_UINT32: {
      auto flat = tensor.flat<uint32>();
      for (int i = 0; i < flat.size(); ++i) flat(i) = 1;
      break;
    }
    case DT_INT32: {
      auto flat = tensor.flat<int32>();
      for (int i = 0; i < flat.size(); ++i) flat(i) = 1;
      break;
    }
    case DT_INT64: {
      auto flat = tensor.flat<int64>();
      for (int i = 0; i < flat.size(); ++i) flat(i) = 1;
      break;
    }
    case DT_UINT64: {
      auto flat = tensor.flat<uint64>();
      for (int i = 0; i < flat.size(); ++i) flat(i) = 1;
      break;
    }
    default:
      LOG(ERROR) << "can't support dtype:" << tensor.dtype();
  }

  return tensor;
}

Call CreateWarmupParams(SignatureDef& sig_def) {
  Call call;
  for (auto it : sig_def.inputs()) {
    const auto& tensor = CreateTensor(it.second);
    call.request.inputs.emplace_back(it.first, tensor);
  }

  for (auto it : sig_def.outputs()) {
    call.request.output_tensor_names.emplace_back(it.first);
  }

  return call; 
}

Call CreateWarmupParams(SignatureDef& sig_def,
                        const std::string& warmup_file_name) {
  // Parse warmup file
  eas::PredictRequest request;
  std::fstream input(warmup_file_name, std::ios::in | std::ios::binary);
  request.ParseFromIstream(&input);
  input.close();

  Call call;
  for (auto& input : request.inputs()) {
    call.request.inputs.emplace_back(input.first,
        util::Proto2Tensor(input.second));
  }

  call.request.output_tensor_names =
      std::vector<std::string>(request.output_filter().begin(),
                               request.output_filter().end());

  // User need to set fetches
  if (call.request.output_tensor_names.size() == 0) {
    LOG(FATAL) << "warmup file must be contain fetches.";
  }

  return call; 
}

bool ShouldWarmup(SignatureDef& sig_def) {
  for (auto it : sig_def.inputs()) {
    if (it.second.dtype() == DT_STRING) return false;
  }
  return true;
}

} // namespace

LocalSessionInstance::LocalSessionInstance(
    SessionOptions* sess_options,
    RunOptions* run_options) :
    warmup_file_name_(""),
    session_options_(sess_options),
    run_options_(run_options) {
}

Status LocalSessionInstance::Init(ModelConfig* config,
    ModelStore* model_store) {
  model_store->GetLatestVersion(version_);
  while (version_.SavedModelEmpty() || version_.CkptEmpty()) {
    // Wait until saved model meta file ready
    LOG(INFO) << "[Model Instance] SavedModel or Checkpoint dir is empty,"
              << "will try 1 minute later, current version: "
              << version_.DebugString();
    sleep(60);
    model_store->GetLatestVersion(version_);
  }

  TF_RETURN_IF_ERROR(ReadMetaGraphDefFromSavedModel(
        version_.savedmodel_dir.c_str(),
        {kSavedModelTagServe}, &meta_graph_def_));

  warmup_file_name_ = config->warmup_file_name;

  GraphOptimizerOption option;
  option.native_tf_mode = true;

  optimizer_ = new SavedModelOptimizer(config->signature_name,
      &meta_graph_def_, option);
  TF_RETURN_IF_ERROR(optimizer_->Optimize());
  
  TF_RETURN_IF_ERROR(ReadModelSignature(config));

  session_mgr_ = new ModelSessionMgr(meta_graph_def_,
      session_options_, run_options_);

  // Load full model
  TF_RETURN_IF_ERROR(session_mgr_->CreateModelSession(version_,
        version_.full_ckpt_name.c_str(),
        /*is_incr_ckpt*/false, config));

  // Load delta model if existed
  if (version_.delta_ckpt_name.empty()) {
    return Status::OK();
  }
  return session_mgr_->CreateModelSession(version_,
      version_.delta_ckpt_name.c_str(),
      /*is_incr_ckpt*/true, config);
}

Status LocalSessionInstance::ReadModelSignature(ModelConfig* model_config) {
  auto model_signatures = meta_graph_def_.signature_def();
  for (auto it : model_signatures) {
    if (it.first == model_config->signature_name) {
      model_signature_ = it;
      return Status::OK();
    }
  }
  return Status(error::Code::INVALID_ARGUMENT,
      "Invalid signature name, please check signature_name in model config");
}

Status LocalSessionInstance::Predict(Request& req, Response& resp) {
  return session_mgr_->LocalPredict(req, resp);
}

Status LocalSessionInstance::Warmup(
    ModelSession* warmup_session) {
  if (warmup_file_name_.empty() &&
      !ShouldWarmup(model_signature_.second)) {
    return Status::OK();
  }

  Call call;
  if (warmup_file_name_.empty()) {
    call = CreateWarmupParams(model_signature_.second);
  } else {
    call = CreateWarmupParams(model_signature_.second,
                              warmup_file_name_);
  }

  if (warmup_session) {
    return warmup_session->LocalPredict(
        call.request, call.response);
  }

  return session_mgr_->LocalPredict(
      call.request, call.response);
}

std::string LocalSessionInstance::DebugString() {
  return model_signature_.second.DebugString();
}

Status LocalSessionInstance::FullModelUpdate(
    const Version& version, ModelConfig* model_config) {
  ModelSession* new_model_session = nullptr;

  TF_RETURN_IF_ERROR(
      session_mgr_->CreateModelSession(version,
          version.full_ckpt_name.c_str(),
          /*is_incr_ckpt*/false, model_config,
          &new_model_session));

  // warmup model
  Warmup(new_model_session);

  session_mgr_->ResetServingSession(new_model_session);
  UpdateVersion(new_model_session->GetVersion());

  return Status::OK();
}

Status LocalSessionInstance::DeltaModelUpdate(
    const Version& version, ModelConfig* model_config) {
  TF_RETURN_IF_ERROR(
      session_mgr_->CreateModelSession(version,
          version.delta_ckpt_name.c_str(),
          /*is_incr_ckpt*/true, model_config));

  // Delta model update: No need to warmup model and
  // reset serving session, we don't create a new session.

  UpdateVersion(version);

  return Status::OK();
}

RemoteSessionInstance::RemoteSessionInstance(
    SessionOptions* sess_options,
    RunOptions* run_options,
    StorageOptions* storage_options) :
    session_options_(sess_options),
    run_options_(run_options),
    storage_options_(storage_options) {
}

Status RemoteSessionInstance::ReadModelSignature(ModelConfig* model_config) {
  auto model_signatures = meta_graph_def_.signature_def();
  for (auto it : model_signatures) {
    if (it.first == model_config->signature_name) {
      model_signature_ = it;
      return Status::OK();
    }
  }
  return Status(error::Code::INVALID_ARGUMENT,
      "Invalid signature name, please check signature_name in model config");
}

Status RemoteSessionInstance::RecursionCreateSession(const Version& version,
    IFeatureStoreMgr* sparse_storage, ModelConfig* model_config) {
  TF_RETURN_IF_ERROR(session_mgr_->CreateModelSession(version,
        version.full_ckpt_name.c_str(),
        sparse_storage, /*is_incr_ckpt*/false,
        /*is_initialize*/storage_options_->is_init_storage_,
        model_config));

  if (version.delta_ckpt_name.empty()) {
    return Status::OK();
  } else {
    return session_mgr_->CreateModelSession(version,
        version.delta_ckpt_name.c_str(),
        sparse_storage, /*is_incr_ckpt*/true,
        /*is_initialize*/storage_options_->is_init_storage_,
        model_config);
  }
}

Status RemoteSessionInstance::Init(ModelConfig* model_config,
    ModelStore* model_store, bool active) {
  ModelConfig serving_model_config(*model_config);
  serving_model_config.redis_db_idx =
      storage_options_->serving_storage_db_index_;
  serving_storage_ = new FeatureStoreMgr(&serving_model_config);

  ModelConfig backup_model_config(*model_config);
  backup_model_config.redis_db_idx =
      storage_options_->backup_storage_db_index_;
  backup_storage_ = new FeatureStoreMgr(&backup_model_config);

  warmup_file_name_ = model_config->warmup_file_name;

  // set active flag
  serving_storage_->SetStorageActiveStatus(active);

  Version version;
  model_store->GetLatestVersion(version);

  while (version.SavedModelEmpty()) {
    // Wait until saved model meta file ready
    LOG(INFO) << "[Model Instance] SavedModel dir is empty,"
              << "will try 1 minute later.";
    sleep(60);
    model_store->GetLatestVersion(version);
  }

  auto savedmodel_dir = version.savedmodel_dir;
  TF_RETURN_IF_ERROR(ReadMetaGraphDefFromSavedModel(
        savedmodel_dir.c_str(),
        {kSavedModelTagServe}, &meta_graph_def_));

  GraphOptimizerOption option;
  option.native_tf_mode = false;
  optimizer_ = new SavedModelOptimizer(model_config->signature_name,
      &meta_graph_def_, option);
  TF_RETURN_IF_ERROR(optimizer_->Optimize());

  session_mgr_ = new ModelSessionMgr(meta_graph_def_,
      session_options_, run_options_);

  TF_RETURN_IF_ERROR(ReadModelSignature(model_config));

  while (version.CkptEmpty()) {
    LOG(INFO) << "[Model Instance] Checkpoint dir is empty,"
              << "will try 1 minute later.";
    sleep(60);
    model_store->GetLatestVersion(version);
  }

  // update instance version
  version_ = version;

  return RecursionCreateSession(version, serving_storage_, model_config);
}

Status RemoteSessionInstance::Predict(Request& req, Response& resp) {
  return session_mgr_->Predict(req, resp);
}

Status RemoteSessionInstance::Warmup(
    ModelSession* warmup_session) {
  if (warmup_file_name_.empty() &&
      !ShouldWarmup(model_signature_.second)) {
    return Status::OK();
  }

  Call call;
  if (warmup_file_name_.empty()) {
    call = CreateWarmupParams(model_signature_.second);
  } else {
    call = CreateWarmupParams(model_signature_.second,
                              warmup_file_name_);
  }

  if (warmup_session) {
    return warmup_session->Predict(
        call.request, call.response);
  }

  return session_mgr_->Predict(
      call.request, call.response);
}

Status RemoteSessionInstance::FullModelUpdate(
    const Version& version, ModelConfig* model_config) {
  ModelSession* new_model_session = nullptr;

  // Logically backup_storage_ shouldn't serving now.
  backup_storage_->Reset();
  TF_RETURN_IF_ERROR(session_mgr_->CreateModelSession(version,
      version.full_ckpt_name.c_str(), backup_storage_,
      /*is_incr_ckpt*/false, /*is_initialize*/false,
      model_config, &new_model_session));

  // warmup model
  Warmup(new_model_session);

  session_mgr_->ResetServingSession(new_model_session);
  std::swap(backup_storage_, serving_storage_);
  UpdateVersion(new_model_session->GetVersion());

  return Status::OK();
}

Status RemoteSessionInstance::DeltaModelUpdate(
    const Version& version, ModelConfig* model_config) {
  ModelSession* new_model_session = nullptr;

  TF_RETURN_IF_ERROR(
      session_mgr_->CreateModelSession(version,
          version.delta_ckpt_name.c_str(), serving_storage_,
          /*is_incr_ckpt*/true, /*is_initialize*/false,
          model_config, &new_model_session));

  // warmup model
  Warmup(new_model_session);

  session_mgr_->ResetServingSession(new_model_session);
  UpdateVersion(new_model_session->GetVersion());

  return Status::OK();
}

std::string RemoteSessionInstance::DebugString() {
  return model_signature_.second.DebugString();
}

LocalSessionInstanceMgr::LocalSessionInstanceMgr(ModelConfig* config)
    : ModelUpdater(config) {
  session_options_ = new SessionOptions();
  //session_options_->target = target;
  session_options_->config.set_intra_op_parallelism_threads(config->inter_threads);
  session_options_->config.set_inter_op_parallelism_threads(config->intra_threads);
  //session_options_->config.mutable_gpu_options()->set_allocator_type("CPU");
  run_options_ = new RunOptions();
}

LocalSessionInstanceMgr::~LocalSessionInstanceMgr() {
  is_stop_ = true;

  delete instance_;
  delete session_options_;
  delete run_options_;
}

Status LocalSessionInstanceMgr::Init() {
  instance_ = new LocalSessionInstance(session_options_, run_options_);
  TF_RETURN_IF_ERROR(instance_->Init(model_config_,
      model_store_));
  TF_RETURN_IF_ERROR(instance_->Warmup());

  thread_ = new std::thread(&ModelUpdater::WorkLoop, this);
  return Status::OK();
}

Status LocalSessionInstanceMgr::Predict(Request& req, Response& resp) {
  return instance_->Predict(req, resp);
}

Status LocalSessionInstanceMgr::Rollback() {
  return Status(error::Code::NOT_FOUND, "TF Processor can't support Rollback.");
}

std::string LocalSessionInstanceMgr::DebugString() {
  return instance_->DebugString();
}

Status LocalSessionInstanceMgr::FullModelUpdate(
    const Version& version, ModelConfig* model_config) {
  return instance_->FullModelUpdate(
      version, model_config);
}

Status LocalSessionInstanceMgr::DeltaModelUpdate(
    const Version& version, ModelConfig* model_config) {
  return instance_->DeltaModelUpdate(
      version, model_config);
}

Version LocalSessionInstanceMgr::GetVersion() {
  return instance_->GetVersion();
}


RemoteSessionInstanceMgr::RemoteSessionInstanceMgr(ModelConfig* config)
    : ModelUpdater(config) {
  session_options_ = new SessionOptions();
  //session_options_->target = target;
  session_options_->config.set_intra_op_parallelism_threads(config->inter_threads);
  session_options_->config.set_inter_op_parallelism_threads(config->intra_threads);
  //session_options_->config.mutable_gpu_options()->set_allocator_type("CPU");
  run_options_ = new RunOptions();

  std::unique_ptr<FeatureStoreMgr> tmp_storage(
      new FeatureStoreMgr(model_config_));
  // Get 'active' and 'model_version' of DB-0 and DB-1
  // This will be abstracted as an interface below.
  StorageMeta storage_meta;
  Status s = tmp_storage->GetStorageMeta(&storage_meta);
  if (!s.ok()) {
    LOG(FATAL) << "Get storage meta data failed. "
               << s.error_message();
  }

  tmp_storage->GetStorageOptions(
      storage_meta,
      &cur_inst_storage_options_,
      &base_inst_storage_options_);
}

RemoteSessionInstanceMgr::~RemoteSessionInstanceMgr() {
  is_stop_ = true;

  delete base_instance_;
  delete cur_instance_;
  delete session_options_;
  delete run_options_;
  delete cur_inst_storage_options_;
  delete base_inst_storage_options_;
}

Status RemoteSessionInstanceMgr::Init() {
  /*Version version;
  auto status = model_store_->GetLatestVersion(version);
  if (!status.ok()) {
    return status;
  }*/

  TF_RETURN_IF_ERROR(CreateInstances());
  
  thread_ = new std::thread(&ModelUpdater::WorkLoop, this);
  return Status::OK();
}

Status RemoteSessionInstanceMgr::CreateInstances() {
  cur_instance_ = new RemoteSessionInstance(
      session_options_, run_options_,
      cur_inst_storage_options_);
  TF_RETURN_IF_ERROR(cur_instance_->Init(model_config_,
      model_store_, true));
  TF_RETURN_IF_ERROR(cur_instance_->Warmup());

  base_instance_ = new RemoteSessionInstance(
      session_options_, run_options_,
      base_inst_storage_options_);
  TF_RETURN_IF_ERROR(base_instance_->Init(model_config_,
      model_store_, false));
  return base_instance_->Warmup();
}

Status RemoteSessionInstanceMgr::Predict(Request& req, Response& resp) {
  return cur_instance_->Predict(req, resp);
}

Status RemoteSessionInstanceMgr::Rollback() {
  if (cur_instance_->GetVersion() == base_instance_->GetVersion()) {
    LOG(WARNING) << "[Processor] Already rollback to base model.";
    return Status::OK();
  }
  std::swap(cur_instance_, base_instance_);
  // TODO: Reset base_instance, shouldn't rollback again.
  // base_instance_->Reset();
  return Status::OK();
}

std::string RemoteSessionInstanceMgr::DebugString() {
  return cur_instance_->DebugString();
}

Status RemoteSessionInstanceMgr::FullModelUpdate(const Version& version,
                                       ModelConfig* model_config) {
  return cur_instance_->FullModelUpdate(
      version, model_config);
}

Status RemoteSessionInstanceMgr::DeltaModelUpdate(const Version& version,
                                        ModelConfig* model_config) {
  // NOTE: will initialize base_instance storage once after
  // a newly full model was updated.
  if (cur_instance_->GetVersion().IsSameFullModel(version) &&
      !base_instance_->GetVersion().IsSameFullModel(version)) {
    TF_RETURN_IF_ERROR(base_instance_->FullModelUpdate(
          cur_instance_->GetVersion(), model_config));
  }

  return cur_instance_->DeltaModelUpdate(
      version, model_config);
}

Version RemoteSessionInstanceMgr::GetVersion() {
  return cur_instance_->GetVersion();
}


ModelUpdater::ModelUpdater(ModelConfig* config)
    : model_store_(new ModelStore(config)),
      model_config_(config) {
  model_store_->Init();
}

ModelUpdater::~ModelUpdater() {
  is_stop_ = true;
  if (thread_) {
    thread_->join();
    delete thread_;
  }

  delete model_store_;
}

Status ModelUpdater::ModelUpdate(const Version& version,
                             ModelConfig* model_config) {
  if (version.IsFullModel()) {
    return FullModelUpdate(version, model_config);
  } else {
    return DeltaModelUpdate(version, model_config);
  }
}

void ModelUpdater::WorkLoop() {
  while(!is_stop_) {
    Version version;
    auto status = model_store_->GetLatestVersion(version);
    LOG(INFO) << "[Processor] ModelUpdater::WorkLoop get latest version: "
              << version.DebugString();
    if (!status.ok()) {
      LOG(WARNING) << "[Processor] Can't get latest model, will try 60 seconds later. "
                   << status.error_message() << std::endl;
    }

    if (GetVersion() < version) {
      auto status = ModelUpdate(version, model_config_);
      if (!status.ok()) {
        LOG(ERROR) << status.error_message() << std::endl;
      }
    }

    sleep(_60_Seconds);
  }
}

} // processor
} // tensorflow
