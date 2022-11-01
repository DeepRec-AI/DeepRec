#include <fstream>
#include "serving/processor/serving/message_coding.h"
#include "serving/processor/serving/model_instance.h"
#include "serving/processor/serving/model_partition.h"
#include "serving/processor/serving/model_session.h"
#include "serving/processor/serving/predict.pb.h"
#include "serving/processor/serving/util.h"
#include "serving/processor/storage/model_store.h"
#include "serving/processor/storage/feature_store_mgr.h"
#include "serving/processor/framework/graph_optimizer.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/platform/protobuf_internal.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"

using tensorflow::kPredictMethodName;

namespace tensorflow {
namespace processor {
namespace {
constexpr int _60_Seconds = 60;
constexpr int MAX_TRY_COUNT = 10;
constexpr int WARMUP_COUNT = 5;

Tensor CreateTensor(const TensorInfo& tensor_info) {
  auto real_ts = tensor_info.tensor_shape();
  // set batch_size to 1 when the default value is -1
  if (real_ts.dim(0).size() < 0) {
    real_ts.mutable_dim(0)->set_size(1);
  }
  Tensor tensor(tensor_info.dtype(), TensorShape(real_ts));

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

Status CreateWarmupParams(SignatureDef& sig_def, Call* call) {
  for (auto it : sig_def.inputs()) {
    const auto& tensor = CreateTensor(it.second);
    call->request.inputs.emplace_back(it.second.name(), tensor);
  }

  for (auto it : sig_def.outputs()) {
    call->request.output_tensor_names.emplace_back(it.second.name());
  }

  return Status::OK();
}

Status CreateWarmupParams(SignatureDef& sig_def,
                          const std::string& warmup_file_name,
                          Call* call, IParser* parser,
                          const SignatureInfo& signature_info) {
  // Parse warmup file
  eas::PredictRequest request;
  std::fstream input(warmup_file_name, std::ios::in | std::ios::binary);
  bool success = request.ParseFromIstream(&input);
  if (!success) {
    LOG(ERROR) << "Read warmp file failed: " << warmup_file_name;
    return Status(error::Code::INTERNAL,
        "Read warmp file failed, please check warmp file path");
  }
  input.close();

  return parser->ParseRequest(request, &signature_info, *call);
}

bool ShouldWarmup(SignatureDef& sig_def) {
  for (auto it : sig_def.inputs()) {
    if (it.second.dtype() == DT_STRING) return false;
  }
  return true;
}

void StringReplace(std::string& strBig, const std::string& strsrc,
                   const std::string& strdst) {
  std::string::size_type pos = 0;
  std::string::size_type srclen = strsrc.size();
  std::string::size_type dstlen = strdst.size();

  while ((pos = strBig.find(strsrc, pos)) != std::string::npos) {
    strBig.replace(pos, srclen, strdst);
    pos += dstlen;
  }
}
void GenerateJsonSignatureFormat(
    const std::pair<std::string, SignatureDef>& signature,
    std::string& json_signature) {
  std::map<int, std::string> dtype_to_string = {
      {1, "DT_FLOAT"}, {2, "DT_DOUBLE"}, {3, "DT_INT32"}, {4, "DT_UINT8"},
      {6, "DT_INT8"},  {7, "DT_STRING"}, {9, "DT_INT64"}, {10, "DT_BOOL"}};
  std::ostringstream model_signature;
  if (signature.second.method_name() == kPredictMethodName) {
    model_signature << "{";
    model_signature << "\"signature_name\": \"" << signature.first << "\",";
    model_signature << "\"inputs\": [";
    LOG(INFO) << "Inputs:";
    for (auto& input : (signature.second).inputs()) {
      model_signature << "{";
      model_signature << "\"name\": \"" << input.first << "\",";
      std::stringstream signature_input_info;
      signature_input_info << input.first + ": [";
      model_signature << "\"shape\": [";
      int dims = input.second.tensor_shape().dim_size();
      if (dims > 0) {
        for (int i = 0; i < dims - 1; i++) {
          signature_input_info << input.second.tensor_shape().dim(i).size();
          model_signature << input.second.tensor_shape().dim(i).size()
                          << ", ";
          signature_input_info << ", ";
        }
        signature_input_info
            << input.second.tensor_shape().dim(dims - 1).size();
        model_signature << input.second.tensor_shape().dim(dims - 1).size();
      }
      signature_input_info << "]; ";
      model_signature << "],";
      signature_input_info << dtype_to_string[input.second.dtype()];
      model_signature << "\"type\": \""
                      << dtype_to_string[input.second.dtype()] << "\"";
      LOG(INFO) << signature_input_info.str();
      model_signature << "},";
    }
    model_signature << "],";
    LOG(INFO) << "Outputs:";
    model_signature << "\"outputs\": [";
    for (auto& output : (signature.second).outputs()) {
      model_signature << "{";
      model_signature << "\"name\": \"" << output.first << "\",";
      std::stringstream signature_output_info;
      signature_output_info << output.first + ": [";
      model_signature << "\"shape\": [";
      int dims = output.second.tensor_shape().dim_size();
      if (dims > 0) {
        for (int i = 0; i < dims - 1; i++) {
          signature_output_info
              << output.second.tensor_shape().dim(i).size();
          model_signature << output.second.tensor_shape().dim(i).size()
                          << ", ";
          signature_output_info << ", ";
        }
        signature_output_info
            << output.second.tensor_shape().dim(dims - 1).size();
        model_signature
            << output.second.tensor_shape().dim(dims - 1).size();
      }
      signature_output_info << "]; ";
      model_signature << "],";
      signature_output_info << dtype_to_string[output.second.dtype()];
      model_signature << "\"type\": \""
                      << dtype_to_string[output.second.dtype()] << "\"";
      LOG(INFO) << signature_output_info.str();
      model_signature << "},";
    }
    model_signature << "]}";
  }
  json_signature = model_signature.str();
  StringReplace(json_signature, "},]", "}]");
}

void InternalGetSignatureInfo(
    const std::pair<std::string, SignatureDef>& signature,
    SignatureInfo& signature_info) {
  int idx = 0;
  for (auto& iter : signature.second.inputs()) {
    signature_info.input_key.emplace_back(iter.first);
    signature_info.input_value_name.emplace_back(iter.second.name());
    signature_info.input_key_idx[iter.first] = idx;
    signature_info.input_value_name_idx[iter.second.name()] = idx;
    ++idx;
  }

  idx = 0;
  for (auto& iter : signature.second.outputs()) {
    signature_info.output_key.emplace_back(iter.first);
    signature_info.output_value_name.emplace_back(iter.second.name());
    signature_info.output_key_idx[iter.first] = idx;
    signature_info.output_value_name_idx[iter.second.name()] = idx;
    ++idx;
  }
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
  // init local partition policy
  PartitionPolicy::GetGlobalPolicy()->Init(config);

  model_store->GetLatestVersion(version_);
  while (version_.SavedModelEmpty() ||
         (config->enable_incr_model_update && version_.CkptEmpty())) {
    if (config->enable_incr_model_update) {
      // Wait until saved model meta file ready
      LOG(INFO) << "[Model Instance] SavedModel or Checkpoint dir is empty,"
                << "will try 1 minute later, current version: "
                << version_.DebugString();
    } else {
      LOG(INFO) << "[Model Instance] SavedModel dir is empty,"
                << "will try 1 minute later, current version: "
                << version_.DebugString();
    }

    sleep(60);
    model_store->GetLatestVersion(version_);
  }

  TF_RETURN_IF_ERROR(ReadMetaGraphDefFromSavedModel(
        version_.savedmodel_dir.c_str(),
        {kSavedModelTagServe}, &meta_graph_def_));

  warmup_file_name_ = config->warmup_file_name;
  parser_ = ParserFactory::GetInstance(config->serialize_protocol, 4);

  GraphOptimizerOption option;
  option.native_tf_mode = true;
  if (config->shard_embedding) {
    option.shard_embedding = config->shard_embedding;
    option.shard_embedding_names = config->shard_embedding_names;
    option.partition_id = PartitionPolicy::GetGlobalPolicy()->GetEmbeddingGroupId();
    option.shard_instance_count =
        PartitionPolicy::GetGlobalPolicy()->GetShardInstanceCount();
  }

  option.st = config->storage_type;
  option.path = config->storage_path;
  option.size = config->storage_size;

  optimizer_ = new SavedModelOptimizer(config->signature_name,
      &meta_graph_def_, option);
  TF_RETURN_IF_ERROR(optimizer_->Optimize());

  TF_RETURN_IF_ERROR(ReadModelSignature(config));

  session_mgr_ = new ModelSessionMgr(meta_graph_def_,
      session_options_, run_options_);

  if (config->enable_incr_model_update) {
    return LoadModelFromCheckpoint(config);
  } else {
    return LoadSavedModel(config);
  }
}

Status LocalSessionInstance::LoadModelFromCheckpoint(
    ModelConfig* config) {
  // Load full model
  TF_RETURN_IF_ERROR(session_mgr_->CreateModelSession(version_,
      version_.full_ckpt_name.c_str(),
      version_.delta_ckpt_name.c_str(),
      /*is_incr_ckpt*/false, config));

  // Load delta model if existed
  if (version_.delta_ckpt_name.empty()) {
    return Status::OK();
  }

  return session_mgr_->CreateModelSession(version_,
      version_.full_ckpt_name.c_str(),
      version_.delta_ckpt_name.c_str(),
      /*is_incr_ckpt*/true, config);
}

Status LocalSessionInstance::LoadSavedModel(
    ModelConfig* config) {
  return session_mgr_->CreateModelSession(version_,
      version_.savedmodel_dir.c_str(), config);
}

Status LocalSessionInstance::ReadModelSignature(ModelConfig* model_config) {
  auto model_signatures = meta_graph_def_.signature_def();
  for (auto it : model_signatures) {
    if (it.first == model_config->signature_name) {
      model_signature_ = it;
      GenerateJsonSignatureFormat(model_signature_,
                                  model_json_signature_);
      InternalGetSignatureInfo(model_signature_,
                               signature_info_);
      return Status::OK();
    }
  }
  return Status(error::Code::INVALID_ARGUMENT,
      "Invalid signature name, please check signature_name in model config");
}

Status LocalSessionInstance::Predict(Request& req, Response& resp) {
  return session_mgr_->LocalPredict(req, resp);
}

Status LocalSessionInstance::GetServingModelInfo(
    ServingModelInfo& model_info) {
  return session_mgr_->GetServingModelInfo(model_info);
}

Status LocalSessionInstance::Warmup(
    ModelSession* warmup_session) {
  if (warmup_file_name_.empty() &&
      !ShouldWarmup(model_signature_.second)) {
    return Status::OK();
  }

  LOG(INFO) << "Try to warmup model: " << warmup_file_name_;
  Status s;
  Call call;
  if (warmup_file_name_.empty()) {
    s = CreateWarmupParams(model_signature_.second, &call);
  } else {
    s = CreateWarmupParams(model_signature_.second,
                           warmup_file_name_, &call,
                           parser_, signature_info_);
  }
  if (!s.ok()) {
    LOG(ERROR) << "Create warmup params failed, warmup will be canceled.";
    return s;
  }

  int left_try_count = WARMUP_COUNT;
  while (left_try_count > 0) {
    if (warmup_session) {
      s = warmup_session->Warmup(
          call.request, call.response);
    } else {
      s = session_mgr_->Warmup(
          call.request, call.response);
    }
    if (!s.ok()) return s;

    --left_try_count;
    call.response.outputs.clear();
  }
  LOG(INFO) << "Warmup model successful: " << warmup_file_name_;

  return Status::OK();
}

std::string LocalSessionInstance::DebugString() {
  return model_json_signature_;
}

SignatureDef LocalSessionInstance::GetServingSignatureDef() {
  return model_signature_.second;
}

const SignatureInfo* LocalSessionInstance::GetSignatureInfo() {
  return &signature_info_;
}

Status LocalSessionInstance::FullModelUpdate(
    const Version& version, ModelConfig* model_config) {
  ModelSession* new_model_session = nullptr;

  TF_RETURN_IF_ERROR(
      session_mgr_->CreateModelSession(version,
          version.full_ckpt_name.c_str(),
          version.delta_ckpt_name.c_str(),
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
          version.full_ckpt_name.c_str(),
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
      GenerateJsonSignatureFormat(model_signature_,
                                  model_json_signature_);
      InternalGetSignatureInfo(model_signature_,
                               signature_info_);
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
  parser_ = ParserFactory::GetInstance(model_config->serialize_protocol, 4);

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

Status RemoteSessionInstance::GetServingModelInfo(
    ServingModelInfo& model_info) {
  return session_mgr_->GetServingModelInfo(model_info);
}

Status RemoteSessionInstance::Warmup(
    ModelSession* warmup_session) {
  if (warmup_file_name_.empty() &&
      !ShouldWarmup(model_signature_.second)) {
    return Status::OK();
  }

  Status s;
  Call call;
  if (warmup_file_name_.empty()) {
    s = CreateWarmupParams(model_signature_.second, &call);
  } else {
    s = CreateWarmupParams(model_signature_.second,
                           warmup_file_name_, &call,
                           parser_, signature_info_);
  }
  if (!s.ok()) {
    LOG(ERROR) << "Create warmup params failed, warmup will be canceled.";
    return s;
  }

  int left_try_count = WARMUP_COUNT;
  while (left_try_count > 0) {
    if (warmup_session) {
      s = warmup_session->Warmup(
          call.request, call.response, false);
    } else {
      s = session_mgr_->Warmup(
          call.request, call.response, false);
    }
    if (!s.ok()) return s;

    --left_try_count;
    call.response.outputs.clear();
  }

  return Status::OK();
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
  return model_json_signature_;
}

SignatureDef RemoteSessionInstance::GetServingSignatureDef() {
  return model_signature_.second;
}

const SignatureInfo* RemoteSessionInstance::GetSignatureInfo() {
  return &signature_info_;
}

LocalSessionInstanceMgr::LocalSessionInstanceMgr(ModelConfig* config)
    : ModelUpdater(config) {
  session_options_ = new SessionOptions();
  //session_options_->target = target;
  session_options_->config.set_inter_op_parallelism_threads(config->inter_threads);
  session_options_->config.set_intra_op_parallelism_threads(config->intra_threads);
  session_options_->config.set_use_per_session_threads(config->use_per_session_threads);
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

  if (model_config_->enable_incr_model_update) {
    thread_ = new std::thread(&ModelUpdater::WorkLoop, this);
  }

  return Status::OK();
}

Status LocalSessionInstanceMgr::Predict(Request& req, Response& resp) {
  return instance_->Predict(req, resp);
}

Status LocalSessionInstanceMgr::GetServingModelInfo(
    ServingModelInfo& model_info) {
  return instance_->GetServingModelInfo(model_info);
}

Status LocalSessionInstanceMgr::Rollback() {
  return Status(error::Code::NOT_FOUND, "TF Processor can't support Rollback.");
}

std::string LocalSessionInstanceMgr::DebugString() {
  return instance_->DebugString();
}

SignatureDef LocalSessionInstanceMgr::GetServingSignatureDef() {
  return instance_->GetServingSignatureDef();
}

const SignatureInfo* LocalSessionInstanceMgr::GetSignatureInfo() {
  return instance_->GetSignatureInfo();
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
  session_options_->config.set_inter_op_parallelism_threads(config->inter_threads);
  session_options_->config.set_intra_op_parallelism_threads(config->intra_threads);
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

Status RemoteSessionInstanceMgr::GetServingModelInfo(
    ServingModelInfo& model_info) {
  return cur_instance_->GetServingModelInfo(model_info);
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

SignatureDef RemoteSessionInstanceMgr::GetServingSignatureDef() {
  return cur_instance_->GetServingSignatureDef();
}

const SignatureInfo* RemoteSessionInstanceMgr::GetSignatureInfo() {
  return cur_instance_->GetSignatureInfo();
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
    ModelConfig* model_config, bool new_full_ckpt_generated) {
  // Load new full model vesion
  if (version.IsFullModel()) {
    return FullModelUpdate(version, model_config);
  } else {
    // Load new full model vesion before incremental model be loaded.
    if (new_full_ckpt_generated) {
      TF_RETURN_IF_ERROR(FullModelUpdate(version, model_config));
    }
    return DeltaModelUpdate(version, model_config);
  }
}

void ModelUpdater::WorkLoop() {
  int try_count = 0;
  while(!is_stop_) {
    Version version;
    auto status = model_store_->GetLatestVersion(version);
    LOG(INFO) << "[Processor] ModelUpdater::WorkLoop get latest version: "
              << version.DebugString();
    if (!version.IsValid()) {
      try_count++;
      LOG(ERROR) << "[Processor] Found a invalid model, "
                 << "please check other error message, "
                 << "we will try 60 seconds later. status: " << status.error_message()
                 << "version debug string: " << version.DebugString();
      if (try_count >= MAX_TRY_COUNT) {
        LOG(FATAL) << "Try to get the latest model failed " << try_count << " times, "
                   << "please check the model directory or network.";
      }
    } else {
      try_count = 0;
      if (!status.ok()) {
        LOG(WARNING) << "[Processor] Not found full model or incremental model directory. "
                     << "Please ignore this warning if you confirm it. "
                     << "And we will try 60 seconds later. Warning message: "
                     << status.error_message();
      }

      // New model directory is generated or the version step is greater than the pre.
      Version pre_version = GetVersion();
      bool new_full_ckpt_generated = version.IsValid() &&
          (pre_version.full_ckpt_name != version.full_ckpt_name);
      if (new_full_ckpt_generated || pre_version < version) {
        LOG(INFO) << "Start to load new version model: " << version.DebugString();
        auto status = ModelUpdate(version, model_config_,
                                  new_full_ckpt_generated);
        if (!status.ok()) {
          LOG(ERROR) << "Load new version model failed: " << status.error_message()
                     << ", version info: " << version.DebugString();
        } else {
          LOG(INFO) << "Load new version model successful: " << version.DebugString();
        }
      }
    }

    sleep(_60_Seconds);
  }
}

} // processor
} // tensorflow
