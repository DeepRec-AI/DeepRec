#include "odl_processor/serving/model_instance.h"
#include "odl_processor/serving/model_session.h"
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
Status LoadMetaGraphIntoSession(const MetaGraphDef& meta_graph_def,
                                const SessionOptions& session_options,
                                Session** session) {
  TF_RETURN_IF_ERROR(NewSession(session_options, session));
  return (*session)->Create(meta_graph_def.graph_def());
}

Tensor CreateStringTensor(const string& value) {
  Tensor tensor(DT_STRING, TensorShape({}));
  tensor.scalar<string>()() = value;
  return tensor;
}

Status GetAssetFileDefs(const MetaGraphDef& meta_graph_def,
                        std::vector<AssetFileDef>* asset_file_defs) {
  const auto& collection_def_map = meta_graph_def.collection_def();
  const auto assets_it = collection_def_map.find(kSavedModelAssetsKey);
  if (assets_it == collection_def_map.end()) {
    return Status::OK();
  }
  const auto& any_assets = assets_it->second.any_list().value();
  for (const auto& any_asset : any_assets) {
    AssetFileDef asset_file_def;
    TF_RETURN_IF_ERROR(
        ParseAny(any_asset, &asset_file_def, "tensorflow.AssetFileDef"));
    asset_file_defs->push_back(asset_file_def);
  }
  return Status::OK();
}

void AddAssetsTensorsToInputs(const StringPiece export_dir,
                              const std::vector<AssetFileDef>& asset_file_defs,
                              std::vector<std::pair<string, Tensor>>* inputs) {
  if (asset_file_defs.empty()) {
    return;
  }
  for (auto& asset_file_def : asset_file_defs) {
    Tensor assets_file_path_tensor = CreateStringTensor(io::JoinPath(
        export_dir, kSavedModelAssetsDirectory, asset_file_def.filename()));
    inputs->push_back(
        {asset_file_def.tensor_info().name(), assets_file_path_tensor});
  }
}

// Like Session::Run(), but uses the Make/Run/ReleaseCallable() API to avoid
// leaving behind non-GC'ed state.
//
// Detailed motivation behind this approach, from ashankar@:
//
// Each call to Session::Run() that identifies a new subgraph (based on feeds
// and fetches) creates some datastructures that live as long as the session
// (the partitioned graph, associated executors etc.).
//
// A pathological case of this would be if say the initialization op
// (main_op/legacy_init_op) involves the use of a large constant. Then we
// allocate memory for that large constant that will just stick around till the
// session dies. With this Callable mechanism, that memory will be released
// right after ReleaseCallable returns.
//
// However, the resource manager state remains.
Status RunOnce(const RunOptions& run_options,
               const std::vector<std::pair<string, Tensor>>& inputs,
               const std::vector<string>& output_tensor_names,
               const std::vector<string>& target_node_names,
               std::vector<Tensor>* outputs, RunMetadata* run_metadata,
               Session* session) {
  CallableOptions callable_options;
  std::vector<Tensor> feed_tensors;
  *callable_options.mutable_run_options() = run_options;
  for (const auto& input : inputs) {
    const string& name = input.first;
    const Tensor& tensor = input.second;
    callable_options.add_feed(name);
    feed_tensors.push_back(tensor);
  }
  for (const string& output_tensor_name : output_tensor_names) {
    callable_options.add_fetch(output_tensor_name);
  }
  for (const string& target_node_name : target_node_names) {
    callable_options.add_target(target_node_name);
  }

  Session::CallableHandle callable_handle;
  TF_RETURN_IF_ERROR(session->MakeCallable(callable_options, &callable_handle));
  const Status run_status = session->RunCallable(callable_handle, feed_tensors,
                                                 outputs, run_metadata);
  // Be sure to call ReleaseCallable() regardless of the outcome of
  // RunCallable().
  session->ReleaseCallable(callable_handle).IgnoreError();
  return run_status;
}

bool HasMainOp(const MetaGraphDef& meta_graph_def) {
  const auto& collection_def_map = meta_graph_def.collection_def();
  if (collection_def_map.find(kSavedModelMainOpKey) !=
      collection_def_map.end()) {
    return true;
  }
  return false;
}

Status RunMainOp(const RunOptions& run_options, const string& export_dir,
                 const MetaGraphDef& meta_graph_def,
                 const std::vector<AssetFileDef>& asset_file_defs,
                 Session* session, const string& main_op_key) {
  LOG(INFO) << "Running MainOp with key " << main_op_key
            << " on SavedModel bundle.";
  const auto& collection_def_map = meta_graph_def.collection_def();
  const auto main_op_it = collection_def_map.find(main_op_key);
  if (main_op_it != collection_def_map.end()) {
    if (main_op_it->second.node_list().value_size() != 1) {
      return errors::FailedPrecondition(
          strings::StrCat("Expected exactly one main op in : ", export_dir));
    }
    std::vector<std::pair<string, Tensor>> inputs;
    AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);
    RunMetadata run_metadata;
    const StringPiece main_op_name = main_op_it->second.node_list().value(0);
    return RunOnce(run_options, inputs, {}, {string(main_op_name)},
                   nullptr /* outputs */, &run_metadata, session);
  }
  return Status::OK();
}

Status RunRestore(const RunOptions& run_options, const string& export_dir,
                  const StringPiece restore_op_name,
                  const StringPiece variable_filename_const_op_name,
                  const std::vector<AssetFileDef>& asset_file_defs,
                  Session* session) {
  LOG(INFO) << "Restoring SavedModel bundle.";
  // Find path to variables to be restored in export directory.
  const string variables_directory =
      io::JoinPath(export_dir, kSavedModelVariablesDirectory);
  // Check for saver checkpoints in v2 format. Models exported in the checkpoint
  // v2 format will have a variables.index file. The corresponding
  // variables are stored in the variables.data-?????-of-????? files.
  const string variables_index_path = io::JoinPath(
      variables_directory, MetaFilename(kSavedModelVariablesFilename));
  if (!Env::Default()->FileExists(variables_index_path).ok()) {
    LOG(INFO) << "The specified SavedModel has no variables; no checkpoints "
                 "were restored. File does not exist: "
              << variables_index_path;
    return Status::OK();
  }
  const string variables_path =
      io::JoinPath(variables_directory, kSavedModelVariablesFilename);

  // Add variables to the graph.
  Tensor variables_path_tensor(DT_STRING, TensorShape({}));
  variables_path_tensor.scalar<string>()() = variables_path;

  std::vector<std::pair<string, Tensor>> inputs = {
      {string(variable_filename_const_op_name), variables_path_tensor}};

  AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);

  RunMetadata run_metadata;
  return RunOnce(run_options, inputs, {}, {string(restore_op_name)},
                 nullptr /* outputs */, &run_metadata, session);
}

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

bool ShouldWarmup(SignatureDef& sig_def) {
  for (auto it : sig_def.inputs()) {
    if (it.second.dtype() == DT_STRING) return false;
  }
  return true;
}
}
SingleSessionInstance::SingleSessionInstance(
    SessionOptions* sess_options,
    RunOptions* run_options) :
    session_options_(sess_options), run_options_(run_options) {
}

Status SingleSessionInstance::Init(ModelConfig* config,
    ModelStore* model_store) {
  model_store->GetLatestVersion(version_);
  if (version_.SavedModelEmpty()) {
    return Status(error::Code::NOT_FOUND, "SavedModel dir is invalid.");
  }

  TF_RETURN_IF_ERROR(ReadMetaGraphDefFromSavedModel(
        version_.savedmodel_dir.c_str(),
        {kSavedModelTagServe}, &meta_graph_def_));

  GraphOptimizerOption option;
  option.native_tf_mode = true;

  optimizer_ = new SavedModelOptimizer(config->signature_name,
      &meta_graph_def_, option);
  TF_RETURN_IF_ERROR(optimizer_->Optimize());
  
  TF_RETURN_IF_ERROR(ReadModelSignature(config));

  return LoadSavedModel(version_.savedmodel_dir); 
}

Status SingleSessionInstance::LoadSavedModel(
    const std::string& export_dir) {
  TF_RETURN_IF_ERROR(LoadMetaGraphIntoSession(
      meta_graph_def_, *session_options_, &session_));

  std::vector<AssetFileDef> asset_file_defs;
  TF_RETURN_IF_ERROR(
      GetAssetFileDefs(meta_graph_def_, &asset_file_defs));
  TF_RETURN_IF_ERROR(
      RunRestore(*run_options_, export_dir,
                 meta_graph_def_.saver_def().restore_op_name(),
                 meta_graph_def_.saver_def().filename_tensor_name(),
                 asset_file_defs, session_));
  if (HasMainOp(meta_graph_def_)) {
    TF_RETURN_IF_ERROR(RunMainOp(*run_options_, export_dir,
                                 meta_graph_def_, asset_file_defs,
                                 session_, kSavedModelMainOpKey));
  } else {
    TF_RETURN_IF_ERROR(RunMainOp(
        *run_options_, export_dir, meta_graph_def_, asset_file_defs,
        session_, kSavedModelLegacyInitOpKey));
  }
  return Status::OK(); 
}

Status SingleSessionInstance::ReadModelSignature(ModelConfig* model_config) {
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

Status SingleSessionInstance::Predict(Request& req, Response& resp) {
  return session_->Run(req.inputs, req.output_tensor_names, {},
      &resp.outputs);
}

Status SingleSessionInstance::Warmup() {
  if (!ShouldWarmup(model_signature_.second)) return Status::OK();
  Call call = CreateWarmupParams(model_signature_.second);
  return Predict(call.request, call.response);
}

std::string SingleSessionInstance::DebugString() {
  return model_signature_.second.DebugString();
}

MultipleSessionInstance::MultipleSessionInstance(
    SessionOptions* sess_options,
    RunOptions* run_options,
    StorageOptions* storage_options) :
    session_options_(sess_options),
    run_options_(run_options),
    storage_options_(storage_options) {
}

Status MultipleSessionInstance::ReadModelSignature(ModelConfig* model_config) {
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

Status MultipleSessionInstance::RecursionCreateSession(const Version& version,
    IFeatureStoreMgr* sparse_storage) {
  TF_RETURN_IF_ERROR(session_mgr_->CreateModelSession(version,
        version.full_ckpt_name.c_str(), sparse_storage, false));

  if (version.delta_ckpt_name.empty()) {
    return Status::OK();
  } else {
    return session_mgr_->CreateModelSession(version,
        version.delta_ckpt_name.c_str(), sparse_storage, true);
  }
}

Status MultipleSessionInstance::Init(ModelConfig* model_config,
    ModelStore* model_store) {
  int timeout = model_config->init_timeout_minutes;
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

  ModelConfig serving_model_config(*model_config);
  serving_model_config.redis_db_idx =
      storage_options_->serving_storage_db_index_;
  serving_storage_ = new FeatureStoreMgr(&serving_model_config);

  ModelConfig backup_model_config(*model_config);
  backup_model_config.redis_db_idx =
      storage_options_->backup_storage_db_index_;
  backup_storage_ = new FeatureStoreMgr(&backup_model_config);
  
  while (version.CkptEmpty()) {
    LOG(INFO) << "[Model Instance] Checkpoint dir is empty,"
              << "will try 1 minute later.";
    sleep(60);
    model_store->GetLatestVersion(version);
  }

  // update instance version
  version_ = version;

  return RecursionCreateSession(version, serving_storage_);
}

Status MultipleSessionInstance::Predict(Request& req, Response& resp) {
  return session_mgr_->Predict(req, resp);
}

Status MultipleSessionInstance::Warmup() {
  if (!ShouldWarmup(model_signature_.second)) return Status::OK();
  Call call = CreateWarmupParams(model_signature_.second);
  return Predict(call.request, call.response);
}

Status MultipleSessionInstance::FullModelUpdate(const Version& version) {
  // Logically backup_storage_ shouldn't serving now.
  backup_storage_->Reset();
  TF_RETURN_IF_ERROR(session_mgr_->CreateModelSession(version,
      version.full_ckpt_name.c_str(), backup_storage_, false));
 
  std::swap(backup_storage_, serving_storage_);
  return Status::OK();
}

Status MultipleSessionInstance::DeltaModelUpdate(const Version& version) {
  return session_mgr_->CreateModelSession(version,
      version.delta_ckpt_name.c_str(), serving_storage_, true);
}

std::string MultipleSessionInstance::DebugString() {
  return model_signature_.second.DebugString();
}

TFInstanceMgr::TFInstanceMgr(ModelConfig* config)
    : model_store_(new ModelStore(config)), model_config_(config) {
  model_store_->Init();
  session_options_ = new SessionOptions();
  //session_options_->target = target;
  session_options_->config.set_intra_op_parallelism_threads(config->inter_threads);
  session_options_->config.set_inter_op_parallelism_threads(config->intra_threads);
  //session_options_->config.mutable_gpu_options()->set_allocator_type("CPU");
  run_options_ = new RunOptions();
}

TFInstanceMgr::~TFInstanceMgr() {
  delete instance_;
  delete session_options_;
  delete run_options_;
  delete model_store_;
}

Status TFInstanceMgr::Init() {
  instance_ = new SingleSessionInstance(session_options_, run_options_);
  TF_RETURN_IF_ERROR(instance_->Init(model_config_,
      model_store_));
  return instance_->Warmup();
}

Status TFInstanceMgr::Predict(Request& req, Response& resp) {
  return instance_->Predict(req, resp);
}

Status TFInstanceMgr::Rollback() {
  return Status(error::Code::NOT_FOUND, "TF Processor can't support Rollback.");
}

std::string TFInstanceMgr::DebugString() {
  return instance_->DebugString();
}

ODLInstanceMgr::ODLInstanceMgr(ModelConfig* config)
    : model_storage_(new ModelStore(config)), model_config_(config) {
  model_storage_->Init();
  session_options_ = new SessionOptions();
  //session_options_->target = target;
  session_options_->config.set_intra_op_parallelism_threads(config->inter_threads);
  session_options_->config.set_inter_op_parallelism_threads(config->intra_threads);
  //session_options_->config.mutable_gpu_options()->set_allocator_type("CPU");
  run_options_ = new RunOptions();
  cur_inst_storage_options_ = new StorageOptions(0, 1);
  base_inst_storage_options_ = new StorageOptions(2, 3);
}

ODLInstanceMgr::~ODLInstanceMgr() {
  is_stop_ = true;
  thread_->join();
  delete thread_;

  delete base_instance_;
  delete cur_instance_;
  delete session_options_;
  delete run_options_;
  delete model_storage_;
  delete cur_inst_storage_options_;
  delete base_inst_storage_options_;
}

Status ODLInstanceMgr::Init() {
  /*Version version;
  auto status = model_storage_->GetLatestVersion(version);
  if (!status.ok()) {
    return status;
  }*/

  TF_RETURN_IF_ERROR(CreateInstances());
  
  thread_ = new std::thread(&ODLInstanceMgr::WorkLoop, this);
  return Status::OK();
}

Status ODLInstanceMgr::CreateInstances() {
  cur_instance_ = new MultipleSessionInstance(
      session_options_, run_options_,
      cur_inst_storage_options_);
  TF_RETURN_IF_ERROR(cur_instance_->Init(model_config_,
      model_storage_));
  TF_RETURN_IF_ERROR(cur_instance_->Warmup());

  base_instance_ = new MultipleSessionInstance(
      session_options_, run_options_,
      base_inst_storage_options_);
  TF_RETURN_IF_ERROR(base_instance_->Init(model_config_,
      model_storage_));
  return base_instance_->Warmup();
}

Status ODLInstanceMgr::Predict(Request& req, Response& resp) {
  return cur_instance_->Predict(req, resp);
}

Status ODLInstanceMgr::Rollback() {
  if (cur_instance_->GetVersion() == base_instance_->GetVersion()) {
    LOG(WARNING) << "[Processor] Already rollback to base model.";
    return Status::OK();
  }
  std::swap(cur_instance_, base_instance_);
  // TODO: Reset base_instance, shouldn't rollback again.
  // base_instance_->Reset();
  return Status::OK();
}

Status ODLInstanceMgr::FullModelUpdate(const Version& version) {
  TF_RETURN_IF_ERROR(cur_instance_->FullModelUpdate(version));
  cur_instance_->UpdateVersion(version);
  return cur_instance_->Warmup();
}

Status ODLInstanceMgr::DeltaModelUpdate(const Version& version) {
  // NOTE: will initialize base_instance storage once after
  // a newly full model was updated.
  if (cur_instance_->GetVersion().IsSameFullModel(version) &&
      !base_instance_->GetVersion().IsSameFullModel(version)) {
    TF_RETURN_IF_ERROR(base_instance_->FullModelUpdate(
          cur_instance_->GetVersion()));
    TF_RETURN_IF_ERROR(base_instance_->Warmup());
  }

  TF_RETURN_IF_ERROR(cur_instance_->DeltaModelUpdate(version));
  cur_instance_->UpdateVersion(version);
  return cur_instance_->Warmup();
}

Status ODLInstanceMgr::ModelUpdate(const Version& version) {
  if (version.IsFullModel()) {
    return FullModelUpdate(version);
  } else {
    return DeltaModelUpdate(version);
  }
}

void ODLInstanceMgr::WorkLoop() {
  while(!is_stop_) {
    Version version;
    auto status = model_storage_->GetLatestVersion(version);
    if (!status.ok()) {
      status = Status(error::Code::NOT_FOUND,
          "[TensorFlow] Can't get latest model name, will try 60 seconds later.");
      LOG(ERROR) << status.error_message() << std::endl;
    } else {
      if (cur_instance_->GetVersion() < version) {
        auto status = ModelUpdate(version);
        if (!status.ok()) {
          LOG(ERROR) << status.error_message() << std::endl;
        }
      }
    }

    sleep(_60_Seconds);
  }
}

std::string ODLInstanceMgr::DebugString() {
  return cur_instance_->DebugString();
}

} // processor
} // tensorflow
