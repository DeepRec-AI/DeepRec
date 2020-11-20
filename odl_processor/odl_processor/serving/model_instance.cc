#include "odl_processor/serving/model_instance.h"
#include "odl_processor/serving/model_storage.h"
#include "odl_processor/serving/model_config.h"
#include "odl_processor/serving/model_storage.h"
#include "odl_processor/serving/run_predict.h"
#include "odl_processor/core/graph_optimizer.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/platform/protobuf_internal.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"

namespace tensorflow {
namespace {
constexpr int _60_Seconds = 60;

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

Tensor CreateStringTensor(const string& value) {
  Tensor tensor(DT_STRING, TensorShape({}));
  tensor.scalar<string>()() = value;
  return tensor;
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
}
namespace processor {
Status ModelSessionMgr::CreateModelSession(const MetaGraphDef& meta_graph_def,
    SessionOptions* session_options, RunOptions* run_options,
    const Version& version, const char* model_dir) {
  Session* session = nullptr;
  TF_RETURN_IF_ERROR(NewSession(*session_options, &session));
  TF_RETURN_IF_ERROR(session->Create(meta_graph_def.graph_def()));

  std::vector<AssetFileDef> asset_file_defs;
  TF_RETURN_IF_ERROR(GetAssetFileDefs(meta_graph_def, &asset_file_defs));

  TF_RETURN_IF_ERROR(
      RunRestore(*run_options, model_dir,
          meta_graph_def.saver_def().restore_op_name(),
          meta_graph_def.saver_def().filename_tensor_name(),
          asset_file_defs, session));

  if (HasMainOp(meta_graph_def)) {
    TF_RETURN_IF_ERROR(RunMainOp(*run_options, model_dir,
        meta_graph_def, asset_file_defs, session, kSavedModelMainOpKey));
  } else {
    TF_RETURN_IF_ERROR(RunMainOp(
        *run_options, model_dir, meta_graph_def,
        asset_file_defs, session, kSavedModelLegacyInitOpKey));
  }

  ResetServingSession(session, version);
  return Status::OK();
}

void ModelSessionMgr::ResetServingSession(Session* session,
    const Version& version) {
  auto model_session = new ModelSession(session, version);
  auto tmp = serving_session_;
  serving_session_ = model_session;

  if (serving_session_->counter_ > 0) {
    // TODO: free it in active object.
    sessions_.emplace_back(tmp);
  } else {
    delete tmp;
  }
}

ModelInstance::ModelInstance(SessionOptions* sess_options,
    RunOptions* run_options) :
    session_options_(sess_options), run_options_(run_options) {
}

Status ModelInstance::ReadModelSignature(ModelConfig* model_config) {
  auto model_signatures = meta_graph_def_.signature_def();
  for (auto it : model_signatures) {
    if (it.first == model_config->signature_name) {
      model_signature_ = it;
      break;
    }
  }
  return Status::OK();
}

Status ModelInstance::CreateSession(const Version& version, 
    const char* model_dir) {
  return session_mgr_->CreateModelSession(meta_graph_def_, session_options_,
      run_options_, version, model_dir);
}

Status ModelInstance::RecursionCreateSession(const Version& version) {
  TF_RETURN_IF_ERROR(CreateSession(version, version.full_model_name.c_str()));
  return CreateSession(version, version.delta_model_name.c_str());
}

Status ModelInstance::Init(const Version& version,
    ModelConfig* model_config, SparseStorage* sparse_storage) {
  TF_RETURN_IF_ERROR(ReadMetaGraphDefFromSavedModel(
        version.full_model_name.c_str(),
        {kSavedModelTagServe}, &meta_graph_def_));

  optimizer_ = new SavedModelOptimizer(model_config->signature_name,
        &meta_graph_def_);
  TF_RETURN_IF_ERROR(optimizer_->Optimize());

  TF_RETURN_IF_ERROR(ReadModelSignature(model_config));
  
  sparse_storage_ = sparse_storage;
  TF_RETURN_IF_ERROR(sparse_storage_->Create(version));

  TF_RETURN_IF_ERROR(RecursionCreateSession(version));
  return Status::OK();
}

Status ModelInstance::Warmup() {
  RunRequest request;
  request.SetSignatureName(model_signature_.first);
  for (auto it : model_signature_.second.inputs()) {
    request.AddFeed(it.first, it.second);
  }

  RunResponse response;
  return Predict(request, &response);
}

Status ModelInstance::Predict(const eas::PredictRequest& req,
    eas::PredictResponse* resp) {
  return Status::OK();
}

Status ModelInstance::Predict(const RunRequest& req,
    RunResponse* resp) {
  return Status::OK();
}

Status ModelInstance::FullModelUpdate(const Version& version) {
  TF_RETURN_IF_ERROR(sparse_storage_->Create(version));
  return CreateSession(version, version.full_model_name.c_str());
}

Status ModelInstance::DeltaModelUpdate(const Version& version) {
  return CreateSession(version, version.delta_model_name.c_str());
}

std::string ModelInstance::DebugString() {
  return model_signature_.second.DebugString();
}

ModelInstanceMgr::ModelInstanceMgr(const char* root_dir, ModelConfig* config) :
    model_storage_(new ModelStorage()),
    model_config_(config) {
  model_storage_->Init(root_dir);
}

ModelInstanceMgr::~ModelInstanceMgr() {
  thread_->join();

  delete base_instance_;
  delete cur_instance_;
  delete model_storage_;
}

Status ModelInstanceMgr::Init(SessionOptions* sess_options,
    RunOptions* run_options) {
  Version version;
  auto status = model_storage_->GetLatestVersion(version);
  if (!status.ok()) {
    return status;
  }

  TF_RETURN_IF_ERROR(CreateInstances(version));
  
  thread_ = new std::thread(&ModelInstanceMgr::WorkLoop, this);
  return Status::OK();
}

Status ModelInstanceMgr::CreateInstances(const Version& version) {
  cur_instance_ = new ModelInstance(session_options_, run_options_);
  TF_RETURN_IF_ERROR(cur_instance_->Init(version, model_config_,
      model_storage_->GetSparseStorage(version)));

  base_instance_ = new ModelInstance(session_options_, run_options_);
  return base_instance_->Init(version, model_config_,
      model_storage_->GetSparseStorage(version));
}

Status ModelInstanceMgr::Predict(const eas::PredictRequest& req,
    eas::PredictResponse* resp) {
  return Status::OK();
}

Status ModelInstanceMgr::FullModelUpdate(const Version& version) {
  return cur_instance_->FullModelUpdate(version);
}

Status ModelInstanceMgr::DeltaModelUpdate(const Version& version) {
  if (cur_instance_->GetVersion().IsSameFullModel(version) &&
      !base_instance_->GetVersion().IsSameFullModel(version)) {
    TF_RETURN_IF_ERROR(base_instance_->FullModelUpdate(
          cur_instance_->GetVersion()));
  }

  return cur_instance_->DeltaModelUpdate(version);
}

Status ModelInstanceMgr::ModelUpdate(const Version& version) {
  if (version.IsFullModel()) {
    return FullModelUpdate(version);
  } else {
    return DeltaModelUpdate(version);
  }
}

void ModelInstanceMgr::WorkLoop() {
  while(!is_stop) {
    Version version;
    auto status = model_storage_->GetLatestVersion(version);
    if (!status.ok()) {
      status = Status(error::Code::NOT_FOUND,
          "[TensorFlow] Can't get latest model name, will try 60 seconds later.");
      std::cerr << status.error_message() << std::endl;
    } else {
      if (version != cur_instance_->GetVersion()) {
        auto status = ModelUpdate(version);
        if (!status.ok()) {
          std::cerr << status.error_message() << std::endl;
        }
      }
    }

    sleep(_60_Seconds);
  }
}

std::string ModelInstanceMgr::DebugString() {
  return std::string();
}

} // processor
} // tensorflow
