#include "odl_processor/serving/model_instance.h"
#include "odl_processor/serving/model_storage.h"
#include "odl_processor/serving/model_config.h"
#include "odl_processor/core/graph_optimizer.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/platform/protobuf_internal.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"
#include "tensorflow/core/framework/tensor.h"

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
  return collection_def_map.find(kSavedModelMainOpKey) !=
    collection_def_map.end();
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

Status RunInitSparseGraph(const MetaGraphDef& meta_graph_def,
                          const RunOptions& run_options,
                          Session* session) {
  std::string init_op_name;
  for (auto sdef : meta_graph_def.signature_def()) {
    if (sdef.first == tensorflow::processor::GetInitDefKey()) {
      TensorInfo tinfo;
      for (auto output : sdef.second.outputs()) {
        if (output.first == "init_op") {
          tinfo = output.second;
          break;
        }
      }
      init_op_name = tinfo.name();
      int offset = init_op_name.find(":");
      init_op_name = init_op_name.substr(0, offset);
      break;
    }
  }

  RunMetadata run_metadata;
  return RunOnce(run_options, {}, {}, {init_op_name},
                 nullptr /* outputs */, &run_metadata, session);
}
}
namespace processor {
ModelSessionMgr::ModelSessionMgr(const MetaGraphDef& meta_graph_def,
    SessionOptions* session_options, RunOptions* run_options) :
  meta_graph_def_(meta_graph_def), session_options_(session_options),
  run_options_(run_options) {
}

Status ModelSessionMgr::CreateSession(Session** session) {
  TF_RETURN_IF_ERROR(NewSession(*session_options_, session));
  TF_RETURN_IF_ERROR((*session)->Create(meta_graph_def_.graph_def()));
  return GetAssetFileDefs(meta_graph_def_, &asset_file_defs_);
}

Status ModelSessionMgr::RunRestoreOps(const char* model_dir,
    Session* session, SparseStorage* sparse_storage) {
  /*TODO: sparse_storage pass to graph by placeholder*/
  TF_RETURN_IF_ERROR(
      RunRestore(*run_options_, model_dir,
          meta_graph_def_.saver_def().restore_op_name(),
          meta_graph_def_.saver_def().filename_tensor_name(),
          asset_file_defs_, session));

  if (HasMainOp(meta_graph_def_)) {
    return RunMainOp(*run_options_, model_dir,
        meta_graph_def_, asset_file_defs_, session, kSavedModelMainOpKey);
  } else {
    return RunMainOp(
        *run_options_, model_dir, meta_graph_def_,
        asset_file_defs_, session, kSavedModelLegacyInitOpKey);
  }
}

Status ModelSessionMgr::Predict(
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_tensor_names,
    std::vector<Tensor>* outputs) {
  return serving_session_->session_->Run(inputs, output_tensor_names,
      {}, outputs);
}

Status ModelSessionMgr::CreateDeltaModelSession(
    const Version& version, const char* model_dir,
    SparseStorage* sparse_storage) {
  Session* session = nullptr;
  TF_RETURN_IF_ERROR(CreateSession(&session));
  TF_RETURN_IF_ERROR(RunRestoreOps(model_dir, session,
        sparse_storage));

  ResetServingSession(session, version, sparse_storage);
  return Status::OK();
}

Status ModelSessionMgr::CreateFullModelSession(
    const Version& version, const char* model_dir,
    SparseStorage* sparse_storage) {
  Session* session = nullptr;
  TF_RETURN_IF_ERROR(CreateSession(&session));
  
  TF_RETURN_IF_ERROR(RunInitSparseGraph(meta_graph_def_,
        *run_options_, session));
  
  TF_RETURN_IF_ERROR(RunRestoreOps(model_dir, session,
        sparse_storage));
  ResetServingSession(session, version, sparse_storage);
  return Status::OK();
}

void ModelSessionMgr::ResetServingSession(Session* session,
    const Version& version, SparseStorage* sparse_storage) {
  auto model_session = new ModelSession(session, version,
      sparse_storage);
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

Status ModelInstance::RecursionCreateSession(const Version& version,
    SparseStorage* sparse_storage) {
  TF_RETURN_IF_ERROR(session_mgr_->CreateFullModelSession(version,
        version.full_model_name.c_str(), sparse_storage));

  if (version.delta_model_name.empty()) {
    return Status::OK();
  } else {
    return session_mgr_->CreateDeltaModelSession(version,
        version.delta_model_name.c_str(), sparse_storage);
  }
}

Status ModelInstance::Init(const Version& version,
    ModelConfig* model_config, ModelStorage* model_storage) {
  TF_RETURN_IF_ERROR(ReadMetaGraphDefFromSavedModel(
        version.full_model_name.c_str(),
        {kSavedModelTagServe}, &meta_graph_def_));

  session_mgr_ = new ModelSessionMgr(meta_graph_def_,
      session_options_, run_options_);

  optimizer_ = new SavedModelOptimizer(model_config->signature_name,
        &meta_graph_def_);
  TF_RETURN_IF_ERROR(optimizer_->Optimize());

  TF_RETURN_IF_ERROR(ReadModelSignature(model_config));

  serving_storage_ = model_storage->CreateSparseStorage(version);
  backup_storage_ = model_storage->CreateSparseStorage(version);

  return RecursionCreateSession(version, serving_storage_);
}

Status ModelInstance::Predict(
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_tensor_names,
    std::vector<Tensor>* outputs) {
  return session_mgr_->Predict(inputs, output_tensor_names, outputs);
}

Tensor ModelInstance::CreateTensor(const TensorInfo& tensor_info) {
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

Status ModelInstance::CreateWarmupParams(
    std::vector<std::pair<std::string, Tensor>>& inputs,
    std::vector<std::string>& output_tensor_names) {
  for (auto it : model_signature_.second.inputs()) {
    const auto& tensor = CreateTensor(it.second);
    inputs.emplace_back(it.first, tensor);
  }

  for (auto it : model_signature_.second.outputs()) {
    output_tensor_names.emplace_back(it.first);
  }
  return Status::OK();
}

Status ModelInstance::Warmup() {
  std::vector<std::pair<std::string, Tensor>> inputs;
  std::vector<std::string> output_tensor_names;
  TF_RETURN_IF_ERROR(CreateWarmupParams(inputs,
        output_tensor_names));

  // Ignore output results, only care about return status.
  std::vector<Tensor> outputs;
  return Predict(inputs, output_tensor_names, &outputs);
}

Status ModelInstance::FullModelUpdate(const Version& version) {
  // Logically backup_storage_ shouldn't serving now.
  backup_storage_->Reset();
  TF_RETURN_IF_ERROR(session_mgr_->CreateFullModelSession(version,
      version.full_model_name.c_str(), backup_storage_));
 
  std::swap(backup_storage_, serving_storage_);
  return Status::OK();
}

Status ModelInstance::DeltaModelUpdate(const Version& version) {
  return session_mgr_->CreateDeltaModelSession(version,
      version.delta_model_name.c_str(), serving_storage_);
}

std::string ModelInstance::DebugString() {
  return model_signature_.second.DebugString();
}

ModelInstanceMgr::ModelInstanceMgr(const char* root_dir, ModelConfig* config)
  : model_storage_(new ModelStorage()), model_config_(config) {
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
      model_storage_));
  TF_RETURN_IF_ERROR(cur_instance_->Warmup());

  base_instance_ = new ModelInstance(session_options_, run_options_);
  TF_RETURN_IF_ERROR(base_instance_->Init(version, model_config_,
      model_storage_));
  return base_instance_->Warmup();
}

Status ModelInstanceMgr::Predict(
    const std::vector<std::pair<std::string, Tensor>>& inputs,
    const std::vector<std::string>& output_tensor_names,
    std::vector<Tensor>* outputs) {
  return cur_instance_->Predict(inputs, output_tensor_names, outputs);
}

Status ModelInstanceMgr::Rollback() {
  if (cur_instance_->GetVersion() == base_instance_->GetVersion()) {
    LOG(WARNING) << "[Processor] Already rollback to base model.";
    return Status::OK();
  }
  std::swap(cur_instance_, base_instance_);
  // TODO: Reset base_instance, shouldn't rollback again.
  // base_instance_->Reset();
  return Status::OK();
}

Status ModelInstanceMgr::FullModelUpdate(const Version& version) {
  cur_instance_->FullModelUpdate(version);
  return cur_instance_->Warmup();
}

Status ModelInstanceMgr::DeltaModelUpdate(const Version& version) {
  if (cur_instance_->GetVersion().IsSameFullModel(version) &&
      !base_instance_->GetVersion().IsSameFullModel(version)) {
    TF_RETURN_IF_ERROR(base_instance_->FullModelUpdate(
          cur_instance_->GetVersion()));
    TF_RETURN_IF_ERROR(base_instance_->Warmup());
  }

  TF_RETURN_IF_ERROR(cur_instance_->DeltaModelUpdate(version));
  return cur_instance_->Warmup();
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
  return cur_instance_->DebugString();
}

} // processor
} // tensorflow
