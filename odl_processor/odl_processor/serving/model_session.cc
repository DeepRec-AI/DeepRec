#include "odl_processor/serving/model_session.h"
#include "odl_processor/serving/model_message.h"
#include "odl_processor/storage/model_store.h"
#include "odl_processor/storage/feature_store_mgr.h"
#include "odl_processor/framework/graph_optimizer.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/platform/protobuf_internal.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"

namespace tensorflow {
namespace {
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
                 Session* session, const string& main_op_key,
                 std::pair<std::string, Tensor> sparse_storage_tensor) {
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
    inputs.emplace_back(sparse_storage_tensor);
    RunMetadata run_metadata;
    const StringPiece main_op_name = main_op_it->second.node_list().value(0);
    return RunOnce(run_options, inputs, {}, {string(main_op_name)},
                   nullptr /* outputs */, &run_metadata, session);
  }
  return Status::OK();
}

Status RunRestore(const RunOptions& run_options,
                  const std::string& ckpt_name,
                  const std::string& savedmodel_dir,
                  const StringPiece restore_op_name,
                  const StringPiece variable_filename_const_op_name,
                  const std::vector<AssetFileDef>& asset_file_defs,
                  Session* session,
                  std::vector<std::pair<std::string, Tensor>>& extra_tensors) {
  LOG(INFO) << "Restoring SavedModel bundle.";
  // Find path to variables to be restored in export directory.

  // Add variables to the graph.
  Tensor variables_path_tensor(DT_STRING, TensorShape({}));
  variables_path_tensor.scalar<string>()() = ckpt_name;

  std::vector<std::pair<string, Tensor>> inputs = {
      {string(variable_filename_const_op_name), variables_path_tensor}};
  for (auto t : extra_tensors) {
    inputs.emplace_back(t);
  }

  AddAssetsTensorsToInputs(savedmodel_dir, asset_file_defs, &inputs);

  RunMetadata run_metadata;
  return RunOnce(run_options, inputs, {}, {string(restore_op_name)},
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

Status ModelSessionMgr::RunRestoreOps(
    const char* ckpt_name, int64 full_ckpt_version,
    const char* savedmodel_dir, Session* session,
    IFeatureStoreMgr* sparse_storage) {
  std::vector<std::pair<std::string, Tensor>> extra_tensors;
  Tensor sparse_storage_tensor(DT_UINT64, TensorShape({}));
  sparse_storage_tensor.scalar<uint64>()() =
      reinterpret_cast<uint64>(sparse_storage);
  auto sparse_storage_tensor_pair = std::make_pair(
      GetStoragePointerNodeName(), sparse_storage_tensor);
  extra_tensors.emplace_back(sparse_storage_tensor_pair);

  Tensor version_tensor(DT_UINT64, TensorShape({}));
  version_tensor.scalar<uint64>()() = full_ckpt_version;
  auto version_tensor_pair = std::make_pair(
      GetModelVersionNodeName(), version_tensor);
  extra_tensors.emplace_back(version_tensor_pair);

  TF_RETURN_IF_ERROR(
      RunRestore(*run_options_, ckpt_name, savedmodel_dir,
          meta_graph_def_.saver_def().restore_op_name(),
          meta_graph_def_.saver_def().filename_tensor_name(),
          asset_file_defs_, session, extra_tensors));

  if (HasMainOp(meta_graph_def_)) {
    return RunMainOp(*run_options_, savedmodel_dir,
        meta_graph_def_, asset_file_defs_, session, kSavedModelMainOpKey,
        sparse_storage_tensor_pair);
  } else {
    return RunMainOp(
        *run_options_, savedmodel_dir, meta_graph_def_,
        asset_file_defs_, session, kSavedModelLegacyInitOpKey,
        sparse_storage_tensor_pair);
  }
}

ModelSession::ModelSession(Session* s, const Version& version,
    IFeatureStoreMgr* sparse_storage) : session_(s), counter_(0),
    version_(version) {
  Tensor t(DT_UINT64, TensorShape({}));
  t.scalar<uint64>()() = reinterpret_cast<uint64>(sparse_storage);
  sparse_storage_tensor_ = t;
  sparse_storage_name_ = GetStoragePointerNodeName();

  Tensor t_version(DT_UINT64, TensorShape({}));
  t_version.scalar<uint64>()() = version.full_ckpt_version;
  model_version_tensor_ = t_version;
  model_version_name_ = GetModelVersionNodeName();
}

Status ModelSession::Predict(Request& req, Response& resp) {
  req.inputs.emplace_back(sparse_storage_name_, sparse_storage_tensor_);
  req.inputs.emplace_back(model_version_name_, model_version_tensor_);
  ++counter_;
  auto status = session_->Run(req.inputs, req.output_tensor_names,
      {}, &resp.outputs);
  --counter_;
  return status;
}

Status ModelSessionMgr::Predict(Request& req, Response& resp) {
  return serving_session_->Predict(req, resp);
}

Status ModelSessionMgr::CreateModelSession(
    const Version& version, const char* ckpt_name,
    IFeatureStoreMgr* sparse_storage) {
  Session* session = nullptr;
  TF_RETURN_IF_ERROR(CreateSession(&session));
  TF_RETURN_IF_ERROR(RunRestoreOps(ckpt_name,
        version.full_ckpt_version,
        version.savedmodel_dir.c_str(),
        session, sparse_storage));
  ResetServingSession(session, version, sparse_storage);
  return Status::OK();
}

Status ModelSessionMgr::CleanupModelSession() {
  mutex_lock lock(mu_);
  sessions_.erase(
      std::remove_if(sessions_.begin(), sessions_.end(),
        [this](ModelSession* sess){
          if (sess->counter_ <= 0) {
            delete sess;
            return true;
          } else {
            return false;
          }
        }), sessions_.end());

  return Status::OK();
}

void ModelSessionMgr::ResetServingSession(Session* session,
    const Version& version, IFeatureStoreMgr* sparse_storage) {
  auto model_session = new ModelSession(session, version,
      sparse_storage);
  auto tmp = serving_session_;
  serving_session_ = model_session;

  if (tmp == nullptr) return;

  if (tmp->counter_ > 0) {
    // TODO: free it in active object.
    mutex_lock lock(mu_);
    sessions_.emplace_back(tmp);
  } else {
    delete tmp;
  }
}

} // processor
} // tensorflow
