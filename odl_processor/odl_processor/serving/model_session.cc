#include <random>
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
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"

namespace tensorflow {
namespace processor {

namespace {
constexpr int _30_Seconds = 30;

int GetRandomNum() {
  std::random_device device("/dev/urandom");
  std::mt19937 r(device());
  return r();
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

void ModifyPathName(std::string* path, int version) {
  // Full  ckpt: /you_path/model.ckpt-num
  // Delta ckpt: /you_path/incremental_model.ckpt-num
  int len = path->length(), i;
  for (i = len - 1; len >= 0; --i) {
    if ((*path)[i] == '-') break;
  }
  *path = path->substr(0, i+1);
  *path += std::to_string(version);
}

void MaybeUpdateRealVersion(Version* version, int64_t full_version,
                            int latest_version) {
  if (version->full_ckpt_version != full_version) {
    version->full_ckpt_version = full_version;
    ModifyPathName(&version->full_ckpt_name, full_version);
  }

  if (version->delta_ckpt_version != latest_version) {
    version->delta_ckpt_version = latest_version;
    // No delta version
    if (latest_version == 0) {
      version->delta_ckpt_name = "";
    } else {
      ModifyPathName(&version->delta_ckpt_name, latest_version);
    }
  }
}

int LockOrWait(int64_t latest_version,
               IFeatureStoreMgr* sparse_storage,
               Version* real_version,
               ModelConfig* config, bool* locked) {
  *locked = false;

  // Only one instance can get the distribute lock,
  // other instances should wait here,
  // and check the model version in the storage.
  // Wait here when the lock timeout is not reached.
  // If lock timeout, other instances should
  // try to get the lock again.
  do {
    auto lock_value = GetRandomNum();
    Status status = sparse_storage->GetStorageLock(
        lock_value, config->lock_timeout, locked);
    if (status.ok() && locked) {
      return lock_value;
    }
    *locked = false;

    // WAIT and CHECK version in the storage
    int64_t curr_full_version = -1;
    int64_t model_version = -1;
    int time_cost = 0;
    // try util lock timeout
    while (time_cost < config->lock_timeout) {
      // check model_version from storage like redis.
      status = sparse_storage->GetModelVersion(
          &curr_full_version, &model_version);
      if (!status.ok()) {
        LOG(WARNING) << "Check latest version error, will try again.";
      } else {
        LOG(INFO) << "Waiting for the latest model to be updated. version is "
                  << latest_version << ", current redis model version is "
                  << model_version;
        if (model_version >= latest_version) {
          MaybeUpdateRealVersion(real_version, curr_full_version,
                                 model_version);
          return 0;
        }
      }

      sleep(_30_Seconds);
      time_cost += _30_Seconds;
    }

    LOG(WARNING) << "Lock timeout, try to grab the lock again.";
    // continue: lock timeout, try to get lock again

  } while(true);

  return 0;
}

Status RunRestore(const RunOptions& run_options,
                  const std::string& ckpt_name,
                  const std::string& savedmodel_dir,
                  const StringPiece restore_op_name,
                  const StringPiece variable_filename_const_op_name,
                  const std::vector<AssetFileDef>& asset_file_defs,
                  Session* session, bool update_sparse, int64_t latest_version,
                  IFeatureStoreMgr* sparse_storage,
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

  std::string dense_restore_op_name =
      std::string(restore_op_name) +
      tensorflow::processor::GetDenseRestoreAllNameSuffix();
  std::string kv_restore_op_name =
      std::string(restore_op_name) +
      tensorflow::processor::GetKvRestoreAllNameSuffix();

  RunMetadata run_metadata;
  // 1) update dense variable
  Status s = RunOnce(
      run_options, inputs, {}, {dense_restore_op_name},
      nullptr /* outputs */, &run_metadata, session);
  if (!s.ok()) return s;

  // 2) update kv variable
  if (update_sparse) {
    // only one instance can update sparse variable
    return RunOnce(
        run_options, inputs, {}, {kv_restore_op_name},
        nullptr /* outputs */, &run_metadata, session);
  }

  return s;
}

} // namespace

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
    IFeatureStoreMgr* sparse_storage,
    bool is_incr_ckpt, bool update_sparse,
    int64_t latest_version) {
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

  Tensor is_incr_ckpt_tensor(DT_BOOL, TensorShape({}));
  is_incr_ckpt_tensor.scalar<bool>()() = is_incr_ckpt;
  auto is_incr_ckpt_tensor_pair = std::make_pair(
      GetIncrCkptNodeName(), is_incr_ckpt_tensor);
  extra_tensors.emplace_back(is_incr_ckpt_tensor_pair);

  TF_RETURN_IF_ERROR(
      RunRestore(*run_options_, ckpt_name, savedmodel_dir,
          meta_graph_def_.saver_def().restore_op_name(),
          meta_graph_def_.saver_def().filename_tensor_name(),
          asset_file_defs_, session, update_sparse,
          latest_version, sparse_storage, extra_tensors));

  if (HasMainOp(meta_graph_def_)) {
    return RunMainOp(*run_options_, savedmodel_dir,
        meta_graph_def_, asset_file_defs_, session,
        kSavedModelMainOpKey, sparse_storage_tensor_pair);
  } else {
    return RunMainOp(
        *run_options_, savedmodel_dir, meta_graph_def_,
        asset_file_defs_, session, kSavedModelLegacyInitOpKey,
        sparse_storage_tensor_pair);
  }
}

ModelSession::ModelSession(Session* s, const Version& version,
    IFeatureStoreMgr* sparse_storage) : session_(s), counter_(0) {
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
    IFeatureStoreMgr* sparse_storage, bool is_incr_ckpt,
    bool is_initialize, ModelConfig* config) {
  Session* session = nullptr;
  TF_RETURN_IF_ERROR(CreateSession(&session));

  Version real_version = version;
  Status status;

  int64_t latest_version =
      version.full_ckpt_version;;
  if (is_incr_ckpt) {
    latest_version = version.delta_ckpt_version;
  }

  int lock_value = 0;
  bool locked = false;
  if (is_initialize) {
    // get redis lock and update, or wait
    //status = sparse_storage->GetStorageLock(
    //    lock_value, config->lock_timeout, &locked);
    lock_value = LockOrWait(latest_version, sparse_storage,
                            &real_version, config, &locked);
  } else {
    // check redis version, compare to version, update or wait

    // the full version in redis.
    int64_t curr_full_version = -1;
    // the latest version in redis, maybe delta version.
    int64_t model_version = -1;
    TF_RETURN_IF_ERROR(sparse_storage->GetModelVersion(
        &curr_full_version, &model_version));
    // Remote storage already has the latest version
    if (model_version >= latest_version) {
      locked = false;
      // Update the model which version matched to redis in local,
      // Avoid the mismatch between dense and sparse variable.
      latest_version = model_version;
      MaybeUpdateRealVersion(&real_version, curr_full_version,
                             model_version);
      LOG(INFO) << "Latest variable have been updated.";
    } else {
      //status = sparse_storage->GetStorageLock(
      //    lock_value, config->lock_timeout, &locked);

      lock_value = LockOrWait(latest_version, sparse_storage,
                              &real_version, config, &locked);
    }
  }
  TF_RETURN_IF_ERROR(status);

  // only one instance can restore sparse variable
  status = RunRestoreOps(ckpt_name,
        real_version.full_ckpt_version,
        real_version.savedmodel_dir.c_str(),
        session, sparse_storage, is_incr_ckpt,
        locked, latest_version);

  // Update model_version and then Release lock after import
  if (locked) {
    if (status.ok()) {
      int64_t full_version = version.full_ckpt_version;
      int64_t delta_version = version.delta_ckpt_version;
      if (!is_incr_ckpt) {
        delta_version = 0;
      }
      sparse_storage->SetModelVersion(full_version, delta_version);
    }

    TF_RETURN_IF_ERROR(
        sparse_storage->ReleaseStorageLock(lock_value));
  }
  TF_RETURN_IF_ERROR(status);

  // version(real_version) maybe modfied across
  // the version returned by remote storage.
  ResetServingSession(session, real_version, sparse_storage);
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
