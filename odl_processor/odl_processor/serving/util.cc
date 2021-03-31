#include "odl_processor/serving/util.h"
#include "odl_processor/framework/graph_optimizer.h"

namespace tensorflow {
namespace processor {
namespace util {

namespace {

Tensor CreateStringTensor(const string& value) {
  Tensor tensor(DT_STRING, TensorShape({}));
  tensor.scalar<string>()() = value;
  return tensor;
}

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

Status RunRestoreCheckpoint(
    const RunOptions& run_options,
    const std::string& ckpt_name,
    const std::string& savedmodel_dir,
    const StringPiece restore_op_name,
    const StringPiece variable_filename_const_op_name,
    const std::vector<AssetFileDef>& asset_file_defs,
    Session* session) {
  LOG(INFO) << "Restoring checkpoint.";
  // Find path to variables to be restored in export directory.
  // Add variables to the graph.
  Tensor variables_path_tensor(DT_STRING, TensorShape({}));
  variables_path_tensor.scalar<string>()() = ckpt_name;

  std::vector<std::pair<string, Tensor>> inputs = { 
      {string(variable_filename_const_op_name), variables_path_tensor}};

  util::AddAssetsTensorsToInputs(savedmodel_dir, asset_file_defs, &inputs);

  RunMetadata run_metadata;
  return util::RunOnce(run_options, inputs, {}, {string(restore_op_name)},
                       nullptr /* outputs */, &run_metadata, session);
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

  util::AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);

  RunMetadata run_metadata;
  return util::RunOnce(run_options, inputs, {}, {string(restore_op_name)},
                       nullptr /* outputs */, &run_metadata, session);
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
    util::AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);
    RunMetadata run_metadata;
    const StringPiece main_op_name = main_op_it->second.node_list().value(0);
    return util::RunOnce(run_options, inputs, {}, {string(main_op_name)},
                         nullptr /* outputs */, &run_metadata, session);
  }
  return Status::OK();
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
    util::AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);
    inputs.emplace_back(sparse_storage_tensor);
    RunMetadata run_metadata;
    const StringPiece main_op_name = main_op_it->second.node_list().value(0);
    return util::RunOnce(run_options, inputs, {}, {string(main_op_name)},
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
                  Session* session, bool update_sparse, int64_t latest_version,
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

  util::AddAssetsTensorsToInputs(savedmodel_dir, asset_file_defs, &inputs);

  std::string dense_restore_op_name =
      std::string(restore_op_name) +
      tensorflow::processor::GetDenseRestoreAllNameSuffix();
  std::string kv_restore_op_name =
      std::string(restore_op_name) +
      tensorflow::processor::GetKvRestoreAllNameSuffix();

  RunMetadata run_metadata;
  // 1) update dense variable
  Status s = util::RunOnce(
      run_options, inputs, {}, {dense_restore_op_name},
      nullptr /* outputs */, &run_metadata, session);
  if (!s.ok()) return s;

  // 2) update kv variable
  if (update_sparse) {
    // only one instance can update sparse variable
    return util::RunOnce(
        run_options, inputs, {}, {kv_restore_op_name},
        nullptr /* outputs */, &run_metadata, session);
  }

  return s;
}

} // namespace util
} // namespace processor
} // namespace tensorflow
