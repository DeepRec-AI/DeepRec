#include "tensorflow/core/common_runtime/custom_thread_pool.h"
#include "serving/processor/serving/util.h"
#include "serving/processor/framework/graph_optimizer.h"

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
  // With SavedModel v2, we write asset file def into metagraph instead of
  // collection, so read from metagraph first.
  if (meta_graph_def.asset_file_def_size() > 0) {
    for (const auto& asset : meta_graph_def.asset_file_def()) {
      asset_file_defs->push_back(asset);
    }
    return Status::OK();
  }

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

Status RunOnce(
    const RunOptions& run_options,
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& output_tensor_names,
    const std::vector<string>& target_node_names,
    std::vector<Tensor>* outputs, RunMetadata* run_metadata,
    Session* session,
    thread::ThreadPoolOptions thread_opt) {
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
                                                 outputs, run_metadata, thread_opt);
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
    bool restore_incr_checkpoint,
    const RunOptions& run_options,
    const std::string& full_ckpt_name,
    const std::string& incr_ckpt_name,
    const std::string& savedmodel_dir,
    const StringPiece restore_op_name,
    const StringPiece variable_filename_const_op_name,
    const StringPiece incr_variable_filename_const_op_name,
    const std::vector<AssetFileDef>& asset_file_defs,
    Session* session, thread::ThreadPoolOptions thread_opt) {
  LOG(INFO) << "Restoring checkpoint.";
  // Find path to variables to be restored in export directory.
  // Add variables to the graph.
  Tensor variables_path_tensor(DT_STRING, TensorShape({}));
  variables_path_tensor.scalar<string>()() = full_ckpt_name;

  std::vector<std::pair<string, Tensor>> inputs = {
      {string(variable_filename_const_op_name), variables_path_tensor}};

  if (restore_incr_checkpoint) {
    Tensor incr_variables_path_tensor(DT_STRING, TensorShape({}));
    incr_variables_path_tensor.scalar<string>()() = incr_ckpt_name;
    inputs.push_back(
        {string(incr_variable_filename_const_op_name), incr_variables_path_tensor});
  }

  util::AddAssetsTensorsToInputs(savedmodel_dir, asset_file_defs, &inputs);

  RunMetadata run_metadata;
  return util::RunOnce(run_options, inputs, {}, {string(restore_op_name)},
                       nullptr /* outputs */, &run_metadata, session, thread_opt);
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

// RunInitOp will return OK if the initialization op was run successfully.
// An empty init_op_name indicates that there are no init ops to run.
Status RunInitOp(const RunOptions& run_options, const string& export_dir,
                 const MetaGraphDef& meta_graph_def,
                 const std::vector<AssetFileDef>& asset_file_defs,
                 Session* session, const string& init_op_name) {
  if (!init_op_name.empty()) {
    LOG(INFO) << "Running initialization op on SavedModel bundle at path: "
              << export_dir;
    std::vector<std::pair<string, Tensor>> inputs;
    AddAssetsTensorsToInputs(export_dir, asset_file_defs, &inputs);
    RunMetadata run_metadata;
    return RunOnce(run_options, inputs, {}, {init_op_name},
                   nullptr /* outputs */, &run_metadata, session);
  }
  return Status::OK();
}

Status GetInitOp(const string& export_dir, const MetaGraphDef& meta_graph_def,
                 string* init_op_name) {
  const auto& sig_def_map = meta_graph_def.signature_def();
  const auto& init_op_sig_it =
      meta_graph_def.signature_def().find(kSavedModelInitOpSignatureKey);
  if (init_op_sig_it != sig_def_map.end()) {
    const auto& sig_def_outputs = init_op_sig_it->second.outputs();
    const auto& sig_def_outputs_it =
        sig_def_outputs.find(kSavedModelInitOpSignatureKey);
    if (sig_def_outputs_it == sig_def_outputs.end()) {
      return errors::FailedPrecondition("Could not find output ",
                                        kSavedModelInitOpSignatureKey);
    }
    *init_op_name = sig_def_outputs_it->second.name();
    return Status::OK();
  }

  const auto& collection_def_map = meta_graph_def.collection_def();
  string init_op_collection_key;
  if (collection_def_map.find(kSavedModelMainOpKey) !=
      collection_def_map.end()) {
    init_op_collection_key = kSavedModelMainOpKey;
  } else {
    init_op_collection_key = kSavedModelLegacyInitOpKey;
  }

  const auto init_op_it = collection_def_map.find(init_op_collection_key);
  if (init_op_it != collection_def_map.end()) {
    if (init_op_it->second.node_list().value_size() != 1) {
      return errors::FailedPrecondition(
          strings::StrCat("Expected exactly one main op in : ", export_dir));
    }
    *init_op_name = init_op_it->second.node_list().value(0);
  }
  return Status::OK();
}

namespace {
Status ValidateNode(const NodeDef& node) {
  const auto node_iterator = node.attr().find("value");
  if (node_iterator != node.attr().end()) {
    AttrValue node_value = node_iterator->second;
    if (node_value.has_tensor()) {
      const PartialTensorShape node_shape(node_value.tensor().tensor_shape());
      if (node_shape.num_elements() < 0) {
        return errors::FailedPrecondition(
            "Saved model contains node \"", node.name(), "\" (op \"", node.op(),
            "\") which initializes from a tensor with ",
            node_shape.num_elements(), " elements");
      }
    }
  } else if (node.op() == "Const") {
    return errors::FailedPrecondition(
        "Saved model contains node \"", node.name(),
        "\" which is a constant tensor but no value has been provided");
  }
  return Status::OK();
}

Status ValidateFunctionNotRecursive(const FunctionDef& function) {
  const auto& function_name = function.signature().name();
  for (const auto& node : function.node_def()) {
    if (node.op() == function_name) {
      return errors::FailedPrecondition(
          "Function ", function_name,
          " is self recursive and TensorFlow does not support this scenario.");
    }
  }

  return Status::OK();
}
}

Status ValidateSavedTensors(const GraphDef& graph_def) {
  for (const auto& node : graph_def.node()) {
    TF_RETURN_IF_ERROR(ValidateNode(node));
  }

  if (graph_def.has_library()) {
    const FunctionDefLibrary& library = graph_def.library();
    for (const auto& function : library.function()) {
      for (const auto& node : function.node_def()) {
        TF_RETURN_IF_ERROR(ValidateNode(node));
      }

      // Also check that there is no recursivity in the library
      // TODO(mihaimaruseac): Do more than self-recursivity
      TF_RETURN_IF_ERROR(ValidateFunctionNotRecursive(function));
    }
  }

  return Status::OK();
}

TensorWithStatus Proto2Tensor(const std::string& key,
                              const eas::ArrayProto& input) {
  TensorShape tensor_shape;
  int64 total_size = 1;
  for (int i = 0; i < input.array_shape().dim_size(); ++i) {
    tensor_shape.AddDim(input.array_shape().dim(i));
    total_size *= input.array_shape().dim(i);
  }

  TensorWithStatus ret;
  ret.status = Status::OK();
  switch (input.dtype()) {
    case tensorflow::eas::DT_FLOAT: {
      if (total_size != input.float_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.float_val_size()));
        break;
      }
      Tensor tensor(DT_FLOAT, tensor_shape);
      auto flat = tensor.flat<float>();
      memcpy(flat.data(), input.float_val().data(),
          input.float_val_size() * sizeof(float));

      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_DOUBLE: {
      if (total_size != input.double_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.double_val_size()));
        break;
      }
      Tensor tensor(DT_DOUBLE, tensor_shape);
      auto flat = tensor.flat<double>();
      memcpy(flat.data(), input.double_val().data(),
          input.double_val_size() * sizeof(double));
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_INT32: {
      if (total_size != input.int_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.int_val_size()));
        break;
      }
      Tensor tensor(DT_INT32, tensor_shape);
      auto flat = tensor.flat<int>();
      memcpy(flat.data(), input.int_val().data(),
          input.int_val_size() * sizeof(int));
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_UINT8: {
      if (total_size != input.int_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.int_val_size()));
        break;
      }
      Tensor tensor(tensorflow::DT_UINT8, tensor_shape);
      auto flat = tensor.flat<uint8>();
      for (int i = 0; i < input.int_val_size(); i++) {
        flat(i) = (uint8)input.int_val(i);
      }
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_INT16: {
      if (total_size != input.int_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.int_val_size()));
        break;
      }
      Tensor tensor(tensorflow::DT_INT16, tensor_shape);
      auto flat = tensor.flat<int16>();
      for (int i = 0; i < input.int_val_size(); i++) {
        flat(i) = (int16)input.int_val(i);
      }
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_UINT16: {
      if (total_size != input.int_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.int_val_size()));
        break;
      }
      Tensor tensor(tensorflow::DT_UINT16, tensor_shape);
      auto flat = tensor.flat<uint16>();
      for (int i = 0; i < input.int_val_size(); i++) {
        flat(i) = (uint16)input.int_val(i);
      }
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_INT8: {
      if (total_size != input.int_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.int_val_size()));
        break;
      }
      Tensor tensor(tensorflow::DT_INT8, tensor_shape);
      auto flat = tensor.flat<int8>();
      for (int i = 0; i < input.int_val_size(); i++) {
        flat(i) = (int8)input.int_val(i);
      }
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_STRING: {
      if (total_size != input.string_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.string_val_size()));
        break;
      }
      Tensor tensor(tensorflow::DT_STRING, tensor_shape);
      auto flat = tensor.flat<std::string>();
      for (int i = 0; i < input.string_val_size(); i++) {
        flat(i) = input.string_val(i);
      }
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_COMPLEX64: {
      if (total_size != input.float_val_size() / 2) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.float_val_size()));
        break;
      }
      Tensor tensor(tensorflow::DT_COMPLEX64, tensor_shape);
      auto flat = tensor.flat<complex64>();
      for (int i = 0; i < input.float_val_size(); i += 2) {
        flat(i) = complex64(input.float_val(i), input.float_val(i + 1));
      }
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_COMPLEX128: {
      if (total_size != input.double_val_size() / 2) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.double_val_size()));
        break;
      }
      Tensor tensor(tensorflow::DT_COMPLEX128, tensor_shape);
      auto flat = tensor.flat<complex128>();
      for (int i = 0; i < input.double_val_size(); i += 2) {
        flat(i) = complex64(input.double_val(i), input.double_val(i + 1));
      }
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_INT64: {
      if (total_size != input.int64_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.int64_val_size()));
        break;
      }
      Tensor tensor(DT_INT64, tensor_shape);
      auto flat = tensor.flat<int64>();
      memcpy(flat.data(), input.int64_val().data(),
          input.int64_val_size() * sizeof(int64));
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_BOOL: {
      if (total_size != input.bool_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.bool_val_size()));
        break;
      }
      Tensor tensor(DT_BOOL, tensor_shape);
      auto flat = tensor.flat<bool>();
      for (int i = 0; i < input.bool_val_size(); ++i) {
        flat(i) = input.bool_val(i);
      }
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_QINT8: {
      if (total_size != input.int_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.int_val_size()));
        break;
      }
      Tensor tensor(DT_QINT8, tensor_shape);
      auto flat = tensor.flat<qint8>();
      for (int i = 0; i < input.int_val_size(); ++i) {
        flat(i) = qint8(input.int_val(i));
      }
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_QUINT8: {
      if (total_size != input.int_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.int_val_size()));
        break;
      }
      Tensor tensor(DT_QUINT8, tensor_shape);
      auto flat = tensor.flat<quint8>();
      for (int i = 0; i < input.int_val_size(); ++i) {
        flat(i) = quint8(input.int_val(i));
      }
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_QINT32: {
      if (total_size != input.int_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.int_val_size()));
        break;
      }
      Tensor tensor(DT_QINT32, tensor_shape);
      auto flat = tensor.flat<qint32>();
      for (int i = 0; i < input.int_val_size(); ++i) {
        flat(i) = qint32(input.int_val(i));
      }
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_QINT16: {
      if (total_size != input.int_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.int_val_size()));
        break;
      }
      Tensor tensor(DT_QINT16, tensor_shape);
      auto flat = tensor.flat<qint16>();
      for (int i = 0; i < input.int_val_size(); ++i) {
        flat(i) = qint16(input.int_val(i));
      }
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_QUINT16: {
      if (total_size != input.int_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.int_val_size()));
        break;
      }
      Tensor tensor(DT_QUINT16, tensor_shape);
      auto flat = tensor.flat<quint16>();
      for (int i = 0; i < input.int_val_size(); ++i) {
        flat(i) = quint16(input.int_val(i));
      }
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_BFLOAT16: {
      if (total_size != input.float_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.float_val_size()));
        break;
      }
      Tensor tensor(DT_BFLOAT16, tensor_shape);
      auto flat = tensor.flat<bfloat16>();
      tensorflow::FloatToBFloat16(input.float_val().data(),
                                  flat.data(),
                                  input.float_val_size());
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_HALF: {
      if (total_size != input.float_val_size()) {
        ret.status = errors::InvalidArgument(
            "Invalid input: ", key, " . shape size VS input size = ",
            std::to_string(total_size), " VS ", std::to_string(input.float_val_size()));
        break;
      }
      Tensor tensor(DT_HALF, tensor_shape);
      auto flat = tensor.flat<Eigen::half>();
      for (int i = 0; i < input.float_val_size(); ++i) {
        flat(i) = Eigen::half(input.float_val(i));
      }
      ret.tensor = std::move(tensor);
    }
    case tensorflow::eas::DT_RESOURCE: {
      ret.status = errors::InvalidArgument(
          "Input Tensor: ", key, ", Not Support this DataType: DT_RESOURCE");
      break;
    }
    case tensorflow::eas::DT_VARIANT: {
      ret.status = errors::InvalidArgument(
          "Input Tensor: ", key, ", Not Support this DataType: DT_VARIANT");
      break;
    }
    default: {
      ret.status = errors::InvalidArgument(
          "Input Tensor: ", key, ", Not Support this DataType");
      break;
    }
  }

  return ret;
}

eas::PredictResponse Tensor2Response(
    const processor::Request& req,
    const processor::Response& resp,
    const SignatureInfo* signature_info) {
  eas::PredictResponse response;
  const auto& output_tensor_names = req.output_tensor_names;
  const auto & outputs = resp.outputs;

  for (size_t i = 0; i < outputs.size(); ++i) {
    eas::ArrayProto output;
    int64 total_dim_size = 1;
    for (int j = 0; j < outputs[i].dims(); ++j) {
      int64 dim_size = outputs[i].dim_size(j);
      output.mutable_array_shape()->add_dim(dim_size);
      total_dim_size *= dim_size;
    }

    switch (outputs[i].dtype()) {
      case DT_FLOAT: {
        output.set_dtype(eas::DT_FLOAT);
        auto flat = outputs[i].flat<float>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_float_val(flat(j));
        }
        break;
      }
      case DT_DOUBLE: {
        output.set_dtype(eas::DT_DOUBLE);
        auto flat = outputs[i].flat<double>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_double_val(flat(j));
        }
        break;
      }
      case DT_INT32: {
        output.set_dtype(eas::DT_INT32);
        auto flat = outputs[i].flat<int>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val(flat(j));
        }
        break;
      }
      case DT_UINT8: {
        output.set_dtype(eas::DT_UINT8);
        auto flat = outputs[i].flat<uint8>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val((int)flat(j));
        }
        break;
      }
      case DT_INT16: {
        output.set_dtype(eas::DT_INT16);
        auto flat = outputs[i].flat<int16>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val((int)flat(j));
        }
        break;
      }
      case DT_INT8: {
        output.set_dtype(eas::DT_INT8);
        auto flat = outputs[i].flat<int8>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val((int)flat(j));
        }
        break;
      }
      case DT_QINT8: {
        output.set_dtype(eas::DT_QINT8);
        auto flat = outputs[i].flat<qint8>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val(flat(j).value);
        }
        break;
      }
      case DT_QUINT8: {
        output.set_dtype(eas::DT_QUINT8);
        auto flat = outputs[i].flat<quint8>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val(flat(j).value);
        }
        break;
      }
      case DT_QINT32: {
        output.set_dtype(eas::DT_QINT32);
        auto flat = outputs[i].flat<qint32>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val(flat(j).value);
        }
        break;
      }
      case DT_QINT16: {
        output.set_dtype(eas::DT_QINT16);
        auto flat = outputs[i].flat<qint16>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val(flat(j).value);
        }
        break;
      }
      case DT_QUINT16: {
        output.set_dtype(eas::DT_QUINT16);
        auto flat = outputs[i].flat<quint16>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val(flat(j).value);
        }
        break;
      }
      case DT_UINT16: {
        output.set_dtype(eas::DT_UINT16);
        auto flat = outputs[i].flat<uint16>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int_val((int)flat(j));
        }
        break;
      }
      case DT_INT64: {
        output.set_dtype(eas::DT_INT64);
        auto flat = outputs[i].flat<int64>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_int64_val(flat(j));
        }
        break;
      }
      case DT_BOOL: {
        output.set_dtype(eas::DT_BOOL);
        auto flat = outputs[i].flat<bool>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_bool_val(flat(j));
        }
        break;
      }
      case DT_STRING: {
        output.set_dtype(eas::DT_STRING);
        auto flat = outputs[i].flat<std::string>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_string_val(flat(j));
        }
        break;
      }
      case DT_COMPLEX64: {
        output.set_dtype(eas::DT_COMPLEX64);
        auto flat = outputs[i].flat<complex64>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_float_val(flat(j).real());
          output.add_float_val(flat(j).imag());
        }
        break;
      }
      case DT_COMPLEX128: {
        output.set_dtype(eas::DT_COMPLEX128);
        auto flat = outputs[i].flat<complex128>();
        for (int64 j = 0; j < total_dim_size; ++j) {
          output.add_double_val(flat(j).real());
          output.add_double_val(flat(j).imag());
        }
        break;
      }
      case DT_HALF: {
        output.set_dtype(eas::DT_HALF);
        auto flat = outputs[i].flat<Eigen::half>();
        for (int64 j = 0; j < total_dim_size; j++)
          output.add_float_val((float)flat(j));
        break;
      }
      case DT_BFLOAT16: {
        output.set_dtype(eas::DT_BFLOAT16);
        auto flat = outputs[i].flat<bfloat16>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++) {
          float value;
          BFloat16ToFloat(&flat(j), &value, 1);
          output.add_float_val(value);
        }
        break;
      }
      case tensorflow::eas::DT_RESOURCE: {
        LOG(ERROR) << "Output Tensor Not Support this DataType: DT_RESOURCE";
        break;
      }
      case tensorflow::eas::DT_VARIANT: {
        LOG(ERROR) << "Output Tensor Not Support this DataType: DT_VARIANT";
        break;
      }
      default:
        LOG(ERROR) << "Output Tensor Not Support this DataType";
        break;
    }
    if (signature_info->output_value_name_idx.find(output_tensor_names[i]) ==
        signature_info->output_value_name_idx.end()) {
      LOG(ERROR) << "Response contain invalid output tensor name: "
                 << output_tensor_names[i];
    }
    std::string key =
        signature_info->output_key[signature_info->output_value_name_idx.at(output_tensor_names[i])];
    (*response.mutable_outputs())[key] = output;
  }
  return response;
}

} // namespace util
} // namespace processor
} // namespace tensorflow
