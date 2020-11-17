#include <dirent.h>
#include <string.h>
#include <unistd.h>
#include <sstream>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"
#include "freeze_bundle.h"
#include "run_predict.h"
#include "saved_model_loader.h"
#include "odl_processor/serving/tf_predict.pb.h"

using tensorflow::ERROR;
using tensorflow::Status;
using tensorflow::TensorShape;
using tensorflow::kSavedModelTagServe;
using tensorflow::kPredictMethodName;
using tensorflow::kClassifyMethodName;
using tensorflow::kRegressMethodName;
using tensorflow::kDefaultServingSignatureDefKey;
using tensorflow::MaybeSavedModelDirectory;
using tensorflow::LoadSavedModel;
using tensorflow::SessionOptions;
using tensorflow::RunOptions;
using tensorflow::eas::ArrayProto;

namespace tensorflow {
namespace serving {
class SessionBundleConfig {
 public:
  SessionBundleConfig() : is_set_threadpool_index(false) {}

  ConfigProto* mutable_session_config() { return &config; }

  ConfigProto session_config() { return config; }

  string session_target() { return target; }

  bool has_session_run_load_threadpool_index() {
    return is_set_threadpool_index;
  }

  int session_run_threadpool_index_value() { return threadpool_index; }

 private:
  ConfigProto config;
  string target;
  int threadpool_index;
  bool is_set_threadpool_index;
};
}
}

namespace {
std::string MapKeysToString(std::vector<std::string>& tensor_alias_name) {
  std::string result = "";
  for (size_t i = 0; i < tensor_alias_name.size(); i++) {
    if (result.empty())
      result += tensor_alias_name[i];
    else
      result += ", " + tensor_alias_name[i];
  }
  return result;
}

SessionOptions GetSessionOptions(SessionBundleConfig& config) {
  SessionOptions options;
  options.target = config.session_target();
  options.config = config.session_config();
  return options;
}

RunOptions GetRunOptions(SessionBundleConfig& config) {
  RunOptions run_options;
  if (config.has_session_run_load_threadpool_index()) {
    run_options.set_inter_op_thread_pool(
        config.session_run_threadpool_index_value());
  }
  return run_options;
}

bool GetModelPath(const std::string& root, std::string* model_path) {
  DIR* pdir;
  struct dirent* ent;
  char buf[BUFSIZ];
  char* pos;
  memset(buf, 0, BUFSIZ);

  if ((pdir = opendir(root.c_str())) == NULL) {
    LOG(ERROR) << "Cannot open path: " << root;
    return false;
  }
  while ((ent = readdir(pdir))) {
    if (ent->d_type & DT_DIR) {
      if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
        continue;
      snprintf(buf, sizeof(buf), "%s/%s", root.c_str(), ent->d_name);
      GetModelPath(buf, model_path);
    } else {
      if (strcmp(ent->d_name, "saved_model.pb") == 0 ||
          strcmp(ent->d_name, "saved_model.pbtxt") == 0) {
        *model_path = root;
        break;
      }
      if (!(pos = strstr(ent->d_name, ".meta"))) continue;
      if (*(pos + 5) != '\0') continue;
      *model_path = root;
      break;
    }
  }
  closedir(pdir);
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
}  // namespace

namespace processor {
Signature::Signature(SignatureDef* sig_def, const std::string& name) :
  signature_def(sig_def), signature_name(name) {}
}

SavedModelLoader::~SavedModelLoader() {
  delete saved_model_bundle;
  delete freeze_bundle;
  delete config;
}

SavedModelLoader::SavedModelLoader(int inter_op_parallelism_threads,
                                   int intra_op_parallelism_threads,
                                   double per_process_gpu_memory_fraction,
                                   bool allow_growth, bool allow_soft_placement,
                                   bool log_device_placement,
                                   bool allow_bfc_allocator)
    : saved_model_bundle(NULL),
      freeze_bundle(NULL),
      config(NULL),
      is_saved_model(true),
      is_freeze_model(false) {
  config = new SessionBundleConfig;
  config->mutable_session_config()->set_intra_op_parallelism_threads(
      intra_op_parallelism_threads);
  config->mutable_session_config()->set_inter_op_parallelism_threads(
      inter_op_parallelism_threads);
  config->mutable_session_config()
      ->mutable_gpu_options()
      ->set_per_process_gpu_memory_fraction(per_process_gpu_memory_fraction);
  config->mutable_session_config()->mutable_gpu_options()->set_allow_growth(
      allow_growth);
  config->mutable_session_config()->set_allow_soft_placement(
      allow_soft_placement);
  config->mutable_session_config()->set_log_device_placement(
      log_device_placement);
  if (allow_bfc_allocator)
    config->mutable_session_config()->mutable_gpu_options()->set_allocator_type(
        "BFC");
}

processor::Signature* SavedModelLoader::GetModelSignatureInfo() {
  return model_signature_info;
}

int SavedModelLoader::LoadModel(const std::string& path) {
  Status status;
  std::string model_path;

  if (processor::MaybeFreezeModelFile(path)) {
    is_freeze_model = true;
    LOG(INFO) << "Attempting to load native FreezeModel from " << path;
    freeze_bundle = new processor::FreezeBundle;
    status =
        processor::LoadFreezeModel(GetSessionOptions(*config), path, freeze_bundle);
    if (status.ok()) {
      return 0;
    } else {
      LOG(ERROR) << "LoadModel Error: " << status.error_message();
      return -1;
    }
  }

  if (!GetModelPath(path, &model_path)) {
    LOG(ERROR) << "Cannot find Model Path from " << path;
    return -1;
  }
  if (MaybeSavedModelDirectory(model_path)) {
    LOG(INFO) << "Attempting to load native SavedModel from " << model_path;
    std::map<int, std::string> dtype_to_string = {
        {1, "DT_FLOAT"}, {2, "DT_DOUBLE"}, {3, "DT_INT32"}, {4, "DT_UINT8"},
        {6, "DT_INT8"},  {7, "DT_STRING"}, {9, "DT_INT64"}, {10, "DT_BOOL"}};
    saved_model_bundle = new SavedModelBundle;
    status =
        LoadSavedModel(GetSessionOptions(*config), GetRunOptions(*config),
                       model_path, {kSavedModelTagServe}, saved_model_bundle);
    LOG(INFO) << "Show SignatureDef Information:";
    std::ostringstream model_signature;
    auto iter = saved_model_bundle->meta_graph_def.signature_def().begin();
    for (; iter != saved_model_bundle->meta_graph_def.signature_def().end();
         iter++) {
      if ((iter->second).method_name() == kPredictMethodName) {
        model_signature_info = new processor::Signature(
            new SignatureDef(iter->second), iter->first);
      }
    }
  } else {
    LOG(ERROR)
        << "SessionBundle or SavedModel not found at specified export location:"
        << model_path;
    return -1;
  }
  if (status.ok()) {
    return 0;
  } else {
    LOG(ERROR) << "LoadModel Error: " << status.error_message();
    return -1;
  }
}

int SavedModelLoader::Predict(const RunRequest& request,
                              RunResponse* response) {
  if (is_freeze_model) return FreezeModelPredict(request, response);
  if (is_saved_model) {
    return SavedModelPredict(request, response);
  }
}

int SavedModelLoader::Predict(const PredictRequest& request,
                              PredictResponse* response) {
  if (is_freeze_model) return FreezeModelPredict(request, response);
  if (is_saved_model) {
    return SavedModelPredict(request, response);
  }
}

int SavedModelLoader::FreezeModelPredict(const PredictRequest& request,
                                         PredictResponse* response) {
  std::string error_message;
  std::stringstream stream;
  std::vector<std::pair<std::string, Tensor> > inputs;
  for (auto& input : request.inputs()) {
    const std::string& tensor_name = input.first;
    if (ConvertProtoToTensor(&inputs, tensor_name, request, tensor_name) < 0)
      return -1;
  }
  std::vector<std::string> output_tensor_names(request.output_filter().begin(),
                                               request.output_filter().end());

  std::vector<Tensor> outputs;
  Status status =
      freeze_bundle->session->Run(inputs, output_tensor_names, {}, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << "Call Session Run Error: " << status.error_message();
    error_message = "Call Session Run Error: " + status.error_message();
    throw std::runtime_error(error_message);
  }

  if (outputs.size() != output_tensor_names.size()) {
    LOG(ERROR) << "The number of output tensors error: " << outputs.size();
    stream.str("");
    stream << "The number of output tensors error: " << outputs.size();
    throw std::runtime_error(stream.str());
  }
  if (ConvertTensorToProto(outputs, output_tensor_names, response) < 0)
    return -1;
  return 0;
}

int SavedModelLoader::FreezeModelPredict(const RunRequest& request,
                                         RunResponse* response) {
  std::string error_message;
  std::stringstream stream;
  std::vector<std::pair<std::string, Tensor> > inputs;
  const std::map<std::string, Tensor*>& inputTensors = request.GetInputs();
  inputs.reserve(inputTensors.size());
  for (auto& input : inputTensors) {
    inputs.emplace_back(std::make_pair(input.first, *(input.second)));
  }

  std::vector<std::string> output_tensor_names(
      request.GetOutputAliasNames().begin(),
      request.GetOutputAliasNames().end());
  std::vector<Tensor> outputs;
  Status status =
      freeze_bundle->session->Run(inputs, output_tensor_names, {}, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << "Call Session Run Error: " << status.error_message();
    error_message = "Call Session Run Error: " + status.error_message();
    throw std::runtime_error(error_message);
  }
  if (outputs.size() != output_tensor_names.size()) {
    LOG(ERROR) << "The number of output tensors error: " << outputs.size();
    stream.str("");
    stream << "The number of output tensors error: " << outputs.size();
    throw std::runtime_error(stream.str());
  }
  for (int i = 0; i < outputs.size(); i++) {
    response->SetOutputTensor(output_tensor_names[i], outputs[i]);
  }

  return 0;
}

int SavedModelLoader::SavedModelPredict(const RunRequest& request,
                                        RunResponse* response) {
  std::string error_message;
  std::stringstream stream;
  const std::string signature_name = request.GetSignatureName().empty()
                                         ? kDefaultServingSignatureDefKey
                                         : request.GetSignatureName();
  auto iter = saved_model_bundle->meta_graph_def.signature_def().begin();
  std::vector<std::string> signature_def_names;
  for (; iter != saved_model_bundle->meta_graph_def.signature_def().end();
       iter++) {
    signature_def_names.push_back(iter->first);
    if (signature_name == iter->first) break;
  }
  if (iter == saved_model_bundle->meta_graph_def.signature_def().end()) {
    LOG(ERROR) << "signatureDef name: " << signature_name
               << ", which not found in set {"
               << MapKeysToString(signature_def_names) << "}.";
    error_message = "signatureDef name: " + signature_name +
                    ", which not found in set {" +
                    MapKeysToString(signature_def_names) + "}.";
    throw std::runtime_error(error_message);
  }
  SignatureDef signature = iter->second;

  std::vector<std::pair<std::string, Tensor> > inputs;
  std::vector<std::string> output_tensor_names;
  std::vector<std::string> output_tensor_aliases;

  if (signature.method_name() != kPredictMethodName &&
      signature.method_name() != kClassifyMethodName &&
      signature.method_name() != kRegressMethodName) {
    LOG(ERROR) << "Expected prediction signature method_name must be one of {"
               << kPredictMethodName << ", " << kClassifyMethodName << ", "
               << kRegressMethodName << "}. Was: " << signature.method_name();
    error_message =
        "Expected prediction signature method_name must be one of {" +
        std::string(kPredictMethodName) + ", " +
        std::string(kClassifyMethodName) + ", " +
        std::string(kRegressMethodName) + "}. Was: " + signature.method_name();
    throw std::runtime_error(error_message);
  }
  if (signature.inputs().empty()) {
    LOG(ERROR) << "Expected at least one input Tensor in prediction signature";
    error_message =
        "Expected at least one input Tensor in prediction signature";
    throw std::runtime_error(error_message);
  }
  if (signature.outputs().empty()) {
    LOG(ERROR) << "Expected at least one output Tensor in prediction signature";
    error_message =
        "Expected at least one output Tensor in prediction signature";
    throw std::runtime_error(error_message);
  }
  const std::map<std::string, Tensor*>& inputTensors = request.GetInputs();
  inputs.reserve(inputTensors.size());
  for (auto& input : inputTensors) {
    const std::string& alias = input.first;
    auto iter = signature.inputs().begin();
    std::vector<std::string> tensor_alias_name;
    tensor_alias_name.reserve(signature.inputs().size());
    for (; iter != signature.inputs().end(); iter++) {
      tensor_alias_name.push_back(iter->first);
      if (alias == iter->first) break;
    }
    if (iter == signature.inputs().end()) {
      LOG(ERROR) << "Input tensor alias not found in signature: " << alias
                 << ". Inputs expected to be in the set {"
                 << MapKeysToString(tensor_alias_name) << "}.";
      error_message = "Input tensor alias not found in signature: " + alias +
                      ". Inputs expected to be in the set {" +
                      MapKeysToString(tensor_alias_name) + "}.";
      throw std::runtime_error(error_message);
    }
    inputs.emplace_back(std::make_pair(iter->second.name(), *(input.second)));
  }

  std::set<std::string> seen_outputs;
  std::vector<std::string> output_filter(request.GetOutputAliasNames().begin(),
                                         request.GetOutputAliasNames().end());
  output_tensor_names.reserve(signature.outputs().size());
  output_tensor_aliases.reserve(signature.outputs().size());
  for (auto& alias : output_filter) {
    auto iter = signature.outputs().begin();
    std::vector<std::string> tensor_alias_name;
    tensor_alias_name.reserve(signature.outputs().size());
    for (; iter != signature.outputs().end(); iter++) {
      tensor_alias_name.push_back(iter->first);
      if (alias == iter->first) break;
    }
    if (iter == signature.outputs().end()) {
      LOG(ERROR) << "Output tensor alias not found in signature: " << alias
                 << " Outputs expected to be in the set {"
                 << MapKeysToString(tensor_alias_name) << "}.";
      error_message = "Output tensor alias not found in signature: " + alias +
                      ". Outputs expected to be in the set {" +
                      MapKeysToString(tensor_alias_name) + "}.";
      throw std::runtime_error(error_message);
    }
    if (seen_outputs.find(alias) != seen_outputs.end()) {
      LOG(ERROR) << "duplicate output tensor alias: " << alias;
      error_message = "duplicate output tensor alias: " + alias;
      throw std::runtime_error(error_message);
    }
    seen_outputs.insert(alias);
    output_tensor_names.emplace_back(iter->second.name());
    output_tensor_aliases.emplace_back(alias);
  }
  if (output_tensor_names.empty()) {
    for (auto& iter : signature.outputs()) {
      output_tensor_names.emplace_back(iter.second.name());
      output_tensor_aliases.emplace_back(iter.first);
    }
  }

  std::vector<Tensor> outputs;
  Status status = saved_model_bundle->session->Run(inputs, output_tensor_names,
                                                   {}, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << "Call Session Run Error:" << status.error_message();
    error_message = "Call Session Run Error: " + status.error_message();
    throw std::runtime_error(error_message);
  }

  if (outputs.size() != output_tensor_names.size()) {
    LOG(ERROR) << "The number of output tensors error: " << outputs.size();
    stream.str("");
    stream << "The number of output tensors error: " << outputs.size();
    throw std::runtime_error(stream.str());
  }
  for (int i = 0; i < outputs.size(); i++) {
    response->SetOutputTensor(output_tensor_aliases[i], outputs[i]);
  }

  return 0;
}

int SavedModelLoader::SavedModelPredict(const PredictRequest& request,
                                        PredictResponse* response) {
  std::string error_message;
  std::stringstream stream;
  std::string signature_name = request.signature_name();
  auto iter = saved_model_bundle->meta_graph_def.signature_def().begin();
  std::vector<std::string> signature_def_names;
  for (; iter != saved_model_bundle->meta_graph_def.signature_def().end();
       iter++) {
	if (signature_name.empty()) {
      if (std::string((iter->second).method_name()) == "tensorflow/serving/predict") {
        signature_name = iter->first;
        break;
      }
	} else {
	  if (signature_name == iter->first) break;
	}
    signature_def_names.push_back(iter->first);
  }
  if (iter == saved_model_bundle->meta_graph_def.signature_def().end()) {
    LOG(ERROR) << "signatureDef name: " << signature_name
               << ", which not found in set {"
               << MapKeysToString(signature_def_names) << "}.";
    error_message = "signatureDef name: " + signature_name +
                    ", which not found in set {" +
                    MapKeysToString(signature_def_names) + "}.";
    throw std::runtime_error(error_message);
  }
  SignatureDef signature = iter->second;

  std::vector<std::pair<std::string, Tensor> > inputs;
  std::vector<std::string> output_tensor_names;
  std::vector<std::string> output_tensor_aliases;
  int pre_status =
      PreProcessPrediction(signature, request, &inputs, &output_tensor_names,
                           &output_tensor_aliases);
  if (pre_status < 0) {
    return -1;
  }
  std::vector<Tensor> outputs;
  Status status = saved_model_bundle->session->Run(inputs, output_tensor_names,
                                                   {}, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << "Call Session Run Error: " << status.error_message();
    error_message = "Call Session Run Error: " + status.error_message();
    throw std::runtime_error(error_message);
  }
  if (outputs.size() != output_tensor_aliases.size()) {
    LOG(ERROR) << "The number of output tensors error: " << outputs.size();
    stream.str("");
    stream << "The number of output tensors error: " << outputs.size();
    throw std::runtime_error(stream.str());
  }
  if (ConvertTensorToProto(outputs, output_tensor_aliases, response) < 0)
    return -1;

  return 0;
}

int SavedModelLoader::PreProcessPrediction(
    const SignatureDef& signature, const PredictRequest& request,
    std::vector<std::pair<std::string, Tensor> >* inputs,
    std::vector<std::string>* output_tensor_names,
    std::vector<std::string>* output_tensor_aliases) {
  std::string error_message;
  std::stringstream stream;
  if (signature.method_name() != kPredictMethodName &&
      signature.method_name() != kClassifyMethodName &&
      signature.method_name() != kRegressMethodName) {
    LOG(ERROR) << "Expected prediction signature method_name must be one of {"
               << kPredictMethodName << ", " << kClassifyMethodName << ", "
               << kRegressMethodName << "}. Was: " << signature.method_name();
    error_message =
        "Expected prediction signature method_name must be one of {" +
        std::string(kPredictMethodName) + ", " +
        std::string(kClassifyMethodName) + ", " +
        std::string(kRegressMethodName) + "}. Was: " + signature.method_name();
    throw std::runtime_error(error_message);
  }
  if (signature.inputs().empty()) {
    LOG(ERROR) << "Expected at least one input Tensor in prediction signature";
    error_message =
        "Expected at least one input Tensor in prediction signature";
    throw std::runtime_error(error_message);
  }
  if (signature.outputs().empty()) {
    LOG(ERROR) << "Expected at least one output Tensor in prediction signature";
    error_message =
        "Expected at least one output Tensor in prediction signature";
    throw std::runtime_error(error_message);
  }
  for (auto& input : request.inputs()) {
    const std::string& alias = input.first;
    auto iter = signature.inputs().begin();
    std::vector<std::string> tensor_alias_name;
    tensor_alias_name.reserve(signature.inputs().size());
    for (; iter != signature.inputs().end(); iter++) {
      tensor_alias_name.push_back(iter->first);
      if (alias == iter->first) break;
    }
    if (iter == signature.inputs().end()) {
      LOG(ERROR) << "Input tensor alias not found in signature: " << alias
                 << ". Inputs expected to be in the set {"
                 << MapKeysToString(tensor_alias_name) << "}.";
      error_message = "Input tensor alias not found in signature: " + alias +
                      ". Inputs expected to be in the set {" +
                      MapKeysToString(tensor_alias_name) + "}.";
      throw std::runtime_error(error_message);
    }
    if (ConvertProtoToTensor(inputs, iter->second.name(), request, alias) < 0)
      return -1;
  }

  std::set<std::string> seen_outputs;
  std::vector<std::string> output_filter(request.output_filter().begin(),
                                         request.output_filter().end());
  output_tensor_names->reserve(signature.outputs().size());
  output_tensor_aliases->reserve(signature.outputs().size());
  for (auto& alias : output_filter) {
    auto iter = signature.outputs().begin();
    std::vector<std::string> tensor_alias_name;
    tensor_alias_name.reserve(signature.outputs().size());
    for (; iter != signature.outputs().end(); iter++) {
      tensor_alias_name.push_back(iter->first);
      if (alias == iter->first) break;
    }
    if (iter == signature.outputs().end()) {
      LOG(ERROR) << "Output tensor alias not found in signature: " << alias
                 << " Outputs expected to be in the set {"
                 << MapKeysToString(tensor_alias_name) << "}.";
      error_message = "Output tensor alias not found in signature: " + alias +
                      ". Outputs expected to be in the set {" +
                      MapKeysToString(tensor_alias_name) + "}.";
      throw std::runtime_error(error_message);
    }
    if (seen_outputs.find(alias) != seen_outputs.end()) {
      LOG(ERROR) << "duplicate output tensor alias: " << alias;
      error_message = "duplicate output tensor alias: " + alias;
      throw std::runtime_error(error_message);
    }
    seen_outputs.insert(alias);
    output_tensor_names->emplace_back(iter->second.name());
    output_tensor_aliases->emplace_back(alias);
  }
  if (output_tensor_names->empty()) {
    for (auto& iter : signature.outputs()) {
      output_tensor_names->emplace_back(iter.second.name());
      output_tensor_aliases->emplace_back(iter.first);
    }
  }
  return 0;
}

int SavedModelLoader::ConvertProtoToTensor(
    std::vector<std::pair<std::string, Tensor> >* inputs,
    const std::string& tensorName, const PredictRequest& request,
    const std::string& tensorAliasName) {
  TensorShape tensorShape;
  std::stringstream stream;
  auto iter = request.inputs().begin();
  for (; iter != request.inputs().end(); iter++)
    if (tensorAliasName == iter->first) break;
  ArrayProto proto = iter->second;
  tensorflow::int64 total_dim_size = 1;
  for (int i = 0; i < proto.array_shape().dim_size(); i++) {
    if (proto.array_shape().dim(i) <= 0) {
      LOG(ERROR) << "The shape of the input tensor " << tensorAliasName
                 << " should be greater than 0";
      stream.str("");
      stream << "The shape of the input tensor " << tensorAliasName
             << " should be greater than 0";
      throw std::runtime_error(stream.str());
    }
    // Program will crash when AddDim -1 since tf1.12
    tensorShape.AddDim(proto.array_shape().dim(i));
    total_dim_size *= proto.array_shape().dim(i);
  }

  switch (proto.dtype()) {
    case tensorflow::eas::DT_FLOAT: {
      if (total_dim_size != proto.float_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor float_tensor(tensorflow::DT_FLOAT, tensorShape);
      auto float_tensor_fat = float_tensor.flat<float>();
      memcpy(float_tensor_fat.data(), proto.float_val().data(),
             proto.float_val_size() * sizeof(float));
      inputs->emplace_back(std::make_pair(tensorName, float_tensor));
      break;
    }
    case tensorflow::eas::DT_DOUBLE: {
      if (total_dim_size != proto.double_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor double_tensor(tensorflow::DT_DOUBLE, tensorShape);
      auto double_tensor_fat = double_tensor.flat<double>();
      memcpy(double_tensor_fat.data(), proto.double_val().data(),
             proto.double_val_size() * sizeof(double));
      inputs->emplace_back(std::make_pair(tensorName, double_tensor));
      break;
    }
    case tensorflow::eas::DT_INT32: {
      if (total_dim_size != proto.int_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor int32_tensor(tensorflow::DT_INT32, tensorShape);
      auto int32_tensor_fat = int32_tensor.flat<int>();
      memcpy(int32_tensor_fat.data(), proto.int_val().data(),
             proto.int_val_size() * sizeof(int));
      inputs->emplace_back(std::make_pair(tensorName, int32_tensor));
      break;
    }
    case tensorflow::eas::DT_UINT8: {
      if (total_dim_size != proto.int_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor uint8_tensor(tensorflow::DT_UINT8, tensorShape);
      auto uint8_tensor_fat = uint8_tensor.flat<tensorflow::uint8>();
      for (int i = 0; i < proto.int_val_size(); i++)
        uint8_tensor_fat(i) = (tensorflow::uint8)proto.int_val(i);
      inputs->emplace_back(std::make_pair(tensorName, uint8_tensor));
      break;
    }
    case tensorflow::eas::DT_INT16: {
      if (total_dim_size != proto.int_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor int16_tensor(tensorflow::DT_INT16, tensorShape);
      auto int16_tensor_fat = int16_tensor.flat<tensorflow::int16>();
      for (int i = 0; i < proto.int_val_size(); i++)
        int16_tensor_fat(i) = (tensorflow::int16)proto.int_val(i);
      inputs->emplace_back(std::make_pair(tensorName, int16_tensor));
      break;
    }
    case tensorflow::eas::DT_INT8: {
      if (total_dim_size != proto.int_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor int8_tensor(tensorflow::DT_INT8, tensorShape);
      auto int8_tensor_fat = int8_tensor.flat<tensorflow::int8>();
      for (int i = 0; i < proto.int_val_size(); i++)
        int8_tensor_fat(i) = (tensorflow::int8)proto.int_val(i);
      inputs->emplace_back(std::make_pair(tensorName, int8_tensor));
      break;
    }
    case tensorflow::eas::DT_STRING: {
      if (total_dim_size != proto.string_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor string_tensor(tensorflow::DT_STRING, tensorShape);
      auto string_tensor_fat = string_tensor.flat<std::string>();
      for (int i = 0; i < proto.string_val_size(); i++)
        string_tensor_fat(i) = proto.string_val(i);
      inputs->emplace_back(std::make_pair(tensorName, string_tensor));
      break;
    }
    case tensorflow::eas::DT_COMPLEX64: {
      if (total_dim_size != proto.float_val_size() / 2) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor complex64_tensor(tensorflow::DT_COMPLEX64, tensorShape);
      auto complex64_tensor_fat =
          complex64_tensor.flat<tensorflow::complex64>();
      for (int i = 0; i < proto.float_val_size(); i += 2)
        complex64_tensor_fat(i) =
            tensorflow::complex64(proto.float_val(i), proto.float_val(i + 1));
      inputs->emplace_back(std::make_pair(tensorName, complex64_tensor));
      break;
    }
    case tensorflow::eas::DT_INT64: {
      if (total_dim_size != proto.int64_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor long_tensor(tensorflow::DT_INT64, tensorShape);
      auto long_tensor_fat = long_tensor.flat<tensorflow::int64>();
      memcpy(long_tensor_fat.data(), proto.int64_val().data(),
             proto.int64_val_size() * sizeof(tensorflow::int64));
      inputs->emplace_back(std::make_pair(tensorName, long_tensor));
      break;
    }
    case tensorflow::eas::DT_BOOL: {
      if (total_dim_size != proto.bool_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor bool_tensor(tensorflow::DT_BOOL, tensorShape);
      auto bool_tensor_fat = bool_tensor.flat<bool>();
      for (int i = 0; i < proto.bool_val_size(); i++)
        bool_tensor_fat(i) = proto.bool_val(i);
      inputs->emplace_back(std::make_pair(tensorName, bool_tensor));
      break;
    }
    case tensorflow::eas::DT_QINT8: {
      if (total_dim_size != proto.int_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor qint8_tensor(tensorflow::DT_QINT8, tensorShape);
      auto qint8_tensor_fat = qint8_tensor.flat<tensorflow::qint8>();
      for (int i = 0; i < proto.int_val_size(); i++)
        qint8_tensor_fat(i) = tensorflow::qint8(proto.int_val(i));
      inputs->emplace_back(std::make_pair(tensorName, qint8_tensor));
      break;
    }
    case tensorflow::eas::DT_QUINT8: {
      if (total_dim_size != proto.int_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor quint8_tensor(tensorflow::DT_QUINT8, tensorShape);
      auto quint8_tensor_fat = quint8_tensor.flat<tensorflow::quint8>();
      for (int i = 0; i < proto.int_val_size(); i++)
        quint8_tensor_fat(i) = tensorflow::quint8(proto.int_val(i));
      inputs->emplace_back(std::make_pair(tensorName, quint8_tensor));
      break;
    }
    case tensorflow::eas::DT_QINT32: {
      if (total_dim_size != proto.int_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor qint32_tensor(tensorflow::DT_QINT32, tensorShape);
      auto qint32_tensor_fat = qint32_tensor.flat<tensorflow::qint32>();
      for (int i = 0; i < proto.int_val_size(); i++)
        qint32_tensor_fat(i) = tensorflow::qint32(proto.int_val(i));
      inputs->emplace_back(std::make_pair(tensorName, qint32_tensor));
      break;
    }
    case tensorflow::eas::DT_BFLOAT16: {
      if (total_dim_size != proto.float_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor bfloat16_tensor(tensorflow::DT_BFLOAT16, tensorShape);
      auto bfloat16_tensor_fat = bfloat16_tensor.flat<tensorflow::bfloat16>();
      tensorflow::FloatToBFloat16(proto.float_val().data(),
                                  bfloat16_tensor_fat.data(),
                                  proto.float_val_size());
      inputs->emplace_back(std::make_pair(tensorName, bfloat16_tensor));
      break;
    }
    case tensorflow::eas::DT_QINT16: {
      if (total_dim_size != proto.int_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor qint16_tensor(tensorflow::DT_QINT16, tensorShape);
      auto qint16_tensor_fat = qint16_tensor.flat<tensorflow::qint16>();
      for (int i = 0; i < proto.int_val_size(); i++)
        qint16_tensor_fat(i) = tensorflow::qint16(proto.int_val(i));
      inputs->emplace_back(std::make_pair(tensorName, qint16_tensor));
      break;
    }
    case tensorflow::eas::DT_QUINT16: {
      if (total_dim_size != proto.int_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor quint16_tensor(tensorflow::DT_QUINT16, tensorShape);
      auto quint16_tensor_fat = quint16_tensor.flat<tensorflow::quint16>();
      for (int i = 0; i < proto.int_val_size(); i++)
        quint16_tensor_fat(i) = tensorflow::quint16(proto.int_val(i));
      inputs->emplace_back(std::make_pair(tensorName, quint16_tensor));
      break;
    }
    case tensorflow::eas::DT_UINT16: {
      if (total_dim_size != proto.int_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor uint16_tensor(tensorflow::DT_UINT16, tensorShape);
      auto uint16_tensor_fat = uint16_tensor.flat<tensorflow::uint16>();
      for (int i = 0; i < proto.int_val_size(); i++)
        uint16_tensor_fat(i) = (tensorflow::uint16)proto.int_val(i);
      inputs->emplace_back(std::make_pair(tensorName, uint16_tensor));
      break;
    }
    case tensorflow::eas::DT_COMPLEX128: {
      if (total_dim_size != proto.double_val_size() / 2) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor complex128_tensor(tensorflow::DT_COMPLEX128, tensorShape);
      auto complex128_tensor_fat =
          complex128_tensor.flat<tensorflow::complex128>();
      for (int i = 0; i < proto.double_val_size(); i += 2)
        complex128_tensor_fat(i) = tensorflow::complex128(
            proto.double_val(i), proto.double_val(i + 1));
      inputs->emplace_back(std::make_pair(tensorName, complex128_tensor));
      break;
    }
    case tensorflow::eas::DT_HALF: {
      if (total_dim_size != proto.float_val_size()) {
        LOG(ERROR) << "The shape and content of the input tensor "
                   << tensorAliasName << " are not the same size";
        stream.str("");
        stream << "The shape and content of the input tensor "
               << tensorAliasName << " are not the same size";
        throw std::runtime_error(stream.str());
      }
      Tensor half_tensor(tensorflow::DT_HALF, tensorShape);
      auto half_tensor_fat = half_tensor.flat<Eigen::half>();
      for (int i = 0; i < proto.float_val_size(); i++)
        half_tensor_fat(i) = Eigen::half(proto.float_val(i));
      inputs->emplace_back(std::make_pair(tensorName, half_tensor));
      break;
    }
    case tensorflow::eas::DT_RESOURCE: {
      LOG(ERROR) << "Input Tensor Not Support this DataType: DT_RESOURCE";
      stream.str("");
      stream << "Input Tensor Not Support this DataType: DT_RESOURCE";
      throw std::runtime_error(stream.str());
    }
    case tensorflow::eas::DT_VARIANT: {
      LOG(ERROR) << "Input Tensor Not Support this DataType: DT_VARIANT";
      stream.str("");
      stream << "Input Tensor Not Support this DataType: DT_VARIANT";
      throw std::runtime_error(stream.str());
    }
    default:
      LOG(ERROR) << "Input Tensor Not Support this DataType";
      stream.str("");
      stream << "Input Tensor Not Support this DataType";
      throw std::runtime_error(stream.str());
  }
  return 0;
}

int SavedModelLoader::ConvertTensorToProto(
    std::vector<Tensor>& outputs,
    std::vector<std::string>& output_tensor_aliases,
    PredictResponse* response) {
  std::stringstream stream;
  for (int i = 0; i < outputs.size(); i++) {
    ArrayProto arrayProto;
    tensorflow::int64 total_dim_size = 1;
    for (int j = 0; j < outputs[i].dims(); j++) {
      tensorflow::int64 dim_size = outputs[i].dim_size(j);
      arrayProto.mutable_array_shape()->add_dim(dim_size);
      total_dim_size *= dim_size;
    }
    switch (outputs[i].dtype()) {
      case tensorflow::eas::DT_FLOAT: {
        arrayProto.set_dtype(tensorflow::eas::DT_FLOAT);
        auto float_tensor_fat = outputs[i].flat<float>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++)
          arrayProto.add_float_val(float_tensor_fat(j));
        break;
      }
      case tensorflow::eas::DT_DOUBLE: {
        arrayProto.set_dtype(tensorflow::eas::DT_DOUBLE);
        auto double_tensor_fat = outputs[i].flat<double>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++)
          arrayProto.add_double_val(double_tensor_fat(j));
        break;
      }
      case tensorflow::eas::DT_INT32: {
        arrayProto.set_dtype(tensorflow::eas::DT_INT32);
        auto int32_tensor_fat = outputs[i].flat<int>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++)
          arrayProto.add_int_val(int32_tensor_fat(j));
        break;
      }
      case tensorflow::eas::DT_UINT8: {
        arrayProto.set_dtype(tensorflow::eas::DT_UINT8);
        auto uint8_tensor_fat = outputs[i].flat<tensorflow::uint8>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++)
          arrayProto.add_int_val((int)uint8_tensor_fat(j));
        break;
      }
      case tensorflow::eas::DT_INT16: {
        arrayProto.set_dtype(tensorflow::eas::DT_INT16);
        auto int16_tensor_fat = outputs[i].flat<tensorflow::int16>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++)
          arrayProto.add_int_val((int)int16_tensor_fat(j));
        break;
      }
      case tensorflow::eas::DT_INT8: {
        arrayProto.set_dtype(tensorflow::eas::DT_INT8);
        auto int8_tensor_fat = outputs[i].flat<tensorflow::int8>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++)
          arrayProto.add_int_val((int)int8_tensor_fat(j));
        break;
      }
      case tensorflow::eas::DT_STRING: {
        arrayProto.set_dtype(tensorflow::eas::DT_STRING);
        auto string_tensor_fat = outputs[i].flat<std::string>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++)
          arrayProto.add_string_val(string_tensor_fat(j));
        break;
      }
      case tensorflow::eas::DT_COMPLEX64: {
        arrayProto.set_dtype(tensorflow::eas::DT_COMPLEX64);
        auto complex64_tensor_fat = outputs[i].flat<tensorflow::complex64>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++) {
          arrayProto.add_float_val(complex64_tensor_fat(j).real());
          arrayProto.add_float_val(complex64_tensor_fat(j).imag());
        }
        break;
      }
      case tensorflow::eas::DT_INT64: {
        arrayProto.set_dtype(tensorflow::eas::DT_INT64);
        auto int64_tensor_fat = outputs[i].flat<tensorflow::int64>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++)
          arrayProto.add_int64_val(int64_tensor_fat(j));
        break;
      }
      case tensorflow::eas::DT_BOOL: {
        arrayProto.set_dtype(tensorflow::eas::DT_BOOL);
        auto bool_tensor_fat = outputs[i].flat<bool>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++)
          arrayProto.add_bool_val(bool_tensor_fat(j));
        break;
      }
      case tensorflow::eas::DT_QINT8: {
        arrayProto.set_dtype(tensorflow::eas::DT_QINT8);
        auto qint8_tensor_fat = outputs[i].flat<tensorflow::qint8>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++)
          arrayProto.add_int_val(qint8_tensor_fat(j).value);
        break;
      }
      case tensorflow::eas::DT_QUINT8: {
        arrayProto.set_dtype(tensorflow::eas::DT_QUINT8);
        auto quint8_tensor_fat = outputs[i].flat<tensorflow::quint8>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++)
          arrayProto.add_int_val(quint8_tensor_fat(j).value);
        break;
      }
      case tensorflow::eas::DT_QINT32: {
        arrayProto.set_dtype(tensorflow::eas::DT_QINT32);
        auto qint32_tensor_fat = outputs[i].flat<tensorflow::qint32>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++)
          arrayProto.add_int_val(qint32_tensor_fat(j).value);
        break;
      }
      case tensorflow::eas::DT_BFLOAT16: {
        arrayProto.set_dtype(tensorflow::eas::DT_BFLOAT16);
        auto bfloat16_tensor_fat = outputs[i].flat<tensorflow::bfloat16>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++) {
          float value;
          tensorflow::BFloat16ToFloat(&bfloat16_tensor_fat(j), &value, 1);
          arrayProto.add_float_val(value);
        }
        break;
      }
      case tensorflow::eas::DT_QINT16: {
        arrayProto.set_dtype(tensorflow::eas::DT_QINT16);
        auto qint16_tensor_fat = outputs[i].flat<tensorflow::qint16>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++)
          arrayProto.add_int_val(qint16_tensor_fat(j).value);
        break;
      }
      case tensorflow::eas::DT_QUINT16: {
        arrayProto.set_dtype(tensorflow::eas::DT_QUINT16);
        auto quint16_tensor_fat = outputs[i].flat<tensorflow::quint16>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++)
          arrayProto.add_int_val(quint16_tensor_fat(j).value);
        break;
      }
      case tensorflow::eas::DT_UINT16: {
        arrayProto.set_dtype(tensorflow::eas::DT_UINT16);
        auto uint16_tensor_fat = outputs[i].flat<tensorflow::uint16>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++)
          arrayProto.add_int_val((int)uint16_tensor_fat(j));
        break;
      }
      case tensorflow::eas::DT_COMPLEX128: {
        arrayProto.set_dtype(tensorflow::eas::DT_COMPLEX128);
        auto complex128_tensor_fat = outputs[i].flat<tensorflow::complex128>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++) {
          arrayProto.add_double_val(complex128_tensor_fat(j).real());
          arrayProto.add_double_val(complex128_tensor_fat(j).imag());
        }
        break;
      }
      case tensorflow::eas::DT_HALF: {
        arrayProto.set_dtype(tensorflow::eas::DT_HALF);
        auto half_tensor_fat = outputs[i].flat<Eigen::half>();
        for (tensorflow::int64 j = 0; j < total_dim_size; j++)
          arrayProto.add_float_val((float)half_tensor_fat(j));
        break;
      }
      case tensorflow::eas::DT_RESOURCE: {
        LOG(ERROR) << "Output Tensor Not Support this DataType: DT_RESOURCE";
        stream.str("");
        stream << "Input Tensor Not Support this DataType: DT_RESOURCE";
        throw std::runtime_error(stream.str());
      }
      case tensorflow::eas::DT_VARIANT: {
        LOG(ERROR) << "Output Tensor Not Support this DataType: DT_VARIANT";
        stream.str("");
        stream << "Input Tensor Not Support this DataType: DT_VARIANT";
        throw std::runtime_error(stream.str());
      }
      default:
        LOG(ERROR) << "Output Tensor Not Support this DataType";
        stream.str("");
        stream << "Input Tensor Not Support this DataType";
        throw std::runtime_error(stream.str());
    }
    (*response->mutable_outputs())[output_tensor_aliases[i]] = arrayProto;
  }
  return 0;
}
