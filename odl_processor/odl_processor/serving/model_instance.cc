#include "odl_processor/serving/model_instance.h"
#include "odl_processor/serving/model_config.h"
#include "odl_processor/serving/model_session.h"
#include "odl_processor/storage/model_storage.h"
#include "odl_processor/storage/sparse_storage.h"
#include "odl_processor/framework/graph_optimizer.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace {
constexpr int _60_Seconds = 60;
}
namespace processor {
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
  TF_RETURN_IF_ERROR(session_mgr_->CreateModelSession(version,
        version.full_model_name.c_str(), sparse_storage));

  if (version.delta_model_name.empty()) {
    return Status::OK();
  } else {
    return session_mgr_->CreateModelSession(version,
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

Status ModelInstance::Predict(Request& req, Response& resp) {
  return session_mgr_->Predict(req, resp);
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

Call ModelInstance::CreateWarmupParams() {
  Call call;
  for (auto it : model_signature_.second.inputs()) {
    const auto& tensor = CreateTensor(it.second);
    call.request.inputs.emplace_back(it.first, tensor);
  }

  for (auto it : model_signature_.second.outputs()) {
    call.request.output_tensor_names.emplace_back(it.first);
  }
  return call;
}

Status ModelInstance::Warmup() {
  Call call = CreateWarmupParams();
  return Predict(call.request, call.response);
}

Status ModelInstance::FullModelUpdate(const Version& version) {
  // Logically backup_storage_ shouldn't serving now.
  backup_storage_->Reset();
  TF_RETURN_IF_ERROR(session_mgr_->CreateModelSession(version,
      version.full_model_name.c_str(), backup_storage_));
 
  std::swap(backup_storage_, serving_storage_);
  return Status::OK();
}

Status ModelInstance::DeltaModelUpdate(const Version& version) {
  return session_mgr_->CreateModelSession(version,
      version.delta_model_name.c_str(), serving_storage_);
}

std::string ModelInstance::DebugString() {
  return model_signature_.second.DebugString();
}

ModelInstanceMgr::ModelInstanceMgr(const char* root_dir, ModelConfig* config)
  : model_storage_(new ModelStorage(config)), model_config_(config) {
  model_storage_->Init(root_dir);
  session_options_ = new SessionOptions();
  //session_options_->target = target;
  session_options_->config.set_intra_op_parallelism_threads(config->inter_threads);
  session_options_->config.set_inter_op_parallelism_threads(config->intra_threads);
  //session_options_->config.mutable_gpu_options()->set_allocator_type("CPU");
  run_options_ = new RunOptions();
}

ModelInstanceMgr::~ModelInstanceMgr() {
  is_stop_ = true;
  thread_->join();
  delete thread_;

  delete base_instance_;
  delete cur_instance_;
  delete session_options_;
  delete run_options_;
  delete model_storage_;
}

Status ModelInstanceMgr::Init() {
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

Status ModelInstanceMgr::Predict(Request& req, Response& resp) {
  return cur_instance_->Predict(req, resp);
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
  while(!is_stop_) {
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
