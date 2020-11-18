#include "odl_processor/core/graph_optimizer.h"
#include "odl_processor/serving/tf_predict.pb.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "model_serving.h"
#include "model_config.h"
#include "run_predict.h"

namespace tensorflow {
namespace processor {
/*namespace {
struct Signature {
  std::shared_ptr<SignatureDef> signature_def;
  std::string signature_name;

  Signature(SignatureDef* sig_def, const std::string& name) :
    signature_def(sig_def), signature_name(name) {
  }
};*/
//}

class ModelImpl {
 public:
  virtual Status Load(const char* model_dir) = 0;
  virtual Status Predict(const eas::PredictRequest& req,
                         const eas::PredictResponse& resp) = 0;
  virtual void GraphOptimize() = 0;
  virtual ~ModelImpl() {}

  virtual Status Load(const char* model_dir) = 0;
  virtual Status Predict(const eas::PredictRequest& req,
                         eas::PredictResponse* resp) = 0;
  virtual Status Predict(const RunRequest& req, RunResponse* resp) = 0;

  virtual Status Warmup() = 0;
  virtual std::string DebugString() = 0;
};

class FreezeSavedModelImpl : public ModelImpl {
 public:
  ~FreezeSavedModelImpl() override {
    // TODO
  }

  Status Load(const char* model_dir) override {
    return Status::OK();
  }

  Status Predict(const eas::PredictRequest& req,
                 eas::PredictResponse* resp) override {
    return Status::OK();
  }

  void GraphOptimize() override {
  }

  Status Predict(const RunRequest& req, RunResponse* resp) override {
    return Status::OK();
  }
  
  Status Warmup() override {
    return Status::OK();
  }

  std::string DebugString() override {
    return std::string();
  }
};

class SavedModelImpl : public ModelImpl {
 public:
  ~SavedModelImpl() override {
    delete saved_model_bundle_;
    delete optimizer_;
  }

  explicit SavedModelImpl(ModelConfig* config) :
    saved_model(new SavedModelBundle),
    model_config(config) {
  }

  Status Load(const char* model_dir) override {
    /*auto status = LoadSavedModel(session_options, run_options,
        model_dir, {kSavedModelTagServe}, saved_model);
    if (!status.ok()) {
      LOG(ERROR) << "[TensorFlow] Processor can't load model"
                 << ", model_dir:" << model_dir;
      return status;
    }
    
    auto model_signatures = saved_model->meta_graph_def.signature_def();
    for (auto it : model_signatures) {
      if ((it.second).method_name() == model_config->signature_name) {
        warmup_signature = std::make_pair(it.first, it.second);
      }
    }*/

    // 1 meta_graph_def = ReadMetaGraphDefFromSavedModel
    // 2 bundle->meta_graph_def = OptimizeGraph(signature_name, meta_graph_def)
    // 3 bundle->session = LoadMetaGraphIntoSession
    // 4 RunRestore(bundle)
    // 5 UpdateData(oss, embedding_service)
    //
    // 2 bundle object --> VersionManager

    return Status::OK();
  }

  Status Predict(const eas::PredictRequest& req,
                 eas::PredictResponse* resp) override {
    // cur_version->bundle->Run();
    return Status::OK();
  }
 
 private:
  void GraphOptimize() override {
    GraphOptimizerOptions opts;
    // TODO: please set flag 'cache_sparse_locally',
    // can get the value from config, default is false.
    optimizer_ = new SavedModelOptimizer(saved_model_bundle_, opts);
    optimizer_->Optimize();
  }

  Status Predict(const RunRequest& req, RunResponse* resp) override {
    return Status::OK();
  }

  Status Warmup() override {
    RunRequest request;
    request.SetSignatureName(warmup_signature.first);
    for (auto it : warmup_signature.second.inputs()) {
      request.AddFeed(it.first, it.second);
    }

    RunResponse response;
    return Predict(request, &response);
  }

  std::string DebugString() override {
    return warmup_signature.second.DebugString();
  }

 private:
  

 private:
  SavedModelBundle* saved_model_bundle_;
  Signature* model_signature; 
  SavedModelOptimizer* optimizer_;
  
  SavedModelBundle* saved_model;
  ModelConfig* model_config;

  SessionOptions session_options;
  RunOptions run_options;

  //std::map<std::string, SignatureDef> model_signatures; 
  std::pair<std::string, SignatureDef> warmup_signature;
};

class ModelImplFactory {
 public:
  static ModelImpl* Create(ModelConfig* config) {
    return new SavedModelImpl(config);
  }
};

Model::Model(const char* model_config) {
  config_ = ModelConfigFactory::Create(model_config);
  impl_ = ModelImplFactory::Create(config_);
}

Model::~Model() {
  delete impl_;
  delete config_;
}

Status Model::Load(const char* model_dir) {
  return impl_->Load(model_dir);
}

Status Model::Predict(const eas::PredictRequest& req,
                      eas::PredictResponse* resp) {
  return impl_->Predict(req, resp);
}

Status Model::Predict(const RunRequest& req,
                      RunResponse* resp) {
  return Status::OK();
}

Status Model::Warmup() {
  return impl_->Warmup();
}

std::string Model::DebugString() {
  return impl_->DebugString();
}

} // processor
} // tensorflow
