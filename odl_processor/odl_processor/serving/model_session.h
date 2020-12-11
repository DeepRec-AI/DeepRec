#ifndef TENSORFLOW_SERVING_MODEL_SESSION_H
#define TENSORFLOW_SERVING_MODEL_SESSION_H

#include "odl_processor/framework/model_version.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include <thread>
#include <atomic>

namespace tensorflow {
class SessionOptions;
class RunOptions;
class Session;
class Tensor;

namespace processor {
class FeatureStoreMgr;
class Request;
class Response;
struct ModelSession {
  ModelSession(Session* s, const Version& version,
      FeatureStoreMgr* sparse_storage);

  Status Predict(Request& req, Response& resp);

  Session* session_ = nullptr;
  FeatureStoreMgr* sparse_storage_ = nullptr;
  
  std::string sparse_storage_name_;
  Tensor sparse_storage_tensor_;
  std::atomic<int64> counter_;
  Version version_;
};

class ModelSessionMgr {
 public:
  ModelSessionMgr(const MetaGraphDef& meta_graph_def,
      SessionOptions* session_options, RunOptions* run_options);
  
  Status Predict(Request& req, Response& resp);

  Status CreateModelSession(const Version& version,
      const char* ckpt_name, FeatureStoreMgr* sparse_storage);

 private:
  Status CreateSession(Session** sess);
  Status RunRestoreOps(const char* ckpt_name, const char* savedmodel_dir,
      Session* session, FeatureStoreMgr* sparse_storage);
  void ResetServingSession(Session* session, const Version& version,
      FeatureStoreMgr* sparse_storage);

 private:
  ModelSession* serving_session_ = nullptr;
  std::vector<ModelSession*> sessions_;

  MetaGraphDef meta_graph_def_;
  SessionOptions* session_options_;
  RunOptions* run_options_;
  std::vector<AssetFileDef> asset_file_defs_;
};

} // processor
} // tensorflow

#endif // TENSORFLOW_SERVING_MODEL_SESSION_H

