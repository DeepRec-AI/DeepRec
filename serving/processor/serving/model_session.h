#ifndef SERVING_PROCESSOR_SERVING_MODEL_SESSION_H
#define SERVING_PROCESSOR_SERVING_MODEL_SESSION_H

#include "serving/processor/framework/model_version.h"
#include "serving/processor/serving/model_config.h"
#include "serving/processor/serving/model_message.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include <thread>
#include <atomic>

namespace tensorflow {
class SessionOptions;
class RunOptions;
class SessionGroup;
class Session;
class Tensor;

namespace processor {
class IFeatureStoreMgr;
class Request;
class Response;
enum SelectSessionPolicy {
  MOD = 1,
  RR = 2
};
struct ModelSession {
  ModelSession(SessionGroup* s, const std::string& select_session_policy,
      const Version& version, IFeatureStoreMgr* sparse_storage);
  ModelSession(SessionGroup* s, const std::string& select_session_policy,
      const Version& version);
  virtual ~ModelSession();

  Status Predict(Request& req, Response& resp);
  Status Predict(Request& req, Response& resp, int sess_id);
  Status LocalPredict(Request& req, Response& resp);
  Status LocalPredict(Request& req, Response& resp, int sess_id);
  Version GetVersion() {return version_;}
  void UpdateVersion(const Version& v) { version_ = v; }
  Session* GetSession();
  Status Warmup(Request& req, Response& resp, bool local=true);

  SessionGroup* session_group_ = nullptr;
  SelectSessionPolicy select_session_policy_ =
      SelectSessionPolicy::MOD;
  //IFeatureStoreMgr* sparse_storage_ = nullptr;
  
  std::string sparse_storage_name_;
  Tensor sparse_storage_tensor_;
  std::string model_version_name_;
  Tensor model_version_tensor_;
  std::atomic<int64> counter_;
  // Local storage or remote storage for sparse variable.
  bool is_local_ = true;
  Version version_;

 private:
  int GetServingSessionId();
  Status InternalPredict(Request& req, Response& resp, int sess_id);
  Status InternalLocalPredict(Request& req, Response& resp, int sess_id);
};

class ModelSessionMgr {
 public:
  ModelSessionMgr(const MetaGraphDef& meta_graph_def,
      SessionOptions* session_options, RunOptions* run_options);
  virtual ~ModelSessionMgr();

  Status Predict(Request& req, Response& resp);
  Status LocalPredict(Request& req, Response& resp);
  Status Warmup(Request& req, Response& resp, bool local=true);

  Status CreateModelSession(
      const Version& version,
      const char* saved_model_path,
      ModelConfig* config);

  Status CreateModelSession(
      const Version& version, const char* ckpt_name,
      IFeatureStoreMgr* sparse_storage,
      bool is_incr_ckpt, bool is_initialize,
      ModelConfig* config);

  Status CreateModelSession(
      const Version& version, const char* ckpt_name,
      IFeatureStoreMgr* sparse_storage,
      bool is_incr_ckpt, bool is_initialize,
      ModelConfig* config,
      ModelSession** new_model_session);

  Status CreateModelSession(
      const Version& version, const char* full_ckpt_name,
      const char* incr_ckpt_name, bool is_incr_ckpt,
      ModelConfig* config);
 
  Status CreateModelSession(
      const Version& version, const char* full_ckpt_name,
      const char* incr_ckpt_name, bool is_incr_ckpt,
      ModelConfig* config, ModelSession** new_model_session);

  Status CleanupModelSession();

  void ResetServingSession(ModelSession* model_session);

  Status GetServingModelInfo(
      tensorflow::processor::ServingModelInfo& model_info);

 private:
  virtual Status CreateSession(Session** sess);
  virtual Status CreateSessionGroup(
      SessionGroup** session_group, ModelConfig* config);
 
  virtual Status RunRestoreOps(
      const char* ckpt_name, int64 full_ckpt_version,
      const char* savedmodel_dir, Session* session,
      IFeatureStoreMgr* sparse_storage,
      bool is_incr_ckpt, bool update_sparse,
      int64_t latest_version);
  
  void ClearLoop();

 protected:
  ModelSession* serving_session_ = nullptr;

  MetaGraphDef meta_graph_def_;
  SessionOptions* session_options_;
  RunOptions* run_options_;
  std::vector<AssetFileDef> asset_file_defs_;

  std::thread* clear_session_thread_ = nullptr;
  std::vector<ModelSession*> sessions_;
  mutex mu_;
  volatile bool is_stop_ = false;
};

} // processor
} // tensorflow

#endif // SERVING_PROCESSOR_SERVING_MODEL_SESSION_H

