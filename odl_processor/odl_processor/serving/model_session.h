#ifndef TENSORFLOW_SERVING_MODEL_SESSION_H
#define TENSORFLOW_SERVING_MODEL_SESSION_H

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "odl_processor/serving/model_version.h"

#include <thread>
#include <atomic>

namespace tensorflow {
class SessionOptions;
class RunOptions;
class Session;
class Tensor;

namespace processor {
class SparseStorage;

struct ModelSession {
  ModelSession(Session* s, const Version& version,
      SparseStorage* sparse_storage) : session_(s), counter_(0),
    version_(version) {}

  Session* session_ = nullptr;
  SparseStorage* sparse_storage_ = nullptr;
  std::atomic<int64> counter_;

  Version version_;
};

class ModelSessionMgr {
 public:
  ModelSessionMgr(const MetaGraphDef& meta_graph_def,
      SessionOptions* session_options, RunOptions* run_options);
  
  Status Predict(
      const std::vector<std::pair<std::string, Tensor>>& inputs,
      const std::vector<std::string>& output_tensor_names,
      std::vector<Tensor>* outputs);

  Status CreateModelSession(const Version& version,
      const char* model_dir, SparseStorage* sparse_storage);

 private:
  Status CreateSession(Session** sess);
  Status RunRestoreOps(const char* model_dir, Session* session,
      SparseStorage* sparse_storage);
  void ResetServingSession(Session* session, const Version& version,
      SparseStorage* sparse_storage);

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

