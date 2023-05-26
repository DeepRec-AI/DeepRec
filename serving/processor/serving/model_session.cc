#include <random>
#include "serving/processor/serving/model_session.h"
#include "serving/processor/serving/model_message.h"
#include "serving/processor/serving/tracer.h"
#include "serving/processor/serving/util.h"
#include "serving/processor/storage/model_store.h"
#include "serving/processor/storage/feature_store_mgr.h"
#include "serving/processor/framework/graph_optimizer.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/common_runtime/custom_thread_pool.h"
#include "tensorflow/core/platform/protobuf_internal.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"

namespace tensorflow {
namespace processor {

namespace {
constexpr int _30_Seconds = 30;
constexpr int _60_Seconds = 60;

int GetRandomNum() {
  std::random_device device("/dev/urandom");
  std::mt19937 r(device());
  return r();
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

void CreateCustomThreadPool(CustomThreadPoolImpl** tp, mutex& mu,
                            int num, const std::string& name) {
  mutex_lock lock(mu);
  if (!(*tp)) {
    *tp = new CustomThreadPoolImpl(name, num);
  }
}

static CustomThreadPoolImpl* GetModelUpdateInterThreadPool(int num) {
  static CustomThreadPoolImpl* tp = nullptr;
  static mutex mu;
  if (tp) return tp;
  CreateCustomThreadPool(&tp, mu, num, "user_model_update_inter");
  return tp;
}

static CustomThreadPoolImpl* GetModelUpdateIntraThreadPool(int num) {
  static CustomThreadPoolImpl* tp = nullptr;
  static mutex mu;
  if (tp) return tp;
  CreateCustomThreadPool(&tp, mu, num, "user_model_update_intra");
  return tp;
}

} // namespace

ModelSessionMgr::ModelSessionMgr(const MetaGraphDef& meta_graph_def,
    SessionOptions* session_options, RunOptions* run_options) :
  meta_graph_def_(meta_graph_def), session_options_(session_options),
  run_options_(run_options) {
  clear_session_thread_ = new std::thread(&ModelSessionMgr::ClearLoop, this);
}

void ModelSessionMgr::ClearLoop() {
  while(!is_stop_) {
    CleanupModelSession();
    sleep(_60_Seconds);
  }
}

ModelSessionMgr::~ModelSessionMgr() {
  is_stop_ = true;
  clear_session_thread_->join();
  delete clear_session_thread_;
}

Status ModelSessionMgr::CreateSession(Session** session) {
  TF_RETURN_IF_ERROR(NewSession(*session_options_, session));
  TF_RETURN_IF_ERROR((*session)->Create(meta_graph_def_.graph_def()));
  asset_file_defs_.clear();
  return util::GetAssetFileDefs(meta_graph_def_, &asset_file_defs_);
}

Status ModelSessionMgr::CreateSessionGroup(
    SessionGroup** session_group, ModelConfig* config) {
  SessionGroupMetadata metadata;
  metadata.session_count = config->session_num;
  metadata.model_id = 0;
  metadata.gpu_ids = config->gpu_ids;
  metadata.cpusets = config->cpusets;

  ConfigProto* opt_config = const_cast<ConfigProto*>(&(session_options_->config));
  GPUOptions* gpu_opt = opt_config->mutable_gpu_options();
  gpu_opt->mutable_experimental()->clear_virtual_devices();
  TF_RETURN_IF_ERROR(NewSessionGroup(*session_options_,
                                     session_group, metadata));
  TF_RETURN_IF_ERROR((*session_group)->Create(meta_graph_def_.graph_def()));
  asset_file_defs_.clear();
  return util::GetAssetFileDefs(meta_graph_def_, &asset_file_defs_);
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
      util::RunRestore(*run_options_, ckpt_name, savedmodel_dir,
          meta_graph_def_.saver_def().restore_op_name(),
          meta_graph_def_.saver_def().filename_tensor_name(),
          asset_file_defs_, session, update_sparse,
          latest_version, extra_tensors));

  if (util::HasMainOp(meta_graph_def_)) {
    return util::RunMainOp(*run_options_, savedmodel_dir,
        meta_graph_def_, asset_file_defs_, session,
        kSavedModelMainOpKey, sparse_storage_tensor_pair);
  } else {
    return util::RunMainOp(
        *run_options_, savedmodel_dir, meta_graph_def_,
        asset_file_defs_, session, kSavedModelLegacyInitOpKey,
        sparse_storage_tensor_pair);
  }
}

ModelSession::ModelSession(SessionGroup* s,
    const std::string& select_session_policy,
    const Version& version, IFeatureStoreMgr* sparse_storage,
    const std::string& graph_hash_value)
    : session_group_(s), counter_(0), is_local_(false),
      version_(version), graph_hash_value_(graph_hash_value) {
  if (select_session_policy == "MOD") {
    select_session_policy_ = SelectSessionPolicy::MOD;
  } else if (select_session_policy == "RR") {
    select_session_policy_ = SelectSessionPolicy::RR;
  } else {
    LOG(FATAL) << "[ModelSession] select_session_policy must be RR or MOD, current get "
               << select_session_policy;
  }

  Tensor t(DT_UINT64, TensorShape({}));
  t.scalar<uint64>()() = reinterpret_cast<uint64>(sparse_storage);
  sparse_storage_tensor_ = t;
  sparse_storage_name_ = GetStoragePointerNodeName();

  Tensor t_version(DT_UINT64, TensorShape({}));
  t_version.scalar<uint64>()() = version.full_ckpt_version;
  model_version_tensor_ = t_version;
  model_version_name_ = GetModelVersionNodeName();
}

ModelSession::ModelSession(SessionGroup* s,
    const std::string& select_session_policy, const Version& version,
    const std::string& graph_hash_value)
    : session_group_(s), counter_(0), is_local_(true),
      version_(version), graph_hash_value_(graph_hash_value) {
  if (select_session_policy == "MOD") {
    select_session_policy_ = SelectSessionPolicy::MOD;
  } else if (select_session_policy == "RR") {
    select_session_policy_ = SelectSessionPolicy::RR;
  } else {
    LOG(FATAL) << "[ModelSession] select_session_policy must be RR or MOD, current get "
               << select_session_policy;
  }

  Tensor t_version(DT_UINT64, TensorShape({}));
  t_version.scalar<uint64>()() = version.full_ckpt_version;
  model_version_tensor_ = t_version;
  model_version_name_ = GetModelVersionNodeName();
}

ModelSession::~ModelSession() {
  for (auto iter : incr_restore_handler_map) {
    const_cast<Session*>(iter.first)
        ->ReleaseCallable(*iter.second).IgnoreError();
  }

  for (auto iter : main_op_handler_map) {
    const_cast<Session*>(iter.first)
        ->ReleaseCallable(*iter.second).IgnoreError();
  }

  if (session_group_) {
    delete session_group_;
    session_group_ = nullptr;
  }
}

Session::CallableHandle* ModelSession::GetIncrRestoreHandler(const Session* sess) {
  auto iter = incr_restore_handler_map.find(sess);
  if (iter == incr_restore_handler_map.end()) {
    return nullptr;
  }
  return iter->second;
}

Session::CallableHandle* ModelSession::GetMainOpHandler(const Session* sess) {
  auto iter = main_op_handler_map.find(sess);
  if (iter == main_op_handler_map.end()) {
    return nullptr;
  }
  return iter->second;
}

void ModelSession::SetIncrRestoreHandler(const Session* sess,
    Session::CallableHandle* handler) {
  incr_restore_handler_map[sess] = handler;
}

void ModelSession::SetMainOpHandler(const Session* sess,
    Session::CallableHandle* handler) {
  main_op_handler_map[sess] = handler;
}

std::vector<Session*> ModelSession::GetLeaderSessions() {
  return session_group_->GetLeaderSessions();
}

int ModelSession::GetServingSessionId() {
  if (select_session_policy_ ==
      SelectSessionPolicy::RR) {
    return -1;
  }
  static std::atomic<int> counter{0};
  static thread_local int tid = -1;
  if (tid == -1) {
    tid = counter.fetch_add(1);
  }
  return tid;
}

Status ModelSession::Predict(Request& req, Response& resp) {
  return InternalPredict(req, resp, GetServingSessionId());
}

Status ModelSession::Predict(Request& req, Response& resp,
                             int sess_id) {
  return InternalPredict(req, resp, sess_id);
}

Status ModelSession::InternalPredict(Request& req, Response& resp,
                                     int sess_id) {
  if (is_local_) {
    return Status(error::Code::INTERNAL,
        "Local sparse storage, please use LocalPredict.");
  }

  req.inputs.emplace_back(sparse_storage_name_, sparse_storage_tensor_);
  req.inputs.emplace_back(model_version_name_, model_version_tensor_);
  ++counter_;
  Status status;
  tensorflow::RunOptions run_options;
  tensorflow::RunMetadata run_metadata;
  run_metadata.set_graph_signature(graph_hash_value_);
  if (Tracer::GetTracer()->NeedTracing()) {
    run_options.set_trace_level(tensorflow::RunOptions::FULL_TRACE);
    // TODO: which session selected to run on, add some policy here
    status = session_group_->Run(run_options, req.inputs,
        req.output_tensor_names, {}, &resp.outputs,
        &run_metadata, sess_id);
    Tracer::GetTracer()->GenTimeline(run_metadata);
  } else {
    status = session_group_->Run(run_options, req.inputs,
		req.output_tensor_names, {}, &resp.outputs,
		&run_metadata, sess_id);
  }
  --counter_;
  return status;
}

Status ModelSession::LocalPredict(Request& req,
                                  Response& resp) {
  return InternalLocalPredict(req, resp,
      GetServingSessionId());
}

Status ModelSession::LocalPredict(Request& req,
                                  Response& resp,
                                  int sess_id) {
  return InternalLocalPredict(req, resp, sess_id);
}

Status ModelSession::InternalLocalPredict(Request& req,
                                          Response& resp,
                                          int sess_id) {
  if (!is_local_) {
    return Status(error::Code::INTERNAL,
        "Remote sparse storage, please use Predict.");
  }
  ++counter_;
  Status status;
  tensorflow::RunOptions run_options;
  tensorflow::RunMetadata run_metadata;
  run_metadata.set_graph_signature(graph_hash_value_);
  if (Tracer::GetTracer()->NeedTracing()) {
    run_options.set_trace_level(tensorflow::RunOptions::FULL_TRACE);
    // TODO: which session selected to run on, add some policy here
    status = session_group_->Run(run_options, req.inputs,
        req.output_tensor_names, {}, &resp.outputs,
        &run_metadata, sess_id);
    Tracer::GetTracer()->GenTimeline(run_metadata); 
  } else {
    status = session_group_->Run(run_options, req.inputs,
        req.output_tensor_names, {}, &resp.outputs,
        &run_metadata, sess_id);
  }
  --counter_;
  return status;
}

Status ModelSession::Warmup(Request& req, Response& resp, bool local) {
  int N = session_group_->GetSessionNum();
  for (int i = 0; i < N; ++i) {
    Status s;
    if (local) {
      s = LocalPredict(req, resp, i);
    } else {
      s = Predict(req, resp, i);
    }
    if (!s.ok()) return s;
  }

  return Status::OK();
}

Status ModelSessionMgr::Predict(Request& req, Response& resp) {
  return serving_model_session_->Predict(req, resp);
}

Status ModelSessionMgr::LocalPredict(Request& req, Response& resp) {
  return serving_model_session_->LocalPredict(req, resp);
}

Status ModelSessionMgr::Warmup(Request& req, Response& resp, bool local) {
  return serving_model_session_->Warmup(req, resp, local);
}

Status ModelSessionMgr::CreateModelSession(
    const Version& version, const char* ckpt_name,
    IFeatureStoreMgr* sparse_storage, bool is_incr_ckpt,
    bool is_initialize, ModelConfig* config,
    const std::string& graph_hash_value) {
  ModelSession* new_model_session = nullptr;
  TF_RETURN_IF_ERROR(
      CreateModelSession(version, ckpt_name, sparse_storage,
                         is_incr_ckpt, is_initialize, config,
                         &new_model_session, graph_hash_value));
  ResetServingSession(new_model_session);
  return Status::OK();
}

Status ModelSessionMgr::CreateModelSession(
    const Version& version, const char* ckpt_name,
    IFeatureStoreMgr* sparse_storage, bool is_incr_ckpt,
    bool is_initialize, ModelConfig* config,
    ModelSession** new_model_session,
    const std::string& graph_hash_value) {
  SessionGroup* session_group = nullptr;
  std::vector<Session*> sessions;
  TF_RETURN_IF_ERROR(CreateSessionGroup(&session_group, config));
  sessions = session_group->GetLeaderSessions();

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

  for (auto session : sessions) {
    // only one instance can restore sparse variable
    status = RunRestoreOps(ckpt_name,
          real_version.full_ckpt_version,
          real_version.savedmodel_dir.c_str(),
          session, sparse_storage, is_incr_ckpt,
          locked, latest_version);
  }

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
  // ResetServingSession(session, real_version, sparse_storage);
  *new_model_session = new ModelSession(
      session_group, config->select_session_policy,
      version, sparse_storage, graph_hash_value);

  return Status::OK();
}

Status ModelSessionMgr::CreateModelSession(
    const Version& version,
    const char* saved_model_path,
    ModelConfig* config,
    const std::string& graph_hash_value) {
  SessionGroup* session_group = nullptr;
  std::vector<Session*> sessions;
  TF_RETURN_IF_ERROR(CreateSessionGroup(&session_group, config));
  sessions = session_group->GetLeaderSessions();
  TF_RETURN_IF_ERROR(util::ValidateSavedTensors(meta_graph_def_.graph_def()));

  LOG(INFO) << "CreateModelSession get leader sessions count: " << sessions.size();
  for (auto session : sessions) {
    TF_RETURN_IF_ERROR(
        util::RunRestore(*run_options_, saved_model_path,
            meta_graph_def_.saver_def().restore_op_name(),
            meta_graph_def_.saver_def().filename_tensor_name(),
            asset_file_defs_, session));
  }

  string init_op_name;
  TF_RETURN_IF_ERROR(
      util::GetInitOp(saved_model_path, meta_graph_def_, &init_op_name));
  for (auto session : sessions) {
    TF_RETURN_IF_ERROR(util::RunInitOp(*run_options_, saved_model_path,
                                       meta_graph_def_, asset_file_defs_,
                                       session, init_op_name));
  }

  auto new_model_session = new ModelSession(
      session_group, config->select_session_policy,
      version, graph_hash_value);
  ResetServingSession(new_model_session);

  return Status::OK();
}

Status ModelSessionMgr::CreateModelSession(
    const Version& version, const char* full_ckpt_name,
    const char* incr_ckpt_name, bool is_incr_ckpt,
    bool is_initialize, ModelConfig* config,
    const std::string& graph_hash_value) {
  ModelSession* new_model_session = nullptr;
  TF_RETURN_IF_ERROR(
      CreateModelSession(version, full_ckpt_name, incr_ckpt_name,
                         is_incr_ckpt, is_initialize, config,
                         &new_model_session, graph_hash_value));
  if (!is_incr_ckpt) {
    ResetServingSession(new_model_session);
  }

  return Status::OK();
}
 
Status ModelSessionMgr::CreateModelSession(
    const Version& version, const char* full_ckpt_name,
    const char* incr_ckpt_name, bool is_incr_ckpt,
    bool is_initialize, ModelConfig* config,
    ModelSession** new_model_session,
    const std::string& graph_hash_value) {
  std::string restore_op_name =
      meta_graph_def_.saver_def().restore_op_name();
  std::string filename_tensor_name =
      meta_graph_def_.saver_def().filename_tensor_name();
  std::string incr_filename_tensor_name =
      meta_graph_def_.incr_saver_def().filename_tensor_name();
  SessionGroup* session_group = nullptr;
  std::vector<Session*> sessions;
  if (is_incr_ckpt) {
    // Use serving session to update delta model
    sessions = serving_model_session_->GetLeaderSessions();
    restore_op_name =
        meta_graph_def_.incr_saver_def().restore_op_name();
  } else {
    TF_RETURN_IF_ERROR(CreateSessionGroup(&session_group, config));
    sessions = session_group->GetLeaderSessions();
  }

  thread::ThreadPoolOptions thread_opt = thread::ThreadPoolOptions();
  if (!is_initialize && config->model_update_intra_threads > 0) {
    thread_opt.intra_op_threadpool =
        GetModelUpdateIntraThreadPool(config->model_update_intra_threads);
  }

  if (!is_initialize && config->model_update_inter_threads > 0) {
    thread_opt.inter_op_threadpool =
        GetModelUpdateInterThreadPool(config->model_update_inter_threads);
  }

  LOG(INFO) << "CreateModelSession get leader sessions count: " << sessions.size();

  if (is_incr_ckpt) {
    for (auto session : sessions) {
      bool init_create_handler = false;
      Session::CallableHandle* incr_restore_handler = nullptr;
      Session::CallableHandle* main_op_handler = nullptr;
      incr_restore_handler = serving_model_session_->GetIncrRestoreHandler(session);
      main_op_handler = serving_model_session_->GetMainOpHandler(session);
      if (!incr_restore_handler || !main_op_handler) {
        init_create_handler = true;
      }

      TF_RETURN_IF_ERROR(util::RunIncrRestoreCheckpoint(
          *run_options_, full_ckpt_name,
          incr_ckpt_name, version.savedmodel_dir.c_str(),
          restore_op_name, filename_tensor_name,
          incr_filename_tensor_name, asset_file_defs_, session,
          thread_opt, &incr_restore_handler));

      if (util::HasMainOp(meta_graph_def_)) {
        TF_RETURN_IF_ERROR(util::RunIncrMainOp(
            *run_options_, version.savedmodel_dir.c_str(),
            meta_graph_def_, asset_file_defs_,
            session, kSavedModelMainOpKey,
            thread_opt, &main_op_handler));
      } else {
        TF_RETURN_IF_ERROR(util::RunIncrMainOp(
            *run_options_, version.savedmodel_dir.c_str(),
            meta_graph_def_, asset_file_defs_, session,
            kSavedModelLegacyInitOpKey, thread_opt, &main_op_handler));
      }

      if (init_create_handler) {
        serving_model_session_->SetIncrRestoreHandler(
            session, incr_restore_handler);
        serving_model_session_->SetMainOpHandler(
            session, main_op_handler);
      }
    }
  } else {
    for (auto session : sessions) {
      TF_RETURN_IF_ERROR(util::RunRestoreCheckpoint(
          *run_options_, full_ckpt_name,
          version.savedmodel_dir.c_str(),
          restore_op_name, filename_tensor_name,
          incr_filename_tensor_name, asset_file_defs_, session));

      if (util::HasMainOp(meta_graph_def_)) {
        TF_RETURN_IF_ERROR(util::RunMainOp(*run_options_,
            version.savedmodel_dir.c_str(),
            meta_graph_def_, asset_file_defs_,
            session, kSavedModelMainOpKey));
      } else {
        TF_RETURN_IF_ERROR(util::RunMainOp(
            *run_options_, version.savedmodel_dir.c_str(),
            meta_graph_def_, asset_file_defs_, session,
            kSavedModelLegacyInitOpKey));
      }
    }
  }

  if (!is_incr_ckpt) {
    *new_model_session = new ModelSession(
      session_group, config->select_session_policy,
      version, graph_hash_value);
  } else {
    serving_model_session_->UpdateVersion(version);
  }

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

void ModelSessionMgr::ResetServingSession(ModelSession* model_session) {
  auto tmp = serving_model_session_;
  serving_model_session_ = model_session;

  if (tmp == nullptr) return;

  if (tmp->counter_ > 0) {
    // TODO: free it in active object.
    mutex_lock lock(mu_);
    sessions_.emplace_back(tmp);
  } else {
    delete tmp;
  }
}

Status ModelSessionMgr::GetServingModelInfo(
    tensorflow::processor::ServingModelInfo& model_info) {
  model_info.model_path =
      serving_model_session_->GetVersion().full_ckpt_name;
  return Status::OK();
}

} // processor
} // tensorflow
