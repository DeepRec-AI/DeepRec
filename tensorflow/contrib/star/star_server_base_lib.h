#ifndef TENSORFLOW_CONTRIB_STAR_STAR_SERVER_BASE_LIB_H_
#define TENSORFLOW_CONTRIB_STAR_STAR_SERVER_BASE_LIB_H_

#include "grpc++/grpc++.h"
#include "grpc++/security/credentials.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/env.h"


namespace tensorflow {

class Master;
class StarWorker;
class StarWorkerService;
class StarChannelSpec;
class StarPortMgr;

class StarServerBase : public ServerInterface {
protected:
  StarServerBase(const ServerDef& server_def, Env* env);

public:
  virtual ~StarServerBase();

  Status Start() override;
  Status Stop() override;
  Status Join() override;
  const string target() const;

  Status Init();

protected:
  Status ParseChannelSpec(const WorkerCacheFactoryOptions& options,
                          StarChannelSpec* channel_spec);

  size_t ParseServers(const WorkerCacheFactoryOptions& options);

  virtual Status StarWorkerCacheFactory(
      const WorkerCacheFactoryOptions& options,
      WorkerCacheInterface** worker_cache) = 0;
  virtual void CreateEngine(
      size_t server_number, const string& job_name) = 0;

  virtual std::shared_ptr<::grpc::ServerCredentials> GetServerCredentials(
          const ServerDef& server_def) const;
  virtual std::unique_ptr<Master> CreateMaster(MasterEnv* master_env);

  int bound_port() const { return bound_port_; }
  WorkerEnv* worker_env() { return &worker_env_; }
  const ServerDef& server_def() const { return server_def_; }

  virtual RendezvousMgrInterface* CreateRendezvousMgr(WorkerEnv* env);
  virtual StarWorker* CreateWorker(WorkerEnv* env);
  virtual SessionMgr* CreateSessionMgr(WorkerEnv* env,
      const string& default_worker_name,
      WorkerCacheInterface* default_worker_cache,
      std::function<Status(const ServerDef&,
          WorkerCacheInterface**)> worker_cache_factory);
  virtual StarWorkerService* CreateWorkerService(StarWorker* worker);
  virtual bool GetRunGraphModeFlag(const ConfigProto& config);
  virtual bool GetRunGraphModeFlagLite(const ConfigProto& config);

protected:
  const ServerDef server_def_;
  Env* env_;

  int bound_port_ = 0;
  int star_bound_port_ = 0;

  mutex mu_;
  enum State { NEW, STARTED, STOPPED };
  State state_ GUARDED_BY(mu_);

  // Master part, still using grpc.
  MasterEnv master_env_;
  std::unique_ptr<Master> master_impl_;
  AsyncServiceInterface* master_service_ = nullptr;
  std::unique_ptr<Thread> master_thread_ GUARDED_BY(mu_);
  std::unique_ptr<::grpc::Server> server_ GUARDED_BY(mu_);

  WorkerEnv worker_env_;
  StarWorker* worker_impl_;
  StarWorkerService* worker_service_ = nullptr;
  StarPortMgr* star_port_mgr_ = nullptr;
};

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_STAR_SERVER_BASE_LIB_H_
