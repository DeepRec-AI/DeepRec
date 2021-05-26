#ifndef TENSORFLOW_CONTRIB_STAR_SERVER_STAR_SERVER_LIB_H_
#define TENSORFLOW_CONTRIB_STAR_SERVER_STAR_SERVER_LIB_H_

#include "tensorflow/contrib/star/seastar/seastar_server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
class Env;
class RendezvousMgrInterface;
class SeastarWorker;
class ServerInterface;
class SessionMgr;
class WorkerCacheInterface;
class WorkerEnv;

class StarServer : public SeastarServer {
 public:
  StarServer(const ServerDef& server_def, Env* env);
  static Status Create(const ServerDef& server_def, Env* env,
                       std::unique_ptr<ServerInterface>* out_server);

 protected:
  RendezvousMgrInterface* CreateRendezvousMgr(WorkerEnv* env) override;
  StarWorker* CreateWorker(WorkerEnv* env) override;
  SessionMgr* CreateSessionMgr(WorkerEnv* env,
      const string& default_worker_name,
      WorkerCacheInterface* default_worker_cache,
      std::function<Status(const ServerDef&,
          WorkerCacheInterface**)> worker_cache_factory) override;
  StarWorkerService* CreateWorkerService(StarWorker* worker) override;
  bool GetRunGraphModeFlag(const ConfigProto&) override;
};

} //namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_SERVER_STAR_SERVER_LIB_H_
