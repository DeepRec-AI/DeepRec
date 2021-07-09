#include <map>
#include <stdexcept>

#include "grpc++/grpc++.h"
#include "grpc++/security/credentials.h"
#include "grpc++/server_builder.h"
#include "grpc/support/alloc.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/contrib/star_server/star_server_lib.h"
#include "tensorflow/contrib/star_server/tls_rendezvous_mgr.h"
#include "tensorflow/contrib/star_server/tls_session_mgr.h"
#include "tensorflow/contrib/star_server/tls_worker.h"
#include "tensorflow/contrib/star_server/tls_worker_service.h"

namespace tensorflow {

StarServer::StarServer(const ServerDef& server_def, Env* env)
    : SeastarServer(server_def, env) {
}

Status StarServer::Create(const ServerDef& server_def, Env* env,
    std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<StarServer> ret(
      new StarServer(
          server_def, env == nullptr ? Env::Default() : env));

  TF_RETURN_IF_ERROR(ret->Init());
  *out_server = std::move(ret);
  return Status::OK();
}

RendezvousMgrInterface* StarServer::CreateRendezvousMgr(WorkerEnv* env) {
  return new TLSRendezvousMgr(env);
}

StarWorker* StarServer::CreateWorker(WorkerEnv* env) {
  env->lockless = true;
  return NewTLSWorker(env);
}

SessionMgr* StarServer::CreateSessionMgr(WorkerEnv* env,
    const string& default_worker_name,
    WorkerCacheInterface* default_worker_cache,
    std::function<Status(const ServerDef&, WorkerCacheInterface**)> worker_cache_factory) {
  return new TLSSessionMgr(env, default_worker_name,
      default_worker_cache,
      worker_cache_factory);
}

StarWorkerService* StarServer::CreateWorkerService(StarWorker* worker) {
  return NewTLSWorkerService(worker);
}

bool StarServer::GetRunGraphModeFlag(const ConfigProto&) {
  return true;
}

namespace {
class StarServerFactory : public ServerFactory {
public:
  bool AcceptsOptions(const ServerDef& server_def) override {
    return (server_def.protocol() == "star_server" ||
            server_def.protocol() == "star_server_v2") &&
           (server_def.job_name() == "ps");
  }

  Status NewServer(const ServerDef& server_def,
      std::unique_ptr<ServerInterface>* out_server) override {
    return StarServer::Create(server_def, Env::Default(), out_server);
  }
};

class StarServerRegistrar {
public:
  StarServerRegistrar() {
    gpr_allocation_functions alloc_fns;
    memset(&alloc_fns, 0, sizeof(alloc_fns));
    alloc_fns.malloc_fn = port::Malloc;
    alloc_fns.realloc_fn = port::Realloc;
    alloc_fns.free_fn = port::Free;
    gpr_set_allocation_functions(alloc_fns);
    ServerFactory::Register("STAR_SERVER", new StarServerFactory());
}
};

static StarServerRegistrar registrar;
} // namespace

} // namespace tensorflow
