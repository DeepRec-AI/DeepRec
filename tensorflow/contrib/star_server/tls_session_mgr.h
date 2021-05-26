#ifndef TENSORFLOW_CONTRIB_STAR_SERVER_TLS_SESSION_MGR_H_
#define TENSORFLOW_CONTRIB_STAR_SERVER_TLS_SESSION_MGR_H_

#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {

class TLSSessionMgr : public SessionMgr {
 public:
  explicit TLSSessionMgr(
      WorkerEnv* worker_env, const string& default_worker_name,
      WorkerCacheInterface* default_worker_cache,
      WorkerCacheFactory worker_cache_factory);
  virtual ~TLSSessionMgr();

  TLSSessionMgr(const TLSSessionMgr&) = delete;
  TLSSessionMgr& operator=(const TLSSessionMgr&) = delete;
  
  virtual Status CreateSession(const string& session,
                               const ServerDef& server_def);
  virtual Status DeleteSession(const string& session);
  virtual std::shared_ptr<WorkerSession> WorkerSessionForSession(const string& session);
  virtual std::shared_ptr<WorkerSession> LegacySession();

 private:
  SessionMgr* GetImpl();

 private:
  WorkerEnv* worker_env_;
  std::string default_worker_name_;
  WorkerCacheInterface* default_worker_cache_;
  WorkerCacheFactory worker_cache_factory_;

  pthread_key_t key_;
};

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_SERVER_TLS_SESSION_MGR_H_
