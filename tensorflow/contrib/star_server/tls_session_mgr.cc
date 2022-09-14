#include "tensorflow/contrib/star_server/tls_session_mgr.h"

namespace tensorflow {

TLSSessionMgr::TLSSessionMgr(
      WorkerEnv* worker_env, const string& default_worker_name,
      WorkerCacheInterface* default_worker_cache,
      WorkerCacheFactory worker_cache_factory) :
        SessionMgr(worker_env, default_worker_name,
                   std::unique_ptr<WorkerCacheInterface>(default_worker_cache),
                   worker_cache_factory),
        worker_env_(worker_env),
        default_worker_name_(default_worker_name),
        default_worker_cache_(default_worker_cache),
        worker_cache_factory_(worker_cache_factory) {
    pthread_key_create(&key_, nullptr);
}

TLSSessionMgr::~TLSSessionMgr() {
  pthread_key_delete(key_);
}
  
Status TLSSessionMgr::CreateSession(const string& session,
                                    const ServerDef& server_def) {
  return GetImpl()->CreateSession(session, server_def, false);
}

Status TLSSessionMgr::DeleteSession(const string& session) {
  return GetImpl()->DeleteSession(session);
}

std::shared_ptr<WorkerSession> TLSSessionMgr::WorkerSessionForSession(
    const string& session) {
  std::shared_ptr<WorkerSession> out_session;
  Status s = GetImpl()->WorkerSessionForSession(session, &out_session);
  if (!s.ok()) {
    LOG(FATAL) << "[StarServer] Get worker session failed.";
  }
  return out_session;
}

std::shared_ptr<WorkerSession> TLSSessionMgr::LegacySession() {
  return GetImpl()->LegacySession();
}

SessionMgr* TLSSessionMgr::GetImpl() {
  SessionMgr* mgr = static_cast<SessionMgr*>(pthread_getspecific(key_));
  if (mgr == nullptr) {
    mgr = new SessionMgr(worker_env_, default_worker_name_,
                         std::unique_ptr<WorkerCacheInterface>(default_worker_cache_),
                         worker_cache_factory_);
    pthread_setspecific(key_, mgr);
  }
  return mgr;
} 

} // namespace tensorflow
