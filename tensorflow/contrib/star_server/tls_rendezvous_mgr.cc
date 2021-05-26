#include "tensorflow/contrib/star_server/tls_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"

namespace tensorflow {

TLSRendezvousMgr::TLSRendezvousMgr(const WorkerEnv* env) :
    StarRendezvousMgr(env), worker_env_(env) {
  pthread_key_create(&key_, nullptr);
}

TLSRendezvousMgr::~TLSRendezvousMgr() {
  pthread_key_delete(key_);
};

RemoteRendezvous* TLSRendezvousMgr::Find(int64 step_id) {
  return GetImpl()->Find(step_id);
}

void TLSRendezvousMgr::RecvLocalAsync(int64 step_id,
                                      const Rendezvous::ParsedKey& parsed,
                                      Rendezvous::DoneCallback done) {
  GetImpl()->RecvLocalAsync(step_id, parsed, done);
}

Status TLSRendezvousMgr::RecvLocal(int64 step_id,
                                          const Rendezvous::ParsedKey& parsed,
                                          Tensor* val,
                                          bool* is_dead) {
  return GetImpl()->RecvLocal(step_id, parsed, val, is_dead);
}

void TLSRendezvousMgr::FuseRecvLocalAsync(
    int64 step_id, const std::vector<Rendezvous::ParsedKey>& parsed_keys,
    Rendezvous::FuseDoneCallback done) {
  GetImpl()->FuseRecvLocalAsync(step_id, parsed_keys, done);
}

void TLSRendezvousMgr::Cleanup(int64 step_id) {
  GetImpl()->Cleanup(step_id);
}

void TLSRendezvousMgr::CleanupAll() {
  GetImpl()->CleanupAll();
}

RemoteRendezvous* TLSRendezvousMgr::FindInterStepRendezvous() {
  return GetImpl()->FindInterStepRendezvous();
}

RendezvousMgrInterface* TLSRendezvousMgr::GetImpl() {
  RendezvousMgrInterface* mgr = static_cast<RendezvousMgrInterface*>
    (pthread_getspecific(key_));
  if (mgr == nullptr) {
    mgr = new StarRendezvousMgr(worker_env_);
    pthread_setspecific(key_, mgr);
  }
  return mgr;
}

} // namespace tensorflow
