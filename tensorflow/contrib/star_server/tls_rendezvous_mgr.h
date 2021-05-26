#ifndef TENSORFLOW_CONTRIB_STAR_SERVER_TLS_RENDEZVOUS_H_
#define TENSORFLOW_CONTRIB_STAR_SERVER_TLS_RENDEZVOUS_H_

#include "tensorflow/contrib/star/star_rendezvous_mgr.h"

namespace tensorflow {

class WorkerEnv;
class TLSRendezvousMgr : public StarRendezvousMgr {
 public:
  explicit TLSRendezvousMgr(const WorkerEnv* env);
  virtual ~TLSRendezvousMgr();

  virtual RemoteRendezvous* Find(int64 step_id);
  virtual void RecvLocalAsync(int64 step_id,
                              const Rendezvous::ParsedKey& parsed,
                              Rendezvous::DoneCallback done);
  virtual Status RecvLocal(int64 step_id, const Rendezvous::ParsedKey& parsed,
                           Tensor* val, bool* is_dead);
  virtual void FuseRecvLocalAsync(
      int64 step_id, const std::vector<Rendezvous::ParsedKey>& parsed_keys,
      Rendezvous::FuseDoneCallback done);
  virtual void Cleanup(int64 step_id);
  virtual void CleanupAll();
  virtual RemoteRendezvous* FindInterStepRendezvous();

 private:
  RendezvousMgrInterface* GetImpl();

 private:
  const WorkerEnv* worker_env_;
  pthread_key_t key_;
};

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_SERVER_TLS_RENDEZVOUS_H_
