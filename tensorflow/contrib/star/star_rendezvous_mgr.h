#ifndef TENSORFLOW_CONTRIB_STAR_STAR_RENDEZVOUS_MGR_H_
#define TENSORFLOW_CONTRIB_STAR_STAR_RENDEZVOUS_MGR_H_

#include "tensorflow/core/distributed_runtime/base_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/platform/macros.h"


namespace tensorflow {

class DeviceMgr;

class StarRendezvousMgr : public BaseRendezvousMgr {
 public:
  explicit StarRendezvousMgr(const WorkerEnv* env);

 protected:
  BaseRemoteRendezvous* Create(int64 step_id, const WorkerEnv* worker_env);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(StarRendezvousMgr);
};

}  // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_STAR_RENDEZVOUS_MGR_H_
