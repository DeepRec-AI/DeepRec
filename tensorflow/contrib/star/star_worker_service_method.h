#ifndef TENSORFLOW_CONTRIB_PAI_STAR_STAR_WORKER_SERVICE_METHOD_H_
#define TENSORFLOW_CONTRIB_PAI_STAR_STAR_WORKER_SERVICE_METHOD_H_

namespace tensorflow {

enum StarWorkerServiceMethod {
  kGetStatus = 0,
  kGetTopologyStatus,
  kCreateWorkerSession,
  kDeleteWorkerSession,
  kRegisterGraph,
  kDeregisterGraph,
  kRunGraph,
  kStarRunGraph,
  kCleanupGraph,
  kCleanupAll,
  kRecvTensor,
  kFuseRecvTensor,
  kLogging,
  kTracing,
  kRecvBuf,
  kCompleteGroup,
  kCompleteInstance,
  kGetStepSequence,
  kInvalid = 100,
};

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_PAI_STAR_STAR_WORKER_SERVICE_METHOD_H_
