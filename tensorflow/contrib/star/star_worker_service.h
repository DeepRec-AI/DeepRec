#ifndef TENSORFLOW_CONTRIB_STAR_STAR_WORKER_SERVICE_H_
#define TENSORFLOW_CONTRIB_STAR_STAR_WORKER_SERVICE_H_

#include <map>
#include <memory>

#include "tensorflow/contrib/star/star_worker_interface.h"
#include "tensorflow/contrib/star/star_worker_service_method.h"
#include "tensorflow/core/distributed_runtime/worker.h"
#include "tensorflow/core/lib/core/status.h"


namespace tensorflow {

class StarServerTag;
class CallOptions;
class RecvTensorRequest;
class StarTensorResponse;
struct WorkerEnv;

class StarWorker : public Worker, public StarWorkerInterface {
 public:
  typedef std::function<void(const Status&)> StatusCallback;
  explicit StarWorker(WorkerEnv* worker_env);
  virtual ~StarWorker() {}

  // Specialized version of RecvTensor for star.
  void RecvTensorAsync(CallOptions* opts, const RecvTensorRequest* request,
                       StarTensorResponse *response, StatusCallback done);

  void FuseRecvTensorAsync(CallOptions* opts,
                           const FuseRecvTensorRequest* request,
                           StarFuseTensorResponse *response,
                           StatusCallback done);

  void StarRunGraphAsync(StarRunGraphRequest* request,
                         StarRunGraphResponse* response,
                         StatusCallback done);

  WorkerEnv* env();

  virtual void Cleanup(int64 step_id);

 private:
  mutex mu_cm_;
  CancellationManager* cancel_mgr_ GUARDED_BY(mu_cm_);
  mutex graph_count_mu_;
  std::unordered_map<int64, int> pending_graph_count_ GUARDED_BY(graph_count_mu_);
};

class StarWorkerService {
public:
  using HandleRequestFunction = void (StarWorkerService::*)(StarServerTag*);

  explicit StarWorkerService(StarWorker* worker);
  virtual ~StarWorkerService() {}

  HandleRequestFunction GetHandler(StarWorkerServiceMethod methodId);

  void RunGraphHandler(StarServerTag* tag);
  void StarRunGraphHandler(StarServerTag* tag);
  void GetStatusHandler(StarServerTag* tag);
  void GetTopologyStatusHandler(StarServerTag* tag);
  void CreateWorkerSessionHandler(StarServerTag* tag);
  void DeleteWorkerSessionHandler(StarServerTag* tag);
  void CleanupAllHandler(StarServerTag* tag);
  void RegisterGraphHandler(StarServerTag* tag);
  void DeregisterGraphHandler(StarServerTag* tag);
  void CleanupGraphHandler(StarServerTag* tag);
  void LoggingHandler(StarServerTag* tag);
  void TracingHandler(StarServerTag* tag);
  void RecvTensorHandlerRaw(StarServerTag* tag);
  void RecvBufHandler(StarServerTag* tag);
  void CompleteGroupHandler(StarServerTag* tag);
  void CompleteInstanceHandler(StarServerTag* tag);
  void GetStepSequenceHandler(StarServerTag* tag);
  void FuseRecvTensorHandlerRaw(StarServerTag* tag);
  StarWorker* GetWorker() const;

private:
  virtual void Schedule(std::function<void()> f);

private:
  std::map<StarWorkerServiceMethod, HandleRequestFunction> handler_map_;
  StarWorker* worker_;

  TF_DISALLOW_COPY_AND_ASSIGN(StarWorkerService);
};

std::unique_ptr<StarWorker> NewStarWorker(WorkerEnv* worker_env);

std::unique_ptr<StarWorkerService> NewStarWorkerService(StarWorker* worker);

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_STAR_WORKER_SERVICE_H_
