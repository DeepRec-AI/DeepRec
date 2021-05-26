#ifndef TENSORFLOW_CONTRIB_STAR_SERVER_TLS_WORKER_H_
#define TENSORFLOW_CONTRIB_STAR_SERVER_TLS_WORKER_H_

#include "tensorflow/contrib/star/star_worker_service.h"

namespace tensorflow {

class TLSWorker : public StarWorker {
 public:
  explicit TLSWorker(WorkerEnv* worker_env);
  virtual ~TLSWorker();

  void GetStatusAsync(const GetStatusRequest* request,
                      GetStatusResponse* response,
                      StatusCallback done) override;
  void GetStatusAsyncWithOptions(const GetStatusRequest* request,
                                 GetStatusResponse* response,
                                 StatusCallback done,
                                 CallOptions* call_opts) override;
  void CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request,
                                CreateWorkerSessionResponse* response,
                                StatusCallback done) override;
  void RegisterGraphAsync(const RegisterGraphRequest* request,
                          RegisterGraphResponse* response,
                          StatusCallback done) override;
  void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                            DeregisterGraphResponse* response,
                            StatusCallback done) override;
  void RunGraphAsync(CallOptions* opts, RunGraphRequestWrapper* request,
                     MutableRunGraphResponseWrapper* response,
                     StatusCallback done) override;
  void StarRunGraphAsync(StarRunGraphRequest* request,
                         StarRunGraphResponse* response,
                         StatusCallback done) override;
  MutableRunGraphRequestWrapper* CreateRunGraphRequest() override;
  MutableRunGraphResponseWrapper* CreateRunGraphResponse() override;
  void CleanupGraphAsync(const CleanupGraphRequest* request,
                         CleanupGraphResponse* response,
                         StatusCallback done) override;
  void CleanupAllAsync(const CleanupAllRequest* request,
                       CleanupAllResponse* response,
                       StatusCallback done) override;
  void RecvTensorAsync(CallOptions* opts, const RecvTensorRequest* request,
                       StarTensorResponse* response, StatusCallback done) override;
  void FuseRecvTensorAsync(CallOptions* opts,
                           const FuseRecvTensorRequest* request,
                           StarFuseTensorResponse *response,
                           StatusCallback done) override;
  void LoggingAsync(const LoggingRequest* request, LoggingResponse* response,
                    StatusCallback done) override;
  void TracingAsync(const TracingRequest* request, TracingResponse* response,
                    StatusCallback done) override;
  void Cleanup(int64 step_id) override;
 private:
  StarWorker* GetImpl();

 private:
  WorkerEnv* worker_env_;
  pthread_key_t key_;
};

StarWorker* NewTLSWorker(WorkerEnv* worker_env);

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_SERVER_TLS_WORKER_H_
