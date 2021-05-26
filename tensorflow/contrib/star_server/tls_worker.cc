#include "tensorflow/contrib/star_server/tls_worker.h"

namespace tensorflow {

TLSWorker::TLSWorker(WorkerEnv* worker_env) :
  StarWorker(worker_env), worker_env_(worker_env) {
  pthread_key_create(&key_, nullptr);
};

TLSWorker::~TLSWorker() {
  pthread_key_delete(key_);
};

void TLSWorker::GetStatusAsync(const GetStatusRequest* request,
                               GetStatusResponse* response,
                               StatusCallback done) {
  GetImpl()->GetStatusAsync(request, response, done);
}

void TLSWorker::GetStatusAsyncWithOptions(const GetStatusRequest* request,
                                          GetStatusResponse* response,
                                          StatusCallback done,
                                          CallOptions* call_opts) {
  GetImpl()->GetStatusAsyncWithOptions(request, response, done, call_opts);
}

void TLSWorker::CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request,
                                         CreateWorkerSessionResponse* response,
                                         StatusCallback done) {
  GetImpl()->CreateWorkerSessionAsync(request, response, done);
}

void TLSWorker::RegisterGraphAsync(const RegisterGraphRequest* request,
                                   RegisterGraphResponse* response,
                                   StatusCallback done) {
  GetImpl()->RegisterGraphAsync(request, response, done);
}

void TLSWorker::DeregisterGraphAsync(const DeregisterGraphRequest* request,
                                     DeregisterGraphResponse* response,
                                     StatusCallback done) {
  GetImpl()->DeregisterGraphAsync(request, response, done);
}

void TLSWorker::RunGraphAsync(CallOptions* opts, RunGraphRequestWrapper* request,
                              MutableRunGraphResponseWrapper* response,
                              StatusCallback done) {
  GetImpl()->RunGraphAsync(opts, request, response, done);
}

void TLSWorker::StarRunGraphAsync(StarRunGraphRequest* request,
                                  StarRunGraphResponse* response,
                                  StatusCallback done) {
  GetImpl()->StarRunGraphAsync(request, response, done);
}

MutableRunGraphRequestWrapper* TLSWorker::CreateRunGraphRequest() {
  return GetImpl()->CreateRunGraphRequest();
}

MutableRunGraphResponseWrapper* TLSWorker::CreateRunGraphResponse() {
  return GetImpl()->CreateRunGraphResponse();
}

void TLSWorker::CleanupGraphAsync(const CleanupGraphRequest* request,
                                  CleanupGraphResponse* response,
                                  StatusCallback done) {
  GetImpl()->CleanupGraphAsync(request, response, done);
}

void TLSWorker::CleanupAllAsync(const CleanupAllRequest* request,
                                CleanupAllResponse* response,
                                StatusCallback done) {
  GetImpl()->CleanupAllAsync(request, response, done);
}

void TLSWorker::RecvTensorAsync(CallOptions* opts,
                                const RecvTensorRequest* request,
                                StarTensorResponse* response,
                                StatusCallback done) {
  GetImpl()->RecvTensorAsync(opts, request, response, done);
}

void TLSWorker::FuseRecvTensorAsync(CallOptions* opts,
                                    const FuseRecvTensorRequest* request,
                                    StarFuseTensorResponse *response,
                                    StatusCallback done) {
  GetImpl()->FuseRecvTensorAsync(opts, request, response, done);
}

void TLSWorker::LoggingAsync(const LoggingRequest* request,
                             LoggingResponse* response,
                             StatusCallback done) {
  GetImpl()->LoggingAsync(request, response, done);
}

void TLSWorker::TracingAsync(const TracingRequest* request,
                             TracingResponse* response,
                             StatusCallback done) {
  GetImpl()->TracingAsync(request, response, done);
}

void TLSWorker::Cleanup(int64 step_id) {
  GetImpl()->Cleanup(step_id);
}

StarWorker* TLSWorker::GetImpl() {
  StarWorker* worker = static_cast<StarWorker*>(pthread_getspecific(key_));
  if (worker == nullptr) {
    worker = new StarWorker(worker_env_);
    pthread_setspecific(key_, worker);
  }
  return worker;
}

StarWorker* NewTLSWorker(WorkerEnv* worker_env) {
  return new TLSWorker(worker_env);
}

} // namespace tensorflow
