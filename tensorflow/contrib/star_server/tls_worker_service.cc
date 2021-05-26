#include "tensorflow/contrib/star_server/tls_worker_service.h"

namespace tensorflow {

TLSWorkerService::TLSWorkerService(StarWorker* worker) :
     StarWorkerService(worker) {
}

void TLSWorkerService::Schedule(std::function<void()> f) {
  f();
}

StarWorkerService* NewTLSWorkerService(StarWorker* worker) {
  return new TLSWorkerService(worker);
}

} // namespace tensorflow
