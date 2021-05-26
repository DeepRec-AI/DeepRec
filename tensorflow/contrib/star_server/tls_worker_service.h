#ifndef TENSORFLOW_CONTRIB_STAR_SERVER_TLS_WORKER_SERVICE_H_
#define TENSORFLOW_CONTRIB_STAR_SERVER_TLS_WORKER_SERVICE_H_

#include "tensorflow/contrib/star/star_worker_service.h"

namespace tensorflow {

class TLSWorkerService : public StarWorkerService {
public:
  TLSWorkerService(StarWorker* worker);
  virtual ~TLSWorkerService() {}

private:
  virtual void Schedule(std::function<void()> f);
};

StarWorkerService* NewTLSWorkerService(StarWorker* worker);

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_SERVER_TLS_WORKER_SERVICE_H_
