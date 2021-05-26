#ifndef TENSORFLOW_CONTRIB_STAR_STAR_WOKRER_INTERFACE_H_
#define TENSORFLOW_CONTRIB_STAR_STAR_WOKRER_INTERFACE_H_

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class CallOptions;
class RecvTensorRequest;
class StarTensorResponse;
class FuseRecvTensorRequest;
class StarFuseTensorResponse;
class StarRunGraphRequest;
class StarRunGraphResponse;

class StarWorkerInterface {
public:
  virtual void RecvTensorAsync(CallOptions* call_opts,
                               const RecvTensorRequest* request,
                               StarTensorResponse* response,
                               StatusCallback done) = 0;

  virtual void FuseRecvTensorAsync(CallOptions* call_opts,
                                   const FuseRecvTensorRequest* request,
                                   StarFuseTensorResponse* response,
                                   StatusCallback done) = 0;

  virtual void StarRunGraphAsync(StarRunGraphRequest* request,
                                 StarRunGraphResponse* response,
                                 StatusCallback done) = 0;
};

} // namespace tensorflow

#endif // TENSORFLOW_CONTRIB_STAR_STAR_WOKRER_INTERFACE_H_
