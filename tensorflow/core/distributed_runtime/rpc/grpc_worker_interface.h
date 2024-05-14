#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WOKRER_INTERFACE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WOKRER_INTERFACE_H_

namespace tensorflow {

class CallOptions;
class FuseTensorResponse;
class FuseRecvTensorRequest;
class FlowControlRecvTensorRequest;
class TensorResponse;

class GrpcWorkerInterface {
 public:
  virtual void FuseRecvTensorAsync(CallOptions* call_opts,
                                   const FuseRecvTensorRequest* request,
                                   FuseTensorResponse* response,
                                   StatusCallback done) = 0;

  virtual void FlowControlRecvTensorAsync(CallOptions* call_opts,
                 const FlowControlRecvTensorRequest* request,
                 TensorResponse* response, StatusCallback done) = 0;
};

} // namespace tensorflow

#endif //TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WOKRER_INTERFACE_H_
