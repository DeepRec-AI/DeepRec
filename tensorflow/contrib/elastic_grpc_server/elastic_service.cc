/* Copyright 2023 The DeepRec Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

#include "tensorflow/contrib/elastic_grpc_server/elastic_service.h"

#include "tensorflow/contrib/elastic_grpc_server/elastic_grpc_server_lib.h"
#include "tensorflow/core/protobuf/elastic_training.grpc.pb.h"
#include "tensorflow/core/protobuf/elastic_training.pb.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"


#include <grpc/support/log.h>
#include <grpcpp/grpcpp.h>
#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include "grpcpp/server_builder.h"

using namespace deeprec;

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerCompletionQueue;
using grpc::ServerContext;

namespace tensorflow {

class GrpcElasticService : public AsyncServiceInterface {
 public:
  GrpcElasticService(ElasticGrpcServer* elastic_grpc_server,
                     ::grpc::ServerBuilder* builder): 
      elastic_grpc_server_(elastic_grpc_server), builder_(builder) {
    builder_->RegisterService(&elastic_service_);
    cq_ = builder_->AddCompletionQueue();
  }

  ~GrpcElasticService() override { }
  
  void Shutdown() override {
    cq_->Shutdown();
  }

  void HandleRPCsLoop() override {
    new CallData(&elastic_service_, elastic_grpc_server_, cq_.get());
    void* tag;
    bool ok;
    while (true) {
      // Block waiting to read the next event from the completion queue. The
      // event is uniquely identified by its tag, which in this case is the
      // memory address of a CallData instance.
      // The return value of Next should always be checked. This return value
      // tells us whether there is any kind of event or cq_ is shutting down.
      GPR_ASSERT(cq_->Next(&tag, &ok));
      GPR_ASSERT(ok);
      static_cast<CallData*>(tag)->Proceed();
    }
  }

 private:
  // Class encompasing the state and logic needed to serve a request.
  class CallData {
   public:
    // Take in the "service" instance (in this case representing an asynchronous
    // server) and the completion queue "cq" used for asynchronous communication
    // with the gRPC runtime.
    CallData(ElasticTrainingService::AsyncService* service, ElasticGrpcServer* elastic_grpc_server,
        ServerCompletionQueue* cq)
      : service_(service), elastic_grpc_server_(elastic_grpc_server),
        cq_(cq), responder_(&ctx_), status_(CREATE) {
      // Invoke the serving logic right away.
      Proceed();
    }

    void Proceed() {
      if (status_ == CREATE) {
        // Make this instance progress to the PROCESS state.
        status_ = PROCESS;

        // As part of the initial CREATE state, we *request* that the system
        // start processing SayHello requests. In this request, "this" acts are
        // the tag uniquely identifying the request (so that different CallData
        // instances can serve different requests concurrently), in this case
        // the memory address of this CallData instance.
        service_->RequestUpdateServerDef(&ctx_, &request_, &responder_,
                                         cq_, cq_, this);
      } else if (status_ == PROCESS) {
        // Spawn a new CallData instance to serve new clients while we process
        // the one for this CallData. The instance will deallocate itself as
        // part of its FINISH state.
        new CallData(service_, elastic_grpc_server_, cq_);

        // The actual processing.
        Status s = elastic_grpc_server_->Update(request_.cluster_def());
        if (s.ok()) {
          reply_.set_code(Code::OK);
        } else {
          reply_.set_code(Code::INTERNAL);
          reply_.set_msg(s.ToString());
          LOG(ERROR) << "error" << s.ToString();
        }

        // And we are done! Let the gRPC runtime know we've finished, using the
        // memory address of this instance as the uniquely identifying tag for
        // the event.
        status_ = FINISH;
        responder_.Finish(reply_, ::grpc::Status::OK, this);
      } else {
        GPR_ASSERT(status_ == FINISH);
        // Once in the FINISH state, deallocate ourselves (CallData).
        delete this;
      }
    }
   private:
    ElasticGrpcServer* elastic_grpc_server_;
    // The means of communication with the gRPC runtime for an asynchronous
    // server.
    ElasticTrainingService::AsyncService* service_;
    // The producer-consumer queue where for asynchronous server notifications.
    ServerCompletionQueue* cq_;
    // Context for the rpc, allowing to tweak aspects of it such as the use
    // of compression, authentication, as well as to send metadata back to the
    // client.
    ServerContext ctx_;

    // What we get from the client.
    UpdateServerDefRequest request_;
    // What we send back to the client.
    UpdateServerDefResponse reply_;

    // The means to get back to the client.
    ServerAsyncResponseWriter<UpdateServerDefResponse> responder_;

    // Let's implement a tiny state machine with the following states.
    enum CallStatus { CREATE, PROCESS, FINISH };
    CallStatus status_;  // The current serving state.
  };

  ElasticGrpcServer* elastic_grpc_server_;
  ::grpc::ServerBuilder* builder_;
  ElasticTrainingService::AsyncService elastic_service_;
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
};

AsyncServiceInterface* NewElasticGrpcService(
    ElasticGrpcServer* elastic_grpc_server, ::grpc::ServerBuilder* builder) {
  return reinterpret_cast<AsyncServiceInterface*>(new GrpcElasticService(elastic_grpc_server, builder));
}
}