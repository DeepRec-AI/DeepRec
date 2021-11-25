#include <utility>

#include "tensorflow/contrib/star/seastar/seastar_remote_worker.h"
#include "tensorflow/contrib/star/seastar/seastar_client_tag.h"
#include "tensorflow/contrib/star/star_message.h"
#include "tensorflow/contrib/star/star_tensor_coding.h"
#include "tensorflow/contrib/star/star_worker_interface.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

class SeastarRemoteWorker : public WorkerInterface, public StarWorkerInterface {
 public:
  explicit SeastarRemoteWorker(seastar::channel* chan, WorkerCacheLogger* logger, WorkerEnv* env)
      : seastar_channel_(chan),
        logger_(logger),
        env_(env) {
  }

  ~SeastarRemoteWorker() override {}

  void GetStatusAsync(const GetStatusRequest* request,
                      GetStatusResponse* response,
                      StatusCallback done) override {
    GetStatusAsyncWithOptions(request, response, done, nullptr);
  }

  void GetStatusAsyncWithOptions(const GetStatusRequest* request,
                                 GetStatusResponse* response,
                                 StatusCallback done,
                                 CallOptions* call_opts) override {
    env_->compute_pool->Schedule([this, request, response, call_opts, done]() {
      IssueRequest(request, response, StarWorkerServiceMethod::kGetStatus, std::move(done), call_opts);
    });
  }

  void CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request,
                                CreateWorkerSessionResponse* response,
                                StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request, response, StarWorkerServiceMethod::kCreateWorkerSession, std::move(done));
    });
  }

  void DeleteWorkerSessionAsync(CallOptions* call_opts,
                                const DeleteWorkerSessionRequest* request,
                                DeleteWorkerSessionResponse* response,
                                StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done, call_opts] {
	IssueRequest(request,
		     response,
		     StarWorkerServiceMethod::kDeleteWorkerSession,
		     std::move(done),
		     call_opts);
      });
  }

  void RegisterGraphAsync(const RegisterGraphRequest* request,
                          RegisterGraphResponse* response,
                          StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request, response, StarWorkerServiceMethod::kRegisterGraph, std::move(done));
    });
  }

  void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                            DeregisterGraphResponse* response,
                            StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request, response, StarWorkerServiceMethod::kDeregisterGraph, std::move(done));
    });
  }

  void RunGraphAsync(CallOptions* call_opts, const RunGraphRequest* request,
                     RunGraphResponse* response, StatusCallback done) override {
    TRACEPRINTF("Seastar RunGraph: %lld", request->step_id());
    env_->compute_pool->Schedule([this, request, response, call_opts, done]() {
      IssueRequest(request, response, StarWorkerServiceMethod::kRunGraph, std::move(done), call_opts);
    });
  }

  void RunGraphAsync(CallOptions* call_opts, RunGraphRequestWrapper* request,
                     MutableRunGraphResponseWrapper* response,
                     StatusCallback done) override {
    TRACEPRINTF("wrapped Seastar RunGraph: %lld", request->step_id());
    env_->compute_pool->Schedule([this, request, response, call_opts, done]() {
      IssueRequest(&request->ToProto(), get_proto_from_wrapper(response),
                   StarWorkerServiceMethod::kRunGraph, std::move(done), call_opts);
    });
  }

  void StarRunGraphAsync(StarRunGraphRequest* request,
                         StarRunGraphResponse* response,
                         StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request, response, StarWorkerServiceMethod::kStarRunGraph, std::move(done));
    });
  }

  void CleanupGraphAsync(const CleanupGraphRequest* request,
                         CleanupGraphResponse* response,
                         StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request, response, StarWorkerServiceMethod::kCleanupGraph, std::move(done));
    });
  }

  void CleanupAllAsync(const CleanupAllRequest* request,
                       CleanupAllResponse* response,
                       StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request, response, StarWorkerServiceMethod::kCleanupAll, std::move(done));
    });
  }

  void RecvTensorAsync(CallOptions* call_opts, const RecvTensorRequest* request,
                       TensorResponse* response, StatusCallback done) override {
    done(errors::Unimplemented("SeastarWorker::RecvTensorAsync()"));
  }

  void RecvTensorAsync(CallOptions* call_opts, const RecvTensorRequest* request,
                       StarTensorResponse* response, StatusCallback done) override {
    VLOG(1) << "RecvTensorAsync req: " << request->DebugString();
    // Don't propagate dma_ok over gRPC.
    RecvTensorRequest* req_copy = nullptr;
    if (request->dma_ok()) {
      req_copy = new RecvTensorRequest;
      *req_copy = *request;
      req_copy->set_dma_ok(false);
    }
    StatusCallback wrapper_done;
    const StatusCallback* cb_to_use;
    if (req_copy == nullptr) {
      cb_to_use = &done;  // No additional work to do, so just use done directly
    } else {
      wrapper_done = [req_copy, done](Status s) {
        delete req_copy;
        done(s);
      };
      cb_to_use = &wrapper_done;
    }

    IssueRequest(req_copy ? req_copy : request, response,
                 StarWorkerServiceMethod::kRecvTensor,
                 std::move(*cb_to_use), call_opts);
  }

  void FuseRecvTensorAsync(CallOptions* call_opts,
                           const FuseRecvTensorRequest* request,
                           StarFuseTensorResponse* response,
                           StatusCallback done) override {
    VLOG(1) << "FuseRecvTensorAsync req: " << request->DebugString();
    // Don't propagate dma_ok over gRPC.
    FuseRecvTensorRequest* req_copy = nullptr;
    if (request->dma_ok()) {
      req_copy = new FuseRecvTensorRequest;
      *req_copy = *request;
      req_copy->set_dma_ok(false);
    }
    StatusCallback wrapper_done;
    const StatusCallback* cb_to_use;
    if (req_copy == nullptr) {
      cb_to_use = &done;  // No additional work to do, so just use done directly
    } else {
      wrapper_done = [req_copy, done](Status s) {
        delete req_copy;
        done(s);
      };
      cb_to_use = &wrapper_done;
    }

    IssueRequest(req_copy ? req_copy : request, response,
                 StarWorkerServiceMethod::kFuseRecvTensor,
                 std::move(*cb_to_use), call_opts);
  }

  void LoggingAsync(const LoggingRequest* request, LoggingResponse* response,
                    StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request, response, StarWorkerServiceMethod::kLogging, done);
    });
  }

  void TracingAsync(const TracingRequest* request, TracingResponse* response,
                    StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request, response, StarWorkerServiceMethod::kTracing, done);
    });
  }

  void RecvBufAsync(CallOptions* opts, const RecvBufRequest* request,
                    RecvBufResponse* response, StatusCallback done) override {
    done(errors::Unimplemented("SeastarRemoteWorker::RecvBufAsync()"));
  }

  void CompleteGroupAsync(CallOptions* opts,
                          const CompleteGroupRequest* request,
                          CompleteGroupResponse* response,
                          StatusCallback done) override {
    done(errors::Unimplemented("SeastarRemoteWorker::CompleteGroupAsync()"));
  }

  void CompleteInstanceAsync(CallOptions* ops,
                             const CompleteInstanceRequest* request,
                             CompleteInstanceResponse* response,
                             StatusCallback done) override {
    done(errors::Unimplemented("SeastarRemoteWorker::CompleteInstanceAsync()"));
  }

  void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                            GetStepSequenceResponse* response,
                            StatusCallback done) override {
    done(errors::Unimplemented("SeastarRemoteWorker::GetStepSequenceAsync()"));
  }

 private:
  void IssueRequest(const protobuf::Message* request,
                    protobuf::Message* response,
                    const StarWorkerServiceMethod method,
                    StatusCallback done,
                    CallOptions* call_opts = nullptr) {
      auto tag = new SeastarClientTag(method, env_);
      InitStarClientTag(const_cast<protobuf::Message*>(request),
                           response, std::move(done), tag, call_opts);
      tag->StartReq(seastar_channel_);
    }

  void IssueRequest(const protobuf::Message* request,
                    StarTensorResponse* response,
                    const StarWorkerServiceMethod method,
                    StatusCallback done,
                    CallOptions* call_opts = nullptr) {
    auto tag = new SeastarClientTag(method, env_, 1);
    InitStarClientTag(const_cast<protobuf::Message*>(request),
                      response, std::move(done), tag, call_opts);
    tag->StartReq(seastar_channel_);
  }

  void IssueRequest(const protobuf::Message* request,
                    StarFuseTensorResponse* response,
                    const StarWorkerServiceMethod method,
                    StatusCallback done,
                    CallOptions* call_opts = nullptr) {
    auto tag = new SeastarClientTag(method, env_, response->GetFuseCount());
    InitStarClientTag(const_cast<protobuf::Message*>(request),
                      response, std::move(done), tag, call_opts);
    tag->StartReq(seastar_channel_);
  }

  void IssueRequest(StarRunGraphRequest* request,
                    StarRunGraphResponse* response,
                    const StarWorkerServiceMethod method,
                    StatusCallback done) {
    auto tag = new SeastarClientTag(method, env_,
                                    request->fetch_names_.size(),
                                    request->feed_names_.size());
    InitStarClientTag(request, response, std::move(done), tag);
    tag->StartReq(seastar_channel_);
  }

private:
  seastar::channel* seastar_channel_;
  // Support for logging.
  WorkerCacheLogger* logger_;
  WorkerEnv* env_;

  TF_DISALLOW_COPY_AND_ASSIGN(SeastarRemoteWorker);
};

WorkerInterface* NewSeastarRemoteWorker(seastar::channel* seastar_channel,
                                        WorkerCacheLogger* logger,
                                        WorkerEnv* env) {
  return new SeastarRemoteWorker(seastar_channel, logger, env);
}

} // namespace tensorflow
