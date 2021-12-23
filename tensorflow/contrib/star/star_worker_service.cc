#include "tensorflow/contrib/star/star_worker_service.h"
#include "tensorflow/contrib/star/star_server_tag.h"
#include "tensorflow/contrib/star/star_tensor_coding.h"
#include "tensorflow/contrib/verbs/verbs_util.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/memory_planner.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#endif // GOOGLE_CUDA
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/public/session_options.h"


namespace tensorflow {
namespace {

template<class RequestMessage, class ResponseMessage>
class StarCall {
public:
  RequestMessage req_;
  ResponseMessage resp_;
};

} // namespace

using HandleRequestFunction = void (StarWorkerService::*)(StarServerTag*);

StarWorkerService::StarWorkerService(StarWorker* worker)
  : worker_(worker) {
  handler_map_[StarWorkerServiceMethod::kRunGraph] = &StarWorkerService::RunGraphHandler;
  handler_map_[StarWorkerServiceMethod::kStarRunGraph] = &StarWorkerService::StarRunGraphHandler;
  handler_map_[StarWorkerServiceMethod::kRecvTensor] = &StarWorkerService::RecvTensorHandlerRaw;
  handler_map_[StarWorkerServiceMethod::kFuseRecvTensor] = &StarWorkerService::FuseRecvTensorHandlerRaw;
  handler_map_[StarWorkerServiceMethod::kGetStatus] = &StarWorkerService::GetStatusHandler;
  handler_map_[StarWorkerServiceMethod::kCreateWorkerSession] = &StarWorkerService::CreateWorkerSessionHandler;
  handler_map_[StarWorkerServiceMethod::kDeleteWorkerSession] = &StarWorkerService::DeleteWorkerSessionHandler;
  handler_map_[StarWorkerServiceMethod::kRegisterGraph] = &StarWorkerService::RegisterGraphHandler;
  handler_map_[StarWorkerServiceMethod::kDeregisterGraph] = &StarWorkerService::DeregisterGraphHandler;
  handler_map_[StarWorkerServiceMethod::kCleanupGraph] = &StarWorkerService::CleanupGraphHandler;
  handler_map_[StarWorkerServiceMethod::kCleanupAll] = &StarWorkerService::CleanupAllHandler;
  handler_map_[StarWorkerServiceMethod::kLogging] = &StarWorkerService::LoggingHandler;
  handler_map_[StarWorkerServiceMethod::kTracing] = &StarWorkerService::TracingHandler;
  handler_map_[StarWorkerServiceMethod::kRecvBuf] = &StarWorkerService::RecvBufHandler;
  handler_map_[StarWorkerServiceMethod::kCompleteGroup] = &StarWorkerService::CompleteGroupHandler;
  handler_map_[StarWorkerServiceMethod::kCompleteInstance] = &StarWorkerService::CompleteInstanceHandler;
  handler_map_[StarWorkerServiceMethod::kGetStepSequence] = &StarWorkerService::GetStepSequenceHandler;
}

HandleRequestFunction StarWorkerService::GetHandler(StarWorkerServiceMethod methodId) {
  return handler_map_[methodId];
}

void StarWorkerService::RunGraphHandler(StarServerTag* tag) {
  Schedule([this, tag]() {
      StarCall<RunGraphRequest, RunGraphResponse>
        *call = new StarCall<RunGraphRequest, RunGraphResponse>();
      InitStarServerTag(&call->req_, &call->resp_, tag);

      CallOptions* call_opts = new CallOptions;
      ProtoRunGraphRequest* wrapped_request =
        new ProtoRunGraphRequest(&call->req_);
      NonOwnedProtoRunGraphResponse* wrapped_response =
        new NonOwnedProtoRunGraphResponse(&call->resp_);

      worker_->RunGraphAsync(call_opts, wrapped_request, wrapped_response,
                             [tag, call, call_opts, wrapped_request, wrapped_response](const Status& s) {
        tag->ProcessDone(s);
        delete call_opts;
        delete wrapped_request;
        delete wrapped_response;
        delete call;
      });
    });
}

void StarWorkerService::StarRunGraphHandler(StarServerTag* tag) {
  Schedule([this, tag]() {
    worker_->StarRunGraphAsync(&(tag->star_graph_request_),
                               &(tag->star_graph_response_),
                               [this, tag](const Status& s) {
      auto step_id = tag->star_graph_request_.step_id_;
      tag->ProcessDone(s);
      worker_->Cleanup(step_id);
    });
  });
}

void StarWorkerService::GetStatusHandler(StarServerTag* tag) {
  Schedule([this, tag]() {
      StarCall<GetStatusRequest, GetStatusResponse>
        *call = new StarCall<GetStatusRequest, GetStatusResponse>();
      InitStarServerTag(&call->req_, &call->resp_, tag);

      Status s = worker_->GetStatus(&call->req_, &call->resp_);

      tag->ProcessDone(s);
      delete call;
  });
}

void StarWorkerService::CreateWorkerSessionHandler(StarServerTag* tag) {
  Schedule([this, tag]() {
      StarCall<CreateWorkerSessionRequest, CreateWorkerSessionResponse>
        *call = new StarCall<CreateWorkerSessionRequest, CreateWorkerSessionResponse>();
      InitStarServerTag(&call->req_, &call->resp_, tag);

      Status s = worker_->CreateWorkerSession(&call->req_, &call->resp_);
      tag->ProcessDone(s);
      delete call;
  });
}

void StarWorkerService::DeleteWorkerSessionHandler(StarServerTag* tag) {
  Schedule([this, tag]() {
      StarCall<DeleteWorkerSessionRequest, DeleteWorkerSessionResponse>
	*call = new StarCall<DeleteWorkerSessionRequest,
			     DeleteWorkerSessionResponse>();
      InitStarServerTag(&call->req_, &call->resp_, tag);

      Status s = worker_->DeleteWorkerSession(&call->req_, &call->resp_);
      tag->ProcessDone(s);
      delete call;
    });
}

void StarWorkerService::CleanupAllHandler(StarServerTag* tag) {
  Schedule([this, tag]() {
      StarCall<CleanupAllRequest, CleanupAllResponse>
        *call = new StarCall<CleanupAllRequest, CleanupAllResponse>();
      InitStarServerTag(&call->req_, &call->resp_, tag);

      Status s = worker_->CleanupAll(&call->req_, &call->resp_);
      tag->ProcessDone(s);
      delete call;
  });
}

void StarWorkerService::RegisterGraphHandler(StarServerTag* tag) {
  Schedule([this, tag]() {
      StarCall<RegisterGraphRequest, RegisterGraphResponse>
        *call = new StarCall<RegisterGraphRequest, RegisterGraphResponse>();
      InitStarServerTag(&call->req_, &call->resp_, tag);

      Status s = worker_->RegisterGraph(&call->req_, &call->resp_);
      tag->ProcessDone(s);
      delete call;
  });
}

void StarWorkerService::DeregisterGraphHandler(StarServerTag* tag) {
  Schedule([this, tag]() {
      StarCall<DeregisterGraphRequest, DeregisterGraphResponse>
        *call = new StarCall<DeregisterGraphRequest, DeregisterGraphResponse>();
      InitStarServerTag(&call->req_, &call->resp_, tag);

      Status s = worker_->DeregisterGraph(&call->req_, &call->resp_);
      tag->ProcessDone(s);
      delete call;
  });
}

void StarWorkerService::CleanupGraphHandler(StarServerTag* tag) {
  Schedule([this, tag]() {
      StarCall<CleanupGraphRequest, CleanupGraphResponse>
        *call = new StarCall<CleanupGraphRequest, CleanupGraphResponse>();
      InitStarServerTag(&call->req_, &call->resp_, tag);

      Status s = worker_->CleanupGraph(&call->req_, &call->resp_);
      tag->ProcessDone(s);
      delete call;
  });
}

void StarWorkerService::LoggingHandler(StarServerTag* tag) {
  Schedule([this, tag]() {
      StarCall<LoggingRequest, LoggingResponse>
        *call = new StarCall<LoggingRequest, LoggingResponse>();
      InitStarServerTag(&call->req_, &call->resp_, tag);

      Status s = worker_->Logging(&call->req_, &call->resp_);
      tag->ProcessDone(s);
      delete call;
  });
}

void StarWorkerService::TracingHandler(StarServerTag* tag) {
  Schedule([this, tag]() {
      StarCall<TracingRequest, TracingResponse>
        *call = new StarCall<TracingRequest, TracingResponse>();
      InitStarServerTag(&call->req_, &call->resp_, tag);

      Status s = worker_->Tracing(&call->req_, &call->resp_);
      tag->ProcessDone(s);
      delete call;
  });
}

void StarWorkerService::RecvBufHandler(StarServerTag* tag) {
  tag->ProcessDone(errors::Unimplemented(
      "StarWorkerService::RecvBufHandler()"));
}

void StarWorkerService::CompleteGroupHandler(StarServerTag* tag) {
  tag->ProcessDone(errors::Unimplemented(
      "StarWorkerService::CompleteGroupHandler()"));
}

void StarWorkerService::CompleteInstanceHandler(StarServerTag* tag) {
  tag->ProcessDone(errors::Unimplemented(
      "StarWorkerService::CompleteInstanceHandler()"));
}

void StarWorkerService::GetStepSequenceHandler(StarServerTag* tag) {
  tag->ProcessDone(errors::Unimplemented(
      "StarWorkerService::GetStepSequenceHandler()"));
}

void StarWorkerService::RecvTensorHandlerRaw(StarServerTag *tag) {
  // LOG(INFO) << "StarWorkerService::RecvTensorHandlerRaw";
  Schedule([this, tag]() {
      CallOptions* call_opts = new CallOptions;

      StarCall<RecvTensorRequest, StarTensorResponse> *call =
          new StarCall<RecvTensorRequest, StarTensorResponse>();

      InitStarServerTag(&call->req_, &call->resp_, tag, [call] (const Status& s) {delete call;});

      worker_->RecvTensorAsync(call_opts, &call->req_, &call->resp_,
                               [tag, call, call_opts
                               ](const Status& s) {
                                 delete call_opts;
                                 tag->ProcessDone(s);
                               });
    });
}

void StarWorkerService::FuseRecvTensorHandlerRaw(StarServerTag *tag) {
  // LOG(INFO) << "StarWorkerService::FuseRecvTensorHandlerRaw";
  Schedule([this, tag]() {
      CallOptions* call_opts = new CallOptions;

      StarCall<FuseRecvTensorRequest, StarFuseTensorResponse> *call =
          new StarCall<FuseRecvTensorRequest, StarFuseTensorResponse>();

      InitStarServerTag(&call->req_, &call->resp_, tag, [call] (const Status& s) {delete call;});

      worker_->FuseRecvTensorAsync(call_opts, &call->req_, &call->resp_,
                                   [tag, call, call_opts
                                   ](const Status& s) {
                                     delete call_opts;
                                     tag->ProcessDone(s);
                                   });
    });
}

void StarWorkerService::Schedule(std::function<void()> f) {
  worker_->env()->compute_pool->Schedule(std::move(f));
}

StarWorker* StarWorkerService::GetWorker() const {
  return worker_;
}

WorkerEnv* StarWorker::env() {
  return env_;
}

StarWorker::StarWorker(WorkerEnv* worker_env)
    : Worker(worker_env), cancel_mgr_(new CancellationManager) {
}

void StarWorker::Cleanup(int64 step_id) {
  if (env()->lockless) {
    if (--pending_graph_count_[step_id] == 0) {
      pending_graph_count_.erase(step_id);
      env()->rendezvous_mgr->Cleanup(step_id);
    }
    return;
  }

  int left = -1;
  {
    mutex_lock l(graph_count_mu_);
    left = --pending_graph_count_[step_id];
    if (left == 0) {
      pending_graph_count_.erase(step_id);
    }
  }
  if (left == 0) {
    env()->rendezvous_mgr->Cleanup(step_id);
  }
}

void StarWorker::StarRunGraphAsync(StarRunGraphRequest* request,
                                   StarRunGraphResponse* response,
                                   StatusCallback done) {
  // Enable prmalloc optimization
  ScopedMemoryCollector scoped_memory_collector;

  CallOptions* opts = &(request->opts_);
  const int64 step_id = request->step_id_;
  WorkerSession* session = env_->session_mgr->LegacySession().get();

  if (env()->lockless) {
    if (pending_graph_count_.find(step_id) ==
        pending_graph_count_.end()) {
        pending_graph_count_[step_id] = request->ps_graph_count_;
    }
  } else {
    mutex_lock l(graph_count_mu_);
    if (pending_graph_count_.find(step_id) ==
        pending_graph_count_.end()) {
        pending_graph_count_[step_id] = request->ps_graph_count_;
    }
  }

  static std::atomic<int64> global_step_id(0);
  auto current_step_id = global_step_id.fetch_add(1);

  static int64 last_time = 0;
  if (current_step_id % 100000 == 0) {
    int64_t cur_time = Env::Default()->NowMicros();
    if (last_time != 0) {
      double qps = 100000 / ((cur_time - last_time) / 1000000.0);
      VLOG(1) << "zero copy run graph qps:" << qps;
    }
    last_time = cur_time;
  }

  std::map<std::string, bool> is_send_dead;
  GraphMgr::NamedTensors in;
  GraphMgr::NamedTensors* out = new GraphMgr::NamedTensors;
  static Tensor empty_tensor(DT_FLOAT);
  for (size_t i = 0; i < request->feed_tensors_.size(); ++i) {
    in.insert({request->feed_names_[i], request->feed_tensors_[i]});
    is_send_dead.insert({request->feed_names_[i], request->is_dead_[i]});
  }

  for (size_t i = 0; i < request->fetch_names_.size(); ++i) {
    out->insert({request->fetch_names_[i], empty_tensor});
  }

  // TODO: No collector, No GPU, No execution options
  ExecutorOpts exec_opts;
  StepStatsCollector* collector = nullptr;

  CancellationManager* cm = new CancellationManager;
  opts->SetCancelCallback([this, cm, step_id]() {
    cm->StartCancel();
    AbortStep(step_id);
  });
  CancellationToken token;
  {
    mutex_lock l(mu_cm_);
    token = cancel_mgr_->get_cancellation_token();
    bool already_cancelled = !cancel_mgr_->RegisterCallback(
        token, [cm]() { cm->StartCancel(); });
    if (already_cancelled) {
      opts->ClearCancelCallback();
      delete cm;
      delete collector;
      delete out;
      done(errors::Aborted("Call was aborted"));
      return;
    }
  }

  // TODO: response should set to nullptr in run graph op
  session->graph_mgr->ExecuteAsync(
      request->graph_handle_, step_id, session, exec_opts,
      collector, nullptr, cm, in, is_send_dead,
       [this, step_id, response, session, cm, out, token, collector, opts, done
       ](Status s) {

        std::map<std::string, bool> is_out_dead;
        if (s.ok()) {
          s = session->graph_mgr->RecvOutputs(step_id, out, &is_out_dead);
        }

        opts->ClearCancelCallback();
        {
          mutex_lock l(mu_cm_);
          cancel_mgr_->DeregisterCallback(token);
        }
        delete cm;

        if (s.ok()) {
          response->fetch_names_.resize(out->size());
          response->fetch_tensors_.resize(out->size());
          response->is_dead_.resize(out->size());
          int idx = 0;
          for (const auto& p : *out) {
            response->fetch_names_[idx] = p.first;
            response->fetch_tensors_[idx] = p.second;
            response->is_dead_[idx++] = is_out_dead[p.first];
          }
        }

        delete collector;
        delete out;
        done(s);
      });
}

void StarWorker::RecvTensorAsync(CallOptions* opts,
                                 const RecvTensorRequest* request,
                                 StarTensorResponse* response,
                                 StatusCallback done) {
    const int64 step_id = request->step_id();
    const string& key = request->rendezvous_key();
    Rendezvous::ParsedKey parsed;

    Status s = Rendezvous::ParseKey(key, &parsed);
    Device* src_dev = nullptr;
    if (s.ok()) {
      s = PrepareRecvTensor(parsed, &src_dev);
    }
    if (!s.ok()) {
      LOG(ERROR) << "PrepareRecvTensor failed, tensor:" << key;
      done(s);
      return;
    }

    // TODO(rangeng.llb): make call opts useful.
    // opts->SetCancelCallback([this, step_id]() { AbortStep(step_id); });
    env_->rendezvous_mgr->RecvLocalAsync(
      step_id, parsed,
      [opts, request, response, done, src_dev, key](const Status& status,
                                           const Rendezvous::Args& send_args,
                                           const Rendezvous::Args& recv_args,
                                           const Tensor& val, const bool is_dead) {
        //opts->ClearCancelCallback();

        if (!status.ok()) {
          LOG(ERROR) << "env_->rendezvous_mgr->RecvLocalAsync failed, error msg is: "
                     << status.error_message();
        }

        if (status.ok()) {
          response->SetIsDead(is_dead);
          bool can_memcpy = DataTypeCanUseMemcpy(val.dtype());

          if (src_dev->tensorflow_gpu_device_info() &&
              (!send_args.alloc_attrs.on_host())) {
#if GOOGLE_CUDA
            CHECK(send_args.device_context)
              << "send dev name: " << src_dev->name()
              << " gpu_info: " << src_dev->tensorflow_gpu_device_info();

            if (can_memcpy) {
              Allocator* alloc = GPUProcessState::singleton()->GetGpuHostAllocator(0);
              Tensor* cpu_copy = new Tensor(alloc, val.dtype(), val.shape());

              GPUUtil::CopyGPUTensorToCPU(src_dev, send_args.device_context, &val, cpu_copy,
                                          [response, cpu_copy, done](const Status& s) {
                                            CHECK(s.ok()) << "copy tensor from gpu sync";
                                            response->SetTensor(*cpu_copy);
                                            delete cpu_copy;
                                            done(s);
                                          });
            } else {
              // TODO(rangeng.llb): Should not be executed currrently.
              Tensor* copy = new Tensor(val);
              GPUUtil::SetProtoFromGPU(*copy,
                                       src_dev,
                                       send_args.device_context,
                                       &response->GetTensorProto(),
                                       is_dead,
                                       [response, copy, done] (const Status& s) {
                                         CHECK(s.ok()) << "copy proto from gpu sync";
                                         response->SetTensor(*copy);
                                         delete copy;
                                         done(s);
                                       });
            }
#else
            done(errors::Internal("No GPU device in process"));
#endif
          } else {
            // tensor is in CPU memory.
            response->SetTensor(val);
            if (!can_memcpy) {
              val.AsProtoTensorContent(&response->GetTensorProto());
            }
            done(Status());
          }
        } else {
          // !s.ok()
          done(status);
        }
      });
}

void StarWorker::FuseRecvTensorAsync(CallOptions* opts,
                                     const FuseRecvTensorRequest* request,
                                     StarFuseTensorResponse* response,
                                     StatusCallback done) {
    const int64 step_id = request->step_id();
    int fuse_count = request->rendezvous_key_size();
    std::vector<Rendezvous::ParsedKey> parsed_keys(fuse_count);
    std::vector<Device*>* src_devs = new std::vector<Device*>(fuse_count, nullptr);

    for (int idx = 0; idx < fuse_count; ++idx) {
      const string& key = request->rendezvous_key(idx);
      Status s = Rendezvous::ParseKey(key, &parsed_keys[idx]);
      // LOG(INFO) << "parsed_keys at index " << idx << " is " << parsed_keys[idx].FullKey()
      //          << " incarnation is " << parsed_keys[idx].src_incarnation;
      if (s.ok()) {
        s = PrepareRecvTensor(parsed_keys[idx], &(*src_devs)[idx]);
      }

      if (!s.ok()) {
        LOG(ERROR) << "PrepareRecvTensor failed, tensor:" << key;
        delete src_devs;
        done(s);
        return;
      }
    }

    // make call opts useful.
    // opts->SetCancelCallback([this, step_id]() { AbortStep(step_id); });
    env_->rendezvous_mgr->FuseRecvLocalAsync(
      step_id, parsed_keys,
      [opts, request, response, done, fuse_count, src_devs](
          const Status& status,
          const std::vector<Rendezvous::Args>& send_argses,
          const Rendezvous::Args& recv_args,
          const std::vector<Tensor>& vals,
          const std::vector<bool>& is_deads) {
        // opts->ClearCancelCallback();

        if (!status.ok()) {
          LOG(ERROR) << "env_->rendezvous_mgr->FuseRecvLocalAsync failed, error msg is: "
                     << status.error_message();
        }

        if (status.ok()) {
          response->Init(fuse_count);
          int *fuse_counter = new int(fuse_count);

          for (int idx = 0; idx < fuse_count; ++idx) {
            response->SetIsDeadByIndex(idx, is_deads[idx]);
            bool can_memcpy = DataTypeCanUseMemcpy(vals[idx].dtype());

            if ((*src_devs)[idx]->tensorflow_gpu_device_info() &&
                (!send_argses[idx].alloc_attrs.on_host())) {
#if GOOGLE_CUDA
              CHECK(send_argses[idx].device_context)
                << "send dev name: " << (*src_devs)[idx]->name()
                << " gpu_info: " << (*src_devs)[idx]->tensorflow_gpu_device_info();

              if (can_memcpy) {
                Allocator* alloc = GPUProcessState::singleton()->GetGpuHostAllocator(0);
                Tensor* cpu_copy = new Tensor(alloc, vals[idx].dtype(), vals[idx].shape());

                GPUUtil::CopyGPUTensorToCPU((*src_devs)[idx], send_argses[idx].device_context,
                                            &vals[idx], cpu_copy,
                                            [response, cpu_copy, done,
                                             src_devs, fuse_counter, idx](const Status& s) {
                                              CHECK(s.ok()) << "copy tensor from gpu sync";
                                              response->SetTensorByIndex(idx, *cpu_copy);
                                              delete cpu_copy;
                                              if (__sync_sub_and_fetch(fuse_counter, 1) == 0) {
                                                done(s);
                                                delete src_devs;
                                                delete fuse_counter;
                                              }
                                            });
              } else {
                // Should not be executed currrently.
                Tensor* copy = new Tensor(vals[idx]);
                GPUUtil::SetProtoFromGPU(*copy,
                                         (*src_devs)[idx],
                                         send_argses[idx].device_context,
                                         &response->GetTensorProtoByIndex(idx),
                                         is_deads[idx],
                                         [response, copy, done,
                                          src_devs, fuse_counter, idx] (const Status& s) {
                                           CHECK(s.ok()) << "copy proto from gpu sync";
                                           response->SetTensorByIndex(idx, *copy);
                                           delete copy;
                                           if (__sync_sub_and_fetch(fuse_counter, 1) == 0) {
                                             done(s);
                                             delete src_devs;
                                             delete fuse_counter;
                                           }
                                         });
              }
#else
              done(errors::Internal("No GPU device in process"));
#endif
            } else {
              // tensor is in CPU memory.
              response->SetTensorByIndex(idx, vals[idx]);
              if (!can_memcpy) {
                vals[idx].AsProtoTensorContent(&response->GetTensorProtoByIndex(idx));
              }
              if (__sync_sub_and_fetch(fuse_counter, 1) == 0) {
                done(Status());
                delete src_devs;
                delete fuse_counter;
              }
            }
          } // end of cycle for with fuse_count
        } else { // !s.ok()
          delete src_devs;
          done(status);
        }
      });
}

std::unique_ptr<StarWorker> NewStarWorker(WorkerEnv* worker_env) {
  return std::unique_ptr<StarWorker>(new StarWorker(worker_env));
}

std::unique_ptr<StarWorkerService> NewStarWorkerService(StarWorker* worker) {
  return std::unique_ptr<StarWorkerService>(
      new StarWorkerService(worker));
}

} // namespace tensorflow
