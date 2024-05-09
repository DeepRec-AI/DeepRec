/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"

#include <unordered_set>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_interface.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

class RpcRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  RpcRemoteRendezvous(const WorkerEnv* env, int64 step_id)
      : BaseRemoteRendezvous(env, step_id) {}

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

  void FuseRecvFromRemoteAsync(
      const std::vector<Rendezvous::ParsedKey>& parsed_keys,
      const Rendezvous::Args& args,
      FuseDoneCallback done) override;

  void FlowControlRecvFromRemoteAsync(const StringPiece& tag,
      const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
      DoneCallback done) override;

 private:
  ~RpcRemoteRendezvous() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(RpcRemoteRendezvous);
};

// Used only to retrieve tensors from remote processes.
class RpcRecvTensorCall : public BaseRecvTensorCall {
 public:
  RpcRecvTensorCall() : wi_(nullptr), dst_device_(nullptr) {}

  void Init(WorkerInterface* wi, int64 step_id, StringPiece key,
            AllocatorAttributes alloc_attrs, Device* dst_device,
            const Rendezvous::Args& recv_args, Rendezvous::DoneCallback done) {
    wi_ = wi;
    alloc_attrs_ = alloc_attrs;
    dst_device_ = dst_device;
    recv_args_ = recv_args;
    done_ = std::move(done);
    req_.set_step_id(step_id);
    req_.set_rendezvous_key(key.data(), key.size());
    req_.set_request_id(GetUniqueRequestId());
  }

  void Reset() {
    // The RpcRemoteRendezvous using this object is responsible for calling
    // ReleaseWorker() before Reset().
    DCHECK_EQ(static_cast<WorkerInterface*>(nullptr), wi_)
        << "Leaking WorkerInterface in RpcRecvTensorCall::Reset().";

    alloc_attrs_ = AllocatorAttributes();
    dst_device_ = nullptr;
    // We don't clear opts_ and assume that Init will set up the state for
    // opts_ appropriately.
    req_.Clear();
    resp_.Clear();
    {
      mutex_lock l(mu_);
      status_ = Status::OK();
    }
    done_ = nullptr;
  }

  ~RpcRecvTensorCall() override {
    // Since only the RpcRecvTensorFreeList will delete an
    // RpcRecvTensorCall, we require that ReleaseWorker() has been called before
    // the user releases a Call object to the free list.
    CHECK_EQ(static_cast<WorkerInterface*>(nullptr), wi_)
        << "Leaking WorkerInterface in RpcRecvTensorCall destructor.";
  }

  void Start(std::function<void()> recv_done) override {
    StartRTCall(std::move(recv_done));
  }

  void StartAbort(const Status& s) override {
    {
      mutex_lock l(mu_);
      status_.Update(s);
    }
    opts_.StartCancel();
  }

  Status status() const override {
    mutex_lock l(mu_);
    return status_;
  }

  void ReleaseWorker(WorkerCacheInterface* worker_cache) {
    DCHECK_NE(static_cast<WorkerInterface*>(nullptr), wi_)
        << "RpcRecvTensorCall::ReleaseWorker() called twice.";
    worker_cache->ReleaseWorker(src_worker_, wi_);
    wi_ = nullptr;
  }

  const Tensor& tensor() const { return resp_.tensor(); }

  bool is_dead() const { return resp_.metadata().is_dead(); }

  Device* dst_device() const { return dst_device_; }
  const Rendezvous::Args& recv_args() const { return recv_args_; }
  const Rendezvous::DoneCallback& done() const { return done_; }

 private:
  friend class RpcRemoteRendezvous;

  // Start the main RecvTensor call, checking for an async abort.
  void StartRTCall(std::function<void()> recv_done) {
    resp_.InitAlloc(dst_device_, alloc_attrs_);
    using namespace std::placeholders;
    StatusCallback cb = std::bind(
        [this](std::function<void()> recv_done,
               // Begin unbound arguments.
               const Status& s) {
          if (!s.ok()) {
            mutex_lock l(mu_);
            status_.Update(s);
          }
          recv_done();
        },
        std::move(recv_done), _1);
    wi_->RecvTensorAsync(&opts_, &req_, &resp_, std::move(cb));
  }

  string src_worker_;
  string src_rel_device_;
  WorkerInterface* wi_;  // Not owned.
  AllocatorAttributes alloc_attrs_;
  Device* dst_device_;
  CallOptions opts_;
  RecvTensorRequest req_;
  TensorResponse resp_;
  Rendezvous::Args recv_args_;
  Rendezvous::DoneCallback done_;

  mutable mutex mu_;
  Status status_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RpcRecvTensorCall);
};

class RpcRecvTensorFreeList {
 public:
  RpcRecvTensorFreeList() {}
  ~RpcRecvTensorFreeList() {
    for (size_t i = 0; i < objects_.size(); i++) {
      delete objects_[i];
    }
  }

  RpcRecvTensorCall* New() {
    {
      mutex_lock l(mu_);
      if (!objects_.empty()) {
        RpcRecvTensorCall* result = objects_.back();
        objects_.pop_back();
        return result;
      }
    }
    return new RpcRecvTensorCall;
  }

  void Release(RpcRecvTensorCall* obj) {
    obj->Reset();
    {
      mutex_lock l(mu_);
      if (objects_.size() < kMaxObjects) {
        objects_.push_back(obj);
        return;
      }
    }
    delete obj;
  }

 private:
  static const int kMaxObjects = 1000;

  mutex mu_;
  std::vector<RpcRecvTensorCall*> objects_ GUARDED_BY(mu_);
};

static RpcRecvTensorFreeList* get_call_freelist() {
  static RpcRecvTensorFreeList* call_freelist = new RpcRecvTensorFreeList();
  return call_freelist;
}

void RpcRemoteRendezvous::RecvFromRemoteAsync(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
    DoneCallback done) {
  CHECK(is_initialized());
  Status s;

  // Prepare a RecvTensor call that can handle being aborted.
  RpcRecvTensorCall* call = get_call_freelist()->New();

  // key.src_device identifies a remote device.
  if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &call->src_worker_,
                                        &call->src_rel_device_)) {
    s = errors::Internal(parsed.src_device,
                         " is invalid remote source device.");
  }
  WorkerSession* sess = session();
  // The worker will be released in a subsequent call to
  // `sess->worker_cache->ReleaseWorker()` (if the call has not yet been
  // initialized) or `call->ReleaseWorker()` (if it has been initialized).
  WorkerInterface* rwi =
      sess->worker_cache->GetOrCreateWorker(call->src_worker_);
  if (s.ok() && rwi == nullptr) {
    s = errors::Internal("No worker known as ", call->src_worker_);
  }

  Device* dst_device;
  if (s.ok()) {
    s = sess->device_mgr()->LookupDevice(parsed.dst_device, &dst_device);
  }
  if (!s.ok()) {
    if (rwi != nullptr) {
      sess->worker_cache->ReleaseWorker(call->src_worker_, rwi);
    }
    get_call_freelist()->Release(call);
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }

  call->Init(rwi, step_id_, parsed.FullKey(), recv_args.alloc_attrs, dst_device,
             recv_args, std::move(done));

  // Record "call" in active_ so that it can be aborted cleanly.
  RegisterCall(call, recv_args);

  // RendezvousMgr already aborted, shouldn't send RPC call any more
  if (!call->status().ok()) {
    // NOTE: `*sess` can potentially be deleted before we return from
    // `call->done()(...)`, so we must release the worker before calling the
    // callback.
    call->ReleaseWorker(sess->worker_cache.get());
    call->done()(call->status(), Args(), Args(), Tensor(), false);
    get_call_freelist()->Release(call);
    return;
  }

  // Start "call".
  Ref();
  call->Start([this, call]() {
    // Removes "call" from active_. Prevent StartAbort().
    DeregisterCall(call);
    // If StartAbort was called prior to DeregisterCall, then the
    // current status should be bad.
    Status s = call->status();
    // NOTE: `*session()` can potentially be deleted before we return from
    // `call->done()(...)`, so we must release the worker before calling the
    // callback.
    call->ReleaseWorker(session()->worker_cache.get());
    call->done()(s, Args(), call->recv_args(), call->tensor(), call->is_dead());
    get_call_freelist()->Release(call);
    Unref();
  });
}

class FuseRpcRecvTensorCall : public BaseRecvTensorCall {
 public:
  FuseRpcRecvTensorCall() :
    wi_(nullptr),
    dst_device_(nullptr) {
  }

  void Init(WorkerInterface* wi, int64 step_id,
            const std::vector<Rendezvous::ParsedKey>& parsed_keys,
            AllocatorAttributes alloc_attrs, Device* dst_device,
            const Rendezvous::Args& recv_args,
            Rendezvous::FuseDoneCallback done) {
    wi_ = wi;
    grpc_wi_ = dynamic_cast<GrpcWorkerInterface*>(wi_);
    alloc_attrs_ = alloc_attrs;
    dst_device_ = dst_device;
    recv_args_ = recv_args;
    fuse_done_ = std::move(done);
    fuse_req_.set_step_id(step_id);
    fuse_req_.set_request_id(GetUniqueRequestId());

    fuse_count_ = parsed_keys.size();
    for (int i = 0; i < fuse_count_; ++i) {
      StringPiece key = parsed_keys[i].FullKey();
      fuse_req_.add_rendezvous_key(key.data(), key.size());
    }
    fuse_resp_.Init(fuse_count_);
  }

  void Reset(WorkerCacheInterface* wc) {
    wc->ReleaseWorker(src_worker_, wi_);
    wi_ = nullptr;
    grpc_wi_ = nullptr;
    alloc_attrs_ = AllocatorAttributes();
    dst_device_ = nullptr;
    // We don't clear opts_ and assume that Init will set up the state for
    // opts_ appropriately.
    fuse_req_.Clear();
    fuse_resp_.Clear();
    {
      mutex_lock l(mu_);
      status_ = Status::OK();
    }
    fuse_done_ = nullptr;
  }

  ~FuseRpcRecvTensorCall() override {
    // Since only the RpcRecvTensorFreeList will delete an
    // RpcRecvTensorCall, and it always sets this->wi_ to null when
    // a call object is released to it, we can assert that this->wi_ is
    // always null at the point of deletion.
    CHECK_EQ(static_cast<WorkerInterface*>(nullptr), wi_)
        << "Leaking WorkerInterface in FuseRpcRecvTensorCall destructor.";
  }

  void Start(std::function<void()> recv_done) override {
    StartRTCall(std::move(recv_done));
  }

  void StartAbort(const Status& s) override {
    {
      mutex_lock l(mu_);
      status_.Update(s);
    }
    opts_.StartCancel();
  }

  Status status() const override {
    mutex_lock l(mu_);
    return status_;
  }

  const std::vector<Tensor>& tensors() const { return fuse_resp_.GetTensors(); }
  const std::vector<bool>& is_deads() const { return fuse_resp_.GetIsDeads(); }
  const Rendezvous::Args& recv_args() const { return recv_args_; }
  const Rendezvous::FuseDoneCallback& fuse_done() const { return fuse_done_; }

 private:
  friend class RpcRemoteRendezvous;

  void StartRTCall(std::function<void()> recv_done) {
    fuse_resp_.InitAlloc(dst_device_, alloc_attrs_);
    using namespace std::placeholders;
    StatusCallback cb = std::bind(
        [this](std::function<void()> recv_done,
               // Begin unbound arguments.
               const Status& s) {
          if (!s.ok()) {
            mutex_lock l(mu_);
            status_.Update(s);
          }
          recv_done();
        },
        std::move(recv_done), _1);

    grpc_wi_->FuseRecvTensorAsync(&opts_, &fuse_req_, &fuse_resp_,
                                  std::move(cb));
  }

  int fuse_count_;
  string src_worker_;
  string src_rel_device_;
  WorkerInterface* wi_;
  GrpcWorkerInterface* grpc_wi_;
  AllocatorAttributes alloc_attrs_;
  Device* dst_device_;
  CallOptions opts_;
  FuseRecvTensorRequest fuse_req_;
  FuseTensorResponse fuse_resp_;
  Rendezvous::Args recv_args_;
  Rendezvous::FuseDoneCallback fuse_done_;

  mutable mutex mu_;
  Status status_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(FuseRpcRecvTensorCall);
};

class FuseRpcRecvTensorFreeList {
 public:
  FuseRpcRecvTensorFreeList() {}
  ~FuseRpcRecvTensorFreeList() {
    for (size_t i = 0; i < objects_.size(); i++) {
      delete objects_[i];
    }
  }

  FuseRpcRecvTensorCall* New() {
    {
      mutex_lock l(mu_);
      if (!objects_.empty()) {
        FuseRpcRecvTensorCall* result = objects_.back();
        objects_.pop_back();
        return result;
      }
    }
    return new FuseRpcRecvTensorCall();
  }

  void Release(FuseRpcRecvTensorCall* obj, WorkerCacheInterface* wc) {
    obj->Reset(wc);
    {
      mutex_lock l(mu_);
      if (objects_.size() < kMaxObjects) {
        objects_.push_back(obj);
        return;
      }
    }
    delete obj;
  }

 private:
  static const int kMaxObjects = 1000;

  mutex mu_;
  std::vector<FuseRpcRecvTensorCall*> objects_ GUARDED_BY(mu_);
};

static FuseRpcRecvTensorFreeList* get_fuse_call_freelist() {
  static FuseRpcRecvTensorFreeList* fuse_call_freelist =
      new FuseRpcRecvTensorFreeList();
  return fuse_call_freelist;
}

void RpcRemoteRendezvous::FuseRecvFromRemoteAsync(
    const std::vector<Rendezvous::ParsedKey>& parsed_keys,
    const Rendezvous::Args& recv_args, FuseDoneCallback done) {
  CHECK(is_initialized());
  int fuse_count = parsed_keys.size();
  Status s;

  // Prepare a RecvTensor call that can handle being aborted.
  FuseRpcRecvTensorCall* call = get_fuse_call_freelist()->New();

  // key.src_device identifies a remote device.
  if (!DeviceNameUtils::SplitDeviceName(parsed_keys[0].src_device,
                                        &call->src_worker_,
                                        &call->src_rel_device_)) {
    s = errors::Internal(parsed_keys[0].src_device,
                         " is invalid remote source device.");
  }

  WorkerSession* sess = session();
  WorkerInterface* rwi = sess->worker_cache->GetOrCreateWorker(call->src_worker_);
  if (s.ok() && rwi == nullptr) {
    s = errors::Internal("No worker known as ", call->src_worker_);
  }

  Device* dst_device;
  if (s.ok()) {
    s = sess->device_mgr()->LookupDevice(parsed_keys[0].dst_device, &dst_device);
  }
  if (!s.ok()) {
    if (rwi != nullptr) {
      sess->worker_cache->ReleaseWorker(call->src_worker_, rwi);
    }
    get_fuse_call_freelist()->Release(call, sess->worker_cache.get());
    done(s, std::vector<Args>(fuse_count), recv_args,
         std::vector<Tensor>(fuse_count),
         std::vector<bool>(fuse_count, false));
    return;
  }
  call->Init(rwi, step_id_,
             parsed_keys, recv_args.alloc_attrs, dst_device,
             recv_args, std::move(done));

  // Record "call" in active_ so that it can be aborted cleanly.
  RegisterCall(call, recv_args);

  // RendezvousMgr already aborted, shouldn't send RPC call any more
  if (!call->status().ok()) {
    call->fuse_done()(call->status(), std::vector<Args>(call->fuse_count_), Args(),
                      std::vector<Tensor>(call->fuse_count_),
                      std::vector<bool>(call->fuse_count_, false));
    session()->worker_cache->ReleaseWorker(call->src_worker_, call->wi_);
    call->wi_ = nullptr;
    get_fuse_call_freelist()->Release(call, session()->worker_cache.get());
    return;
  }

  // Start "call".
  Ref();
  call->Start([this, call]() {
    // Removes "call" from active_. Prevent StartAbort().
    DeregisterCall(call);
    // If StartAbort was called prior to DeregisterCall, then the
    // current status should be bad.
    Status s = call->status();
    call->fuse_done()(s, std::vector<Args>(call->fuse_count_), call->recv_args(),
                      call->tensors(), call->is_deads());
    session()->worker_cache->ReleaseWorker(call->src_worker_, call->wi_);
    call->wi_ = nullptr;
    get_fuse_call_freelist()->Release(call, session()->worker_cache.get());
    Unref();
  });
}



class FlowControlRpcRecvTensorCall : public BaseRecvTensorCall {
 public:
  FlowControlRpcRecvTensorCall()
    : wi_(nullptr), dst_device_(nullptr) {}

  void Init(WorkerInterface* wi, int64 step_id, const StringPiece& tag,
            const StringPiece& key, AllocatorAttributes alloc_attrs,
            Device* dst_device, const Rendezvous::Args& recv_args,
            Rendezvous::DoneCallback done) {
    wi_ = wi;
    grpc_wi_ = dynamic_cast<GrpcWorkerInterface*>(wi_);
    alloc_attrs_ = alloc_attrs;
    dst_device_ = dst_device;
    recv_args_ = recv_args;
    done_ = std::move(done);
    req_.set_step_id(step_id);
    req_.set_tag(tag.data(), tag.size());
    req_.set_request_id(GetUniqueRequestId());
    req_.set_rendezvous_key(key.data(), key.size());
  }

  void Reset() {
    // The FlowControlRpcRemoteRendezvous using this object is responsible for
    // calling ReleaseWorker() before Reset().
    DCHECK_EQ(static_cast<WorkerInterface*>(nullptr), wi_)
        << "Leaking WorkerInterface in RpcRecvTensorCall::Reset().";

    alloc_attrs_ = AllocatorAttributes();
    dst_device_ = nullptr;
    // We don't clear opts_ and assume that Init will set up the state for
    // opts_ appropriately.
    req_.Clear();
    resp_.Clear();
    {
      mutex_lock l(mu_);
      status_ = Status::OK();
    }
    done_ = nullptr;
  }

  ~FlowControlRpcRecvTensorCall() override {
    // Since only the FlowControlRpcRecvTensorFreeList will delete an
    // FlowControlRpcRecvTensorCall, and it always sets this->wi_ to null when
    // a call object is released to it, we can assert that this->wi_ is
    // always null at the point of deletion.
    CHECK_EQ(static_cast<WorkerInterface*>(nullptr), wi_)
      << "Leaking WorkerInterface in FlowControlRpcRecvTensorCall destructor.";
  }

  void Start(std::function<void()> recv_done) override {
    StartRTCall(std::move(recv_done));
  }

  void StartAbort(const Status& s) override {
    {
      mutex_lock l(mu_);
      status_.Update(s);
    }
    opts_.StartCancel();
  }

  Status status() const override {
    mutex_lock l(mu_);
    return status_;
  }

  void ReleaseWorker(WorkerCacheInterface* worker_cache) {
    DCHECK_NE(static_cast<WorkerInterface*>(nullptr), wi_)
      << "FlowControlRpcRecvTensorCall::ReleaseWorker() called twice.";
    worker_cache->ReleaseWorker(src_worker_, wi_);
    wi_ = nullptr;
    grpc_wi_ = nullptr;
  }

  const Tensor& tensor() const { return resp_.tensor(); }

  bool is_dead() const { return resp_.metadata().is_dead(); }

  Device* dst_device() const { return dst_device_; }
  const Rendezvous::Args recv_args() const { return recv_args_; }
  const Rendezvous::DoneCallback& done() const { return done_; }

 private:
  friend class RpcRemoteRendezvous;

  // Start the main RecvTensor call, checking for an async abort.
  void StartRTCall(std::function<void()> recv_done) {
    resp_.InitAlloc(dst_device_, alloc_attrs_);
    using namespace std::placeholders;
    StatusCallback cb = std::bind(
        [this](std::function<void()> recv_done,
               // Begin unbound arguments.
               const Status& s) {
          if (!s.ok()) {
            mutex_lock l(mu_);
            status_.Update(s);
          }
          recv_done();
        },
        std::move(recv_done), _1);
    grpc_wi_->FlowControlRecvTensorAsync(&opts_, &req_, &resp_, std::move(cb));
  }

  string src_worker_;
  string src_rel_device_;
  WorkerInterface* wi_;  // Not owned.
  GrpcWorkerInterface* grpc_wi_;
  AllocatorAttributes alloc_attrs_;
  Device* dst_device_;
  CallOptions opts_;
  FlowControlRecvTensorRequest req_;
  TensorResponse resp_;
  Rendezvous::Args recv_args_;
  Rendezvous::DoneCallback done_;

  mutable mutex mu_;
  Status status_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(FlowControlRpcRecvTensorCall);
};

class FlowControlRpcRecvTensorFreeList {
 public:
  FlowControlRpcRecvTensorFreeList() {}
  ~FlowControlRpcRecvTensorFreeList() {
    for (size_t i = 0; i < objects_.size(); i++) {
      delete objects_[i];
    }
  }

  FlowControlRpcRecvTensorCall* New() {
    {
      mutex_lock l(mu_);
      if (!objects_.empty()) {
        FlowControlRpcRecvTensorCall* result = objects_.back();
        objects_.pop_back();
        return result;
      }
    }
    return new FlowControlRpcRecvTensorCall;
  }

  void Release(FlowControlRpcRecvTensorCall* obj) {
    obj->Reset();
    {
      mutex_lock l(mu_);
      if (objects_.size() < kMaxObjects) {
        objects_.push_back(obj);
        return;
      }
    }
    delete obj;
  }

 private:
  static const int kMaxObjects = 1000;

  mutex mu_;
  std::vector<FlowControlRpcRecvTensorCall*> objects_ GUARDED_BY(mu_);
};

static FlowControlRpcRecvTensorFreeList* get_flow_control_call_freelist() {
  static FlowControlRpcRecvTensorFreeList* call_freelist = \
    new FlowControlRpcRecvTensorFreeList();
  return call_freelist;
}

void RpcRemoteRendezvous::FlowControlRecvFromRemoteAsync(
       const StringPiece& tag, const Rendezvous::ParsedKey& parsed,
       const Rendezvous::Args& recv_args, DoneCallback done) {
  CHECK(is_initialized());
  Status s;

  // Prepare a FlowControlRecvTensor call that can handle being aborted.
  FlowControlRpcRecvTensorCall* call = get_flow_control_call_freelist()->New();

  // key.src_device identifies a remote device.
  if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &call->src_worker_,
                                        &call->src_rel_device_)) {
    s = errors::Internal(parsed.src_device,
                         " is invalid remote source device.");
  }

  WorkerSession* sess = session();
  WorkerInterface* rwi =
      sess->worker_cache->GetOrCreateWorker(call->src_worker_);
  if (s.ok() && rwi == nullptr) {
    s = errors::Internal("No worker known as ", call->src_worker_);
  }

  Device* dst_device;
  if (s.ok()) {
    s = sess->device_mgr()->LookupDevice(parsed.dst_device, &dst_device);
  }
  if (!s.ok()) {
    if (rwi != nullptr) {
      sess->worker_cache->ReleaseWorker(call->src_worker_, rwi);
    }
    get_flow_control_call_freelist()->Release(call);
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }

  call->Init(rwi, step_id_, tag, parsed.FullKey(), recv_args.alloc_attrs,
             dst_device, recv_args, std::move(done));

  // Record "call" in active_ so that it can be aborted cleanly.
  RegisterCall(call, recv_args);

  // RendezvousMgr already aborted, shouldn't send RPC call any more
  if (!call->status().ok()) {
    // NOTE: `*sess` can potentially be deleted before we return from
    // `call->done()(...)`, so we must release the worker before calling the
    // callback.
    call->ReleaseWorker(sess->worker_cache.get());
    call->done()(call->status(), Args(), Args(), Tensor(), false);
    get_flow_control_call_freelist()->Release(call);
    return;
  }

  // Start "call".
  Ref();
  call->Start([this, call]() {
    // Removes "call" from active_. Prevent StartAbort().
    DeregisterCall(call);
    // If StartAbort was called prior to DeregisterCall, then the
    // current status should be bad.
    Status s = call->status();
    // NOTE: `*session()` can potentially be deleted before we return from
    // `call->done()(...)`, so we must release the worker before calling the
    // callback.
    call->ReleaseWorker(session()->worker_cache.get());
    call->done()(s, Args(), call->recv_args(), call->tensor(), call->is_dead());
    get_flow_control_call_freelist()->Release(call);
    Unref();
  });

}

}  // namespace

RpcRendezvousMgr::RpcRendezvousMgr(const WorkerEnv* env)
    : BaseRendezvousMgr(env) {}

BaseRemoteRendezvous* RpcRendezvousMgr::Create(int64 step_id,
                                               const WorkerEnv* worker_env) {
  return new RpcRemoteRendezvous(worker_env, step_id);
}

}  // end namespace tensorflow
