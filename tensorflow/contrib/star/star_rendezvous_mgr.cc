#include "tensorflow/contrib/star/star_rendezvous_mgr.h"
#include "tensorflow/contrib/star/star_tensor_coding.h"
#include "tensorflow/contrib/star/star_worker_interface.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
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

class StarRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  StarRemoteRendezvous(const WorkerEnv* env, int64 step_id)
      : BaseRemoteRendezvous(env, step_id) {}

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

  void FuseRecvFromRemoteAsync(
      const std::vector<Rendezvous::ParsedKey>& parsed_keys,
      const Rendezvous::Args& args,
      FuseDoneCallback done) override;

 private:
  ~StarRemoteRendezvous() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(StarRemoteRendezvous);
};

// Used only to retrieve tensors from remote processes.
class StarRecvTensorCall : public BaseRecvTensorCall {
 public:
  StarRecvTensorCall() : wi_(nullptr), dst_device_(nullptr) {}

  void Init(WorkerInterface* wi, int64 step_id, StringPiece key,
            AllocatorAttributes alloc_attrs, Device* dst_device,
            const Rendezvous::Args& recv_args, Rendezvous::DoneCallback done) {
    wi_ = wi;
    star_wi_ = dynamic_cast<StarWorkerInterface*>(wi_);
    alloc_attrs_ = alloc_attrs;
    dst_device_ = dst_device;
    recv_args_ = recv_args;
    done_ = std::move(done);
    req_.set_step_id(step_id);
    req_.set_rendezvous_key(key.data(), key.size());
  }

  void Reset(WorkerCacheInterface* wc) {
    wc->ReleaseWorker(src_worker_, wi_);
    wi_ = nullptr;
    star_wi_ = nullptr;
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

  ~StarRecvTensorCall() override {
    // Since only the StarRecvTensorFreeList will delete an
    // StarRecvTensorCall, and it always sets this->wi_ to null when
    // a call object is released to it, we can assert that this->wi_ is
    // always null at the point of deletion.
    CHECK_EQ(static_cast<WorkerInterface*>(nullptr), wi_)
        << "Leaking WorkerInterface in StarRecvTensorCall destructor.";
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

  const Tensor& tensor() const { return resp_.GetTensor(); }
  bool is_dead() const { return resp_.GetIsDead(); }
  const Rendezvous::Args& recv_args() const { return recv_args_; }
  const Rendezvous::DoneCallback& done() const { return done_; }

 private:
  friend class StarRemoteRendezvous;

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
    star_wi_->RecvTensorAsync(&opts_, &req_, &resp_, std::move(cb));
  }

private:
  string src_worker_;
  string src_rel_device_;
  WorkerInterface* wi_;
  StarWorkerInterface* star_wi_;
  AllocatorAttributes alloc_attrs_;
  Device* dst_device_;
  CallOptions opts_;
  RecvTensorRequest req_;
  StarTensorResponse resp_;
  Rendezvous::Args recv_args_;
  Rendezvous::DoneCallback done_;

  mutable mutex mu_;
  Status status_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(StarRecvTensorCall);
};

class StarRecvTensorFreeList {
 public:
  virtual ~StarRecvTensorFreeList() {
    for (size_t i = 0; i < objects_.size(); i++) {
      delete objects_[i];
    }
  }

  StarRecvTensorCall* New() {
    {
      mutex_lock l(mu_);
      if (!objects_.empty()) {
        StarRecvTensorCall* result = objects_.back();
        objects_.pop_back();
        return result;
      }
    }
    return new StarRecvTensorCall;
  }

  void Release(StarRecvTensorCall* obj, WorkerCacheInterface* wc) {
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
  std::vector<StarRecvTensorCall*> objects_ GUARDED_BY(mu_);
};

static StarRecvTensorFreeList* get_call_freelist() {
  static StarRecvTensorFreeList* call_freelist =
    new StarRecvTensorFreeList();
  return call_freelist;
}

void StarRemoteRendezvous::RecvFromRemoteAsync(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
    DoneCallback done) {
  CHECK(is_initialized());
  Status s;

  // Prepare a RecvTensor call that can handle being aborted.
  StarRecvTensorCall* call = get_call_freelist()->New();

  // key.src_device identifies a remote device.
  if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &call->src_worker_,
                                        &call->src_rel_device_)) {
    s = errors::Internal(parsed.src_device,
                         " is invalid remote source device.");
  }
  WorkerSession* sess = session();
  WorkerInterface* rwi = sess->worker_cache->GetOrCreateWorker(call->src_worker_);
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
    get_call_freelist()->Release(call, sess->worker_cache.get());
    done(s, Args(), recv_args, Tensor{}, false);
    LOG(ERROR) << "RecvFromRemoteAsync failed, detail " << s.error_message();
    return;
  }
  call->Init(rwi, step_id_, parsed.FullKey(), recv_args.alloc_attrs, dst_device,
             recv_args, std::move(done));

  // Record "call" in active_ so that it can be aborted cleanly.
  RegisterCall(call, recv_args);
  Ref();
  if (!call->status().ok()) {
    LOG(WARNING) << "Rendezvous has been aborted, ignore the rpc call."
                 << ", rendezvous key: " << parsed.FullKey()
                 << ", step id: " << step_id_;
    call->done()(s, Args(), recv_args, Tensor(), false);
    session()->worker_cache->ReleaseWorker(call->src_worker_, call->wi_);
    call->wi_ = nullptr;
    get_call_freelist()->Release(call, session()->worker_cache.get());
    Unref();
    return;
  }

  // Start "call".
  call->Start([this, call]() {
    // Removes "call" from active_. Prevent StartAbort().
    DeregisterCall(call);
    // If StartAbort was called prior to DeregisterCall, then the
    // current status should be bad.
    Status s = call->status();
    call->done()(s, Args(), call->recv_args(), call->tensor(), call->is_dead());
    session()->worker_cache->ReleaseWorker(call->src_worker_, call->wi_);
    call->wi_ = nullptr;
    get_call_freelist()->Release(call, session()->worker_cache.get());
    Unref();
  });
}

class StarFuseRecvTensorCall : public BaseRecvTensorCall {
public:
  StarFuseRecvTensorCall() : wi_(nullptr), dst_device_(nullptr) {}

  void Init(WorkerInterface* wi, int64 step_id,
            const std::vector<Rendezvous::ParsedKey>& parsed_keys,
            AllocatorAttributes alloc_attrs, Device* dst_device,
            const Rendezvous::Args& recv_args,
            Rendezvous::FuseDoneCallback done) {
    wi_ = wi;
    star_wi_ = dynamic_cast<StarWorkerInterface*>(wi_);
    alloc_attrs_ = alloc_attrs;
    dst_device_ = dst_device;
    recv_args_ = recv_args;
    fuse_done_ = std::move(done);
    fuse_req_.set_step_id(step_id);

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
    star_wi_ = nullptr;
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

  ~StarFuseRecvTensorCall() override {
    // Since only the StarRecvTensorFreeList will delete an
    // StarRecvTensorCall, and it always sets this->wi_ to null when
    // a call object is released to it, we can assert that this->wi_ is
    // always null at the point of deletion.
    CHECK_EQ(static_cast<WorkerInterface*>(nullptr), wi_)
        << "Leaking WorkerInterface in StarRecvTensorCall destructor.";
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
  friend class StarRemoteRendezvous;

  // Start the main FuseRecvTensor call, checking for an async abort.
  void StartRTCall(std::function<void()> recv_done) {
    // LOG(INFO) << "StartRTCall for fuse tensor recv";
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
    star_wi_->FuseRecvTensorAsync(&opts_, &fuse_req_, &fuse_resp_,
                                     std::move(cb));
  }

private:
  string src_worker_;
  string src_rel_device_;
  WorkerInterface* wi_;
  StarWorkerInterface* star_wi_;
  AllocatorAttributes alloc_attrs_;
  Device* dst_device_;
  CallOptions opts_;
  Rendezvous::FuseDoneCallback fuse_done_;
  int fuse_count_;
  FuseRecvTensorRequest fuse_req_;
  StarFuseTensorResponse fuse_resp_;
  Rendezvous::Args recv_args_;

  mutable mutex mu_;
  Status status_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(StarFuseRecvTensorCall);
};

class StarFuseRecvTensorFreeList {
 public:
  virtual ~StarFuseRecvTensorFreeList() {
    for (size_t i = 0; i < objects_.size(); i++) {
      delete objects_[i];
    }
  }

  StarFuseRecvTensorCall* New() {
    {
      mutex_lock l(mu_);
      if (!objects_.empty()) {
        StarFuseRecvTensorCall* result = objects_.back();
        objects_.pop_back();
        return result;
      }
    }
    return new StarFuseRecvTensorCall;
  }

  void Release(StarFuseRecvTensorCall* obj, WorkerCacheInterface* wc) {
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
  std::vector<StarFuseRecvTensorCall*> objects_ GUARDED_BY(mu_);
};

static StarFuseRecvTensorFreeList* get_fuse_call_freelist() {
  static StarFuseRecvTensorFreeList* fuse_call_freelist =
    new StarFuseRecvTensorFreeList();
  return fuse_call_freelist;
}

void StarRemoteRendezvous::FuseRecvFromRemoteAsync(
    const std::vector<Rendezvous::ParsedKey>& parsed_keys,
    const Rendezvous::Args& recv_args, FuseDoneCallback done) {
  CHECK(is_initialized());
  int fuse_count = parsed_keys.size();
  Status s;

  // Prepare a FuseRecvTensor call that can handle being aborted.
  StarFuseRecvTensorCall* call = get_fuse_call_freelist()->New();

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
  Ref();
  if (!s.ok()) {
    LOG(WARNING) << "Rendezvous has been aborted, ignore the rpc call."
                 << ", rendezvous key: " << parsed_keys[0].FullKey()
                 << ", step id: " << step_id_;
    call->fuse_done()(s, std::vector<Args>(fuse_count), recv_args,
                      std::vector<Tensor>(fuse_count),
                      std::vector<bool>(fuse_count, false));
    session()->worker_cache->ReleaseWorker(call->src_worker_, call->wi_);
    call->wi_ = nullptr;
    get_fuse_call_freelist()->Release(call, session()->worker_cache.get());
    Unref();
    return;
  }

  // Start "call".
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

} // namespace

StarRendezvousMgr::StarRendezvousMgr(const WorkerEnv* env)
  : BaseRendezvousMgr(env) {}

BaseRemoteRendezvous* StarRendezvousMgr::Create(int64 step_id,
                                                const WorkerEnv* worker_env) {
  return new StarRemoteRendezvous(worker_env, step_id);
}

} // namespace tensorflow
