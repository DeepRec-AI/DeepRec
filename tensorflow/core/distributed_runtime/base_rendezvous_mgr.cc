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

#include "tensorflow/core/distributed_runtime/base_rendezvous_mgr.h"

#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

namespace {
  uint64 kGlobalStepId = 0x100000000000000uLL;
  int64 kFlowControlMaxSize = 16;
} // namespace anonymous

static void StartAbortRendevous(Rendezvous* rendez, const Status& s) {
  rendez->StartAbort(s);
  rendez->Unref();
}

BaseRendezvousMgr::BaseRendezvousMgr(const WorkerEnv* worker_env)
    : worker_env_(worker_env) {}

BaseRendezvousMgr::~BaseRendezvousMgr() {
  for (auto& p : table_) {
    auto rendez = p.second;
    StartAbortRendevous(rendez, errors::Aborted("Shutdown"));
  }
}

RemoteRendezvous* BaseRendezvousMgr::Find(int64 step_id) {
  return FindOrCreate(step_id);
}

BaseRemoteRendezvous* BaseRendezvousMgr::FindOrCreate(int64 step_id) {
  mutex_lock l(mu_);
  auto iter = table_.find(step_id);
  if (iter == table_.end()) {
    auto rr = Create(step_id, worker_env_);
    iter = table_.insert({step_id, rr}).first;
  }
  iter->second->Ref();
  return iter->second;
}

void BaseRendezvousMgr::RecvLocalAsync(int64 step_id,
                                       const Rendezvous::ParsedKey& parsed,
                                       Rendezvous::DoneCallback done) {
  auto rendez = FindOrCreate(step_id);
  using namespace std::placeholders;
  Rendezvous::DoneCallback done_cb = std::bind(
      [rendez](Rendezvous::DoneCallback done,
               // Begin unbound arguments.
               const Status& s, const Rendezvous::Args& send_args,
               const Rendezvous::Args& recv_args, const Tensor& v, bool dead) {
        rendez->Unref();
        done(s, send_args, recv_args, v, dead);
      },
      std::move(done), _1, _2, _3, _4, _5);
  rendez->RecvLocalAsync(parsed, std::move(done_cb));
}

Status BaseRendezvousMgr::RecvLocal(int64 step_id,
                                    const Rendezvous::ParsedKey& parsed,
                                    Tensor* val, bool* is_dead) {
  Status ret;
  Notification n;
  RecvLocalAsync(step_id, parsed,
                 [val, is_dead, &ret, &n](const Status& s,
                                          const Rendezvous::Args& send_args,
                                          const Rendezvous::Args& recv_args,
                                          const Tensor& v, const bool dead) {
                   ret = s;
                   *val = v;
                   *is_dead = dead;
                   n.Notify();
                 });
  n.WaitForNotification();
  return ret;
}

void BaseRendezvousMgr::FuseRecvLocalAsync(
    int64 step_id, const std::vector<Rendezvous::ParsedKey>& parsed_keys,
    Rendezvous::FuseDoneCallback done) {
  auto rendez = FindOrCreate(step_id);
  using namespace std::placeholders;
  Rendezvous::FuseDoneCallback done_cb = std::bind(
      [rendez](Rendezvous::FuseDoneCallback done,
               // Begin unbound arguments.
               const Status& s,
               const std::vector<Rendezvous::Args>& send_argses,
               const Rendezvous::Args& recv_args,
               const std::vector<Tensor>& vals,
               const std::vector<bool>& deads) {
        rendez->Unref();
        done(s, send_argses, recv_args, vals, deads);
      },
      std::move(done), _1, _2, _3, _4, _5);
  rendez->FuseRecvLocalAsync(parsed_keys, std::move(done_cb));
}

void BaseRendezvousMgr::FlowControlRecvLocalAsync(int64 step_id,
       const StringPiece& tag, const Rendezvous::ParsedKey& parsed,
       Rendezvous::DoneCallback done) {
  auto rendez = FindOrCreate(step_id);
  using namespace std::placeholders;
  Rendezvous::DoneCallback done_cb = std::bind(
      [rendez](Rendezvous::DoneCallback done,
               // Begin unbound arguments.
               const Status& s, const Rendezvous::Args& send_args,
               const Rendezvous::Args& recv_args, const Tensor& v, bool dead) {
        rendez->Unref();
        done(s, send_args, recv_args, v, dead);
      },
      std::move(done), _1, _2, _3, _4, _5);
  rendez->FlowControlRecvLocalAsync(tag, parsed, std::move(done_cb));
}

void BaseRendezvousMgr::Cleanup(int64 step_id) {
  Rendezvous* rendez = nullptr;
  {
    mutex_lock l(mu_);
    auto iter = table_.find(step_id);
    if (iter != table_.end()) {
      rendez = iter->second;
      table_.erase(iter);
    }
  }
  if (rendez) {
    StartAbortRendevous(rendez, errors::Aborted("Cleanup ", step_id));
  }
}

void BaseRendezvousMgr::CleanupAll() {
  std::vector<Rendezvous*> rendezs;
  BaseRemoteRendezvous* inter_step_rendez = nullptr;
  {
    mutex_lock l(mu_);
    for (const auto& entry : table_) {
      if (kGlobalStepId == entry.first) {
        inter_step_rendez = entry.second;
      } else {
        rendezs.push_back(entry.second);
      }
    }
    table_.clear();
    // NOTE(yuanman.ym): InterStepRendezvous should not be cleared since
    // one session run might recv from a session run already finished.
    if (nullptr != inter_step_rendez) {
      table_.insert({kGlobalStepId, inter_step_rendez});
    }
  }
  for (auto rendez : rendezs) {
    StartAbortRendevous(rendez, errors::Aborted("Shutdown"));
  }
}

RemoteRendezvous* BaseRendezvousMgr::FindInterStepRendezvous() {
  return FindOrCreate(kGlobalStepId);
}

BaseRemoteRendezvous::BaseRemoteRendezvous(const WorkerEnv* env, int64 step_id)
    : env_(env),
      step_id_(step_id),
      local_(NewLocalRendezvous()),
      session_(nullptr),
      flow_control_num_(0) {
  Status s = ReadInt64FromEnvVar("REMOTE_RENDEZVOUS_FLOW_CONTROL_MAX_SIZE",
        kFlowControlMaxSize, &flow_control_max_size_);
  if (!s.ok()) {
    LOG(ERROR) << "Read REMOTE_RENDEZVOUS_FLOW_CONTROL_MAX_SIZE env error: "
               << s.error_message();
  }
  VLOG(2) << "BaseRemoteRendezvous set flow control max size: "
          << flow_control_max_size_;
}

BaseRemoteRendezvous::~BaseRemoteRendezvous() {
  CHECK(active_.empty());
  local_->Unref();
}

// Returns true if "device_name" is a valid full name of local device
// of the "worker".  This helper is purely based on the worker name
// and device name and does no lookups in the worker->device_mgr.
static bool IsLocalDevice(const StringPiece worker_name,
                          const StringPiece device_name) {
  return absl::StartsWith(device_name, worker_name);
}

Status BaseRemoteRendezvous::Initialize(WorkerSession* session) {
  CHECK_NE(session, nullptr) << "session must not be null!";
  std::vector<DeferredCall> deferred_calls;
  {
    mutex_lock l(mu_);
    if (session_ != nullptr) {
      if (worker_name_ != session->worker_name) {
        Status s = errors::Internal(
            "Double init! Worker names would have changed from: ",
            worker_name_, " -> ", session->worker_name);
        LOG(WARNING) << s;
        return s;
      }
    }
    session_ = session;
    worker_name_ = session->worker_name;
    std::swap(deferred_calls, deferred_calls_);
  }
  for (auto& call : deferred_calls) {
    RecvLocalAsyncInternal(call.parsed, std::move(call.done));
  }

  std::vector<DeferredFuseCall> deferred_fuse_calls;
  {
    mutex_lock l(mu_);
    std::swap(deferred_fuse_calls, deferred_fuse_calls_);
  }
  for (auto& fuse_call : deferred_fuse_calls) {
    FuseRecvLocalAsyncInternal(fuse_call.parsed_keys,
                               std::move(fuse_call.done));
  }

  std::vector<DeferredFlowControlCall> deferred_flow_control_calls;
  {
    mutex_lock l(mu_);
    std::swap(deferred_flow_control_calls, deferred_flow_control_calls_);
  }
  for (auto& fc_call : deferred_flow_control_calls) {
    FlowControlRecvLocalAsyncInternal(fc_call.tag, fc_call.parsed,
                                      std::move(fc_call.done));
  }

  return Status::OK();
}

WorkerSession* BaseRemoteRendezvous::session() {
  tf_shared_lock l(mu_);
  return session_;
}

bool BaseRemoteRendezvous::is_initialized() {
  tf_shared_lock l(mu_);
  return is_initialized_locked();
}

Status BaseRemoteRendezvous::Send(const Rendezvous::ParsedKey& parsed,
                                  const Rendezvous::Args& args,
                                  const Tensor& val, const bool is_dead) {
  VLOG(1) << "BaseRemoteRendezvous Send " << this << " " << parsed.FullKey();
  {
    tf_shared_lock l(mu_);
    if (!status_.ok()) return status_;
    DCHECK(is_initialized_locked());
    if (!IsLocalDevice(session_->worker_name, parsed.src_device)) {
      return errors::InvalidArgument(
          "Invalid rendezvous key (src): ", parsed.FullKey(), " @ ",
          session_->worker_name);
    }
  }
  // Buffers "val" and "device_context" in local_.
  return local_->Send(parsed, args, val, is_dead);
}

Status BaseRemoteRendezvous::Send(const ParsedKey& parsed,
                                  const Rendezvous::Args& args,
                                  Tensor* val, mutex* mu,
                                  const bool is_dead) {
  VLOG(1) << "BaseRemoteRendezvous Send " << this << " " << parsed.FullKey();
  {
    mutex_lock l(mu_);
    if (!status_.ok()) return status_;
    DCHECK(is_initialized_locked());
    if (!IsLocalDevice(session_->worker_name, parsed.src_device)) {
      return errors::InvalidArgument(
          "Invalid rendezvous key (src): ", parsed.FullKey(), " @ ",
          session_->worker_name);
    }
  }
  // Buffers "val" and "device_context" in local_.
  return local_->Send(parsed, args, val, mu, is_dead);
}

Status BaseRemoteRendezvous::FlowControlSend(const StringPiece& tag,
                                             const ParsedKey& parsed,
                                             const Args& args,
                                             const Tensor& val,
                                             const bool is_dead,
                                             const int64 timeout_millis) {
  VLOG(1) << "BaseRemoteRendezvous FlowControlSend " << this << " "
          << parsed.FullKey();
  const std::string tag_string(tag.data(), tag.size());
  {
    mutex_lock l(mu_);
    while(status_.ok() && flow_control_num_ >= flow_control_max_size_) {
      if (flow_control_cv_.wait_for(
            l, std::chrono::milliseconds(timeout_millis)) == \
          std::cv_status::timeout) {
        return errors::DeadlineExceeded("FlowControlSend has timed out.");
      }
    }

    if (!status_.ok()) return status_;
    DCHECK(is_initialized_locked());
    if (!IsLocalDevice(session_->worker_name, parsed.src_device)) {
      return errors::InvalidArgument(
          "Invalid rendezvous key (src): ", parsed.FullKey(), " @ ",
          session_->worker_name);
    }

    flow_control_num_++;
    if (flow_control_counters_.count(tag_string) == 0) {
      flow_control_counters_[tag_string] = 0;
    }
    flow_control_counters_[tag_string]++;
  }
  // Buffers "val" and "device_context" in local_.
  return local_->Send(parsed, args, val, is_dead);
}

Status BaseRemoteRendezvous::ValidateDevices(const ParsedKey& parsed,
                                             bool is_src) {
  // Cache session pointer to avoid repeatedly taking & releasing the lock
  // (e.g. calling session())
  WorkerSession* sess = nullptr;
  {
    tf_shared_lock l(mu_);
    if (!status_.ok()) return status_;
    if (!is_initialized_locked()) {
      return errors::Internal("ValidateDevices called before initialization.");
    }
    sess = session_;
  }
  if (is_src && !IsLocalDevice(sess->worker_name, parsed.src_device)) {
    return errors::InvalidArgument("Invalid rendezvous key (src): ",
                                   parsed.FullKey(), " @ ", sess->worker_name);
  }
  if (!is_src && !IsLocalDevice(sess->worker_name, parsed.dst_device)) {
    return errors::InvalidArgument("Invalid rendezvous key (dst): ",
                                   parsed.FullKey(), " @ ", sess->worker_name);
  }
  return Status::OK();
}

void BaseRemoteRendezvous::SameWorkerRecvDone(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& send_args,
    const Rendezvous::Args& recv_args, const Tensor& in, Tensor* out,
    StatusCallback done) {
  // Do a quick copy (sharing the underlying buffer) if both tensors
  // are on host memory.
  const bool src_host =
      (send_args.alloc_attrs.on_host() || parsed.src.type == "CPU");
  const bool dst_host =
      (recv_args.alloc_attrs.on_host() || parsed.dst.type == "CPU");
  if (src_host && dst_host) {
    *out = in;
    done(Status::OK());
    return;
  }

  // This copy must involve a GPU. Hence, "in" must support DMA
  // (e.g., string tensors do not work on GPU).  Variant copy DMA
  // checks happen inside CopyTensor::ViaDMA.
  if (!DMAHelper::CanUseDMA(&in) && in.dtype() != DT_VARIANT &&
      in.dtype() != DT_RESOURCE) {
    done(errors::InvalidArgument(
        "Non-DMA-safe ", DataTypeString(in.dtype()),
        " tensor may not be copied from/to a device. Key: ", parsed.FullKey()));
    return;
  }

  WorkerSession* sess = session();
  Device* src_device;
  Status s = sess->device_mgr()->LookupDevice(parsed.src_device, &src_device);
  if (!s.ok()) {
    done(s);
    return;
  }
  Device* dst_device;
  s = sess->device_mgr()->LookupDevice(parsed.dst_device, &dst_device);
  if (!s.ok()) {
    done(s);
    return;
  }

  AllocatorAttributes attr = recv_args.alloc_attrs;
  attr.set_gpu_compatible(send_args.alloc_attrs.gpu_compatible() ||
                          recv_args.alloc_attrs.gpu_compatible());
  Allocator* out_allocator = dst_device->GetAllocator(attr);
  AllocationAttributes allocation_attr;
  uint64 safe_alloc_frontier = dst_device->SafeAllocFrontier(0);
  bool sync_dst_compute = (safe_alloc_frontier == 0);
  std::function<uint64()> freed_by_func = [dst_device, &safe_alloc_frontier]() {
    safe_alloc_frontier = dst_device->SafeAllocFrontier(safe_alloc_frontier);
    return safe_alloc_frontier;
  };
  if (!sync_dst_compute) {
    allocation_attr.freed_by_func = &freed_by_func;
  }
  if (in.dtype() != DT_VARIANT) {
    // Variants are handled by CopyTensor::ViaDMA.
    Tensor copy(out_allocator, in.dtype(), in.shape(), allocation_attr);
    *out = copy;
  }

  // The following function takes care of cpu->gpu, gpu->cpu, gpu->gpu copies,
  // etc.
  CopyTensor::ViaDMA(
      parsed.edge_name, send_args.device_context, recv_args.device_context,
      src_device, dst_device, send_args.alloc_attrs, recv_args.alloc_attrs, &in,
      out, 0 /*dev_to_dev_stream_index*/, std::move(done), sync_dst_compute);
}

bool BaseRemoteRendezvous::IsSameWorker(DeviceNameUtils::ParsedName src,
                                        DeviceNameUtils::ParsedName dst) {
  return DeviceNameUtils::IsSameAddressSpace(src, dst);
}

void BaseRemoteRendezvous::RecvAsync(const ParsedKey& parsed,
                                     const Rendezvous::Args& recv_args,
                                     DoneCallback done) {
  VLOG(1) << "RemoteRendezvous Recv " << this << " " << parsed.FullKey();
  Status s = ValidateDevices(parsed, false /*!is_src*/);
  if (s.ok() && !is_initialized()) {
    s.Update(errors::Internal(
        "RecvAsync called when uninitialized (key:", parsed.FullKey(), ")."));
  }
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor(), false);
    return;
  }

  // Are src and dst in the same worker?
  if (IsSameWorker(parsed.src, parsed.dst)) {
    // Recv the tensor from local_.
    local_->RecvAsync(
        parsed, recv_args,
        [this, parsed, done](
            const Status& status, const Rendezvous::Args& send_args,
            const Rendezvous::Args& recv_args, const Tensor& in, bool is_dead) {
          VLOG(2) << "RemoteRendezvous Finished Recv " << this << " "
                  << parsed.FullKey();
          Tensor* out = new Tensor;
          StatusCallback final_callback = [done, send_args, recv_args, out,
                                           is_dead](const Status& s) {
            done(s, send_args, recv_args, *out, is_dead);
            delete out;
          };

          if (status.ok()) {
            SameWorkerRecvDone(parsed, send_args, recv_args, in, out,
                               std::move(final_callback));
          } else {
            final_callback(status);
          }
        });
    return;
  } else {
    RecvFromRemoteAsync(parsed, recv_args, std::move(done));
  }
}

void BaseRemoteRendezvous::FlowControlRecvAsync(const StringPiece& tag,
                                                const ParsedKey& parsed,
                                                const Args& recv_args,
                                                DoneCallback done) {
  VLOG(1) << "RemoteRendezvous FlowControlRecvAsync " << this
          << " " << tag << " " << parsed.FullKey();

  Status s = ValidateDevices(parsed, false /*!is_src*/);
  if (s.ok() && !is_initialized()) {
    s.Update(errors::Internal(
        "FlowControlRecvAsync called when uninitialized (key:",
        parsed.FullKey(), ")."));
  }
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor(), false);
    return;
  }

  // Are src and dst in the same worker?
  if (IsSameWorker(parsed.src, parsed.dst)) {
    // Recv the tensor from local_.
    local_->RecvAsync(
        parsed, recv_args,
        [this, tag, parsed, done](
            const Status& status, const Rendezvous::Args& send_args,
            const Rendezvous::Args& recv_args, const Tensor& in, bool is_dead) {
          VLOG(2) << "RemoteRendezvous Finished Recv " << this << " "
                  << parsed.FullKey();
          Tensor* out = new Tensor;
          StatusCallback final_callback = [done, send_args, recv_args, out,
                                           is_dead](const Status& s) {
            done(s, send_args, recv_args, *out, is_dead);
            delete out;
          };

          if (status.ok()) {
            SameWorkerRecvDone(parsed, send_args, recv_args, in, out,
                               std::move(final_callback));
            const std::string tag_string(tag.data(), tag.size());
            {
              mutex_lock l(mu_);
              flow_control_num_--;
              DCHECK(flow_control_counters_.count(tag_string) != 0);
              flow_control_counters_[tag_string]--;
            }
            flow_control_cv_.notify_one();
          } else {
            final_callback(status);
          }
        });
    return;
  } else {
    FlowControlRecvFromRemoteAsync(tag, parsed, recv_args, std::move(done));
  }

}

void BaseRemoteRendezvous::RecvLocalAsync(const ParsedKey& parsed,
                                          DoneCallback done) {
  {
    mutex_lock l(mu_);
    if (!is_initialized_locked()) {
      // RecvLocalAsync can be called (due to an incoming RecvTensor RPC from a
      // remote worker) before the RunStep (or PartialRunStep) RPC from the
      // master arrives. RecvLocalAsync thus buffers the arguments until after
      // the RemoteRendezvous is Initialize()'d, when it completes the
      // rendezvous logic. At some point after Initialize() is called, a Tensor
      // is produced locally that will then be sent in response to the incoming
      // RPC.
      DeferredCall call(parsed, std::move(done));
      deferred_calls_.push_back(call);
      return;
    }
  }
  RecvLocalAsyncInternal(parsed, std::move(done));
}

void BaseRemoteRendezvous::RecvLocalAsyncInternal(const ParsedKey& parsed,
                                                  DoneCallback done) {
  Status s = ValidateDevices(parsed, true /* is_src */);
  if (!s.ok()) {
    done(s, Args(), Args(), Tensor(), false);
    return;
  }
  local_->RecvAsync(parsed, Args(), std::move(done));
}

void BaseRemoteRendezvous::FuseRecvAsync(
    const std::vector<ParsedKey>& parsed_keys,
    const Rendezvous::Args& recv_args, FuseDoneCallback done) {
  CHECK(is_initialized()) << "RecvAsync called when uninitialized.";

  int fuse_count = parsed_keys.size();
  for (int i = 0; i < fuse_count; ++i) {
    Status s = ValidateDevices(parsed_keys[i], false /*!is_src*/);
    if (!s.ok()) {
      done(s, std::vector<Args>(fuse_count), recv_args,
           std::vector<Tensor>(fuse_count),
           std::vector<bool>(fuse_count, false));
      return;
    }
  }

  // Are src and dst in the same worker?
  if (IsSameWorker(parsed_keys[0].src, parsed_keys[0].dst)) {
    // NOTE(jiankeng.pt): run_graph_op will use the local fuse recv,
    // and we use sync method here.
    FuseRecvLocalSync(parsed_keys, std::move(done));
  } else {
    FuseRecvFromRemoteAsync(parsed_keys, recv_args, std::move(done));
  }
}

// NOTE(jiankeng.pt): The sync method will only be used when
// all parsed_keys' value are ready.
void BaseRemoteRendezvous::FuseRecvLocalSync(
    const std::vector<ParsedKey>& parsed_keys,
    FuseDoneCallback done) {
  int fuse_count = parsed_keys.size();
  std::vector<Tensor> vt(fuse_count);
  std::vector<bool> vd(fuse_count, true);
  std::vector<Rendezvous::Args> vsa(fuse_count);

  for (int i = 0; i < fuse_count; ++i) {
    Status s = ValidateDevices(parsed_keys[i], true /* is_src */);
    if (!s.ok()) {
      if (s.code() != tensorflow::error::OUT_OF_RANGE) {
        LOG(ERROR) << "step_id" << step_id_ << ", "
                   << "send device is not local device. key="
                   << parsed_keys[i].FullKey();
      }
      done(s, vsa, Args(), vt, std::vector<bool>(fuse_count));
      return;
    }
  }

  for (int i = 0; i < fuse_count; ++i) {
    bool recv_done = false;
    Status status;
    local_->RecvAsync(parsed_keys[i], Args(), [i, &vt, &vd, &vsa, &recv_done, &status](
                        const Status& s,
                        const Rendezvous::Args& send_args,
                        const Rendezvous::Args& recv_args,
                        const Tensor& val, bool is_dead) {
                          if (!s.ok()) {
                            status = s;
                          } else {
                            status = Status::OK();
                            vsa[i] = send_args;
                            vt[i] = val;
                            vd[i] = is_dead;
                            recv_done = true;
                          }
                        });
    if (!recv_done) {
      done(status, vsa, Args(), vt, vd);
      return;
    }
  }

  done(Status::OK(), vsa, Args(), vt, vd);
}

void BaseRemoteRendezvous::FuseRecvLocalAsync(
    const std::vector<ParsedKey>& parsed_keys,
    FuseDoneCallback done) {
  {
    mutex_lock l(mu_);
    if (!is_initialized_locked()) {
      // RecvLocalAsync can be called (due to an incoming RecvTensor RPC from a
      // remote worker) before the RunStep (or PartialRunStep) RPC from the
      // master arrives. RecvLocalAsync thus buffers the arguments until after
      // the RemoteRendezvous is Initialize()'d, when it completes the
      // rendezvous logic. At some point after Initialize() is called, a Tensor
      // is produced locally that will then be sent in response to the incoming
      // RPC.
      DeferredFuseCall call(parsed_keys, std::move(done));
      deferred_fuse_calls_.push_back(call);
      return;
    }
  }
  FuseRecvLocalAsyncInternal(parsed_keys, std::move(done));
}

void BaseRemoteRendezvous::FuseRecvLocalAsyncInternal(
    const std::vector<ParsedKey>& parsed_keys, FuseDoneCallback done) {
  int fuse_count = parsed_keys.size();
  std::vector<Tensor> *vt = new std::vector<Tensor>(fuse_count);
  volatile bool *vd = new volatile bool[fuse_count]();
  std::vector<Rendezvous::Args> *vsa = new std::vector<Rendezvous::Args>(fuse_count);

  for (int i = 0; i < fuse_count; ++i) {
    Status s = ValidateDevices(parsed_keys[i], true /* is_src */);
    if (!s.ok()) {
      LOG(ERROR) << "send device is not local device";
      done(s, *vsa, Args(), *vt, std::vector<bool>(fuse_count));
      delete vsa,
      delete vt;
      delete [] vd;
      return;
    }
  }

  Status* status = new Status();
  mutex *mu = new mutex();
  std::atomic<int> *fuse_counter = new std::atomic<int>(fuse_count);

  for (int i = 0; i < fuse_count; ++i) {
    using namespace std::placeholders;
    ParsedKey parsed_key_tmp = parsed_keys[i];
    DoneCallback done_once = std::bind(
        [vsa, vt, vd, status, mu, fuse_count, fuse_counter, i](
            FuseDoneCallback done, const Status& s,
            const Rendezvous::Args& send_args,
            const Rendezvous::Args& recv_args,
            const Tensor& val, bool is_dead) {
          if (!s.ok()) {
            mutex_lock l(*mu);
            *status = s;
          } else {
            (*vsa)[i] = send_args;
            (*vt)[i] = val;
            vd[i] = is_dead;
          }

          if (fuse_counter->fetch_sub(1) == 1) {
            std::vector<bool> vd_tmp(vd, vd + fuse_count);
            done(*status, *vsa, recv_args, *vt, vd_tmp);
            delete vsa;
            delete vt;
            delete [] vd;
            delete status;
            delete fuse_counter;
          }
        },
        // std::move should not be used here, 'done' may be
        // called at multiple context.
        done,
        _1, _2, _3, _4, _5);

    local_->RecvAsync(parsed_keys[i], Args(), std::move(done_once));
  }
}

void BaseRemoteRendezvous::FlowControlRecvLocalAsync(const StringPiece& tag,
                                                     const ParsedKey& parsed,
                                                     DoneCallback done) {
  {
    mutex_lock l(mu_);
    if (!is_initialized_locked()) {
      // FlowControlRecvLocalAsync can be called (due to an incoming RecvTensor
      // RPC from a remote worker) before the RunStep (or PartialRunStep) RPC
      // from the master arrives. RecvLocalAsync thus buffers the arguments
      // until after the RemoteRendezvous is Initialize()'d, when it completes
      // the rendezvous logic. At some point after Initialize() is called, a
      // Tensor is produced locally that will then be sent in response to the
      // incoming RPC.
      DeferredFlowControlCall call(tag, parsed, std::move(done));
      deferred_flow_control_calls_.push_back(call);
      return;
    }
  }
  FlowControlRecvLocalAsyncInternal(tag, parsed, std::move(done));
}

void BaseRemoteRendezvous::FlowControlRecvLocalAsyncInternal(
       const StringPiece& tag, const ParsedKey& parsed, DoneCallback done) {
  Status s = ValidateDevices(parsed, true /* is_src */);
  if (!s.ok()) {
    done(s, Args(), Args(), Tensor(), false);
    return;
  }

  using namespace std::placeholders;
  Rendezvous::DoneCallback done_cb = std::bind(
      [this, tag](Rendezvous::DoneCallback done,
             // Begin unbound arguments.
             const Status& s, const Rendezvous::Args& send_args,
             const Rendezvous::Args& recv_args, const Tensor& v, bool dead) {
        done(s, send_args, recv_args, v, dead);
        if (s.ok()) {
          const std::string tag_string(tag.data(), tag.size());
          {
            mutex_lock l(mu_);
            flow_control_num_--;
            DCHECK(flow_control_counters_.count(tag_string) != 0);
            flow_control_counters_[tag_string]--;
          }
          flow_control_cv_.notify_one();
        }
      },
      std::move(done), _1, _2, _3, _4, _5);

  local_->RecvAsync(parsed, Args(), std::move(done_cb));
}

void BaseRemoteRendezvous::FuseRecvFromRemoteAsync(
        const std::vector<Rendezvous::ParsedKey>& parsed_keys,
        const Rendezvous::Args& args,
        FuseDoneCallback done) {
    CHECK(false) << "FuseRecvFromRemoteAsync Unimplemented";
}

void BaseRemoteRendezvous::FlowControlRecvFromRemoteAsync(
       const StringPiece& tag, const Rendezvous::ParsedKey& parsed,
       const Rendezvous::Args& args, DoneCallback done) {
  CHECK(false) << "FlowControlRecvFromRemoteAsync Unimplemented.";
}

void BaseRemoteRendezvous::RecvAsync(const ParsedKey& parsed,
                                     const Rendezvous::Args& recv_args,
                                     RefDoneCallback done) {
  VLOG(1) << "RemoteRendezvous Recv " << this << " " << parsed.FullKey();
  CHECK(is_initialized()) << "Ref RecvAsync called when uninitialized.";
  Status s = ValidateDevices(parsed, false /*!is_src*/);
  if (!s.ok()) {
    done(s, Args(), Args(), nullptr, nullptr, false);
    return;
  }

  // Require same worker
  if (IsSameWorker(parsed.src, parsed.dst)) {
    // Recv the tensor from local_.
    local_->RecvAsync(
        parsed, recv_args,
        [this, parsed, done](
            const Status& status, const Rendezvous::Args& send_args,
            const Rendezvous::Args& recv_args, Tensor* in, mutex* mu,
            bool is_dead) {
            // only support CPU now
            done(status, send_args, recv_args, in, mu, is_dead);
        });
  } else {
    LOG(ERROR) << "Ref RecvAsync require the same src and dst.";
    done(s, Args(), Args(), nullptr, nullptr, false);
  }
}

int64 BaseRemoteRendezvous::GetAllFlowControlItemNum() {
  mutex_lock l(mu_);
  return flow_control_num_;
}

int64 BaseRemoteRendezvous::GetFlowControlItemNum(StringPiece tag) {
  const std::string tag_string(tag.data(), tag.size());
  mutex_lock l(mu_);
  if (flow_control_counters_.count(tag_string) == 0)
    return 0;
  return flow_control_counters_[tag_string];
}

void BaseRemoteRendezvous::StartAbort(const Status& s) {
  CHECK(!s.ok());
  // Use a "derived" status as the status for the rendezvous. Derived
  // status messages are ignored when aggregating errors across devices: this
  // allows us to prefer our original status message over any cancellation
  // related errors.
  Status derived_status = StatusGroup::MakeDerived(s);

  local_->StartAbort(derived_status);
  {
    // Aborts all active RecvTensor calls.
    mutex_lock l(mu_);
    if (status_.ok()) {
      status_ = derived_status;
      for (auto& entry : active_) {
        entry.first->StartAbort(derived_status);
        entry.second();
      }
      active_.clear();
    }
    flow_control_num_ = 0;
    flow_control_counters_.clear();
  }
  flow_control_cv_.notify_all();
}

void BaseRemoteRendezvous::RegisterCall(BaseRecvTensorCall* call,
                                        const Rendezvous::Args& args) {
  CancellationManager* cm = args.cancellation_manager;
  {
    mutex_lock l(mu_);
    if (!status_.ok()) {
      call->StartAbort(status_);
      return;
    }
    bool already_cancelled = false;
    InactiveCallback callback = [] {};
    if (cm != nullptr) {
      auto token = cm->get_cancellation_token();
      already_cancelled = !cm->RegisterCallback(token, [this, call] {
        {
          mutex_lock l(mu_);
          if (active_.find(call) == active_.end()) return;
          call->StartAbort(
              errors::Cancelled("RecvFromRemoteAsync is cancelled."));
        }
      });
      callback = [cm, token] { cm->TryDeregisterCallback(token); };
    }
    if (already_cancelled) {
      call->StartAbort(errors::Cancelled("RecvFromRemoteAsync is cancelled."));
    } else {
      CHECK(active_.emplace(call, callback).second);
    }
  }
}

void BaseRemoteRendezvous::DeregisterCall(BaseRecvTensorCall* call) {
  mutex_lock l(mu_);
  auto it = active_.find(call);
  if (it != active_.end()) {
    it->second();
    active_.erase(it);
  }
}

BaseRemoteRendezvous::DeferredCall::DeferredCall(const ParsedKey& parsed,
                                                 DoneCallback done)
    : parsed(parsed), done(std::move(done)) {}

BaseRemoteRendezvous::DeferredFuseCall::DeferredFuseCall(
    const std::vector<ParsedKey>& parsed_keys, FuseDoneCallback done)
    : parsed_keys(parsed_keys), done(std::move(done)) {}

BaseRemoteRendezvous::DeferredFlowControlCall::DeferredFlowControlCall(
    const StringPiece& tag, const ParsedKey& parsed, DoneCallback done)
    : tag(tag), parsed(parsed), done(std::move(done)) {}

}  // end namespace tensorflow
