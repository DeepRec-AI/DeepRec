/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <unordered_map>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_resource.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class RunGraphOp : public AsyncOpKernel {
public:
  explicit RunGraphOp(OpKernelConstruction* ctx);

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

private:
  std::string loc_;
  std::string graph_handle_;
  std::vector<std::string> feed_names_;
  std::vector<std::string> fetch_names_;
  std::vector<DataType> data_type_;
  WorkerInterface* worker_;
};

#define GET_ATTR(k, v) {                           \
  const NodeDef &def = ctx->def();                 \
  if (def.attr().find(#k) != def.attr().end()) {   \
    OP_REQUIRES_OK(ctx, ctx->GetAttr(#k, &v));     \
  }                                                \
}

RunGraphOp::RunGraphOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("loc", &loc_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("graph_handle", &graph_handle_));

  GET_ATTR(feed_names, feed_names_);
  GET_ATTR(fetch_names, fetch_names_);
  GET_ATTR(T1, data_type_);

  auto device = dynamic_cast<Device*>(ctx->device());
  OP_REQUIRES(ctx, device != nullptr, errors::Internal("Cast device failed."));
  ResourceMgr* resource_mgr = device->resource_manager();
  WorkerResource* worker_resource = nullptr;
  OP_REQUIRES_OK(ctx, resource_mgr->Lookup("worker_resource",
                 "worker_resource", &worker_resource));
  WorkerCacheInterface* worker_cache = worker_resource->worker_cache;
  WorkerInterface* worker = worker_cache->GetOrCreateWorker(loc_);
  OP_REQUIRES(ctx, worker != nullptr,
              errors::Internal("Create WorkerInterface failed"));
  worker_ = worker;
}

struct CallContext {
  CallContext() {
    req = nullptr;
    res = nullptr;
  }

  ~CallContext() {
    delete req;
    delete res;
  }

  CallOptions opts;
  MutableRunGraphRequestWrapper *req;
  MutableRunGraphResponseWrapper *res;
};

void RunGraphOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  OP_REQUIRES(ctx, ctx->num_inputs() == feed_names_.size(),
              errors::Internal("Op input size must equal to feed_names size, ",
              ctx->num_inputs(), " .vs ", feed_names_.size()));

  OP_REQUIRES(ctx, ctx->num_outputs() == fetch_names_.size(),
              errors::Internal("Op input size must equal to fetch_names size, ",
              ctx->num_inputs(), " .vs ", fetch_names_.size()));

  CallContext *call_ctx = new CallContext();
  call_ctx->req = worker_->CreateRunGraphRequest();
  call_ctx->res = worker_->CreateRunGraphResponse();

  call_ctx->req->set_graph_handle(graph_handle_);
  call_ctx->req->set_step_id(ctx->round_step_id());
  call_ctx->req->set_run_graph_mode(true);

  for(int i = 0; i < feed_names_.size(); i++) {
    CHECK(!IsRefType(data_type_[i])) << "Data type is ref type.";
    call_ctx->req->add_send(feed_names_[i], ctx->input(i));
  }

  std::unordered_map<std::string, int> fetch_name_idxs;
  for (int i = 0; i < fetch_names_.size(); i++) {
    call_ctx->req->add_recv_key(fetch_names_[i]);
    fetch_name_idxs[fetch_names_[i]] = i;
  }

  worker_->RunGraphAsync(&call_ctx->opts, call_ctx->req, call_ctx->res,
                         [ctx, call_ctx, this, done, fetch_name_idxs]
                         (const Status &status) {

    ctx->SetStatus(status);
    if (status.ok()) {
      if (this->fetch_names_.size() != call_ctx->res->num_recvs()) {
        ctx->SetStatus(errors::Internal("run graph failed. num_recvs[",
                       call_ctx->res->num_recvs(), "] != num_fetchs[",
                       this->fetch_names_.size(), "]"));
        delete call_ctx;
        done();
        return;
      }

      for (size_t i = 0; i < this->fetch_names_.size(); i++) {
        Tensor tensor;
        Status s = call_ctx->res->RecvValue(i, &tensor);
        if (!s.ok()) {
          ctx->SetStatus(s);
          break;
        }
        const string &fetch_name = call_ctx->res->recv_key(i);
        auto it = fetch_name_idxs.find(fetch_name);
        if (it == fetch_name_idxs.end()) {
          ctx->SetStatus(errors::Internal("can not find recv_key in fetch names, recv_key:", fetch_name));
          break;
        }
        int idx = it->second;

        ctx->set_output(idx, tensor);
      }
    }
      
    delete call_ctx;
    done();
  });
}

REGISTER_KERNEL_BUILDER(Name("RunGraph").Device(DEVICE_CPU), RunGraphOp);
REGISTER_KERNEL_BUILDER(Name("RunGraph").Device(DEVICE_GPU), RunGraphOp);

}  // namespace tensorflow

