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

#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_resource.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/macros.h"
#if TENSORFLOW_USE_STAR
#include "tensorflow/contrib/star/star_message.h"
#include "tensorflow/contrib/star/star_worker_interface.h"
#endif // TENSORFLOW_USE_STAR

namespace tensorflow {

class StarRunGraphOp : public AsyncOpKernel {
public:
  explicit StarRunGraphOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

private:
  std::string loc_;
  std::string graph_handle_;
  std::vector<std::string> feed_names_;
  std::vector<std::string> fetch_names_;
  std::vector<DataType> data_type_;
  int ps_graph_count_;
  WorkerInterface* worker_;
};

#define GET_ATTR(k, v) {                           \
  const NodeDef &def = ctx->def();                 \
  if (def.attr().find(#k) != def.attr().end()) {   \
    OP_REQUIRES_OK(ctx, ctx->GetAttr(#k, &v));     \
  }                                                \
}

StarRunGraphOp::StarRunGraphOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("loc", &loc_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("graph_handle", &graph_handle_));

  GET_ATTR(feed_names, feed_names_);
  GET_ATTR(fetch_names, fetch_names_);
  GET_ATTR(T1, data_type_);
  GET_ATTR(ps_graph_count, ps_graph_count_);

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

void StarRunGraphOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
#if TENSORFLOW_USE_STAR
  // NOTE(jiankeng.pt) Only grpc++ support zero copy now.
  if (ctx->num_inputs() != feed_names_.size()) {
    LOG(FATAL) << "Op input size must equal to feed_names size, "
               << ctx->num_inputs() << " .vs " << feed_names_.size();
  }
  if (ctx->num_outputs() != fetch_names_.size()) {
    LOG(FATAL) << "Op input size must equal to fetch_names size, "
               << ctx->num_inputs() << " .vs " << fetch_names_.size();
  }
  StarRunGraphRequest* req = new StarRunGraphRequest();
  req->graph_handle_ = graph_handle_;
  req->step_id_ = ctx->round_step_id();
  req->ps_graph_count_ = ps_graph_count_;
  req->feed_names_.resize(feed_names_.size());
  req->feed_tensors_.resize(feed_names_.size());
  req->is_dead_.resize(feed_names_.size());

  for (unsigned i = 0; i < feed_names_.size(); ++i) {
    req->feed_names_[i] = feed_names_[i];
    CHECK(!IsRefType(data_type_[i])) << "Data type is ref type.";
    req->feed_tensors_[i] = ctx->input(i);
    req->is_dead_[i] = ctx->is_input_dead(i);
  }

  std::unordered_map<std::string, int> fetch_name_idxs;
  req->fetch_names_.resize(fetch_names_.size());
  for (unsigned i = 0; i < fetch_names_.size(); ++i) {
    req->fetch_names_[i] = fetch_names_[i];
    fetch_name_idxs[fetch_names_[i]] = i;
  }

  StarRunGraphResponse* resp = new StarRunGraphResponse();
  resp->device_ = ctx->device();

  StarWorkerInterface* seastar_worker = dynamic_cast<StarWorkerInterface*>(worker_);
  if (seastar_worker == nullptr) {
    LOG(FATAL) << "Error worker in star run graph op. Required StarWorkerInterface.";
  }

  seastar_worker->StarRunGraphAsync(req, resp, [this, fetch_name_idxs,
                                                done, ctx, req, resp] (const Status& status) {
    ctx->SetStatus(status);
    if (status.ok()) {
      if (this->fetch_names_.size() != resp->fetch_names_.size()
          || this->fetch_names_.size() != resp->fetch_tensors_.size()) {
        ctx->SetStatus(errors::Internal("run graph failed. num_recv_fetch_names[",
                                        resp->fetch_names_.size() , "],  num_recv_fetch_tensors[",
                                        resp->fetch_tensors_.size(), "], num_fetch_count["
                                        , this->fetch_names_.size(), "]"));
        delete req;
        delete resp;
        done();
        return;
      }

      for (size_t i = 0; i < this->fetch_names_.size(); ++i) {
        auto it = fetch_name_idxs.find(resp->fetch_names_[i]);
        if (it == fetch_name_idxs.end()) {
          ctx->SetStatus(errors::Internal("can not find recv_key in fetch names, recv_key:",
                                          resp->fetch_names_[i]));
          break;
        }
        int idx = it->second;

        // NOTE(jiankeng.pt): ignore dead tensor
        if (!resp->is_dead_[i]) {
          ctx->set_output(idx, resp->fetch_tensors_[i]);
        }
      }
    }
    delete req;
    delete resp;
    done();
  });
#else
  OP_REQUIRES(ctx, false, errors::Unimplemented(
      "OP: 'StarRunGraph' works iff 'STAR' is enabled."));
#endif // TENSORFLOW_USE_STAR
}

REGISTER_KERNEL_BUILDER(Name("StarRunGraph")
                        .Device(DEVICE_CPU),
                        StarRunGraphOp);
REGISTER_KERNEL_BUILDER(Name("StarRunGraph")
                        .Device(DEVICE_GPU),
                        StarRunGraphOp);

}  // namespace tensorflow

