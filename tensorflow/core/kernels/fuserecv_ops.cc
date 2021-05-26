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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/fuserecv_ops.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

static string GetRendezvousKeyPrefix(const string& send_device,
                                     const string& recv_device,
                                     const uint64 send_device_incarnation,
                                     const string& tensor_name) {
  return strings::StrCat(send_device, ";",
                         strings::FpToString(send_device_incarnation), ";",
                         recv_device, ";", tensor_name);
}

static void GetRendezvousKey(const string& key_prefix,
                             const FrameAndIter& frame_iter, string* key) {
  key->clear();
  strings::StrAppend(key, key_prefix, ";", frame_iter.frame_id, ":",
                     frame_iter.iter_id);
}

static FrameAndIter GetFrameAndIter(OpKernelContext* ctx,
                                    bool hostmem_sendrecv) {
  if (hostmem_sendrecv && ctx->call_frame() != nullptr) {
    // Host memory send/recv pairs are added by
    // common_runtime/memory_types.cc.  When the pair of nodes are
    // added inside a function, we need to use the function call frame
    // to formulate the unique rendezvous key.
    return FrameAndIter(reinterpret_cast<uint64>(ctx->call_frame()), 0);
  } else {
    return ctx->frame_iter();
  }
}

FuseRecvOp::FuseRecvOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx), fuse_count_(0) {
  // send devices are not single
  std::vector<string> tensor_names;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_names", &tensor_names));

  std::vector<string> send_devices;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("send_devices", &send_devices));

  std::vector<int64> send_device_incarnations;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device_incarnations",
                                   &send_device_incarnations));

  std::vector<string> recv_devices;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_devices", &recv_devices));

  fuse_count_ = tensor_names.size();
  key_prefixs_.resize(fuse_count_);
  parsed_keys_.resize(fuse_count_);

  for (int i = 0; i < fuse_count_; ++i) {
    key_prefixs_[i] = GetRendezvousKeyPrefix(
        send_devices[i], recv_devices[i],
        static_cast<uint64>(send_device_incarnations[i]),
        tensor_names[i]);
    // The vast majority of Recv nodes are outside any loop context, so
    // proactively cache the rendezvous key for the top-level.
    GetRendezvousKey(key_prefixs_[i], {0, 0}, &parsed_keys_[i].buf_);
    OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(parsed_keys_[i].buf_, &parsed_keys_[i]));
  }

  if (!ctx->GetAttr("_hostmem_sendrecv", &hostmem_sendrecv_).ok()) {
    hostmem_sendrecv_ = false;
  }
}

void FuseRecvOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  //LOG(INFO) << "FuseRecvOp::ComputeAsync";
  OP_REQUIRES(
      ctx, ctx->rendezvous() != nullptr,
      errors::Internal("Op kernel context needs to provide a rendezvous."));

  // NOTE(rangeng.llb): recv args.
  Rendezvous::Args args;
  args.device_context = ctx->op_device_context();
  args.alloc_attrs = ctx->output_alloc_attr(0);

  using namespace std::placeholders;
  Rendezvous::FuseDoneCallback done_cb = std::bind(
      [ctx](DoneCallback done,
            // Begin unbound arguments.
            const Status& s, const std::vector<Rendezvous::Args>& send_args,
            const Rendezvous::Args& recv_args, const std::vector<Tensor>& vals,
            const std::vector<bool>& is_deads) {
        ctx->SetStatus(s);
        if (s.ok()) {
          OpOutputList output;
          OP_REQUIRES_OK(ctx, ctx->output_list("tensors", &output));

          bool all_deads = true;
          for (int i = 0; i < vals.size(); ++i) {
            if (!is_deads[i]) {
              output.set(i, vals[i]);
              all_deads = false;
            }
          }
          // is_output_dead means all the outputs is dead.
          *ctx->is_output_dead() = all_deads;
        }
        done();
      },
      std::move(done), _1, _2, _3, _4, _5);

  FrameAndIter frame_iter = GetFrameAndIter(ctx, hostmem_sendrecv_);
  if (!(frame_iter == FrameAndIter(0, 0))) {
    // if frame_iter has changed, change the parsed keys.
    for (int i = 0; i < fuse_count_; ++i) {
      GetRendezvousKey(key_prefixs_[i], frame_iter, &parsed_keys_[i].buf_);
      OP_REQUIRES_OK_ASYNC(ctx,
                           Rendezvous::ParseKey(parsed_keys_[i].buf_, &parsed_keys_[i]),
                           done);
    }
  }
  ctx->rendezvous()->FuseRecvAsync(parsed_keys_, args, std::move(done_cb));
}

REGISTER_KERNEL_BUILDER(Name("_FuseRecv").Device(DEVICE_CPU), FuseRecvOp);
REGISTER_KERNEL_BUILDER(Name("_FuseRecv").Device(DEVICE_GPU), FuseRecvOp);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("_FuseRecv").Device(DEVICE_SYCL), FuseRecvOp);
#endif // TENSORFLOW_USE_SYCL

REGISTER_KERNEL_BUILDER(Name("_HostFuseRecv").Device(DEVICE_CPU), FuseRecvOp);
REGISTER_KERNEL_BUILDER(
    Name("_HostFuseRecv").Device(DEVICE_GPU).HostMemory("tensors"), FuseRecvOp);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(
    Name("_HostFuseRecv").Device(DEVICE_SYCL).HostMemory("tensors"), FuseRecvOp);
#endif // TENSORFLOW_USE_SYCL

}  // end namespace tensorflow
