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
==============================================================================*/

#include "tensorflow/core/kernels/slice_sendrecv_utils.h"

namespace tensorflow {

namespace slice_sendrecv {

string GetSliceRendezvousKeyPrefix(const string& send_device,
                                   const string& recv_device,
                                   const uint64 send_device_incarnation,
                                   const string& tensor_name) {
  return strings::StrCat(send_device, ";",
                         strings::FpToString(send_device_incarnation), ";",
                         recv_device, ";", tensor_name);
}

void GetSliceRendezvousKey(const string& key_prefix,
                           const string& tensor_name_suffix,
                           const FrameAndIter& frame_iter, string* key) {
  key->clear();
  strings::StrAppend(key, key_prefix, tensor_name_suffix, ";",
                     frame_iter.frame_id, ":", frame_iter.iter_id);
}

FrameAndIter GetFrameAndIter(OpKernelContext* ctx, bool hostmem_sendrecv) {
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

}; // End of namespace slice_sendrecv

}; // End of namespace tensorflow
