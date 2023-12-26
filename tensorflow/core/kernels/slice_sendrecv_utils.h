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

#ifndef TENSORFLOW_CORE_KERNELS_SLICE_SENDRECV_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_SLICE_SENDRECV_UTILS_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace slice_sendrecv {

extern string GetSliceRendezvousKeyPrefix(const string& send_device,
                                          const string& recv_device,
                                          const uint64 send_device_incarnation,
                                          const string& tensor_name);

extern void GetSliceRendezvousKey(const string& key_prefix,
                                  const string& tensor_name_suffix,
                                  const FrameAndIter& frame_iter, string* key);

extern FrameAndIter GetFrameAndIter(OpKernelContext* ctx,
                                    bool hostmem_sendrecv);

}; // End of namespace slice_sendrecv

}; // End of namespace tensorflow

#endif // End of macro TENSORFLOW_CORE_KERNELS_SLICE_SENDRECV_UTILS_H_
