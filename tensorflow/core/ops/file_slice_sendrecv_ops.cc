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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {
REGISTER_OP("_FileSliceSend")
    .Input("file_path: string")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .Attr("client_terminated: bool = false")
    .Attr("slice_size: int >= 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the file from send_device to recv_device.
Supports sending the file of any size.

file_path: The file to send.
tensor_name: The name of the tensor to send.
send_device: The name of the device sending the tensor.
send_device_incarnation: The current incarnation of send_device.
recv_device: The name of the device receiving the tensor.
client_terminated: If set to true, this indicates that the node was added
  to the graph as a result of a client-side feed or fetch of Tensor data,
  in which case the corresponding send or recv is expected to be managed
  locally by the caller.
slice_size: The maximum number of bytes transferred at one time.
)doc");

REGISTER_OP("_FileSliceRecv")
    .Output("file_path: string")
    .Attr("tensor_name: string")
    .Attr("send_device: string")
    .Attr("send_device_incarnation: int")
    .Attr("recv_device: string")
    .Attr("client_terminated: bool = false")
    .Attr("recv_dir: string")
    .Attr("slice_size: int >= 1")
    .Attr("timeout_ms: int >= 0 = 300000")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Receives the file from send_device on recv_device.
Supports recving the file of any size.

file_path: The file to receive.
tensor_name: The name of the tensor to receive.
send_device: The name of the device sending the tensor.
send_device_incarnation: The current incarnation of send_device.
recv_device: The name of the device receiving the tensor.
client_terminated: If set to true, this indicates that the node was added
  to the graph as a result of a client-side feed or fetch of Tensor data,
  in which case the corresponding send or recv is expected to be managed
  locally by the caller.
recv_dir: the directory to store received file.
slice_size: The maximum number of bytes transferred at one time.
timeout_ms: The maximum wait time for receiving a tensor.
)doc");

}; // End of namespace tensorflow
