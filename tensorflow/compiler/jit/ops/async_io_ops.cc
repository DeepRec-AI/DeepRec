/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("_XlaAsyncOutSend")
    .Input("tensor: T")
    .Attr("T: type")
    .Attr("tensor_name: string")
    .Attr("device_name: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sends the named tensor to _XlaAsyncOutRecv on the same device.

tensor: The tensor to send.
tensor_name: The name of the tensor to send.
device_name: The name of the device where Send/Recv occur.
)doc");

REGISTER_OP("_XlaAsyncOutRecv")
    .Output("tensor: T")
    .Attr("T: type")
    .Attr("tensor_name: string")
    .Attr("device_name: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Receives the named tensor from _XlaAsyncOutSend on the same device.

tensor: The tensor to receive.
tensor_name: The name of the tensor to receive.
device_name: The name of the device where Send/Recv occur.
)doc");

REGISTER_OP("_XlaAsyncOutInit")
    .Attr("tensor_names: list(string)")
    .Attr("device_name: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Allocate rendezvous queues for send/recv.

tensor_names: used as keys to index rendezvous queues.
device_name: The name of the device where Send/Recv occur.
)doc");

REGISTER_OP("_XlaAsyncOutDone")
    .Attr("tensor_names: list(string)")
    .Attr("device_name: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Deallocate rendezvous queues for send/recv.

tensor_names: used as keys to index rendezvous queues.
device_name: The name of the device where Send/Recv occur.
)doc");

}  // namespace tensorflow
