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

#include <chrono>
#include <cstddef>
#include <deque>
#include <mutex>
#include <numeric>
#include <vector>

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("TensorBufferPut")
    .Input("record: dtypes")
    .Attr("container: string = ''")
    .Attr("dtypes: list(type)")
    .Attr("shared_name: string = ''")
    .Attr("shared_capacity: int >= 1 = 1")
    .Attr("timeout_millis: int >= 1 = 1000")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("TensorBufferTake")
    .Output("record: dtypes")
    .Attr("container: string = ''")
    .Attr("dtypes: list(type)")
    .Attr("shared_name: string = ''")
    .Attr("shared_capacity: int >= 1 = 1")
    .Attr("shared_threads: int >= 1 = 1")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("TensorBufferCancel")
    .Attr("container: string = ''")
    .Attr("is_cancelled: bool = true")
    .Attr("shared_name: string = ''")
    .Attr("shared_capacity: int >= 1 = 1")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("TensorBufferClose")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("shared_capacity: int >= 1 = 1")
    .SetShapeFn(shape_inference::UnknownShape)
    .SetIsStateful();

REGISTER_OP("TensorBufferSize")
    .Output("size: int32")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("shared_capacity: int >= 1 = 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .SetIsStateful();

}
