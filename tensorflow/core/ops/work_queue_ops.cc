// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset_stateful_op_whitelist.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;

REGISTER_RESOURCE_HANDLE_OP(WorkQueue);

REGISTER_OP("WorkQueueIsInitialized")
    .Output("is_initialized: bool")
    .Input("handle: resource")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Checks whether a work queue has been initialized.

is_initialized: True if the work queue is initialized.
handle: Handle of a work queue.
)doc");

REGISTER_OP("WorkQueueCreate")
    .Input("handle: resource")
    .Attr("shared_name: string")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(R"doc(
Creates a work queue and returns a handle to it.

handle: Handle of a work queue.
shared_name: Name of the work queue.
)doc");

REGISTER_OP("WorkQueueClose")
    .Input("handle: resource")
    .SetShapeFn(shape_inference::NoOutputs)
    .SetIsStateful()
    .Doc(R"doc(
Closes a work queue.

handle: Handle of a work queue.
)doc");

REGISTER_OP("WorkQueueRestore")
    .Input("handle: resource")
    .Input("works: string")
    .SetShapeFn(shape_inference::NoOutputs)
    .SetIsStateful()
    .Doc(R"doc(
Recovers a work queue from saved tensor.

handle: Handle of a work queue.
works: A tensor containing works.
)doc");

REGISTER_OP("WorkQueueSave")
    .Output("works: string")
    .Input("handle: resource")
    .SetShapeFn(shape_inference::ScalarShape)
    .SetIsStateful()
    .Doc(R"doc(
Saves a work queue to tensor.

works: A tensor containing works.
handle: Handle of a work queue.
)doc");

REGISTER_OP("WorkQueueSize")
    .Output("size: int64")
    .Input("handle: resource")
    .SetShapeFn(shape_inference::ScalarShape)
    .SetIsStateful()
    .Doc(R"doc(
Gets size of a work queue.

size: A scalar tensor.
handle: Handle of a work queue.
)doc");

REGISTER_OP("WorkQueuePut")
    .Input("handle: resource")
    .Input("works: string")
    .SetShapeFn(shape_inference::NoOutputs)
    .SetIsStateful()
    .Doc(R"doc(
Puts works to a work queue.

handle: Handle of a work queue.
works: A tensor containing works.
)doc");

REGISTER_OP("WorkQueueTake")
    .Input("handle: resource")
    .Output("work: string")
    .Attr("num_clients: int >= 1 = 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .SetIsStateful()
    .Doc(R"doc(
Take a work from the work queue.

handle: Handle of a work queue.
work: A tensor of taken work.
num_clients:  Number of threads for taking works.
)doc");

REGISTER_OP("SaveLocalWork")
    .Input("work: string")
    .Attr("job_name: string = ''")
    .Attr("task_index: int >= 0 = 0")
    .Attr("restore_works_dir: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      // Make sure that the input is a scalar.
      if (c->Rank(c->input(0)) != 0) {
        return errors::InvalidArgument("input must be a scalar, but has rank: ",
                                       c->Rank(c->input(0)));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Save work to file that will be used when failover for inference job.

work: a tensor containing work.
job_name: name of current tf-worker.
task_index: index of current tf-worker.
restore_works_dir: a directory that restore works for WorkQueue when failover.
)doc");

WHITELIST_STATEFUL_OP_FOR_DATASET_FUNCTIONS("QueueDequeueV2");

} // tensorflow
