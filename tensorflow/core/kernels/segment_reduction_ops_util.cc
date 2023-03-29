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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/core/kernels/segment_reduction_ops.h"

namespace tensorflow {

static void SegmentReductionValidationHelper(OpKernelContext* context,
                                             const Tensor& input,
                                             const Tensor& segment_ids) {
  OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
              errors::InvalidArgument("segment_ids should be a vector."));
  const int64 num_indices = segment_ids.NumElements();
  OP_REQUIRES(context, num_indices == input.dim_size(0),
              errors::InvalidArgument(
                  "segment_ids should be the same size as dimension 0 of"
                  " input."));
}

bool SegmentReductionDoValidation(OpKernelContext* c,
				  const Tensor& input,
				  const Tensor& segment_ids) {
  SegmentReductionValidationHelper(c, input, segment_ids);
  return c->status().ok();
}

static void UnsortedSegmentReductionValidation(OpKernel* op_kernel,
					       OpKernelContext* context,
					       const Tensor& data,
					       const Tensor& segment_ids,
					       const Tensor& num_segments) {
  OP_REQUIRES(
      context, op_kernel->IsLegacyScalar(num_segments.shape()),
      errors::InvalidArgument("num_segments should be a scalar, not shape ",
                              num_segments.shape().DebugString()));
  OP_REQUIRES(
      context, TensorShapeUtils::StartsWith(data.shape(), segment_ids.shape()),
      errors::InvalidArgument("data.shape = ", data.shape().DebugString(),
                              " does not start with segment_ids.shape = ",
                              segment_ids.shape().DebugString()));
}

bool UnsortedSegmentReductionDoValidation(OpKernel* op_kernel,
					  OpKernelContext* context,
					  const Tensor& data,
					  const Tensor& segment_ids,
					  const Tensor& num_segments) {
  UnsortedSegmentReductionValidation(op_kernel, context, data, segment_ids,
                                     num_segments);
  return context->status().ok();
}

} // End of namespace tensorflow
