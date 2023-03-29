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

#ifndef TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_UTIL_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

extern bool SegmentReductionDoValidation(OpKernelContext* c,
				  const Tensor& input,
				  const Tensor& segment_ids);

extern bool UnsortedSegmentReductionDoValidation(OpKernel* op_kernel,
                                          OpKernelContext* context,
                                          const Tensor& data,
                                          const Tensor& segment_ids,
                                          const Tensor& num_segments);

}  // End of namespace tensorflow

#endif // End of TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_OPS_UTIL_H_
