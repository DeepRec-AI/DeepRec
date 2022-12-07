/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <vector>
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/segment_reduction_ali_ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/util.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionMeanAliOp
    : public SparseSegmentReductionAliOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionMeanAliOp(OpKernelConstruction* context)
      : SparseSegmentReductionAliOpBase<Device, T, Index, SegmentId>(
            context, true /*is_mean*/, false /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionMeanWithNumSegmentsAliOp
    : public SparseSegmentReductionAliOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionMeanWithNumSegmentsAliOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionAliOpBase<Device, T, Index, SegmentId>(
            context, true /*is_mean*/, false /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionSqrtNAliOp
    : public SparseSegmentReductionAliOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionSqrtNAliOp(OpKernelConstruction* context)
      : SparseSegmentReductionAliOpBase<Device, T, Index, SegmentId>(
            context, false /*is_mean*/, true /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionSqrtNWithNumSegmentsAliOp
    : public SparseSegmentReductionAliOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionSqrtNWithNumSegmentsAliOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionAliOpBase<Device, T, Index, SegmentId>(
            context, false /*is_mean*/, true /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionSumAliOp
    : public SparseSegmentReductionAliOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionSumAliOp(OpKernelConstruction* context)
      : SparseSegmentReductionAliOpBase<Device, T, Index, SegmentId>(
            context, false /*is_mean*/, false /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionSumWithNumSegmentsAliOp
    : public SparseSegmentReductionAliOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentReductionSumWithNumSegmentsAliOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionAliOpBase<Device, T, Index, SegmentId>(
            context, false /*is_mean*/, false /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

#define REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, index_type) \
  REGISTER_CPU_SPARSE_KERNELS(type, index_type, int32)                         \
  REGISTER_CPU_SPARSE_KERNELS(type, index_type, int64)
#define REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(type)       \
  REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, int32) \
  REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, int64)

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type)       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SparseSegmentSum")                                                \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<type>("T")                                          \
          .TypeConstraint<index_type>("Tidx")                                 \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                   \
      SparseSegmentReductionSumAliOp<CPUDevice, type, index_type,                \
                                  segment_ids_type>);                         \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SparseSegmentSumWithNumSegments")                                 \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<type>("T")                                          \
          .TypeConstraint<index_type>("Tidx")                                 \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                   \
      SparseSegmentReductionSumWithNumSegmentsAliOp<CPUDevice, type, index_type, \
                                                 segment_ids_type>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type)        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("SparseSegmentMean")                                                \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<index_type>("Tidx")                                  \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                    \
      SparseSegmentReductionMeanAliOp<CPUDevice, type, index_type,                \
                                   segment_ids_type>);                         \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("SparseSegmentMeanWithNumSegments")                                 \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<index_type>("Tidx")                                  \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),                    \
      SparseSegmentReductionMeanWithNumSegmentsAliOp<CPUDevice, type, index_type, \
                                                  segment_ids_type>);
TF_CALL_FLOAT_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSqrtN")                                        \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentReductionSqrtNAliOp<CPUDevice, type, index_type,        \
                                    segment_ids_type>);                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSqrtNWithNumSegments")                         \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentReductionSqrtNWithNumSegmentsAliOp<                     \
          CPUDevice, type, index_type, segment_ids_type>);
TF_CALL_FLOAT_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS


template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentSumGradAliOp
    : public SparseSegmentGradAliOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentSumGradAliOp(OpKernelConstruction* context)
      : SparseSegmentGradAliOpBase<Device, T, Index, SegmentId>(
            context, SparseSegmentReductionOperation::kSum) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentMeanGradAliOp
    : public SparseSegmentGradAliOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentMeanGradAliOp(OpKernelConstruction* context)
      : SparseSegmentGradAliOpBase<Device, T, Index, SegmentId>(
            context, SparseSegmentReductionOperation::kMean) {}
};

template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentSqrtNGradAliOp
    : public SparseSegmentGradAliOpBase<Device, T, Index, SegmentId> {
 public:
  explicit SparseSegmentSqrtNGradAliOp(OpKernelConstruction* context)
      : SparseSegmentGradAliOpBase<Device, T, Index, SegmentId>(
            context, SparseSegmentReductionOperation::kSqrtN) {}
};

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSumGrad")                                      \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentSumGradAliOp<CPUDevice, type, index_type, segment_ids_type>);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(float);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(double);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentMeanGrad")                                     \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentMeanGradAliOp<CPUDevice, type, index_type, segment_ids_type>);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(float);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(double);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSqrtNGrad")                                    \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentSqrtNGradAliOp<CPUDevice, type, index_type,             \
                               segment_ids_type>);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(float);
REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(double);
#undef REGISTER_CPU_SPARSE_KERNELS

#undef REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE
#undef REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE

}  // namespace tensorflow