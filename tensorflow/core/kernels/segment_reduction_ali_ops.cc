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
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// ____________________________________________________________________________
// Sparse segment reduction ops.

// Same as SegmentReductionOp but takes as input a "sparse" tensor, represented
// by two dense tensors, one containing the data, and the other containing
// indices into the data.
template <typename Device, class T, typename Tindex, typename Tsegment>
class SparseSegmentReductionAliOpBase : public OpKernel {
 public:
  explicit SparseSegmentReductionAliOpBase(OpKernelConstruction* context,
                                           bool is_mean, bool is_sqrtn,
                                           bool has_num_segments,
                                           T default_value)
      : OpKernel(context),
        has_num_segments_(has_num_segments),
        reducer_(is_mean, is_sqrtn, has_num_segments, default_value) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);

    Tensor num_segments;
    if (has_num_segments_) {
      num_segments = context->input(3);
    }

    Tensor output;
    reducer_.Reduce(context, input, indices, segment_ids, num_segments,
                    context->output_alloc_attr(0), &output);
    context->set_output(0, output);
  }

 private:
  const bool has_num_segments_;
  SparseSegmentReduction<Device, T, Tindex, Tsegment> reducer_;
};

template <typename Device, class T, typename Tindex, typename Tsegment>
class SparseSegmentReductionMeanAliOp
    : public SparseSegmentReductionAliOpBase<Device, T, Tindex, Tsegment> {
 public:
  explicit SparseSegmentReductionMeanAliOp(OpKernelConstruction* context)
      : SparseSegmentReductionAliOpBase<Device, T, Tindex, Tsegment>(
            context, true /*is_mean*/, false /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Tindex, typename Tsegment>
class SparseSegmentReductionMeanWithNumSegmentsAliOp
    : public SparseSegmentReductionAliOpBase<Device, T, Tindex, Tsegment> {
 public:
  explicit SparseSegmentReductionMeanWithNumSegmentsAliOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionAliOpBase<Device, T, Tindex, Tsegment>(
            context, true /*is_mean*/, false /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Tindex, typename Tsegment>
class SparseSegmentReductionSqrtNAliOp
    : public SparseSegmentReductionAliOpBase<Device, T, Tindex, Tsegment> {
 public:
  explicit SparseSegmentReductionSqrtNAliOp(OpKernelConstruction* context)
      : SparseSegmentReductionAliOpBase<Device, T, Tindex, Tsegment>(
            context, false /*is_mean*/, true /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Tindex, typename Tsegment>
class SparseSegmentReductionSqrtNWithNumSegmentsAliOp
    : public SparseSegmentReductionAliOpBase<Device, T, Tindex, Tsegment> {
 public:
  explicit SparseSegmentReductionSqrtNWithNumSegmentsAliOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionAliOpBase<Device, T, Tindex, Tsegment>(
            context, false /*is_mean*/, true /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Tindex, typename Tsegment>
class SparseSegmentReductionSumAliOp
    : public SparseSegmentReductionAliOpBase<Device, T, Tindex, Tsegment> {
 public:
  explicit SparseSegmentReductionSumAliOp(OpKernelConstruction* context)
      : SparseSegmentReductionAliOpBase<Device, T, Tindex, Tsegment>(
            context, false /*is_mean*/, false /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <typename Device, class T, typename Tindex, typename Tsegment>
class SparseSegmentReductionSumWithNumSegmentsAliOp
    : public SparseSegmentReductionAliOpBase<Device, T, Tindex, Tsegment> {
 public:
  explicit SparseSegmentReductionSumWithNumSegmentsAliOp(
      OpKernelConstruction* context)
      : SparseSegmentReductionAliOpBase<Device, T, Tindex, Tsegment>(
            context, false /*is_mean*/, false /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

#ifndef TF_API_COMPATIBLE_1150

#define REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, index_type) \
  REGISTER_CPU_SPARSE_KERNELS(type, index_type, int32)                         \
  REGISTER_CPU_SPARSE_KERNELS(type, index_type, int64)
#define REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(type)       \
  REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, int32) \
  REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, int64)

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSum")                                          \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentReductionSumAliOp<CPUDevice, type, index_type,       \
                                     segment_ids_type>);                \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSumWithNumSegments")                           \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentReductionSumWithNumSegmentsAliOp<                    \
          CPUDevice, type, index_type, segment_ids_type>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentMean")                                         \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentReductionMeanAliOp<CPUDevice, type, index_type,      \
                                      segment_ids_type>);               \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentMeanWithNumSegments")                          \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentReductionMeanWithNumSegmentsAliOp<                   \
          CPUDevice, type, index_type, segment_ids_type>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSqrtN")                                        \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentReductionSqrtNAliOp<CPUDevice, type, index_type,     \
                                       segment_ids_type>);              \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSqrtNWithNumSegments")                         \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentReductionSqrtNWithNumSegmentsAliOp<                  \
          CPUDevice, type, index_type, segment_ids_type>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#undef REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE
#undef REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE

#else

#define REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(type) \
  REGISTER_CPU_SPARSE_KERNELS(type, int32)                    \
  REGISTER_CPU_SPARSE_KERNELS(type, int64)

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type)                      \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("SparseSegmentSum")                                             \
          .Device(DEVICE_CPU)                                              \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<index_type>("Tidx"),                             \
      SparseSegmentReductionSumAliOp<CPUDevice, type, index_type, int32>); \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("SparseSegmentSumWithNumSegments")                              \
          .Device(DEVICE_CPU)                                              \
          .TypeConstraint<type>("T")                                       \
          .TypeConstraint<index_type>("Tidx"),                             \
      SparseSegmentReductionSumWithNumSegmentsAliOp<CPUDevice, type,       \
                                                    index_type, int32>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type)                       \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("SparseSegmentMean")                                             \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<type>("T")                                        \
          .TypeConstraint<index_type>("Tidx"),                              \
      SparseSegmentReductionMeanAliOp<CPUDevice, type, index_type, int32>); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("SparseSegmentMeanWithNumSegments")                              \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<type>("T")                                        \
          .TypeConstraint<index_type>("Tidx"),                              \
      SparseSegmentReductionMeanWithNumSegmentsAliOp<CPUDevice, type,       \
                                                     index_type, int32>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type)                        \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SparseSegmentSqrtN")                                             \
          .Device(DEVICE_CPU)                                                \
          .TypeConstraint<type>("T")                                         \
          .TypeConstraint<index_type>("Tidx"),                               \
      SparseSegmentReductionSqrtNAliOp<CPUDevice, type, index_type, int32>); \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SparseSegmentSqrtNWithNumSegments")                              \
          .Device(DEVICE_CPU)                                                \
          .TypeConstraint<type>("T")                                         \
          .TypeConstraint<index_type>("Tidx"),                               \
      SparseSegmentReductionSqrtNWithNumSegmentsAliOp<CPUDevice, type,       \
                                                      index_type, int32>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#undef REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE
#endif  // TF_API_COMPATIBLE_1150

template <class T, typename Tindex, typename Tsegment>
class SparseSegmentGradAliOpBase : public OpKernel {
 public:
  explicit SparseSegmentGradAliOpBase(OpKernelConstruction* context,
                                      bool is_sqrtn)
      : OpKernel(context), reducer_(is_sqrtn) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);
    const Tensor& output_dim0 = context->input(3);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices should be a vector."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
                errors::InvalidArgument("segment_ids should be a vector."));
    OP_REQUIRES(context, IsLegacyScalar(output_dim0.shape()),
                errors::InvalidArgument("output_dim0 should be a scalar."));

    Tensor output;
    reducer_.ReduceGrad<T, Tindex, Tsegment>(context, input, indices,
                                             segment_ids, output_dim0, output);
    context->set_output(0, output);
  }

 private:
  SparseSegmentReductionGrad reducer_;
};

template <class T, typename Tindex, typename Tsegment>
class SparseSegmentMeanGradAliOp
    : public SparseSegmentGradAliOpBase<T, Tindex, Tsegment> {
 public:
  explicit SparseSegmentMeanGradAliOp(OpKernelConstruction* context)
      : SparseSegmentGradAliOpBase<T, Tindex, Tsegment>(context,
                                                        false /*is_sqrtn*/) {}
};

template <class T, typename Tindex, typename Tsegment>
class SparseSegmentSqrtNGradAliOp
    : public SparseSegmentGradAliOpBase<T, Tindex, Tsegment> {
 public:
  explicit SparseSegmentSqrtNGradAliOp(OpKernelConstruction* context)
      : SparseSegmentGradAliOpBase<T, Tindex, Tsegment>(context,
                                                        true /*is_sqrtn*/) {}
};

#ifndef TF_API_COMPATIBLE_1150

#define REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, index_type) \
  REGISTER_CPU_SPARSE_KERNELS(type, index_type, int32)                         \
  REGISTER_CPU_SPARSE_KERNELS(type, index_type, int64)
#define REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(type)       \
  REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, int32) \
  REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE(type, int64)

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentMeanGrad")                                     \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentMeanGradAliOp<type, index_type, segment_ids_type>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type, segment_ids_type) \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSegmentSqrtNGrad")                                    \
          .Device(DEVICE_CPU)                                           \
          .TypeConstraint<type>("T")                                    \
          .TypeConstraint<index_type>("Tidx")                           \
          .TypeConstraint<segment_ids_type>("Tsegmentids"),             \
      SparseSegmentSqrtNGradAliOp<type, index_type, segment_ids_type>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#undef REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE
#undef REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_SEGMENT_ID_TYPE

#else

#define REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE(type) \
  REGISTER_CPU_SPARSE_KERNELS(type, int32)                    \
  REGISTER_CPU_SPARSE_KERNELS(type, int64)

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type) \
  REGISTER_KERNEL_BUILDER(                            \
      Name("SparseSegmentMeanGrad")                   \
          .Device(DEVICE_CPU)                         \
          .TypeConstraint<type>("T")                  \
          .TypeConstraint<index_type>("Tidx"),        \
      SparseSegmentMeanGradAliOp<type, index_type, int32>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#define REGISTER_CPU_SPARSE_KERNELS(type, index_type) \
  REGISTER_KERNEL_BUILDER(                            \
      Name("SparseSegmentSqrtNGrad")                  \
          .Device(DEVICE_CPU)                         \
          .TypeConstraint<type>("T")                  \
          .TypeConstraint<index_type>("Tidx"),        \
      SparseSegmentSqrtNGradAliOp<type, index_type, int32>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE);
#undef REGISTER_CPU_SPARSE_KERNELS

#undef REGISTER_CPU_SPARSE_KERNELS_FOR_EACH_INDEX_TYPE

#endif  // TF_API_COMPATIBLE_1150

}  // namespace tensorflow