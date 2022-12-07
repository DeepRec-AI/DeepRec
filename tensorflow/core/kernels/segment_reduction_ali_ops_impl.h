#ifndef TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_ALI_OPS_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_ALI_OPS_UTIL_H_

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/util.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

enum class SparseSegmentReductionOperation { kSum, kMean, kSqrtN };

namespace internal {
// Status ValidateSegmentReduction(OpKernelContext* c, const Tensor& input,
//                                 const Tensor& segment_ids);
// Status ValidateUnsortedSegmentReduction(OpKernel* op_kernel,
//                                         OpKernelContext* context,
//                                         const Tensor& data,
//                                         const Tensor& segment_ids,
//                                         const Tensor& num_segments);

Status ValidateSparseSegmentReduction(OpKernelContext* context,
                                      const Tensor& input,
                                      const Tensor& indices,
                                      const Tensor& segment_ids,
                                      bool has_num_segments) {
  if (has_num_segments) {
    const Tensor& num_segments_t = context->input(3);
    if (!TensorShapeUtils::IsScalar(num_segments_t.shape())) {
      return errors::InvalidArgument(
          "num_segments should be a scalar, not shape ",
          num_segments_t.shape().DebugString());
    }
    int64_t output_rows =
        internal::SubtleMustCopy(num_segments_t.dtype() == DT_INT32
                                     ? num_segments_t.scalar<int32>()()
                                     : num_segments_t.scalar<int64_t>()());
    if (output_rows < 0) {
      return errors::InvalidArgument("segment ids must be >= 0");
    }
  }

  if (!TensorShapeUtils::IsVector(indices.shape())) {
    return errors::InvalidArgument("indices should be a vector.");
  }

  if (!TensorShapeUtils::IsVector(segment_ids.shape())) {
    return errors::InvalidArgument("segment_ids should be a vector.");
  }

  const int64_t num_indices = indices.NumElements();
  if (num_indices != segment_ids.NumElements()) {
    return errors::InvalidArgument(
        "segment_ids and indices should have same size.");
  }

  if (input.dims() < 1) {
    return errors::InvalidArgument("Shape must be at least rank 1");
  }

  return Status::OK();
}
}  // namespace internal

// ____________________________________________________________________________
// Sparse segment reduction ops.

// Same as SegmentReductionOp but takes as input a "sparse" tensor, represented
// by two dense tensors, one containing the data, and the other containing
// indices into the data.
//
// The template parameters are:
// * Device: An Eigen device object, on which the kernel will execute.
// * T: The value type.
// * Index: The element type of the indices tensor (int32 or int64).
// * SegmentId: The element type of the segment_ids tensor (int32 or int64).
template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentReductionAliOpBase : public OpKernel {
 public:
  explicit SparseSegmentReductionAliOpBase(OpKernelConstruction* context,
                                        bool is_mean, bool is_sqrtn,
                                        bool has_num_segments, T default_value)
      : OpKernel(context),
        is_mean_(is_mean),
        is_sqrtn_(is_sqrtn),
        has_num_segments_(has_num_segments),
        default_value_(default_value) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);

    OP_REQUIRES_OK(
        context, internal::ValidateSparseSegmentReduction(
                     context, input, indices, segment_ids, has_num_segments_));

    Index output_rows = -1;
    if (has_num_segments_) {
      const Tensor& num_segments = context->input(3);
      // Note that there is a Tnumsegments parameter on the op, but it is not
      // plumbed through to here and so always takes its default value of int32.
      output_rows = internal::SubtleMustCopy(num_segments.scalar<int32>()());
    }
    const int64_t num_indices = indices.NumElements();

    auto input_flat = input.flat_outer_dims<T>();
    const int64_t num_col = input_flat.dimension(1);
    const auto indices_vec = indices.vec<Index>();
    const auto segment_vec = segment_ids.vec<SegmentId>();
    // Note that the current implementation assumes that segment_vec values are
    // sorted.
    const SegmentId last_segment_id_plus_one =
        num_indices > 0
            ? internal::SubtleMustCopy(segment_vec(num_indices - 1)) + 1
            : 0;
    if (has_num_segments_) {
      OP_REQUIRES(
          context, output_rows >= last_segment_id_plus_one,
          errors::InvalidArgument("segment ids must be < num_segments"));
    } else {
      output_rows = last_segment_id_plus_one;
    }
    OP_REQUIRES(context, output_rows >= 0,
                errors::InvalidArgument("segment ids must be >= 0"));

    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, output_rows);

    // Note that we do not initialize the output buffer with a default value, so
    // we need to explicitly set missing indices to the default value.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (num_indices == 0) {
      if (output_rows > 0) {
        output->flat_outer_dims<T>().setConstant(default_value_);
      }
      return;
    }
    OP_REQUIRES(context, output_rows > 0,
                errors::InvalidArgument("segment ids must be >= 0"));
    auto output_flat = output->flat_outer_dims<T>();

    Tensor temp;
    if (input.dtype() == DT_BFLOAT16 || input.dtype() == DT_HALF) {
      temp = tensorflow::Tensor(DT_FLOAT, output_shape);
    }
    auto temp_flat = temp.flat_outer_dims<float>();

    int64_t start = 0, end = 1;
    // Index from which the output is not initialized.
    SegmentId uninitialized_index = 0;
    SegmentId out_index = internal::SubtleMustCopy(segment_vec(start));

    while (true) {
      // We initialize next_index to 0 to avoid "warning: 'next_index' may be
      // used uninitialized in this function" in the Mac build (since the
      // compiler isn't smart enough to realize the code is safe).
      SegmentId next_index = 0;
      if (end < num_indices) {
        next_index = internal::SubtleMustCopy(segment_vec(end));
        if (out_index == next_index) {
          ++end;
          continue;
        }
        // We have a new segment here.  Verify that the segment ids are growing.
        OP_REQUIRES(context, out_index < next_index,
                    errors::InvalidArgument("segment ids are not increasing"));
      }

      OP_REQUIRES(
          context, FastBoundsCheck(out_index, output_rows),
          errors::InvalidArgument(
              "Segment id ", out_index, " out of range [0, ", output_rows,
              "), possibly because 'segment_ids' input is not sorted."));

      // If there is a gap between two indices, we need to set that gap to the
      // default value.
      if (out_index > uninitialized_index) {
        Eigen::DSizes<Eigen::DenseIndex, 2> gap_slice_shape(
            out_index - uninitialized_index, num_col);
        Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Unaligned>
            gap_slice(&output_flat(uninitialized_index, 0), gap_slice_shape);
        gap_slice.setConstant(default_value_);
      }

      auto out = output_flat.template chip<0>(out_index);
      auto temp = temp_flat.template chip<0>(out_index);
      const int bad_offset = Reduce<T, Index>(input_flat, indices_vec, start,
                                              end - start, out, temp);
      OP_REQUIRES(context, bad_offset < 0,
                  errors::InvalidArgument(
                      "Bad: indices[", start + bad_offset,
                      "] == ", indices_vec(start + bad_offset),
                      " out of range [0, ", input_flat.dimension(0), ")"));

      start = end;
      ++end;
      uninitialized_index = out_index + 1;
      out_index = next_index;
      if (end > num_indices) break;
    }

    // Fill the gap at the end with the default value.
    if (uninitialized_index < output_rows) {
      Eigen::DSizes<Eigen::DenseIndex, 2> gap_slice_shape(
          output_rows - uninitialized_index, num_col);
      Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Unaligned>
          gap_slice(&output_flat(uninitialized_index, 0), gap_slice_shape);
      gap_slice.setConstant(default_value_);
    }
  }

 private:
  template <typename Tin>
  using EnableIfBfloat16OrHalf =
      typename std::enable_if<std::is_same<Tin, bfloat16>::value ||
                                  std::is_same<Tin, Eigen::half>::value,
                              int>::type;
  template <typename Tin>
  using EnableIfNotBfloat16OrHalf =
      typename std::enable_if<!std::is_same<Tin, bfloat16>::value &&
                                  !std::is_same<Tin, Eigen::half>::value,
                              int>::type;

  template <typename Tin, typename Tindex, EnableIfNotBfloat16OrHalf<Tin> = 0>
  EIGEN_ALWAYS_INLINE auto fetch_val(
      const typename TTypes<Tin>::ConstMatrix& input_flat, Tindex index) {
    return input_flat.template chip<0>(index);
  }

  template <typename Tin, typename Tindex, EnableIfBfloat16OrHalf<Tin> = 0>
  EIGEN_ALWAYS_INLINE auto fetch_val(
      const typename TTypes<Tin>::ConstMatrix& input_flat, Tindex index) {
    return input_flat.template chip<0>(index).template cast<float>();
  }

  template <typename Tout>
  EIGEN_ALWAYS_INLINE Tout get_scaling_factor(int64_t num) {
    Tout m(1);
    if (is_mean_ && (num < 10)) {
      m = Tout(num);
    }
    if (is_sqrtn_ && (num < 10)) {
      m = Tout(sqrt(num));
    }
    return Tout(1) / m;
  }

  template <typename Tin, typename Tindex, EnableIfNotBfloat16OrHalf<Tin> = 0>
  int64_t Reduce(
      const typename TTypes<Tin>::ConstMatrix& input_flat,
      const typename TTypes<Tindex>::ConstVec& indices_vec, int64_t start,
      int64_t num, Eigen::TensorChippingOp<0, typename TTypes<Tin>::Matrix> out,
      Eigen::TensorChippingOp<0, typename TTypes<float>::Matrix> temp) {
    return ReduceImpl<Tin, Tindex, Tin>(input_flat, indices_vec, start, num,
                                        out, get_scaling_factor<Tin>(num));
  }

  template <typename Tin, typename Tindex, EnableIfBfloat16OrHalf<Tin> = 0>
  int64_t Reduce(
      const typename TTypes<Tin>::ConstMatrix& input_flat,
      const typename TTypes<Tindex>::ConstVec& indices_vec, int64_t start,
      int64_t num, Eigen::TensorChippingOp<0, typename TTypes<Tin>::Matrix> out,
      Eigen::TensorChippingOp<0, typename TTypes<float>::Matrix> temp) {
    int64_t res =
        ReduceImpl<Tin, Tindex, float>(input_flat, indices_vec, start, num,
                                       temp, get_scaling_factor<float>(num));
    out = temp.template cast<Tin>();
    return res;
  }

  template <typename Tin, typename Tindex, typename Tout>
  int64_t ReduceImpl(
      const typename TTypes<Tin>::ConstMatrix& input_flat,
      const typename TTypes<Tindex>::ConstVec& indices_vec, int64_t start,
      int64_t num,
      Eigen::TensorChippingOp<0, typename TTypes<Tout>::Matrix> out,
      const Tout scaling_factor) {
#define INDEX(n, i)                               \
  const auto index##n = indices_vec(start + (i)); \
  if (!FastBoundsCheck(index##n, input_flat.dimension(0))) return (i);

#define L(n) fetch_val<Tin, Tindex>(input_flat, index##n)

    if (num == 1) {
      INDEX(0, 0);
      out = L(0);
    } else {
      int64_t r = num & 7;
      switch (r) {
        case 2: {
          INDEX(0, 0);
          INDEX(1, 1);
          out = (L(0) + L(1)) * scaling_factor;
          break;
        }
        case 3: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          out = (L(0) + L(1) + L(2)) * scaling_factor;
          break;
        }
        case 4: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          out = (L(0) + L(1) + L(2) + L(3)) * scaling_factor;
          break;
        }
        case 5: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          out = (L(0) + L(1) + L(2) + L(3) + L(4)) * scaling_factor;
          break;
        }
        case 6: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5)) * scaling_factor;
          break;
        }
        case 7: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          INDEX(6, 6);
          out =
              (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6)) * scaling_factor;
          break;
        }
        case 0: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          INDEX(6, 6);
          INDEX(7, 7);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7)) *
                scaling_factor;
          r = 8;
          break;
        }
        case 1: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          INDEX(6, 6);
          INDEX(7, 7);
          INDEX(8, 8);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7) + L(8)) *
                scaling_factor;
          r = 9;
          break;
        }
      }
      for (; r < num; r += 8) {
        INDEX(0, r);
        INDEX(1, r + 1);
        INDEX(2, r + 2);
        INDEX(3, r + 3);
        INDEX(4, r + 4);
        INDEX(5, r + 5);
        INDEX(6, r + 6);
        INDEX(7, r + 7);
        out += L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7);
      }
      if (is_mean_ && num >= 10) {
        out = out / static_cast<Tout>(num);
      }
      if (is_sqrtn_ && num >= 10) {
        out = out / static_cast<Tout>(sqrt(num));
      }
    }

    return -1;
#undef L
#undef INDEX
  }

  const bool is_mean_;
  const bool is_sqrtn_;
  const bool has_num_segments_;
  const T default_value_;
};


namespace functor {

template <class Device, typename T, typename Index, typename SegmentId>
struct SparseSegmentGradFunctor {
  void operator()(OpKernelContext* context,
                  SparseSegmentReductionOperation operation,
                  typename TTypes<T>::ConstMatrix input_flat,
                  typename TTypes<Index>::ConstVec indices_vec,
                  typename TTypes<SegmentId>::ConstVec segment_vec,
                  typename TTypes<T>::Matrix output_flat);
};

template <typename T, typename Index, typename SegmentId>
struct SparseSegmentGradFunctor<CPUDevice, T, Index, SegmentId> {
  void operator()(OpKernelContext* context,
                  SparseSegmentReductionOperation operation,
                  typename TTypes<T>::ConstMatrix input_flat,
                  typename TTypes<Index>::ConstVec indices_vec,
                  typename TTypes<SegmentId>::ConstVec segment_vec,
                  typename TTypes<T>::Matrix output_flat) {
    const int64_t N = indices_vec.size();
    const SegmentId M = output_flat.dimension(0);

    // Note that similar to SparseSegmentMean, we assume that segment_vec is
    // already sorted and has non-negative values.
    const SegmentId num_segments = input_flat.dimension(0);
    const SegmentId last_segment_id_plus_one =
        internal::SubtleMustCopy(segment_vec(N - 1)) + 1;
    OP_REQUIRES(context, last_segment_id_plus_one <= num_segments,
                errors::InvalidArgument("Invalid number of segments"));

    // Compute scaling factors for input.
    std::vector<double> scaling(
        (operation == SparseSegmentReductionOperation::kSum ? 0 : num_segments),
        0.0);
    if (operation != SparseSegmentReductionOperation::kSum) {
      for (int64_t i = 0; i < N; ++i) {
        const SegmentId idx = internal::SubtleMustCopy(segment_vec(i));
        OP_REQUIRES(
            context, FastBoundsCheck(idx, num_segments),
            errors::InvalidArgument("Segment id ", idx, " out of range [0, ",
                                    num_segments, ")."));
        scaling[idx] += 1;
      }
      for (size_t i = 0; i < scaling.size(); ++i) {
        switch (operation) {
          case SparseSegmentReductionOperation::kSum: {
            OP_REQUIRES(
                context, false,
                errors::Internal(
                    "Should not happen: sum inside SparseSegmentReductionOp "
                    "scaling generation."));
          }
          case SparseSegmentReductionOperation::kMean: {
            scaling[i] = 1.0 / std::max(scaling[i], 1.0);
            break;
          }
          case SparseSegmentReductionOperation::kSqrtN: {
            scaling[i] = 1.0 / sqrt(std::max(scaling[i], 1.0));
            break;
          }
            // No default to get compiler warnings for missing cases.
        }
      }
    }

    output_flat.setZero();
    std::vector<bool> is_modified(M, false);

    for (int64_t i = 0; i < N; ++i) {
      const Index output_idx = internal::SubtleMustCopy(indices_vec(i));
      OP_REQUIRES(context, FastBoundsCheck(output_idx, M),
                  errors::InvalidArgument("Index ", output_idx,
                                          " out of range [0, ", M, ")."));

      const SegmentId idx = internal::SubtleMustCopy(segment_vec(i));
      OP_REQUIRES(
          context, FastBoundsCheck(idx, num_segments),
          errors::InvalidArgument("Segment id ", idx, " out of range [0, ",
                                  num_segments, ")."));

      const T scale = (operation == SparseSegmentReductionOperation::kSum
                           ? static_cast<T>(1)
                           : static_cast<T>(scaling[idx]));
      if (is_modified[output_idx]) {
        if (scale == 1.0) {
          output_flat.template chip<0>(output_idx) +=
              input_flat.template chip<0>(idx);
        } else {
          output_flat.template chip<0>(output_idx) +=
              input_flat.template chip<0>(idx) * scale;
        }
      } else {
        if (scale == 1.0) {
          output_flat.template chip<0>(output_idx) =
              input_flat.template chip<0>(idx);
        } else {
          output_flat.template chip<0>(output_idx) =
              input_flat.template chip<0>(idx) * scale;
        }
      }
      is_modified[output_idx] = true;
    }
  }
};

}  // namespace functor

// Implements the common logic for the gradients of SparseSegmentReduction
// kernels.
//
// The template parameters are:
// * Device: An Eigen device object, on which the kernel will execute.
// * T: The value type.
// * Index: The element type of the indices tensor (int32 or int64).
// * SegmentId: The element type of the segment_ids tensor (int32 or int64).
template <typename Device, class T, typename Index, typename SegmentId>
class SparseSegmentGradAliOpBase : public OpKernel {
 public:
  explicit SparseSegmentGradAliOpBase(OpKernelConstruction* context,
                                   SparseSegmentReductionOperation operation)
      : OpKernel(context), operation_(operation) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);
    const Tensor& output_dim0 = context->input(3);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices should be a vector."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
                errors::InvalidArgument("segment_ids should be a vector."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(output_dim0.shape()),
                errors::InvalidArgument("output_dim0 should be a scalar."));

    const int64_t N = indices.NumElements();
    OP_REQUIRES(context, N == segment_ids.NumElements(),
                errors::InvalidArgument(
                    "segment_ids and indices should have same size."));
    const SegmentId M = internal::SubtleMustCopy(output_dim0.scalar<int32>()());

    auto input_flat = input.flat_outer_dims<T>();
    const auto indices_vec = indices.vec<Index>();
    const auto segment_vec = segment_ids.vec<SegmentId>();

    TensorShape output_shape = input.shape();
    OP_REQUIRES_OK(context, output_shape.SetDimWithStatus(0, M));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (M == 0 || N == 0) return;

    auto output_flat = output->flat_outer_dims<T>();
    functor::SparseSegmentGradFunctor<Device, T, Index, SegmentId>()(
        context, operation_, input_flat, indices_vec, segment_vec, output_flat);
  }
 private:
  const SparseSegmentReductionOperation operation_;
};

}  // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_ALI_OPS_UTIL_H_