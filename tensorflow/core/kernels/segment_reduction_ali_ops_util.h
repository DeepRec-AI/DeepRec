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
template <typename Device, typename T, typename Tindex, typename Tsegment>
class SparseSegmentReduction {
 public:
  explicit SparseSegmentReduction(bool is_mean, bool is_sqrtn,
                                  bool has_num_segments, T default_value)
      : is_mean_(is_mean),
        is_sqrtn_(is_sqrtn),
        has_num_segments_(has_num_segments),
        default_value_(default_value) {}

  void Reduce(OpKernelContext* context, const Tensor& input,
              const Tensor& indices, const Tensor& segment_ids,
              const Tensor& num_segments, const AllocatorAttributes& attr,
              Tensor* output) {
    Tindex output_rows = -1;
    if (has_num_segments_) {
      OP_REQUIRES(
          context, num_segments.shape().dims() == 0,
          errors::InvalidArgument("num_segments should be a scalar, not shape ",
                                  num_segments.shape().DebugString()));
      output_rows = internal::SubtleMustCopy(num_segments.scalar<int32>()());
      OP_REQUIRES(context, output_rows >= 0,
                  errors::InvalidArgument("segment ids must be >= 0"));
    }

    OP_REQUIRES(context, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices should be a vector."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(segment_ids.shape()),
                errors::InvalidArgument("segment_ids should be a vector."));
    const int64 num_indices = indices.NumElements();
    OP_REQUIRES(context, num_indices == segment_ids.NumElements(),
                errors::InvalidArgument(
                    "segment_ids and indices should have same size."));

    auto input_flat = input.flat_outer_dims<T>();
    const int64 num_col = input_flat.dimension(1);
    const auto segment_vec = segment_ids.vec<Tsegment>();

    // Note that the current implementation assumes that segment_vec values are
    // sorted.
    const Tsegment last_segment_id_plus_one =
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
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(input.dtype(), output_shape, output, attr));
    if (num_indices == 0) {
      if (output_rows > 0) {
        output->flat_outer_dims<T>().setConstant(default_value_);
      }
      return;
    }
    OP_REQUIRES(context, output_rows > 0,
                errors::InvalidArgument("segment ids must be >= 0"));
    auto output_flat = output->flat_outer_dims<T>();

    const auto indices_vec = indices.vec<Tindex>();
    auto work = [this, &context, &output_flat, &input_flat, &indices_vec,
                 &segment_vec, num_col, num_indices,
                 output_rows](int64 start, int64 end) {
      Tsegment uninitialized_index = start;
      // We mannually set start_pos of first thread and end_pos of last thread,
      // which could make sure that unsorted ids would be checked out.
      int64 start_pos =
          start == 0 ? 0 : FirstGreatEqual(segment_vec, start, 0, num_indices);
      const int64 end_pos =
          end == output_rows ? num_indices : FirstGreatEqual(segment_vec, end,
                                                             0, num_indices);
      OP_REQUIRES(context, start_pos <= end_pos,
                  errors::InvalidArgument("segment ids are not increasing"));

      Tsegment out_index;
      bool do_work;
      if (start_pos == num_indices) {
        do_work = false;
      } else {
        out_index = internal::SubtleMustCopy(segment_vec(start_pos));
        do_work = true;
      }
      int64 cur_pos = start_pos + 1;
      while (do_work) {
        // We initialize next_index to 0 to avoid "warning: 'next_index' may be
        // used uninitialized in this function" in the Mac build (since the
        // compiler isn't smart enough to realize the code is safe).
        Tsegment next_index = 0;
        if (cur_pos < end_pos) {
          next_index = internal::SubtleMustCopy(segment_vec(cur_pos));
          if (out_index == next_index) {
            ++cur_pos;
            continue;
          }
          // We have a new segment here.  Verify that the segment ids are
          // growing.
          OP_REQUIRES(
              context, out_index < next_index,
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
          Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>,
                           Eigen::Unaligned>
              gap_slice(&output_flat(uninitialized_index, 0), gap_slice_shape);
          gap_slice.setConstant(default_value_);
        }

        auto out = output_flat.template chip<0>(out_index);
        const int bad_offset = Reduce<Tindex>(
            input_flat, indices_vec, start_pos, cur_pos - start_pos, out);
        OP_REQUIRES(context, bad_offset < 0,
                    errors::InvalidArgument(
                        "Bad: indices[", start_pos + bad_offset,
                        "] == ", indices_vec(start_pos + bad_offset),
                        " out of range [0, ", input_flat.dimension(0), ")"));

        start_pos = cur_pos;
        ++cur_pos;
        uninitialized_index = out_index + 1;
        out_index = next_index;
        if (cur_pos > end_pos) break;
      }

      // Fill the gap at the end with the default value.
      if (uninitialized_index < end) {
        Eigen::DSizes<Eigen::DenseIndex, 2> gap_slice_shape(
            end - uninitialized_index, num_col);
        Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Unaligned>
            gap_slice(&output_flat(uninitialized_index, 0), gap_slice_shape);
        gap_slice.setConstant(default_value_);
      }
    };

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads->num_threads - 1, worker_threads->workers, output_rows,
          num_col /* cost */, work);
  }

 private:
  template <typename Segment>
  int64 FirstGreatEqual(const typename TTypes<Segment>::ConstVec& segment_vec,
                        Segment idx, int64 lb, int64 rb) {
    if (lb == rb) return lb;
    int64 mid = (lb + rb) / 2;
    if (segment_vec(mid) < idx) {
      return FirstGreatEqual(segment_vec, idx, mid + 1, rb);
    }
    return FirstGreatEqual(segment_vec, idx, lb, mid);
  }

  template <typename Index>
  int64 Reduce(const typename TTypes<T>::ConstMatrix& input_flat,
               const typename TTypes<Index>::ConstVec& indices_vec, int64 start,
               int64 num,
               Eigen::TensorChippingOp<0, typename TTypes<T>::Matrix> out) {
#define INDEX(n, i)                               \
  const auto index##n = indices_vec(start + (i)); \
  if (!FastBoundsCheck(index##n, input_flat.dimension(0))) return (i);

#define L(n) input_flat.template chip<0>(index##n)

    if (num == 1) {
      INDEX(0, 0);
      out = L(0);
    } else {
      int64 r = num % 8;
      T m(1);
      if (is_mean_ && (num < 10)) {
        m = T(num);
      }
      if (is_sqrtn_ && (num < 10)) {
        m = T(sqrt(num));
      }
      switch (r) {
        case 2: {
          INDEX(0, 0);
          INDEX(1, 1);
          out = (L(0) + L(1)) / m;
          break;
        }
        case 3: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          out = (L(0) + L(1) + L(2)) / m;
          break;
        }
        case 4: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          out = (L(0) + L(1) + L(2) + L(3)) / m;
          break;
        }
        case 5: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          out = (L(0) + L(1) + L(2) + L(3) + L(4)) / m;
          break;
        }
        case 6: {
          INDEX(0, 0);
          INDEX(1, 1);
          INDEX(2, 2);
          INDEX(3, 3);
          INDEX(4, 4);
          INDEX(5, 5);
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5)) / m;
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
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6)) / m;
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
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7)) / m;
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
          out = (L(0) + L(1) + L(2) + L(3) + L(4) + L(5) + L(6) + L(7) + L(8)) /
                m;
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
        out = out / static_cast<T>(num);
      }
      if (is_sqrtn_ && num >= 10) {
        out = out / static_cast<T>(sqrt(num));
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

class SparseSegmentReductionGrad {
 public:
  explicit SparseSegmentReductionGrad(bool is_sqrtn) : is_sqrtn_(is_sqrtn) {}

  template <typename T, typename Tindex, typename SegmentId>
  void ReduceGrad(OpKernelContext* context, const Tensor& input,
                  const Tensor& indices, const Tensor& segment_ids,
                  const Tensor& output_dim0, Tensor& output) {
    const int64 N = indices.NumElements();
    OP_REQUIRES(context, N == segment_ids.NumElements(),
                errors::InvalidArgument(
                    "segment_ids and indices should have same size."));
    const SegmentId M =
        internal::SubtleMustCopy(output_dim0.scalar<SegmentId>()());

    auto input_flat = input.flat_outer_dims<T>();
    const int64 num_col = input_flat.dimension(1);

    const auto indices_vec = indices.vec<Tindex>();
    const auto segment_vec = segment_ids.vec<SegmentId>();

    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, M);

    const AllocatorAttributes alloc_attr = context->output_alloc_attr(0);
    OP_REQUIRES_OK(context,
                   context->allocate_temp(input.dtype(), output_shape, &output,
                                          alloc_attr));
    if (M == 0 || N == 0) return;

    // Note that similar to SparseSegmentMean, we assume that segment_vec is
    // already sorted and has non-negative values.
    const SegmentId num_segments = input.dim_size(0);
    const SegmentId last_segment_id_plus_one =
        internal::SubtleMustCopy(segment_vec(N - 1)) + 1;
    OP_REQUIRES(context, last_segment_id_plus_one <= num_segments,
                errors::InvalidArgument("Invalid number of segments"));

    auto output_flat = output.flat_outer_dims<T>();
    const bool cnt_layout_by_n = (N <= num_segments);
    std::vector<int> counting(cnt_layout_by_n ? N : num_segments);

    auto do_scan = [&context, &segment_vec, &counting, N, num_segments,
                    cnt_layout_by_n](int64 start, int64 end) {
      int64 start_pos = start;
      if (start > 0) {
        const SegmentId old_idx =
            internal::SubtleMustCopy(segment_vec(start - 1));
        while (start_pos < end &&
               internal::SubtleMustCopy(segment_vec(start_pos)) == old_idx) {
          ++start_pos;
        }
      }
      while (start_pos < end) {
        int64 cur_pos = start_pos + 1;
        const SegmentId in_idx =
            internal::SubtleMustCopy(segment_vec(start_pos));
        while (cur_pos < N &&
               internal::SubtleMustCopy(segment_vec(cur_pos)) == in_idx) {
          ++cur_pos;
        }
        OP_REQUIRES(
            context, FastBoundsCheck(in_idx, num_segments),
            errors::InvalidArgument("Segment id ", in_idx, " out of range [0, ",
                                    num_segments, ")."));
        if (cnt_layout_by_n) {
          for (int i = start_pos; i < cur_pos; ++i) {
            counting[i] = cur_pos - start_pos;
          }
        } else {
          counting[in_idx] = cur_pos - start_pos;
        }
        start_pos = cur_pos;
      }
    };

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads->num_threads - 1, worker_threads->workers, N,
          1 /* cost */, do_scan);
    if (!context->status().ok()) return;

    auto do_write = [this, &context, &output_flat, &input_flat, &indices_vec,
                     &segment_vec, &counting, M, N, num_col,
                     cnt_layout_by_n](int64 start, int64 end) {
      // NOTE(zycao): For outputs with high density, zero setting operations
      // could be considered after all fillings having been done in order to
      // decrease memory writes.
      Eigen::DSizes<Eigen::DenseIndex, 2> gap_slice_shape(end - start, num_col);
      Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Unaligned>
          gap_slice(&output_flat(start, 0), gap_slice_shape);
      gap_slice.setZero();

      // NOTE(zycao): This flag vector was used in community codes. For those
      // cases with dense and high inner rank data, this vector would help
      // decreasing some calculations.
      std::vector<bool> is_modified(end - start, false);
      for (int64 i = 0; i < N; ++i) {
        const Tindex output_idx = internal::SubtleMustCopy(indices_vec(i));
        OP_REQUIRES(context, FastBoundsCheck(output_idx, M),
                    errors::InvalidArgument("Index ", output_idx,
                                            " out of range [0, ", M, ")."));
        if (output_idx < start || output_idx >= end) continue;

        const SegmentId in_idx = internal::SubtleMustCopy(segment_vec(i));
        const int iscale = cnt_layout_by_n ? counting[i] : counting[in_idx];
        if (iscale == 1) {
          if (is_modified[output_idx - start]) {
            output_flat.template chip<0>(output_idx) +=
                input_flat.template chip<0>(in_idx);
          } else {
            output_flat.template chip<0>(output_idx) =
                input_flat.template chip<0>(in_idx);
            is_modified[output_idx - start] = true;
          }
          continue;
        }
        const T scale =
            is_sqrtn_ ? static_cast<T>(1.0 / sqrt(static_cast<double>(iscale)))
                      : static_cast<T>(1.0 / static_cast<double>(iscale));
        if (is_modified[output_idx - start]) {
          output_flat.template chip<0>(output_idx) +=
              input_flat.template chip<0>(in_idx) * scale;
        } else {
          output_flat.template chip<0>(output_idx) =
              input_flat.template chip<0>(in_idx) * scale;
          is_modified[output_idx - start] = true;
        }
      }
    };
    Shard(worker_threads->num_threads - 1, worker_threads->workers, M,
          num_col /* cost */, do_write);
  }

 private:
  const bool is_sqrtn_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SEGMENT_REDUCTION_ALI_OPS_UTIL_H_