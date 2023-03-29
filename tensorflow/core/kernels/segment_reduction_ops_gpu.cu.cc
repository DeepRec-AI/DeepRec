/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/platform/cuda.h"

using stream_executor::cuda::ScopedActivateExecutorContext;
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/platform/rocm.h"
using stream_executor::rocm::ScopedActivateExecutorContext;
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/segment_reduction_ops_gpu.cu.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

/*----------------------------- SortSegment Begin -----------------------------*/

// SortedSegmentReductionFunctor kernel reduces input data just as
// UnsortedSegmentReductionCustomKernel does except that input data
// is partitioned along the outer reduction dimension. This is
// because consecutive rows (elements in a row share the same
// outer dimension index) in the flattened 2D input data likely
// belong to the same segment in sorted segment sum operation.
// Therefore such partitioning strategy has two advantages over
// the UnsortedSegmentReductionFunctor kernel:
// 1. Each thread reduces across multiple rows before writing
// answers to the global memory, we can therefore
// write reduction results to global memory less often.
// 2. We may know that the current thread is the only contributor
// to an output element because of the increasing nature of segment
// ids. In such cases, we do not need to use atomic operations
// to write results to global memory.
// In the flattened view of input data (with only outer and inner
// dimension), every thread processes a strip of input data of
// size OuterDimTileSize x 1. This strip runs across multiple
// rows of input data and all reduction elements share one inner
// dimension index.
template <typename T, typename Index, int OuterDimTileSize, typename ReductionF,
          typename AtomicReductionF>
__global__ void SortedSegmentReductionCustomKernel(
    const Index input_outer_dim_size, const Index inner_dim_size,
    const Index output_outer_dim_size, const Index* __restrict__ segment_ids,
    const T* __restrict__ input, T* __restrict__ output,
    const Index total_stripe_count, const T initial_value) {
  for (int stripe_index : GpuGridRangeX(total_stripe_count)) {
    const Index segment_offset = stripe_index % inner_dim_size;
    const Index input_outer_dim_index_base =
        stripe_index / inner_dim_size * Index(OuterDimTileSize);

    T reduce_res = initial_value;
    Index first_segment_id = segment_ids[input_outer_dim_index_base];
    Index last_output_segment_id = output_outer_dim_size;

    const Index actual_stripe_height =
        min(Index(OuterDimTileSize),
            input_outer_dim_size - input_outer_dim_index_base);
    for (Index j = 0; j < actual_stripe_height; j++) {
      Index current_output_segment_id =
          segment_ids[input_outer_dim_index_base + j];
      // Decide whether to write result to global memory. Result is only written
      // to global memory if we move to another segment. Otherwise we can keep
      // accumulating locally.
      if (current_output_segment_id > last_output_segment_id) {
        const Index output_index =
            last_output_segment_id * inner_dim_size + segment_offset;
        // Decide whether to write result to global memory using atomic
        // operations.
        if (last_output_segment_id == first_segment_id) {
          AtomicReductionF()(output + output_index, reduce_res);
        } else {
          ReductionF()(output + output_index, reduce_res);
        }
        reduce_res = initial_value;
      }
      ReductionF()(
          &reduce_res,
          ldg(input + (input_outer_dim_index_base + j) * inner_dim_size +
              segment_offset));
      last_output_segment_id = current_output_segment_id;
    }
    // For the last result in a strip, always write using atomic operations
    // due to possible race conditions with threads computing
    // the following strip.
    const Index output_index =
        last_output_segment_id * inner_dim_size + segment_offset;
    AtomicReductionF()(output + output_index, reduce_res);
  }
}

template <typename SegmentId, typename Index, typename T>
__global__ void SegmentMeanNormalizeKernel(
    SegmentId nsegments, Index ninner,
    const Index* __restrict__ segment_offsets,  // [nsegments + 1]
    T* __restrict__ output) {                   // [nsegments, ninner]
  for (SegmentId seg : GpuGridRangeY(nsegments)) {
    SegmentId segment_size = segment_offsets[seg + 1] - segment_offsets[seg];
    T norm = static_cast<T>(max(segment_size, Index(1))); // Avoid division by zero
    for (Index i : GpuGridRangeX(ninner)) {
      output[seg * ninner + i] /= norm;
    }
  }
}

template <typename SegmentId, typename Index, typename T>
Status LaunchSegmentMeanNormalizeKernel(
    const GPUDevice& d, SegmentId nsegments, Index ninner,
    const Index* __restrict__ segment_offsets,  // [nsegments + 1]
    T* __restrict__ output) {                   // [nsegments, ninner]
  Gpu2DLaunchConfig config = GetGpu2DLaunchConfig(
      ninner, nsegments, d, SegmentMeanNormalizeKernel<SegmentId, Index, T>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(SegmentMeanNormalizeKernel<SegmentId, Index, T>,
                         config.block_count, config.thread_per_block, 0,
                         d.stream(), nsegments, ninner, segment_offsets,
                         output);
}

template <typename SegmentId, typename Index, typename T>
__global__ void SegmentSetEmptyKernel(
    SegmentId nsegments, Index ninner,
    const Index* __restrict__ segment_offsets,  // [nsegments + 1]
    const T empty_value,
    T* __restrict__ output) {  // [nsegments, ninner]
  for (SegmentId seg : GpuGridRangeY(nsegments)) {
    SegmentId segment_size = segment_offsets[seg + 1] - segment_offsets[seg];
    if (segment_size == 0) {
      for (Index i : GpuGridRangeX(ninner)) {
        output[seg * ninner + i] = empty_value;
      }
    }
  }
}

template <typename SegmentId, typename Index, typename T>
Status LaunchSegmentSetEmptyKernel(
    const GPUDevice& d, SegmentId nsegments, Index ninner,
    const Index* __restrict__ segment_offsets,  // [nsegments + 1]
    const T empty_value,
    T* __restrict__ output) {  // [nsegments, ninner]
  Gpu2DLaunchConfig config = GetGpu2DLaunchConfig(
      ninner, nsegments, d, SegmentSetEmptyKernel<SegmentId, Index, T>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(SegmentSetEmptyKernel<SegmentId, Index, T>,
                         config.block_count, config.thread_per_block, 0,
                         d.stream(), nsegments, ninner, segment_offsets,
                         empty_value, output);
}

template <typename Tindex, typename Tsegmentids>
__global__ void SegmentOffsetsKernel(
    Tindex size, Tsegmentids nsegments,
    const Tsegmentids* __restrict__ segment_ids,  // [size]
    Tindex* __restrict__ segment_offsets) {       // [nsegments + 1]
  GPU_1D_KERNEL_LOOP(i, size + 1) {
    // IDs are clipped to [-1, nsegments] so that out-of-bounds IDs are ignored.
    // Note that we can't report invalid IDs from the GPU without incurring
    // additional overhead.
    auto clip = [&](Tsegmentids id) {
      return min(max(Tsegmentids(-1), id), nsegments);
    };
    const Tsegmentids cur_id = (i < size) ? clip(segment_ids[i]) : nsegments;
    const Tsegmentids prev_id =
        (i == 0) ? Tsegmentids(-1) : clip(segment_ids[i - 1]);
    // At segment boundaries, write the offset for this ID and any missing IDs
    // since the previous one.
    for (Tsegmentids id = prev_id + 1; id <= cur_id; ++id) {
      segment_offsets[id] = i;
    }
  }
}

// Finds the start offset of each segment in the given sorted segment_ids
// vector. Missing IDs are given the same offset as the next ID so that they
// represent empty ranges. Invalid IDs (those that are outside the range
// [0, nsegments)) are ignored. The value at segment_offsets[0] is set to the
// start index of the first valid ID (e.g., 0 if all IDs are valid), and the
// value at segment_offsets[nsegments] is set to the end index of the last valid
// ID (e.g., nsegments if all IDs are valid).
template <typename Tindex, typename Tsegmentids>
Status LaunchSegmentOffsetsKernel(const GPUDevice& d, Tindex size,
                                  Tsegmentids nsegments,
                                  const Tsegmentids* segment_ids,  // [size]
                                  Tindex* segment_offsets) {  // [nsegments + 1]
  GpuLaunchConfig config = GetGpuLaunchConfig(
      size + 1, d, &SegmentOffsetsKernel<Tindex, Tsegmentids>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(SegmentOffsetsKernel<Tindex, Tsegmentids>,
                         config.block_count, config.thread_per_block, 0,
                         d.stream(), size, nsegments, segment_ids,
                         segment_offsets);
}

namespace functor {

template <typename T, typename Index, typename InitialValueF,
          typename EmptySegmentValueF, typename ReductionF>
void SegmentReductionFunctor<
    T, Index, InitialValueF, EmptySegmentValueF,
    ReductionF>::operator()(OpKernelContext* ctx, const GPUDevice& d,
                            const Index output_rows,
                            const TensorShape& segment_ids_shape, bool is_mean,
                            typename TTypes<Index>::ConstFlat segment_ids,
                            const Index data_size, const T* data,
                            typename TTypes<T, 2>::Tensor output) {
  if (output.size() == 0) {
    return;
  }

  // Launch kernel(s) to compute sorted segment reduction.
  // Notes:
  // *) 'input_total_size' is the total number of elements to process.
  // *) 'segment_ids.shape' is a prefix of data's shape.
  // *) 'input_outer_dim_size' is the total number of segments to process.
  const Index input_total_size = data_size;
  const Index input_outer_dim_size = segment_ids.dimension(0);
  const Index input_inner_dim_size = input_total_size / input_outer_dim_size;
  const Index num_segments = output.size() / input_inner_dim_size;

  // Set 'output' to initial value.
  GpuLaunchConfig config = GetGpuLaunchConfig(output.size(), d);
  const T initial_value = InitialValueF()();
  TF_CHECK_OK(GpuLaunchKernel(SetToValue<T>, config.block_count,
                              config.thread_per_block, 0, d.stream(),
                              output.size(), output.data(), initial_value));
  if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
    return;
  }

  const int OuterDimTileSize = 8;

  const Index input_outer_dim_num_stripe =
      Eigen::divup(input_outer_dim_size, Index(OuterDimTileSize));

  const Index total_stripe_count =
      input_inner_dim_size * input_outer_dim_num_stripe;

  config = GetGpuLaunchConfig(total_stripe_count, d);
  TF_CHECK_OK(GpuLaunchKernel(
      SortedSegmentReductionCustomKernel<
          T, Index, OuterDimTileSize,
          typename ReduceUpdateOpFor<ReductionF>::nonatomic_op,
          typename ReduceUpdateOpFor<ReductionF>::atomic_op>,
      config.block_count, config.thread_per_block, 0, d.stream(),
      input_outer_dim_size, input_inner_dim_size, output_rows,
      segment_ids.data(), data, output.data(), total_stripe_count,
      initial_value));

  const T empty_value = EmptySegmentValueF()();
  if (is_mean || initial_value != empty_value) {
    Tensor segment_offsets;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<Index>::value,
                                           TensorShape({num_segments + 1}),
                                           &segment_offsets));
    Index* segment_offsets_ptr = segment_offsets.flat<Index>().data();
    OP_REQUIRES_OK(ctx, LaunchSegmentOffsetsKernel(
                            d, input_outer_dim_size, num_segments,
                            segment_ids.data(), segment_offsets_ptr));

    if (is_mean) {
      OP_REQUIRES_OK(ctx, LaunchSegmentMeanNormalizeKernel(
                              d, num_segments, input_inner_dim_size,
                              segment_offsets_ptr, output.data()));
    }
    if (initial_value != empty_value) {
      OP_REQUIRES_OK(
          ctx, LaunchSegmentSetEmptyKernel(
                   d, num_segments, input_inner_dim_size, segment_offsets_ptr,
                   empty_value, output.data()));
    }
  }
}

#define DEFINE_SORTED_GPU_SPECS_INDEX(T, Index)               \
  template struct SegmentReductionFunctor<                    \
      T, Index, /*InitialValueF=*/functor::Zero<T>,           \
      /*EmptySegmentValueF=*/functor::Zero<T>, functor::Sum>; \
  template struct SegmentReductionFunctor<                    \
      T, Index, /*InitialValueF=*/functor::One<T>,            \
      /*EmptySegmentValueF=*/functor::One<T>, functor::Prod>; \
  template struct SegmentReductionFunctor<                    \
      T, Index, /*InitialValueF=*/functor::Highest<T>,        \
      /*EmptySegmentValueF=*/functor::Zero<T>, functor::Min>; \
  template struct SegmentReductionFunctor<                    \
      T, Index, /*InitialValueF=*/functor::Lowest<T>,         \
      /*EmptySegmentValueF=*/functor::Zero<T>, functor::Max>;

#define DEFINE_SORTED_GPU_SPECS(T)		\
  DEFINE_SORTED_GPU_SPECS_INDEX(T, int32);	\
  DEFINE_SORTED_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_SORTED_GPU_SPECS);
TF_CALL_int32(DEFINE_SORTED_GPU_SPECS);
TF_CALL_int64(DEFINE_SORTED_GPU_SPECS);
TF_CALL_uint32(DEFINE_SORTED_GPU_SPECS);
TF_CALL_uint64(DEFINE_SORTED_GPU_SPECS);

#undef DEFINE_SORTED_GPU_SPECS
#undef DEFINE_SORTED_GPU_SPECS_INDEX
} // end of namespace functor

//  SegmentReductionGPUOp is a segment reduction operator implemented for GPU
//  only.
//  TODO: This implementation of SegmentReductionGPUOp is sometimes slower than
//  its unsorted counterpart (mostly when problem size is small).
//  This is due to the following two main reasons and a cost-effective way
//  to resolve these problems is desirable.
//  1. Sorted segment reduction requires a memory transfer from device to host
//     in order to know the size of the output dimension whereas unsorted
//     segment reduction receives the size of the output dimension as an input
//     parameter.
//  2. Sorted segment reduction is essentially a tiled version of unsorted
//     segment reduction and therefore such optimization comes at an inherent
//     cost. However such cost may not be justified when the problem size is
//     small. When to use the tiled version or the untiled version depends on
//     many factors including data alignments, ratio of calculation to memory
//     traffic and obviously, the problem sizes.
template <class T, class Index, class SegmentReductionFunctor, bool IsMean>
class SegmentReductionGPUOp : public AsyncOpKernel {
 public:
  explicit SegmentReductionGPUOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& input = context->input(0);
    const Tensor& segment_ids = context->input(1);

    OP_REQUIRES_ASYNC(
        context, TensorShapeUtils::IsVector(segment_ids.shape()),
        errors::InvalidArgument("segment_ids should be a vector."), done);

    OP_REQUIRES_ASYNC(context, input.dims() >= 1,
                      errors::InvalidArgument("Shape must be at least rank 1"),
                      done);

    const int64_t num_indices = segment_ids.NumElements();
    OP_REQUIRES_ASYNC(
        context, num_indices == input.dim_size(0),
        errors::InvalidArgument(
            "segment_ids should be the same size as dimension 0 of"
            " input."),
        done);

    if (num_indices == 0) {
      TensorShape output_shape = input.shape();
      output_shape.set_dim(0, 0);

      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(0, output_shape, &output), done);
      done();
      return;
    }

    se::DeviceMemoryBase output_rows_device(
        const_cast<Tensor&>(segment_ids).template flat<Index>().data() +
        (num_indices - 1));
    ScratchSpace<Index> output_rows_host(context, 1, /* on_host */ true);

    auto stream = context->op_device_context()->stream();
    OP_REQUIRES_ASYNC(
        context,
        stream
            ->ThenMemcpy(output_rows_host.mutable_data(), output_rows_device,
                         sizeof(Index))
            .ok(),
        errors::Internal(type_string() +
                         ": failed to copy output_rows from device"),
        done);

    SegmentReductionFunctor functor_;
    auto create_and_check_output = [context, output_rows_host, &input,
                                    &segment_ids, &functor_, done]() {
      // Ensure that within the callback, the proper GPU settings are
      // configured.
      auto stream = context->op_device_context()->stream();
      ScopedActivateExecutorContext scoped_activation{stream->parent()};

      Index output_rows = *output_rows_host.data();
      output_rows++;
      OP_REQUIRES_ASYNC(context, output_rows > 0,
                        errors::InvalidArgument("segment ids must be >= 0"),
                        done);

      TensorShape output_shape = input.shape();
      // Since we're changing the first dimension of the shape, we need to make
      // sure the new shape won't overflow.
      OP_REQUIRES_OK_ASYNC(context,
                           output_shape.SetDimWithStatus(0, output_rows), done);

      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(0, output_shape, &output), done);

      auto output_flat = output->flat_outer_dims<T>();
      auto data_ptr = input.template flat<T>().data();
      auto segment_flat = segment_ids.flat<Index>();
      functor_(context, context->eigen_device<GPUDevice>(), output_rows,
               segment_ids.shape(), IsMean, segment_flat, input.NumElements(),
               data_ptr, output_flat);

      done();
    };

    context->device()
        ->tensorflow_gpu_device_info()
        ->event_mgr->ThenExecute(stream, create_and_check_output);
  }
};

#define REGISTER_GPU_KERNEL_SORTEDSEGMENT(                            \
    name, type, index_type, initial_value_functor,                    \
    empty_segment_value_functor, reduction_kernel_functor, is_mean)   \
  REGISTER_KERNEL_BUILDER(                                            \
      Name(name)                                                      \
          .Device(DEVICE_GPU)                                         \
          .TypeConstraint<type>("T")                                  \
          .TypeConstraint<index_type>("Tindices"),                    \
      SegmentReductionGPUOp<                                          \
          type, index_type,                                           \
          functor::SegmentReductionFunctor<                           \
              type, index_type, initial_value_functor,                \
              empty_segment_value_functor, reduction_kernel_functor>, \
          is_mean>)

#define REGISTER_GPU_SORTED_KERNELS(type, index_type)                         \
  REGISTER_GPU_KERNEL_SORTEDSEGMENT("SegmentSum", type, index_type,           \
                                    functor::Zero<type>, functor::Zero<type>, \
                                    functor::Sum, /*is_mean=*/false);         \
  REGISTER_GPU_KERNEL_SORTEDSEGMENT("SegmentMean", type, index_type,          \
                                    functor::Zero<type>, functor::Zero<type>, \
                                    functor::Sum, /*is_mean=*/true);          \
  REGISTER_GPU_KERNEL_SORTEDSEGMENT("SegmentProd", type, index_type,          \
                                    functor::One<type>, functor::One<type>,   \
                                    functor::Prod, /*is_mean=*/false);        \
  REGISTER_GPU_KERNEL_SORTEDSEGMENT(                                          \
      "SegmentMin", type, index_type, functor::Highest<type>,                 \
      functor::Zero<type>, functor::Min, /*is_mean=*/false);                  \
  REGISTER_GPU_KERNEL_SORTEDSEGMENT(                                          \
      "SegmentMax", type, index_type, functor::Lowest<type>,                  \
      functor::Zero<type>, functor::Max, /*is_mean=*/false);

#define REGISTER_GPU_SORTED_KERNELS_ALL(type) \
  REGISTER_GPU_SORTED_KERNELS(type, int32);   \
  REGISTER_GPU_SORTED_KERNELS(type, int64);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_SORTED_KERNELS_ALL);
TF_CALL_int32(REGISTER_GPU_SORTED_KERNELS_ALL);
TF_CALL_int64(REGISTER_GPU_SORTED_KERNELS_ALL);
TF_CALL_uint32(REGISTER_GPU_SORTED_KERNELS_ALL);
TF_CALL_uint64(REGISTER_GPU_SORTED_KERNELS_ALL);

#undef REGISTER_GPU_SORTED_KERNELS_ALL
#undef REGISTER_GPU_SORTED_KERNELS
#undef REGISTER_GPU_KERNEL_SORTEDSEGMENT

/*----------------------------- SortSegment End -------------------------------*/

/*--------------------------- UnsortedSegment Begin ---------------------------*/

// UnsortedSegmentSumKernel processes 'input_total_size' elements.
// Each element is mapped from input to output by a combination of its
// 'segment_ids' mapping and 'inner_dim_size'.
template <typename T, typename Index, typename KernelReductionFunctor>
__global__ void UnsortedSegmentCustomKernel(const int64 input_outer_dim_size,
                                            const int64 inner_dim_size,
                                            const int64 output_outer_dim_size,
                                            const Index* segment_ids,
                                            const T* input, T* output) {
  const int64 input_total_size = input_outer_dim_size * inner_dim_size;
  for (int64 input_index : GpuGridRangeX(input_total_size)) {
    const int64 input_segment_index = input_index / inner_dim_size;
    const int64 segment_offset = input_index % inner_dim_size;
    const Index output_segment_index = segment_ids[input_segment_index];
    if (output_segment_index < 0 ||
        output_segment_index >= output_outer_dim_size) {
      continue;
    }
    const int64 output_index =
        output_segment_index * inner_dim_size + segment_offset;
    KernelReductionFunctor()(output + output_index, ldg(input + input_index));
  }
}

namespace functor {
template <typename T, typename Index, typename InitialValueF,
          typename ReductionF>
struct UnsortedSegmentFunctor<GPUDevice, T, Index, InitialValueF, ReductionF> {
  void operator()(OpKernelContext* ctx, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  typename TTypes<T, 2>::ConstTensor data,
                  typename TTypes<T, 2>::Tensor output) {
    if (output.size() == 0) {
      return;
    }
    // Set 'output' to initial value.
    GPUDevice d = ctx->template eigen_device<GPUDevice>();
    GpuLaunchConfig config = GetGpuLaunchConfig(output.size(), d);
    TF_CHECK_OK(GpuLaunchKernel(
        SetToValue<T>, config.block_count, config.thread_per_block, 0,
        d.stream(), output.size(), output.data(), InitialValueF()()));
    const int64 data_size = data.size();
    if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
      return;
    }
    // Launch kernel to compute unsorted segment reduction.
    // Notes:
    // *) 'data_size' is the total number of elements to process.
    // *) 'segment_ids.shape' is a prefix of data's shape.
    // *) 'input_outer_dim_size' is the total number of segments to process.
    const int64 input_outer_dim_size = segment_ids.dimension(0);
    const int64 input_inner_dim_size = data.dimension(1);
    const int64 output_outer_dim_size = output.dimension(0);
    config = GetGpuLaunchConfig(data_size, d);

    TF_CHECK_OK(GpuLaunchKernel(
        UnsortedSegmentCustomKernel<T, Index, ReductionF>, config.block_count,
        config.thread_per_block, 0, d.stream(), input_outer_dim_size,
        input_inner_dim_size, output_outer_dim_size, segment_ids.data(),
        data.data(), output.data()));
  }
};

#define DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, Index)                         \
  template struct UnsortedSegmentFunctor<                                      \
      GPUDevice, T, Index, functor::Lowest<T>, functor::AtomicMaxOpGpu>;       \
  template struct UnsortedSegmentFunctor<                                      \
      GPUDevice, T, Index, functor::Highest<T>, functor::AtomicMinOpGpu>;      \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index, functor::One<T>, \
                                         functor::AtomicProdOpGpu>;

// sum is the only op that supports all input types currently
#define DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, Index) \
  template struct UnsortedSegmentFunctor<             \
      GPUDevice, T, Index, functor::Zero<T>, functor::AtomicSumOpGpu>;

#define DEFINE_REAL_GPU_SPECS(T)                  \
  DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, int32); \
  DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, int64);

#define DEFINE_SUM_GPU_SPECS(T)                  \
  DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, int32); \
  DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_REAL_GPU_SPECS);
TF_CALL_int32(DEFINE_REAL_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DEFINE_SUM_GPU_SPECS);
TF_CALL_int32(DEFINE_SUM_GPU_SPECS);

// TODO(rocm): support atomicAdd for complex numbers on ROCm
#if GOOGLE_CUDA
TF_CALL_complex64(DEFINE_SUM_GPU_SPECS);
TF_CALL_complex128(DEFINE_SUM_GPU_SPECS);
#endif

#undef DEFINE_SUM_GPU_SPECS
#undef DEFINE_REAL_GPU_SPECS
#undef DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX
#undef DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX
} // End of namespace functor

#define REGISTER_GPU_KERNEL_UNSORTEDSEGMENT(                                 \
    name, type, index_type, initial_value_functor, reduction_kernel_functor) \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name(name)                                                             \
          .Device(DEVICE_GPU)                                                \
          .HostMemory("num_segments")                                        \
          .TypeConstraint<type>("T")                                         \
          .TypeConstraint<index_type>("Tindices"),                           \
      UnsortedSegmentReductionOp<                                            \
          type, index_type,                                                  \
          functor::UnsortedSegmentFunctor<GPUDevice, type, index_type,       \
                                          initial_value_functor,             \
                                          reduction_kernel_functor> >)

// sum is the only op that supports all input types currently
#define REGISTER_REAL_GPU_UNSORTED_KERNELS(type, index_type)                   \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentMax", type, index_type,  \
                                      functor::Lowest<type>,                   \
                                      functor::AtomicMaxOpGpu);                \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentMin", type, index_type,  \
                                      functor::Highest<type>,                  \
                                      functor::AtomicMinOpGpu);                \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentProd", type, index_type, \
                                      functor::One<type>,                      \
                                      functor::AtomicProdOpGpu);

#define REGISTER_SUM_GPU_UNSORTED_KERNELS(type, index_type)                   \
  REGISTER_GPU_KERNEL_UNSORTEDSEGMENT("UnsortedSegmentSum", type, index_type, \
                                      functor::Zero<type>,                    \
                                      functor::AtomicSumOpGpu);

#define REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL(type) \
  REGISTER_REAL_GPU_UNSORTED_KERNELS(type, int32);   \
  REGISTER_REAL_GPU_UNSORTED_KERNELS(type, int64);

#define REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL(type) \
  REGISTER_SUM_GPU_UNSORTED_KERNELS(type, int32);   \
  REGISTER_SUM_GPU_UNSORTED_KERNELS(type, int64);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL);
TF_CALL_int32(REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL);
TF_CALL_int32(REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL);
// TODO(rocm): support atomicAdd for complex numbers on ROCm
#if GOOGLE_CUDA
TF_CALL_complex64(REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL);
TF_CALL_complex128(REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL);
#endif

#undef REGISTER_SUM_GPU_UNSORTED_KERNELS_ALL
#undef REGISTER_REAL_GPU_UNSORTED_KERNELS_ALL
#undef REGISTER_SUM_GPU_UNSORTED_KERNELS
#undef REGISTER_REAL_GPU_UNSORTED_KERNELS
#undef REGISTER_GPU_KERNEL_UNSORTEDSEGMENT

/*--------------------------- UnsortedSegment End -----------------------------*/

/*---------------------------- SparseSegment Begin ----------------------------*/

template <typename T, typename Index>
__global__ void SparseSegmentSumKernel(const int64 data_sparse_size,
                                       const int64 output_total_size,
                                       const int64 data_inner_dim,
                                       const T* data,
                                       const Index* indices,
                                       const int32* seg_ids,
                                       T* output) {

  for (int input_index : GpuGridRangeX(data_sparse_size)) {
    const Index sparse_row = input_index / data_inner_dim;
    const Index sparse_offset = input_index % data_inner_dim;
    const Index data_row = indices[sparse_row];
    const Index data_idx = data_row * data_inner_dim + sparse_offset;
    const Index output_row = seg_ids[sparse_row];
    const Index output_idx = output_row * data_inner_dim + sparse_offset;
    if (output_idx < 0 || output_idx >= output_total_size) {
      continue;
    }
    functor::AtomicSumOpGpu()(output + output_idx, ldg(data + data_idx));
  }
}
template <typename T, typename Index>
__global__ void SparseSegmentGradKernel(const int64 data_sparse_size,
                                        const int64 output_total_size,
                                        const int64 data_inner_dim,
                                        const T* data,
                                        const int32* seg_ids,
                                        const int32* seg_lens,
                                        const Index* indices,
                                        T* output, const bool is_sqrtn) {

  for (int input_index : GpuGridRangeX(data_sparse_size)) {
    const Index sparse_row = input_index / data_inner_dim;
    const Index sparse_offset = input_index % data_inner_dim;
    const Index data_row = seg_ids[sparse_row];
    const Index data_idx = data_row * data_inner_dim + sparse_offset;
    const Index output_row = indices[sparse_row];
    const Index output_idx = output_row * data_inner_dim + sparse_offset;
    if (output_idx < 0 || output_idx >= output_total_size) {
      continue;
    }
    const int32 seg_len = seg_lens[data_row];
    T scale = seg_len > 1 ? T(1.0)/(is_sqrtn ? std::sqrt(T(seg_len)) : \
        T(seg_len)) : T(1.0);
    functor::AtomicSumOpGpu()(output + output_idx, ldg(data + data_idx) * scale);
  }
}

template <typename T, typename Index>
__global__ void SparseSegmentLenSumKernel(const int64 num_ids,
                                          const int64 output_row_size,
                                          const Index* seg_ids,
                                          T* seg_lens) {
  for (int input_index : GpuGridRangeX(num_ids)) {
    const Index output_idx = seg_ids[input_index];
    if (output_idx < 0 || output_idx>=output_row_size) {
      continue;
    }
    functor::AtomicSumOpGpu()(seg_lens + output_idx, T(1));
  }
}

template <typename T, typename Index>
__global__ void SparseSegmentMeanKernel(const int64 output_total_size,
                                        const int64 data_inner_dim,
                                        T* output,
                                        const T* seg_lens) {
  for (int input_index : GpuGridRangeX(output_total_size)) {
    const Index output_row = input_index / data_inner_dim;
    T count = ldg(seg_lens + output_row);
    output[input_index] /= (count > 0 ? count : 1);
  }
}

template <typename T, typename Index>
__global__ void SparseSegmentSqrtNKernel(const int64 output_total_size,
                                         const int64 data_inner_dim,
                                         T* output,
                                         const T* seg_lens) {
  for (int input_index : GpuGridRangeX(output_total_size)) {
    const Index output_row = input_index / data_inner_dim;
    T count = ldg(seg_lens + output_row);
    output[input_index] /= std::sqrt(count > 0 ? count : 1);
  }
}

namespace functor {
template <typename T, typename Index>
void SparseSegmentReduceFunctor<T, Index>::operator()(OpKernelContext* ctx,
                                                      const Tensor* input,
                                                      const Tensor* indices,
                                                      const Tensor* seg_ids,
                                                      Tensor* output,
                                                      const bool is_mean,
                                                      const bool is_sqrtn) {
  const int64 num_ids = indices->NumElements();
  if (input->NumElements() == 0 || num_ids == 0) {
    return;
  }

  Tensor seg_lens;
  const auto data_flat = input->flat_outer_dims<T>();
  const auto indices_flat = indices->flat<Index>();
  const auto segment_flat = seg_ids->flat<int32>();
  auto output_flat = output->flat<T>();

  const int64 data_inner_dim = data_flat.dimension(1);
  const int64 data_sparse_size = num_ids * data_inner_dim;
  const int64 output_total_size = output->NumElements();
  const int64 output_row_size = output_total_size/data_inner_dim;

  const GPUDevice d = ctx->eigen_device<GPUDevice>();

  // initialize output by default value
  GpuLaunchConfig config = GetGpuLaunchConfig(output_total_size, d);
  SetToValue<<<config.block_count, config.thread_per_block,
      0, d.stream()>>>(static_cast<int>(output_total_size),
      output_flat.data(), T(0.0));

  config = GetGpuLaunchConfig(data_sparse_size, d);
  // launch a kernel
  SparseSegmentSumKernel<T, Index>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          data_sparse_size, output_total_size, data_inner_dim, data_flat.data(),
          indices_flat.data(), segment_flat.data(), output_flat.data());

  if (is_mean | is_sqrtn) {
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(output->dtype(),
                        TensorShape({output_row_size}),
                   &seg_lens));
    auto segment_lens_flat = seg_lens.flat<T>();
    config = GetGpuLaunchConfig(output_row_size, d);
    SetZero<T><<<config.block_count, config.thread_per_block,
      0, d.stream()>>>(static_cast<int>(output_row_size),
      segment_lens_flat.data());
    // launch a kernel to sum the seg_lens for each segment
    config = GetGpuLaunchConfig(num_ids, d);
    SparseSegmentLenSumKernel<T, int32>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          num_ids, output_row_size, segment_flat.data(),
          segment_lens_flat.data());
  }
  // for mean
  if (is_mean) {
    auto segment_lens_flat = seg_lens.flat<T>();
    config = GetGpuLaunchConfig(output_total_size, d);
    SparseSegmentMeanKernel<T, Index>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>
      (output_total_size, data_inner_dim, output_flat.data(),
       segment_lens_flat.data());
  } else if (is_sqrtn) {
    auto segment_lens_flat = seg_lens.flat<T>();
    config = GetGpuLaunchConfig(output_total_size, d);
    SparseSegmentSqrtNKernel<T, Index>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>
      (output_total_size, data_inner_dim, output_flat.data(),
       segment_lens_flat.data());
  }
}

template <typename T, typename Index>
void SparseSegmentReduceGradFunctor<T, Index>::operator()(OpKernelContext* ctx,
                                                          const Tensor* input,
                                                          const Tensor* indices,
                                                          const Tensor* seg_ids,
                                                          Tensor* output,
                                                          const bool is_sqrtn) {
    typedef int32 SegmentId;
    const SegmentId num_segments = input->dim_size(0);
    const int64 N = indices->NumElements();
    auto output_flat = output->flat_outer_dims<T>();
    const int64 seg_lens_size = num_segments;
    // place this on gpu
    Tensor seg_lens;
    // prepare counting
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32,
                        TensorShape({seg_lens_size}),
                   &seg_lens));
    auto segment_lens_flat = seg_lens.flat<int32>();
    const int64 data_inner_dim = output_flat.dimension(1);
    const int64 data_sparse_size = N * data_inner_dim;
    const int64 output_total_size = output->NumElements();
    const int64 output_row_size = output_total_size/data_inner_dim;
    const GPUDevice d = ctx->eigen_device<GPUDevice>();
    // initialize output by default value
    GpuLaunchConfig config = GetGpuLaunchConfig(output_total_size, d);
    SetToValue<<<config.block_count, config.thread_per_block,
        0, d.stream()>>>(static_cast<int>(output_total_size),
        output->flat<T>().data(), T(0.0));
    // obtain seg_lens (counts)
    config = GetGpuLaunchConfig(seg_lens_size, d);
    SetZero<int32><<<config.block_count, config.thread_per_block,
      0, d.stream()>>>(static_cast<int>(seg_lens_size),
      segment_lens_flat.data());
    // launch a kernel to sum the seg_lens for each segment
    config = GetGpuLaunchConfig(N, d);
    SparseSegmentLenSumKernel<int32, SegmentId>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          N, seg_lens_size, seg_ids->flat<SegmentId>().data(),
          segment_lens_flat.data());
    // launch a kernel to sum up values
    config = GetGpuLaunchConfig(data_sparse_size, d);
    SparseSegmentGradKernel<T, Index>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            data_sparse_size, output_total_size, data_inner_dim,
            input->flat<T>().data(),
            seg_ids->flat<SegmentId>().data(),
            segment_lens_flat.data(),
            indices->flat<Index>().data(),
            output->flat<T>().data(), is_sqrtn);
}

// for sparse gpu functors
#define DEFINE_SPARSE_GPU_SPEC_REDUCE(T, Index) \
  template struct SparseSegmentReduceFunctor<T, Index>

#define DEFINE_SPARSE_GPU_SPEC_GRAD_REDUCE(T, Index) \
  template struct SparseSegmentReduceGradFunctor<T, Index>

#define DEFINE_SPARSE_GPU_SPEC_MAX_SEG_ID(Index) \
  template struct FindMaxSegId<Index>;

#define DEFINE_SPARSE_GPU_SPEC_SET_VAL(T) \
  template struct SetValueDefault<T>;

#define DEFINE_SPARSE_GPU_SPEC(T)         \
  DEFINE_SPARSE_GPU_SPEC_REDUCE(T, int32); \
  DEFINE_SPARSE_GPU_SPEC_GRAD_REDUCE(T, int32); \
  DEFINE_SPARSE_GPU_SPEC_REDUCE(T, int64); \
  DEFINE_SPARSE_GPU_SPEC_GRAD_REDUCE(T, int64);

TF_CALL_float(DEFINE_SPARSE_GPU_SPEC)
TF_CALL_double(DEFINE_SPARSE_GPU_SPEC)
TF_CALL_int32(DEFINE_SPARSE_GPU_SPEC_MAX_SEG_ID)
TF_CALL_float(DEFINE_SPARSE_GPU_SPEC_SET_VAL)
TF_CALL_double(DEFINE_SPARSE_GPU_SPEC_SET_VAL)

#undef DEFINE_SPARSE_GPU_SPEC
#undef DEFINE_SPARSE_GPU_SPEC_SET_VAL
#undef DEFINE_SPARSE_GPU_SPEC_MAX_SEG_ID
#undef DEFINE_SPARSE_GPU_SPEC_GRAD_REDUCE
#undef DEFINE_SPARSE_GPU_SPEC_REDUCE

template <typename T>
void SetValueDefault<T>::operator()(OpKernelContext* ctx,
                                    Tensor* target,
                                    T default_val) {
  const GPUDevice d = ctx->eigen_device<GPUDevice>();
  GpuLaunchConfig config = GetGpuLaunchConfig(target->NumElements(), d);
  SetToValue<<<config.block_count, config.thread_per_block,
      0, d.stream()>>>(static_cast<int>(target->NumElements()),
      target->flat<T>().data(), default_val);
}

template <typename Index>
void FindMaxSegId<Index>::operator()(OpKernelContext* ctx,
                                     const Tensor* seg_ids,
                                     Index& max_id) {
  AllocatorAttributes attr;
  attr.set_on_host(true);
  attr.set_gpu_compatible(true);
  Tensor seg_ids_cpu;
  ctx->allocate_temp(seg_ids->dtype(), TensorShape{seg_ids->NumElements()},
                         &seg_ids_cpu, attr);
  if (!ctx->status().ok()) {
    return;
  }
  auto dst_vec = seg_ids_cpu.flat<Index>();
  auto src_vec = seg_ids->flat<Index>();

  // copy data
  ctx->eigen_gpu_device().memcpy(dst_vec.data(),
       src_vec.data(), seg_ids->NumElements()*sizeof(Index));
  ctx->eigen_gpu_device().synchronize();
  // find the max
  max_id = *std::max_element(dst_vec.data(),
                             dst_vec.data() + seg_ids->NumElements());
}

} // end of namespace functor

template <class T, typename Index>
class SparseSegmentReductionGpuOpBase : public OpKernel {
 public:
  explicit SparseSegmentReductionGpuOpBase(OpKernelConstruction* context,
                                           bool is_mean, bool is_sqrtn,
                                           bool has_num_segments,
                                           T default_value)
      : OpKernel(context),
        is_mean_(is_mean),
        is_sqrtn_(is_sqrtn),
        has_num_segments_(has_num_segments),
        default_value_(default_value) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& segment_ids = context->input(2);

    int32 output_rows = -1;
    if (has_num_segments_) {
      const Tensor& num_segments = context->input(3);
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

    if (segment_ids.NumElements() == 0) {
      TensorShape output_shape = input.shape();
      functor::SetValueDefault<T> setval_functor_;
      output_shape.set_dim(0, output_rows < 0 ? 0: output_rows);
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(
        0, output_shape, &output));
      if (output_rows > 0) {
        setval_functor_(context, output, default_value_);
      }
      return;
    }

    typedef int32 OutputRow;
    const auto segment_vec = segment_ids.vec<OutputRow>();
    int32 max_id = 0;
    functor::FindMaxSegId<OutputRow> find_max_seg_functor_;
    find_max_seg_functor_(context, &segment_ids, max_id);

    const OutputRow last_segment_id_plus_one = max_id + 1;

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

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    // set default value to output
    functor::SparseSegmentReduceFunctor<T, Index> reduction_functor_;
    reduction_functor_(context, &input, &indices,
                       &segment_ids, output, is_mean_, is_sqrtn_);
  }

 private:
  const bool is_mean_;
  const bool is_sqrtn_;
  const bool has_num_segments_;
  const T default_value_;
};

template <class T, typename Index>
class SparseSegmentReductionSumGpuOp
    : public SparseSegmentReductionGpuOpBase<T, Index> {
 public:
  explicit SparseSegmentReductionSumGpuOp(OpKernelConstruction* context)
      : SparseSegmentReductionGpuOpBase<T, Index>(
            context, false /*is_mean*/, false /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <class T, typename Index>
class SparseSegmentReductionSumWithNumSegmentsGpuOp
    : public SparseSegmentReductionGpuOpBase<T, Index> {
 public:
  explicit SparseSegmentReductionSumWithNumSegmentsGpuOp(
    OpKernelConstruction* context)
      : SparseSegmentReductionGpuOpBase<T, Index>(
            context, false /*is_mean*/, false /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

template <class T, typename Index>
class SparseSegmentReductionMeanGpuOp
    : public SparseSegmentReductionGpuOpBase<T, Index> {
 public:
  explicit SparseSegmentReductionMeanGpuOp(OpKernelConstruction* context)
      : SparseSegmentReductionGpuOpBase<T, Index>(
            context, true /*is_mean*/, false /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <class T, typename Index>
class SparseSegmentReductionMeanWithNumSegmentsGpuOp
    : public SparseSegmentReductionGpuOpBase<T, Index> {
 public:
  explicit SparseSegmentReductionMeanWithNumSegmentsGpuOp(
    OpKernelConstruction* context)
      : SparseSegmentReductionGpuOpBase<T, Index>(
            context, true /*is_mean*/, false /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

template <class T, typename Index>
class SparseSegmentReductionSqrtNGpuOp
    : public SparseSegmentReductionGpuOpBase<T, Index> {
 public:
  explicit SparseSegmentReductionSqrtNGpuOp(OpKernelConstruction* context)
      : SparseSegmentReductionGpuOpBase<T, Index>(
            context, false /*is_mean*/, true /*is_sqrtn*/,
            false /* has_num_segments */, T(0) /* default_value */) {}
};

template <class T, typename Index>
class SparseSegmentReductionSqrtNWithNumSegmentsGpuOp
    : public SparseSegmentReductionGpuOpBase<T, Index> {
 public:
  explicit SparseSegmentReductionSqrtNWithNumSegmentsGpuOp(
    OpKernelConstruction* context)
      : SparseSegmentReductionGpuOpBase<T, Index>(
            context, false /*is_mean*/, true /*is_sqrtn*/,
            true /* has_num_segments */, T(0) /* default_value */) {}
};

#define REGISTER_GPU_SPARSE_KERNELS(type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentSum")                       \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<index_type>("Tidx"),       \
                          SparseSegmentReductionSumGpuOp<type, index_type>); \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentSumWithNumSegments")        \
        .Device(DEVICE_GPU)                                              \
        .HostMemory("num_segments")                                      \
        .TypeConstraint<type>("T")                                       \
        .TypeConstraint<index_type>("Tidx"),                             \
    SparseSegmentReductionSumWithNumSegmentsGpuOp<type, index_type>);    \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentMean")                      \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<index_type>("Tidx"),       \
                          SparseSegmentReductionMeanGpuOp<type, index_type>); \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentMeanWithNumSegments")        \
        .Device(DEVICE_GPU)                                              \
        .HostMemory("num_segments")                                      \
        .TypeConstraint<type>("T")                                       \
        .TypeConstraint<index_type>("Tidx"),                             \
    SparseSegmentReductionMeanWithNumSegmentsGpuOp<type, index_type>);   \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentSqrtN")                     \
                              .Device(DEVICE_GPU)                        \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<index_type>("Tidx"),       \
                          SparseSegmentReductionSqrtNGpuOp<type, index_type>); \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentSqrtNWithNumSegments")      \
        .Device(DEVICE_GPU)                                              \
        .HostMemory("num_segments")                                      \
        .TypeConstraint<type>("T")                                       \
        .TypeConstraint<index_type>("Tidx"),                             \
    SparseSegmentReductionSqrtNWithNumSegmentsGpuOp<type, index_type>);

REGISTER_GPU_SPARSE_KERNELS(float, int64)
REGISTER_GPU_SPARSE_KERNELS(float, int32)
REGISTER_GPU_SPARSE_KERNELS(double, int32)
REGISTER_GPU_SPARSE_KERNELS(double, int64)
#undef REGISTER_GPU_SPARSE_KERNELS

template <class T, typename Index>
class SparseSegmentGradGpuOpBase : public OpKernel {
 public:
  explicit SparseSegmentGradGpuOpBase(OpKernelConstruction* context,
                                      bool is_sqrtn)
      : OpKernel(context), is_sqrtn_(is_sqrtn) {}

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

    const int64 N = indices.NumElements();
    OP_REQUIRES(context, N == segment_ids.NumElements(),
                errors::InvalidArgument(
                    "segment_ids and indices should have same size."));
    typedef int32 SegmentId;
    SegmentId M = internal::SubtleMustCopy(output_dim0.scalar<SegmentId>()());
    // allocate output tensor
    Tensor* output = nullptr;
    TensorShape output_shape = input.shape();
    output_shape.set_dim(0, M);
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (M == 0 || N == 0) return;
    // invoke gpu functors
    functor::SparseSegmentReduceGradFunctor<T, Index> reduction_grad_functor_;
    reduction_grad_functor_(context, &input, &indices,
                            &segment_ids, output, is_sqrtn_);
  }

 private:
  const bool is_sqrtn_;
};

template <class T, typename Index>
class SparseSegmentMeanGradGpuOp :
  public SparseSegmentGradGpuOpBase<T, Index> {
 public:
  explicit SparseSegmentMeanGradGpuOp(OpKernelConstruction* context)
      : SparseSegmentGradGpuOpBase<T, Index>(context, false /*is_sqrtn*/) {}
};

template <class T, typename Index>
class SparseSegmentSqrtNGradGpuOp :
  public SparseSegmentGradGpuOpBase<T, Index> {
 public:
  explicit SparseSegmentSqrtNGradGpuOp(OpKernelConstruction* context)
      : SparseSegmentGradGpuOpBase<T, Index>(context, true /*is_sqrtn*/) {}
};

#define REGISTER_GPU_SPARSE_KERNELS(type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentMeanGrad")                  \
                              .Device(DEVICE_GPU)                        \
                              .HostMemory("output_dim0")                 \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<index_type>("Tidx"),       \
                          SparseSegmentMeanGradGpuOp<type, index_type>); \
  REGISTER_KERNEL_BUILDER(Name("SparseSegmentSqrtNGrad")                 \
                              .Device(DEVICE_GPU)                        \
                              .HostMemory("output_dim0")                 \
                              .TypeConstraint<type>("T")                 \
                              .TypeConstraint<index_type>("Tidx"),       \
                          SparseSegmentSqrtNGradGpuOp<type, index_type>);
REGISTER_GPU_SPARSE_KERNELS(float, int32)
REGISTER_GPU_SPARSE_KERNELS(float, int64)
REGISTER_GPU_SPARSE_KERNELS(double, int32)
REGISTER_GPU_SPARSE_KERNELS(double, int64)
#undef REGISTER_GPU_SPARSE_KERNELS

/*---------------------------- SparseSegment End ------------------------------*/

namespace functor {



}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
