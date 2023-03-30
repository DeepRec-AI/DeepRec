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

#if GOOGLE_CUDA //|| TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

// We need to include gpu_kernel_helper.h before segment_reduction_ops.h
// See comment in segment_reduction_ops.h for more details.
// clang-format off
#include "tensorflow/core/util/gpu_kernel_helper.h"
// clang-format on

#include "tensorflow/core/kernels/segment_reduction_ops.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_device_functions.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

// SortedSegmentSumFunctor kernel reduces input data just as
// UnsortedSegmentSumCustomKernel does except that input data
// is partitioned along the outer reduction dimension. This is
// because consecutive rows (elements in a row share the same
// outer dimension index) in the flattened 2D input data likely
// belong to the same segment in sorted segment sum operation.
// Therefore such partitioning strategy has two advantages over
// the UnsortedSegmentSumFunctor kernel:
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
template <typename T, typename Index, int OuterDimTileSize>
__global__ void SortedSegmentSumCustomKernel(const Index input_outer_dim_size,
                                             const Index inner_dim_size,
                                             const Index output_outer_dim_size,
                                             const Index* segment_ids,
                                             const T* input, T* output,
                                             const Index total_stripe_count) {
  for (int stripe_index : GpuGridRangeX(total_stripe_count)) {
    const Index segment_offset = stripe_index % inner_dim_size;
    const Index input_outer_dim_index_base =
        stripe_index / inner_dim_size * Index(OuterDimTileSize);

    T sum = T(0);
    Index first_segment_id = segment_ids[input_outer_dim_index_base];
    Index last_output_segment_id = output_outer_dim_size;

    const Index actual_stripe_height =
        min(Index(OuterDimTileSize),
            input_outer_dim_size - input_outer_dim_index_base);
    for (Index j = 0; j < actual_stripe_height; j++) {
      Index current_output_segment_id =
          segment_ids[input_outer_dim_index_base + j];
      // Decide whether to write result to global memory.
      // Result is only written to global memory if we move
      // to another segment. Otherwise we can keep accumulating
      // locally.
      if (current_output_segment_id > last_output_segment_id) {
        const Index output_index =
            last_output_segment_id * inner_dim_size + segment_offset;
        // decide whether to write result to global memory using atomic
        // operations
        if (last_output_segment_id == first_segment_id) {
          GpuAtomicAdd(output + output_index, sum);
        } else {
          *(output + output_index) = sum;
        }
        sum = T(0);
      }
      sum += ldg(input + (input_outer_dim_index_base + j) * inner_dim_size +
                 segment_offset);
      last_output_segment_id = current_output_segment_id;
    }
    // For the last result in a strip, always write using atomic operations
    // due to possible race conditions with threads computing
    // the following strip.
    const Index output_index =
        last_output_segment_id * inner_dim_size + segment_offset;
    GpuAtomicAdd(output + output_index, sum);
  }
}

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
    functor::SumOpGpu<T>()(output + output_idx, ldg(data + data_idx));
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
    functor::SumOpGpu<T>()(output + output_idx, ldg(data + data_idx) * scale);
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
    functor::SumOpGpu<T>()(seg_lens + output_idx, T(1));
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

template <typename T, typename Index>
__global__ void SparseSegmentGradSqrtNKernel(const int64 data_sparse_size,
                                             const int64 output_total_size,
                                             const int64 data_inner_dim,
                                             const T* seg_lens,
                                             const int32* seg_ids,
                                             const Index* indices,
                                             T* output) {
  for (int input_index : GpuGridRangeX(data_sparse_size)) {
    const Index sparse_row = input_index / data_inner_dim;
    const Index sparse_offset = input_index % data_inner_dim;
    const Index seg_idx = seg_ids[sparse_row];
    const Index output_row = indices[sparse_row];
    const Index output_idx = output_row * data_inner_dim + sparse_offset;
    if (output_idx < 0 || output_idx >= output_total_size) {
      continue;
    }
    T count = ldg(seg_lens + seg_idx);
    output[output_idx] /= std::sqrt(count > 0 ? count : 1);
  }
}

template <typename T, typename Index>
__global__ void SparseSegmentGradMeanKernel(const int64 data_sparse_size,
                                            const int64 output_total_size,
                                            const int64 data_inner_dim,
                                            const T* seg_lens,
                                            const int32* seg_ids,
                                            const Index* indices,
                                            T* output) {
  for (int input_index : GpuGridRangeX(data_sparse_size)) {
    const Index sparse_row = input_index / data_inner_dim;
    const Index sparse_offset = input_index % data_inner_dim;
    const Index seg_idx = seg_ids[sparse_row];
    const Index output_row = indices[sparse_row];
    const Index output_idx = output_row * data_inner_dim + sparse_offset;
    if (output_idx < 0 || output_idx >= output_total_size) {
      continue;
    }
    T count = ldg(seg_lens + seg_idx);
    output[output_idx] /= (count > 0 ? count : 1);
  }
}

namespace functor {

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

template <typename T, typename Index>
void SegmentSumFunctor<T, Index>::operator()(
    OpKernelContext* ctx, const GPUDevice& d, const Index output_rows,
    const TensorShape& segment_ids_shape,
    typename TTypes<Index>::ConstFlat segment_ids, const Index data_size,
    const T* data, typename TTypes<T, 2>::Tensor output) {
  if (output.size() == 0) {
    return;
  }
  // Set 'output' to zeros.
  GpuLaunchConfig config = GetGpuLaunchConfig(output.size(), d);
  TF_CHECK_OK(GpuLaunchKernel(SetZero<T>, config.block_count,
                              config.thread_per_block, 0, d.stream(),
                              output.size(), output.data()));
  if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
    return;
  }

  // Launch kernel to compute sorted segment sum.
  // Notes:
  // *) 'input_total_size' is the total number of elements to process.
  // *) 'segment_ids.shape' is a prefix of data's shape.
  // *) 'input_outer_dim_size' is the total number of segments to process.
  const Index input_total_size = data_size;
  const Index input_outer_dim_size = segment_ids.dimension(0);
  const Index input_inner_dim_size = input_total_size / input_outer_dim_size;

  const int OuterDimTileSize = 8;

  const Index input_outer_dim_num_stripe =
      Eigen::divup(input_outer_dim_size, Index(OuterDimTileSize));

  const Index total_stripe_count =
      input_inner_dim_size * input_outer_dim_num_stripe;

  config = GetGpuLaunchConfig(total_stripe_count, d);
  TF_CHECK_OK(GpuLaunchKernel(
      SortedSegmentSumCustomKernel<T, Index, OuterDimTileSize>,
      config.block_count, config.thread_per_block, 0, d.stream(),
      input_outer_dim_size, input_inner_dim_size, output_rows,
      segment_ids.data(), data, output.data(), total_stripe_count));
}

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

#define DEFINE_SORTED_GPU_SPECS_INDEX(T, Index) \
  template struct SegmentSumFunctor<T, Index>

#define DEFINE_SORTED_GPU_SPECS(T)         \
  DEFINE_SORTED_GPU_SPECS_INDEX(T, int32); \
  DEFINE_SORTED_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_SORTED_GPU_SPECS);

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

#define DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, Index)                         \
  template struct UnsortedSegmentFunctor<                                      \
      GPUDevice, T, Index, functor::Lowest<T>, functor::MaxOpGpu<T>>;          \
  template struct UnsortedSegmentFunctor<                                      \
      GPUDevice, T, Index, functor::Highest<T>, functor::MinOpGpu<T>>;         \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index, functor::One<T>, \
                                         functor::ProdOpGpu<T>>;

// sum is the only op that supports all input types currently
#define DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, Index) \
  template struct UnsortedSegmentFunctor<             \
      GPUDevice, T, Index, functor::Zero<T>, functor::SumOpGpu<T>>;

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

#undef DEFINE_SORTED_GPU_SPECS_INDEX
#undef DEFINE_SORTED_GPU_SPECS
#undef DEFINE_SPARSE_GPU_SPEC_REDUCE
#undef DEFINE_SPARSE_GPU_SPEC_SET_VAL
#undef DEFINE_SPARSE_GPU_SPEC_MAX_SEG_ID
#undef DEFINE_SPARSE_GPU_SPEC
#undef DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX
#undef DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX
#undef DEFINE_REAL_GPU_SPECS
#undef DEFINE_SUM_GPU_SPECS

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
